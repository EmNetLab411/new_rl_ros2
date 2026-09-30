#!/usr/bin/env python3
"""
run_pi4_ros2.py  —  ROS2 Node chạy trên Raspberry Pi 4 sử dụng ONNX Runtime
=============================================================================
[Phase 1 — PLAN.md] Bắt camera TRỰC TIẾP bằng OpenCV/MJPEG trong cùng tiến
trình (không qua usb_cam → DDS → cv_bridge nữa) → ONNX inference → solvePnP
→ publish kết quả. Luồng cũ 3 chặng riêng (usb_cam node, DDS, cv_bridge
decode) mỗi chặng xếp hàng riêng, cộng dồn thành 500-1000ms độ trễ dù bản
thân model chỉ mất ~10ms. Bây giờ 1 thread nền liên tục đọc camera, luôn
giữ ĐÚNG 1 frame mới nhất (ghi đè, không xếp hàng); vòng xử lý chính lấy
frame mới nhất đó ngay khi rảnh.

Topics:
  Publish   : /aeroscript/pen_image (sensor_msgs/Image)  — frame annotated (debug, web_video_server)
  Publish   : /aeroscript/pen_xyz   (geometry_msgs/Point)— tọa độ 3D (mm)
  Publish   : /aeroscript/pen_pose  (geometry_msgs/PoseStamped) — đầy đủ

Cài đặt trên Pi 4:
    pip3 install "numpy<2" opencv-python onnxruntime

Cách chạy (không cần chạy usb_cam_node_exe nữa — script này tự mở camera):
    python3 run_pi4_ros2.py --device /dev/video0 --model best.onnx
    python3 run_pi4_ros2.py --device /dev/video0 --model best.onnx --focus 0 --exposure 150
Vẫn chạy `ros2 run web_video_server web_video_server` ở terminal khác nếu muốn xem debug qua trình duyệt.

────────────────────────────────────────────────────────────────────────────
PHÂN TÍCH PIPELINE — bottleneck & song song hoá
────────────────────────────────────────────────────────────────────────────
Chuỗi xử lý mỗi frame trong _process_frame():

    1. CvBridge decode          (~1-2ms)
    2. ONNX preprocess          (~3-5ms)
    3. session.run() inference  (~XX ms)  ← BOTTLENECK THỰC SỰ, không
                                            song song hoá được vì bước 6
                                            (solvePnP) PHỤ THUỘC TRỰC TIẾP
                                            vào output của bước này.
    4. decode + NMS              (~1ms)
    5. KeypointKalmanFilter      (~0.1ms)
    6. solvePnP                  (~1-2ms)  ← phải chờ bước 3 xong
    7. PoseKalmanFilter          (~0.1ms)
    8. Vẽ overlay lên frame      (~2-3ms)
    9. Publish XYZ + PoseStamped (~0.5ms)
   10. Encode + publish ảnh      (~5-8ms)  ← KHÔNG phụ thuộc bước 6-7,
                                            chỉ cần frame có sẵn → tách
                                            ra thread riêng (xem dưới).

Hai chỗ ĐÃ tối ưu được (không đụng vào bottleneck #3, nhưng giảm latency
tổng thể của hệ thống):

  (a) [Phase 1] FrameGrabber đọc camera trực tiếp bằng OpenCV, KHÔNG qua
      usb_cam → DDS → cv_bridge nữa. Trước đây mỗi chặng đó xếp hàng
      riêng, cộng dồn thành 500-1000ms dù model chỉ mất ~10ms. Giờ 1
      thread nền liên tục đọc camera, chỉ giữ ĐÚNG 1 frame mới nhất (ghi
      đè, không xếp hàng); _on_timer() lấy frame đó ngay khi rảnh — cùng
      hiệu quả "luôn xử lý frame mới nhất" như QoS BEST_EFFORT+depth=1
      làm trước đây, nhưng bỏ hẳn lớp trung gian DDS/cv_bridge.

  (b) Tách bước 10 (encode + publish ảnh debug) ra ThreadPoolExecutor:
      Bước này không ảnh hưởng đến bước 6-9 (đã publish XYZ xong trước
      đó), nên không có lý do gì để nó chặn callback nhận frame tiếp
      theo. Tiết kiệm ~5-8ms/frame trên critical path.

Bottleneck #3 chỉ tấn công được bằng:
  - Giảm imgsz (320 → 256 hoặc 224) — giảm trực tiếp FLOPs.
  - Dùng INT8 thay FP32 — ORT có kernel INT8 tối ưu hơn cho ARM NEON.
  - KHÔNG có cách "song song hoá" giả vì input bước 6 là output bước 3.
────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import subprocess
import sys
import time
import argparse
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

import onnxruntime as ort


# ══════════════════════════════════════════════════════════════════════════════
# 0. Frame Grabber — đọc camera trong thread riêng, luôn giữ frame mới nhất
# ══════════════════════════════════════════════════════════════════════════════
class FrameGrabber(threading.Thread):
    """
    Thay cho usb_cam → DDS → cv_bridge: đọc camera trực tiếp bằng OpenCV
    trên thread riêng, chỉ giữ ĐÚNG 1 frame mới nhất (ghi đè frame cũ,
    không xếp hàng). Vòng xử lý chính chỉ cần hỏi get_latest() để lấy frame
    mới nhất tại thời điểm nó rảnh — đúng cơ chế "drop frame cũ" mà QoS
    BEST_EFFORT trước đây làm ở tầng DDS, nhưng giờ làm ngay tại nguồn, bỏ
    hẳn 1 lớp trung gian.

    Vòng lặp tách grab() (lấy frame, rẻ) khỏi retrieve() (giải mã JPEG,
    đắt) và CHỈ giải mã khi vòng chính báo cần frame mới qua request_next()
    — xem chi tiết lý do trong run().
    """
    def __init__(self, device, width=1280, height=720, fourcc="MJPG",
                 focus=None, exposure=None):
        super().__init__(daemon=True)
        idx = int(device) if str(device).isdigit() else device
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if focus is not None:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, focus)
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được camera {device}")

        self._lock = threading.Lock()
        self._frame = None
        self._frame_t = 0.0
        self._frame_id = 0
        self._stop = False

        # Chỉ GIẢI MÃ frame khi vòng xử lý chính thật sự cần — xem run().
        # Đặt sẵn = True để frame đầu tiên được giải mã ngay lúc khởi động.
        self._want = threading.Event()
        self._want.set()

        # Đo riêng thời gian retrieve() (giải mã MJPEG bằng CPU qua
        # libjpeg-turbo) — KHÔNG nằm trong _record_timing của model
        # (pre/invoke/decode) vì chạy ở thread này. Cần số này để biết có
        # đáng offload sang hardware JPEG decode (bcm2835-codec qua
        # GStreamer) hay không — việc đó tốn công dựng, không đáng làm nếu
        # bản thân giải mã vốn đã rẻ.
        self._dec_n = 0
        self._dec_sum = 0.0

    def request_next(self):
        """Vòng xử lý chính gọi hàm này NGAY KHI lấy frame ra để xử lý —
        báo cho thread nền biết được phép giải mã frame kế tiếp. Nhờ vậy
        việc giải mã frame N+1 chạy SONG SONG với việc xử lý frame N, mà
        vẫn chỉ tốn đúng 1 lần giải mã cho mỗi frame thật sự dùng tới."""
        self._want.set()

    def run(self):
        while not self._stop:
            # grab() chỉ lấy frame từ driver V4L2, KHÔNG giải mã JPEG — rất
            # rẻ, và giữ cho hàng đợi driver luôn được rút cạn nên frame lấy
            # ra luôn là mới nhất (độ trễ thấp). retrieve() mới là bước giải
            # mã tốn CPU thật sự.
            #
            # Trước đây dùng read() = grab()+retrieve() chạy hết tốc độ
            # camera (~30fps) trong khi vòng xử lý chỉ tiêu thụ ~5 frame/giây
            # → ~25 lần giải mã JPEG mỗi giây bị vứt đi không ai dùng, đốt
            # CPU và băng thông bộ nhớ mà XNNPACK đang cần (đo được: invoke
            # 176ms trong pipeline thật vs 116ms khi benchmark cô lập).
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            # Mốc "frame tới tay chương trình" để đo E2E latency. Chưa gồm
            # thời gian phơi sáng + nén JPEG + truyền USB bên trong camera
            # (webcam thường thêm ~30-60ms, phần mềm không đo được).
            t_grab = time.perf_counter()
            if not self._want.is_set():
                continue

            t0 = time.perf_counter()
            ok, frame = self.cap.retrieve()
            t1 = time.perf_counter()
            if not ok:
                continue

            with self._lock:
                self._frame = frame
                self._frame_t = t_grab
                self._frame_id += 1
            self._want.clear()

            self._dec_sum += (t1 - t0)
            self._dec_n += 1
            if self._dec_n % 30 == 0:
                print(f"📷 [{self._dec_n} decode] retrieve() (giải mã MJPEG) "
                      f"trung bình={self._dec_sum/self._dec_n*1000:.1f}ms")

    def get_latest(self):
        """(frame, frame_id, thời điểm grab theo time.perf_counter())."""
        with self._lock:
            return self._frame, self._frame_id, self._frame_t

    def stop(self):
        self._stop = True
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pose Inference Engine — 2 backend: ONNX Runtime (mặc định cũ) và TFLite
#    (mới, nhanh hơn ONNX ~25-30% trên Pi 4 — XNNPACK tối ưu tốt hơn cho
#    ARM Cortex-A72 với model conv nhỏ này; xem PLAN.md mục 5b để biết cách
#    đo đúng — so cùng điều kiện, không đem số đo cô lập so với số đo trong
#    pipeline thật). Chọn backend TỰ ĐỘNG theo đuôi
#    file --model (.onnx -> ONNX Runtime, .tflite -> TFLite), không cần cờ
#    riêng. Phần preprocess/decode/NMS giống hệt nhau giữa 2 backend (cùng
#    xuất từ 1 model gốc, cùng layout output YOLOv8-pose) nên gộp chung vào
#    _PoseBackendBase, mỗi backend chỉ khác nhau ở cách load model + gọi
#    inference.
# ══════════════════════════════════════════════════════════════════════════════
class _PoseBackendBase:
    """Preprocess/decode/NMS dùng chung cho mọi backend — KHÔNG phụ thuộc
    ONNX hay TFLite cụ thể, chỉ cần self.imgsz/self.is_nhwc/self._np_dtype
    đã được backend con set đúng trong __init__."""

    def _init_preproc_buffers(self):
        """Cấp phát TRƯỚC mọi buffer mà _preprocess dùng lại ở từng frame.

        Bản cũ cấp phát ~2.7MB MỖI FRAME (canvas 300KB + .astype() 1.2MB +
        phép chia /255.0 tạo thêm 1.2MB nữa) và dùng slice stride âm
        `[:, :, ::-1]` để đổi BGR→RGB — buộc numpy copy theo chiều ngược,
        rất chậm. Đo thật trên Pi 4: 28ms/frame cho việc lẽ ra chỉ ~8ms.
        Giờ mọi thứ ghi thẳng vào buffer có sẵn, không cấp phát gì trong
        vòng lặp, và đổi màu bằng cv2.cvtColor (SIMD, C++).
        """
        s = self.imgsz
        self._canvas    = np.full((s, s, 3), 114, np.uint8)
        self._rgb_u8    = np.empty((s, s, 3), np.uint8)
        self._buf_f32   = np.empty((s, s, 3), self._np_dtype)
        self._inv255    = self._np_dtype(1.0 / 255.0)
        self._last_geom = None
        # NHWC: tensor chỉ là view của _buf_f32 (không tốn gì). NCHW: cần
        # đảo trục nên phải có buffer contiguous riêng.
        self._tensor_nchw = (None if self.is_nhwc
                             else np.empty((1, 3, s, s), self._np_dtype))

    def _preprocess(self, bgr: np.ndarray):
        h0, w0 = bgr.shape[:2]
        r  = self.imgsz / max(h0, w0)
        nw, nh = int(w0 * r), int(h0 * r)
        dw, dh = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2

        # Chỉ tô lại nền xám khi hình dạng ảnh vào thay đổi — với camera cố
        # định thì việc này chạy đúng 1 lần ở frame đầu tiên.
        geom = (nw, nh, dw, dh)
        if geom != self._last_geom:
            self._canvas[:] = 114
            self._last_geom = geom

        cv2.resize(bgr, (nw, nh), dst=self._canvas[dh:dh+nh, dw:dw+nw],
                   interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB, dst=self._rgb_u8)
        np.multiply(self._rgb_u8, self._inv255, out=self._buf_f32,
                    casting="unsafe")

        if self.is_nhwc:
            tensor = self._buf_f32[np.newaxis]
        else:
            np.copyto(self._tensor_nchw[0], self._buf_f32.transpose(2, 0, 1))
            tensor = self._tensor_nchw
        return tensor, r, dw, dh

    def _decode(self, preds, r, dw, dh):
        # Layout: [cx,cy,w,h, conf, kp0x,kp0y,kp0v, ..., kp3v] — 17 fields, không có cls
        if preds.shape[1] < 17:
            return None, None
        # Ghi lại độ tin cậy cao nhất mỗi cửa sổ 30 frame (in ở _record_timing)
        # — phân biệt "model không thấy gì" với "thấy nhưng dưới ngưỡng conf".
        self._win_max_conf = max(getattr(self, "_win_max_conf", 0.0),
                                 float(preds[:, 4].max()))
        mask = preds[:, 4] > self.conf_thresh
        if not np.any(mask):
            return None, None
        f    = preds[mask]
        keep = self._nms(f[:, :4], f[:, 4], self.iou_thresh)
        if not keep:
            return None, None
        best = f[keep[0]]
        kraw = best[5:17].reshape(4, 3)
        kxy  = kraw[:, :2].copy()
        kv   = kraw[:, 2]
        kxy[:, 0] = (kxy[:, 0] - dw) / r
        kxy[:, 1] = (kxy[:, 1] - dh) / r
        return kxy.astype(np.float32), kv.astype(np.float32)

    def _record_timing(self, t_pre: float, t_infer: float, t_dec: float):
        """Đo riêng từng giai đoạn (preprocess/invoke/decode) NGAY TRONG
        pipeline thật đang chạy trên Pi — benchmark_tflite.py chỉ đo invoke()
        cô lập (không FrameGrabber/ROS/publish-thread chạy cùng), nên không
        biết phần chênh lệch giữa 116ms benchmark và ~190-210ms thực đo qua
        FPS node nằm ở giai đoạn nào. In log mỗi 30 frame thay vì mỗi frame
        để không tự làm chậm thêm bằng chi phí print()."""
        s = self._timing_sum
        s["pre"] += t_pre; s["infer"] += t_infer; s["dec"] += t_dec
        self._timing_n += 1
        if self._timing_n % 30 == 0:
            n = self._timing_n
            total = s["pre"] + s["infer"] + s["dec"]
            print(f"⏱️  [{n} frames] pre={s['pre']/n*1000:.1f}ms  "
                  f"invoke={s['infer']/n*1000:.1f}ms  decode={s['dec']/n*1000:.1f}ms  "
                  f"tổng={total/n*1000:.1f}ms (~{n/total:.1f} fps thuần model)  "
                  f"conf_max={getattr(self, '_win_max_conf', 0.0):.2f} (ngưỡng {self.conf_thresh})")
            self._win_max_conf = 0.0

    @staticmethod
    def _nms(boxes, scores, thr):
        if not len(boxes):
            return []
        cx, cy, bw, bh = boxes.T
        x1, y1, x2, y2 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
        areas = bw * bh
        order = scores.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(int(i))
            if order.size == 1:
                break
            ix1 = np.maximum(x1[i], x1[order[1:]]); iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]]); iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= thr]
        return keep


class ONNXPoseInference(_PoseBackendBase):
    """
    Lưu ý hiệu năng: dtype đích (FP32 hoặc FP16) được xác định MỘT LẦN ở
    __init__ và lưu vào self._np_dtype — không kiểm tra lại mỗi frame.
    Bản gốc kiểm tra "float16" in inp_meta.type bên trong __call__() mỗi
    lần gọi, tốn một lần string-compare + một lần allocate array thừa.
    Trên Pi 4 con số này nhỏ (<0.1ms) nhưng không có lý do để lặp lại
    việc đã biết kết quả từ lúc load model.
    """
    def __init__(self, model_path: str, conf_thresh: float = 0.55,
                 iou_thresh: float = 0.45, n_threads: int = 4):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

        opts = ort.SessionOptions()
        # QUAN TRỌNG trên Pi 4 (chỉ 4 core tổng): không nên để 1 lần inference
        # chiếm hết tất cả core. usb_cam, web_video_server, và ROS2 executor
        # của chính node này đều cần CPU để chạy đồng thời. Nếu intra_op=4,
        # mỗi lần session.run() sẽ "khóa" toàn bộ 4 core trong suốt thời gian
        # inference, khiến các tiến trình khác phải context-switch liên tục
        # — đây chính là nguyên nhân Load average > số core đang chạy.
        # Với model nhỏ như yolov8n-pose (320x320), 2 threads thường nhanh
        # GẦN BẰNG 4 threads vì overhead đồng bộ giữa thread đã chiếm phần
        # lớn lợi ích, nhưng để lại 2 core "thở" cho phần còn lại của hệ thống.
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1     # chỉ có 1 graph, không cần song song giữa op-group
        opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"⏳ Khởi tạo ONNX Runtime: {model_path}")
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        inp_meta  = self.session.get_inputs()[0]
        inp_shape = inp_meta.shape
        inp_type  = inp_meta.type   # ví dụ "tensor(float)" hoặc "tensor(float16)"

        if inp_shape[1] == 3:        # NCHW
            self.imgsz   = int(inp_shape[2])
            self.is_nhwc = False
        else:                        # NHWC
            self.imgsz   = int(inp_shape[1])
            self.is_nhwc = True

        # Xác định dtype MỘT LẦN — tránh string-compare mỗi frame
        self._np_dtype = np.float16 if "float16" in inp_type else np.float32

        print(f"✅ ONNX Ready: {inp_shape}  {inp_type}  "
              f"{'NHWC' if self.is_nhwc else 'NCHW'}  threads={n_threads}")

        self._init_preproc_buffers()
        self._timing_n   = 0
        self._timing_sum = {"pre": 0.0, "infer": 0.0, "dec": 0.0}

    def __call__(self, bgr: np.ndarray):
        t0 = time.perf_counter()
        tensor, r, dw, dh = self._preprocess(bgr)
        t1 = time.perf_counter()
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        t2 = time.perf_counter()
        preds = raw[0].T if raw.ndim == 3 else raw[0]
        result = self._decode(preds, r, dw, dh)
        t3 = time.perf_counter()
        self._record_timing(t1 - t0, t2 - t1, t3 - t2)
        return result


class TFLitePoseInference(_PoseBackendBase):
    """
    Backend TFLite + XNNPACK — benchmark thật trên Pi 4 (2026-09-23):
    nhanh hơn ONNX Runtime CPU EP khoảng 25-30% cho cùng model.
    (So cùng imgsz=320 threads=2: TFLite invoke 152ms ĐO TRONG pipeline
    thật — tức đã gánh tranh chấp CPU — vẫn thấp hơn ONNX 194.7ms đo CÔ LẬP.
    Con số "nhanh gấp 2x" ghi ở bản trước là SAI: nó đem TFLite đo cô lập
    so với ONNX đo cả pipeline, hai điều kiện khác nhau.)
    KHÔNG dùng bản .tflite float16 — chậm HƠN float32 trên Pi 4 vì
    Cortex-A72 không có phần cứng tính fp16 (ARMv8.2-FP16), phải quy đổi
    ngược sang fp32 lúc chạy, tốn thêm mà không lợi gì.

    TFLite export luôn ở dạng NHWC (khác ONNX có thể NCHW), và output có
    thể là (1,17,N) hoặc (1,N,17) tuỳ phiên bản export — _normalize_output
    tự phát hiện bằng cách tìm trục có kích thước đúng 17 (5 field box/conf
    + 4 keypoint x 3), không giả định cứng như code ONNX gốc.
    """
    def __init__(self, model_path: str, conf_thresh: float = 0.55,
                 iou_thresh: float = 0.45, n_threads: int = 4):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self._np_dtype   = np.float32

        print(f"⏳ Khởi tạo TFLite: {model_path}")
        self.interpreter = Interpreter(model_path=model_path, num_threads=n_threads)
        self.interpreter.allocate_tensors()
        self._inp = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]

        # Layout KHÔNG cố định: best_float32.tflite cũ là NHWC [1,H,W,3], còn
        # model train lại bằng ultralytics 8.4.x (litert) là NCHW [1,3,H,W].
        # Đọc cứng shape[1] làm imgsz sẽ ra imgsz=3 với bản NCHW.
        shape = [int(d) for d in self._inp["shape"]]
        if shape[1] == 3:               # NCHW
            self.is_nhwc = False
            self.imgsz   = shape[2]
        else:                           # NHWC
            self.is_nhwc = True
            self.imgsz   = shape[1]

        print(f"✅ TFLite Ready: {shape}  "
              f"{'NHWC' if self.is_nhwc else 'NCHW'}  threads={n_threads}")

        self._init_preproc_buffers()
        self._timing_n   = 0
        self._timing_sum = {"pre": 0.0, "infer": 0.0, "dec": 0.0}

    # Cột nào trong 17 field là TOẠ ĐỘ (cx,cy,w,h + kp0-3 x,y) — cần nhân lại
    # với imgsz. Cột còn lại (conf, kp0-3 visibility: index 4,7,10,13,16)
    # đã đúng thang [0,1] sẵn (confidence/objectness), KHÔNG được nhân.
    _COORD_COLS = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15]

    def __call__(self, bgr: np.ndarray):
        t0 = time.perf_counter()
        tensor, r, dw, dh = self._preprocess(bgr)
        t1 = time.perf_counter()
        self.interpreter.set_tensor(self._inp["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self._out["index"])
        t2 = time.perf_counter()
        preds = self._normalize_output(raw)
        result = self._decode(preds, r, dw, dh)
        t3 = time.perf_counter()
        self._record_timing(t1 - t0, t2 - t1, t3 - t2)
        return result

    def _normalize_output(self, raw):
        arr = raw[0] if raw.ndim == 3 else raw
        if arr.shape[0] == 17 and arr.shape[-1] != 17:
            arr = arr.T
        # PHÁT HIỆN THỰC TẾ (inspect_tflite.py, 2026-09-23): export TFLite
        # này xuất toạ độ box/keypoint đã CHUẨN HOÁ về [0,1] (tỉ lệ theo
        # imgsz), khác hẳn ONNX xuất thẳng pixel-space [0,320]. Nếu không
        # nhân lại đúng imgsz, _decode() dùng chung (viết cho quy ước ONNX)
        # sẽ hiểu sai toạ độ, gây sai lệch hàng trăm lần -> solvePnP ra vị
        # trí sai hàng chục mét (đúng lỗi đã gặp thực tế).
        arr = arr.copy()
        arr[:, self._COORD_COLS] *= self.imgsz
        return arr


def load_pose_model(model_path: str, conf_thresh: float, iou_thresh: float, n_threads: int):
    """Chọn backend TỰ ĐỘNG theo đuôi file — .tflite dùng TFLite (nhanh hơn
    ~25-30% trên Pi 4 theo benchmark thật, xem PLAN.md mục 5b), còn lại
    (.onnx) dùng ONNX Runtime như trước. Không cần cờ --backend riêng."""
    if str(model_path).lower().endswith(".tflite"):
        return TFLitePoseInference(model_path, conf_thresh, iou_thresh, n_threads)
    return ONNXPoseInference(model_path, conf_thresh, iou_thresh, n_threads)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Kalman Filter 3D
# ══════════════════════════════════════════════════════════════════════════════
class PoseKalmanFilter:
    """
    QUAN TRỌNG khi FPS thấp (~2.8 FPS, dt~357ms ở threads=1): transition
    matrix F[i, i+3] = dt phải dùng THỜI GIAN THẬT giữa 2 lần update(), không
    phải hằng số 1.0 cố định. Với dt cố định=1 nhưng FPS thực tế dao động
    (357ms khi không bị gì cản, nhưng có thể lên 500-700ms nếu lost frame),
    vận tốc ước lượng (vx,vy,vz) sẽ bị sai theo đúng tỷ lệ sai lệch dt giả
    định so với dt thật — dẫn đến dự đoán lệch hướng/lệch tốc độ khi mất
    tracking tạm thời và cần ngoại suy giữa các lần đo thưa.

    process_noise giảm xuống (so với mặc định cũ 0.5) để TIN TƯỞNG motion
    model nhiều hơn khi đo thưa — bù lại bằng việc dt giờ luôn chính xác,
    nên dự đoán vận tốc đáng tin cậy hơn để filter "lấp" khoảng trống giữa
    các lần đo 357ms.
    """
    def __init__(self, process_noise=0.15, measurement_noise=15.0, max_jump_mm=250.0):
        self.pn = process_noise; self.mn = measurement_noise; self.mj = max_jump_mm
        self.initialized = False; self.last = None; self._last_t = None
        self._init()

    def _init(self):
        self.kf = cv2.KalmanFilter(6, 3, 0)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)  # dt được set trong update()
        H = np.zeros((3,6), dtype=np.float32); H[0,0]=H[1,1]=H[2,2]=1.0
        self.kf.measurementMatrix = H
        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * self.pn
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.mn
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32) * 500.0

    def update(self, tvec, dt: float = None):
        """
        dt: thời gian (giây) thật giữa lần update() này và lần trước. Nếu
        None, ước lượng bằng time.time() nội bộ — luôn truyền dt từ ngoài
        khi có thể (ví dụ tính từ ROS2 message timestamp) để chính xác hơn.
        """
        now = time.time()
        m = np.array([tvec[0][0], tvec[1][0], tvec[2][0]], dtype=np.float32)

        if not self.initialized:
            self._init_at(m, now); return tvec.copy()
        if np.linalg.norm(m - self.last) > self.mj:
            self._init_at(m, now); return tvec.copy()

        if dt is None:
            dt = now - self._last_t
        dt = max(dt, 1e-3)  # tránh dt=0 hoặc âm gây ma trận suy biến

        # Cập nhật transition matrix với dt THẬT của lần gọi này — đây là
        # phần khác biệt cốt lõi so với bản cũ (dt cố định = 1.0)
        F = np.eye(6, dtype=np.float32)
        F[0,3] = F[1,4] = F[2,5] = dt
        self.kf.transitionMatrix = F

        # Process noise cũng cần scale theo dt — nhiễu tích lũy nhiều hơn
        # khi khoảng cách giữa 2 lần đo dài hơn (đúng theo lý thuyết Kalman
        # liên tục: Q ~ dt, không phải hằng số cố định bất kể dt).
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * (self.pn * dt)

        self.last = m.copy(); self._last_t = now
        self.kf.predict()
        return self.kf.correct(m.reshape(3,1))[:3,0].reshape(3,1)

    def _init_at(self, m, now):
        self._init(); s = np.zeros(6, dtype=np.float32); s[:3] = m
        self.kf.statePost = s.reshape(6,1)
        self.last = m.copy(); self._last_t = now; self.initialized = True

    def reset(self):
        self._init(); self.initialized = False; self.last = None; self._last_t = None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Kalman Filter 2D
# ══════════════════════════════════════════════════════════════════════════════
class KeypointKalmanFilter:
    """
    Cùng lý do dt động như PoseKalmanFilter ở trên. Với FPS thấp (~2.8),
    keypoint 2D cũng cần transition matrix scale theo dt thật, không phải
    hằng số 1.0 — nếu không, vận tốc pixel ước lượng sẽ sai lệch khi
    khoảng cách giữa 2 lần đo dao động (357ms bình thường, có thể dài hơn
    khi mất tracking tạm thời rồi bắt lại).
    """
    def __init__(self, process_noise=3e-3, measurement_noise=2.0):
        self.pn = process_noise; self.mn = measurement_noise
        self.initialized = False; self._last_t = None
        self._init()

    def _init(self):
        self.kf = cv2.KalmanFilter(16, 8, 0)
        self.kf.transitionMatrix = np.eye(16, dtype=np.float32)  # dt set trong update()
        self.kf.measurementMatrix = np.zeros((8,16), dtype=np.float32)
        for i in range(8):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov     = np.eye(16, dtype=np.float32) * self.pn
        self.kf.measurementNoiseCov = np.eye(8,  dtype=np.float32) * self.mn
        self.kf.errorCovPost        = np.eye(16, dtype=np.float32) * 100.0

    def update(self, kpts_4x2, dt: float = None):
        now = time.time()
        meas = kpts_4x2.flatten().astype(np.float32)

        if not self.initialized:
            s = np.zeros(16, dtype=np.float32); s[:8] = meas
            self.kf.statePost = s.reshape(16,1)
            self.initialized = True
            self._last_t = now
            return kpts_4x2.copy()

        if dt is None:
            dt = now - self._last_t
        dt = max(dt, 1e-3)

        F = np.eye(16, dtype=np.float32)
        for i in range(8):
            F[i, i+8] = dt
        self.kf.transitionMatrix = F
        self.kf.processNoiseCov  = np.eye(16, dtype=np.float32) * (self.pn * dt)

        self._last_t = now
        self.kf.predict()
        corrected = self.kf.correct(meas.reshape(8,1))
        return corrected[:8, 0].reshape(4, 2)

    def reset(self):
        self.initialized = False; self._last_t = None


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTTP JSON Server
# ══════════════════════════════════════════════════════════════════════════════
class _JSONHandler(BaseHTTPRequestHandler):
    node_ref = None

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.node_ref is None:
            self.send_error(503); return
        d    = self.node_ref.get_state()
        body = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def _start_json_server(node, port: int = 8081):
    _JSONHandler.node_ref = node
    srv = HTTPServer(("0.0.0.0", port), _JSONHandler)
    t   = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Số liệu hệ thống cho bảng overlay (CPU, nhiệt độ, ping)
# ══════════════════════════════════════════════════════════════════════════════
class SysStats(threading.Thread):
    """Đọc /proc mỗi giây trên thread riêng — vòng xử lý chính chỉ đọc lại
    số đã có, không tốn gì. Cùng cách tính với scripts/monitor_resources.py.

    GPU: Pi 4 không có bộ đếm % tải GPU, và model TFLite/ONNX ở đây chạy
    hoàn toàn trên CPU (VideoCore VI không được dùng), nên không có số GPU%
    nào để hiển thị — bảng ghi rõ điều đó thay vì bịa ra một con số.

    Ping: đo tới máy đang xem stream. Mặc định lấy IP laptop từ biến
    SSH_CLIENT (có sẵn khi chạy node qua ssh), đổi bằng --ping-host.
    """
    def __init__(self, ping_host=None):
        super().__init__(daemon=True)
        self.ping_host = ping_host
        self.n_core = os.cpu_count() or 4
        self.cpu_sys = self.cpu_node = self.temp = self.ping_ms = None
        self.rss_mb = None
        self._tick = os.sysconf("SC_CLK_TCK")
        self._page_mb = os.sysconf("SC_PAGE_SIZE") / 1e6
        if ping_host:
            threading.Thread(target=self._ping_loop, daemon=True).start()

    @staticmethod
    def _cpu_total():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v) - v[3] - v[4], sum(v)

    def _self_ticks(self):
        with open("/proc/self/stat") as f:
            p = f.read().rsplit(")", 1)[1].split()
        with open("/proc/self/statm") as f:
            self.rss_mb = int(f.read().split()[1]) * self._page_mb
        return int(p[11]) + int(p[12])

    def run(self):
        busy0, tot0 = self._cpu_total()
        me0, w0 = self._self_ticks(), time.time()
        while True:
            time.sleep(1.0)
            try:
                busy, tot = self._cpu_total()
                me, w = self._self_ticks(), time.time()
                # % cả máy (0-100) và % của node tính theo 1 nhân (0-400
                # trên Pi 4), giống cột CPU% của htop.
                self.cpu_sys = (busy - busy0) / max(tot - tot0, 1) * 100
                self.cpu_node = (me - me0) / self._tick / max(w - w0, 1e-3) * 100
                busy0, tot0, me0, w0 = busy, tot, me, w
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    self.temp = int(f.read()) / 1000
            except (OSError, ValueError, IndexError):
                pass

    def _ping_loop(self):
        while True:
            try:
                r = subprocess.run(["ping", "-c", "1", "-W", "1", self.ping_host],
                                   capture_output=True, text=True, timeout=3)
                m = re.search(r"time[=<]([\d.]+)", r.stdout)
                self.ping_ms = float(m.group(1)) if m else None
            except (OSError, subprocess.SubprocessError):
                self.ping_ms = None
            time.sleep(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Geometry / Camera
# ══════════════════════════════════════════════════════════════════════════════
PEN_3D = np.array([
    [0.0,   0.0, 0.0],
    [0.0,  64.0, 0.0],
    [-11.5, 44.0, 0.0],
    [11.5,  44.0, 0.0],
], dtype=np.float32)

def load_camera_calib(path: str, width: int, height: int):
    """Nạp K/dist từ file calib (scripts/calibrate_camera.py) và quy đổi về
    đúng độ phân giải đang bắt ảnh.

    Bản cũ viết cứng K=[[770,0,320],[0,770,240]] (số đoán cho ảnh 640x480),
    không hề đọc file calib → với camera thật ở 640x360, tiêu cự đúng là
    ~451 chứ không phải 770, nên Z bị báo lớn gấp ~1.7 lần và tâm ảnh lệch
    62px theo chiều dọc làm Y lệch theo.

    K tỉ lệ thuận với độ phân giải NẾU giữ nguyên tỉ lệ khung (vd calib ở
    1280x720, chạy 640x360: nhân 0.5). Đổi tỉ lệ khung (vd 640x480) thì
    camera thường cắt cảm biến khác đi, phép nhân không còn đúng — cần calib
    lại đúng độ phân giải đó.
    """
    d = np.load(path)
    K = d["K"].astype(np.float64)
    dist = d["dist"].astype(np.float64).reshape(-1, 1)
    w0, h0 = (int(v) for v in d["image_size"])
    sx, sy = width / w0, height / h0
    if abs(sx - sy) > 0.01:
        print(f"⚠️  Calib làm ở {w0}x{h0} nhưng đang chạy {width}x{height} — "
              f"KHÁC tỉ lệ khung, K quy đổi có thể sai. Nên calib lại ở "
              f"{width}x{height} hoặc chạy độ phân giải cùng tỉ lệ {w0}:{h0}.")
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    print(f"📐 Calib: {path} ({w0}x{h0} → {width}x{height})  "
          f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  "
          f"sai số calib {float(d['reprojection_error']):.3f}px")
    return K, dist


# ══════════════════════════════════════════════════════════════════════════════
# 6. ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════
class AeroScriptVisionNode(Node):
    def __init__(self, model_path: str, conf: float, device: str,
                 width: int, height: int, fourcc: str, focus, exposure,
                 n_threads: int, target_fps: float = 12.0,
                 trust_motion: float = 1.0, calib_path: str = "calib/c920_720p.npz",
                 no_filter: bool = False, ping_host: str = None):
        super().__init__("aeroscript_vision")
        self.no_filter = no_filter
        self.stats = SysStats(ping_host)
        self.stats.start()
        # Độ trễ từng chặng (ms, làm mượt EMA) cho bảng overlay
        self._lat = {"wait": None, "model": None, "xyz": None, "img": None}
        self._det_hist = []   # 1/0 cho ~2 giây gần nhất -> tỉ lệ bắt được
        self.bridge = CvBridge()
        self.model  = load_pose_model(model_path, conf, 0.45, n_threads)

        # Warmup: lần gọi ONNX Runtime đầu tiên luôn chậm hơn hẳn các lần
        # sau (JIT + graph optimization chạy lúc đó, không phải lúc load
        # model) — chạy thử 1 lần trên ảnh đen giả trước khi vào vòng xử lý
        # camera thật, để không làm sai lệch số đo fps/latency thực tế (nếu
        # không warmup, frame camera thật ĐẦU TIÊN sẽ gánh luôn chi phí này,
        # gây ra 1 lần trễ bất thường ngay lúc khởi động — đúng hiện tượng
        # đã đo được trên Pi thật: "max: 26.503s" chỉ xuất hiện 1 lần).
        _warmup_img = np.zeros((self.model.imgsz, self.model.imgsz, 3), dtype=np.uint8)
        _t0 = time.time()
        self.model(_warmup_img)
        print(f"🔥 Warmup ONNX xong trong {time.time()-_t0:.2f}s (lần đầu luôn chậm hơn bình thường).")

        # Process noise chỉnh lại cho ~15fps. Giá trị cũ (2D 3e-3, 3D 0.15)
        # chỉnh cho 2.8fps, mà Q = process_noise * dt nên khi fps tăng lên 15
        # (dt nhỏ đi 5 lần) bộ lọc càng tin mô hình chuyển động hơn số đo:
        # hệ số cập nhật chỉ còn ~0.01 (2D) và ~0.03 (3D), hai bộ lọc nối
        # tiếp → bút di chuyển mà XYZ phải 0.5-1s mới đuổi kịp.
        # Giá trị mới cho hệ số cập nhật ~0.5 mỗi frame ở 15fps (Q mỗi frame
        # xấp xỉ nhiễu đo R). trust_motion > 1 bám nhanh hơn nhưng rung hơn,
        # < 1 mượt hơn nhưng trễ hơn. --no-filter bỏ hẳn cả 2 bộ lọc.
        self.kf     = PoseKalmanFilter(process_noise=200.0 * trust_motion)
        self.kf_2d  = KeypointKalmanFilter(process_noise=30.0 * trust_motion)
        self.lost   = 0
        self._diag = {"ok": 0, "no_box": 0, "weak_kp": 0, "pnp_fail": 0}
        self._diag_weakest = []
        self._diag_n = 0
        self.last_valid_pose = None
        self.RESET  = 10
        self._fps_t = time.time()
        self._fps   = 0.0

        # Throttle inference rate — đây là cơ chế chính giảm CPU usage.
        # Camera có thể publish ở 15-30 FPS (ảnh nhỏ chạy rất nhanh), nhưng
        # KHÔNG có gì giới hạn _process_frame() chạy nhanh hơn mức cần thiết cho robot
        # arm điều khiển. Nếu target_fps=0, tắt throttle (chạy full speed
        # — hữu ích khi muốn benchmark tốc độ tối đa thực sự của model).
        self._target_fps     = target_fps
        self._min_interval   = (1.0 / target_fps) if target_fps > 0 else 0.0
        self._last_infer_t   = 0.0   # dùng chung: throttle check + dt cho Kalman 2D

        # Publishers
        self.pub_img  = self.create_publisher(Image,       "/aeroscript/pen_image", 1)
        self.pub_xyz  = self.create_publisher(Point,        "/aeroscript/pen_xyz",   1)
        self.pub_pose = self.create_publisher(PoseStamped,  "/aeroscript/pen_pose",  1)

        # Bắt camera trực tiếp — thay cho subscribe /image_raw qua usb_cam/DDS.
        # FrameGrabber tự lo việc "luôn giữ frame mới nhất" (drop frame cũ),
        # đúng vai trò mà QoS BEST_EFFORT+depth=1 làm ở tầng DDS trước đây,
        # nhưng làm ngay tại nguồn — bỏ hẳn 1 lớp trung gian + tránh phải
        # decode ảnh 2 lần (usb_cam encode → cv_bridge decode).
        self.grabber = FrameGrabber(device, width, height, fourcc, focus, exposure)

        # Độ phân giải THẬT camera nhận — có thể khác số yêu cầu nếu camera
        # không hỗ trợ đúng mode đó, mà K phải quy đổi theo số thật.
        real_w = int(self.grabber.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.grabber.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (real_w, real_h) != (width, height):
            print(f"⚠️  Yêu cầu {width}x{height} nhưng camera trả về {real_w}x{real_h}.")
        try:
            self.K, self.dist = load_camera_calib(calib_path, real_w, real_h)
        except FileNotFoundError:
            # Không có calib: dùng tiêu cự ước lượng (FOV ngang ~70°), đặt
            # tâm ảnh đúng giữa khung. XYZ chỉ đúng tương đối, KHÔNG dùng để đo.
            f = 0.71 * real_w
            self.K = np.array([[f, 0, real_w / 2], [0, f, real_h / 2], [0, 0, 1]], np.float64)
            self.dist = np.zeros((5, 1), np.float64)
            print(f"❌ KHÔNG tìm thấy file calib '{calib_path}' — đang dùng K ước "
                  f"lượng, XYZ SẼ SAI. Chép file calib lên Pi (deploy_to_pi.sh).")

        self.grabber.start()
        self._last_frame_id = -1
        # Timer poll nhanh (500Hz trần) — thực tế bị giới hạn bởi thời gian
        # _process_frame() chạy bên trong, không phải bởi chu kỳ timer này.
        self.timer = self.create_timer(0.002, self._on_timer)

        # Thread pool riêng cho encode+publish ảnh debug — KHÔNG nằm trên
        # critical path (inference → solvePnP → publish XYZ). Việc ảnh debug
        # đến web_video_server trễ vài ms không ảnh hưởng gì đến độ trễ điều
        # khiển thực tế, vì XYZ đã được publish trước khi submit vào đây.
        self._publish_pool = ThreadPoolExecutor(max_workers=1)

        # State chia sẻ với JSON HTTP server (port 8081)
        self._state_lock = threading.Lock()
        self._state = {"x": None, "y": None, "z": None,
                       "detected": False, "fps": 0.0}

        _start_json_server(self, port=8081)

        self.get_logger().info(f"✅ ONNX Node sẵn sàng — camera: {device} {width}x{height} {fourcc}")
        if self._target_fps > 0:
            self.get_logger().info(
                f"🎯 Throttle: tối đa {self._target_fps:.1f} inference/giây "
                f"(min_interval={self._min_interval*1000:.0f}ms)")
        else:
            self.get_logger().info("🎯 Throttle: TẮT — chạy full speed")
        self.get_logger().info(
            "📺 Video : http://192.168.50.1:8080/stream?topic=/aeroscript/pen_image&type=mjpeg")
        self.get_logger().info("📊 Data  : http://192.168.50.1:8081/")

    def get_state(self) -> dict:
        with self._state_lock:
            d = dict(self._state)
        st = self.stats
        d.update(frame="camera (X phai, Y xuong, Z ra truoc), goc o dau but",
                 e2e_xyz_ms=self._lat["xyz"], e2e_img_ms=self._lat["img"],
                 model_ms=self._lat["model"], cpu_sys=st.cpu_sys,
                 cpu_node=st.cpu_node, rss_mb=st.rss_mb, temp_c=st.temp,
                 ping_ms=st.ping_ms)
        return d

    @staticmethod
    def _text_box(frame, lines, x0, y0, color, s):
        """Vẽ khối chữ trên nền tối mờ. lines: list chuỗi ASCII (putText
        không vẽ được dấu tiếng Việt)."""
        fs, lh = 0.32 * s, int(11 * s)
        w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for t in lines)
        h, W, H = lh * len(lines) + int(6 * s), frame.shape[1], frame.shape[0]
        x0 = min(max(x0, 0), W - w - 8); y0 = min(max(y0, 0), H - h)
        roi = frame[y0:y0 + h, x0:x0 + w + 8]
        roi[:] = (roi * 0.3).astype(np.uint8)
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (x0 + 4, y0 + lh * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

    def _draw_panels(self, frame, info):
        s = frame.shape[1] / 640          # font to theo độ phân giải
        f = lambda v, fmt: "--" if v is None else fmt.format(v)

        # Góc trên-trái: vị trí ĐẦU BÚT (Tip) trong hệ toạ độ CAMERA.
        # PEN_3D đặt gốc ở Tip nên tvec của solvePnP chính là toạ độ Tip.
        # Hệ camera OpenCV: gốc ở tâm quang học, X sang phải, Y xuống dưới,
        # Z hướng ra trước ống kính (Z = khoảng cách theo trục nhìn).
        if info["valid"]:
            x, y, z, col = info["x"], info["y"], info["z"], (0, 255, 255)
        elif info.get("last_valid_pose") is not None:
            t = info["last_valid_pose"][1]
            x, y, z, col = float(t[0][0]), float(t[1][0]), float(t[2][0]), (150, 150, 150)
        else:
            x = y = z = None; col = (150, 150, 150)
        dist = None if z is None else float(np.sqrt(x * x + y * y + z * z))
        tag = "" if info["valid"] else (" [giu cu]" if z is not None else " [mat but]")
        self._text_box(frame, [
            f"TIP/CAM (mm) X {f(x, '{:+.1f}')}  Y {f(y, '{:+.1f}')}  Z {f(z, '{:.1f}')}",
            f"D {f(dist, '{:.1f}')}  (X phai, Y xuong, Z truoc){tag}",
        ], int(4 * s), int(4 * s), col, s)

        # Góc trên-phải: trạng thái
        cv2.putText(frame, "TRACKING" if info["valid"] else "NO TARGET",
                    (frame.shape[1] - int(75 * s), int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * s, (0, 255, 100) if info["valid"] else (0, 0, 230), 1, cv2.LINE_AA)

        # Góc dưới-trái: hiệu năng. E2E tính từ lúc frame tới chương trình
        # (grab) — chưa gồm phơi sáng/nén/USB bên trong camera (~30-60ms).
        # cam->XYZ ≈ cho model + chay model + vài ms solvePnP/Kalman/publish.
        L, st = self._lat, self.stats
        self._text_box(frame, [
            f"cam->XYZ {f(L['xyz'], '{:.0f}')} ms | cam->anh {f(L['img'], '{:.0f}')} ms",
            f" = frame cho model {f(L['wait'], '{:.0f}')} + chay model {f(L['model'], '{:.0f}')} ms",
            f"FPS {info['fps']:.1f} | Bat but (2s) {info['det_rate']:.0f}%",
            f"CPU may {f(st.cpu_sys, '{:.0f}')}% | node {f(st.cpu_node, '{:.0f}')}%/{st.n_core * 100}",
            f"RAM node {f(st.rss_mb, '{:.0f}')} MB | {f(st.temp, '{:.1f}')} C",
        ], int(4 * s), frame.shape[0], (0, 255, 0), s)

    def _render_and_publish(self, frame: np.ndarray, info: dict):
        """Chạy trên thread riêng — không chặn callback chính. Nhận frame
        THÔ (chưa vẽ gì) + info (số liệu đã tính sẵn ở _process_frame), tự
        vẽ toàn bộ overlay debug rồi encode/publish — tách hẳn phần vẽ ra
        khỏi critical path để _process_frame() trả về nhanh hơn."""
        try:
            if info["valid"]:
                labels = ["Tip", "Tail", "L", "R"]
                colors = [(0,255,0), (0,200,255), (255,100,0), (100,0,255)]
                for i, (px, py) in enumerate(info["kpts"].astype(int)):
                    cv2.circle(frame, (px, py), 6, colors[i], -1)
                    cv2.putText(frame, labels[i], (px+7, py-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
                cv2.drawFrameAxes(frame, self.K, self.dist, info["rvec"], info["tvec"], 40)
            elif info.get("last_valid_pose") is not None:
                # Đóng băng hiển thị khi mất tracking tạm thời (< RESET frames)
                kpts_frozen, tvec_frozen, rvec_frozen = info["last_valid_pose"]
                for px, py in kpts_frozen.astype(int):
                    cv2.circle(frame, (px, py), 6, (80,80,80), -1)
                cv2.drawFrameAxes(frame, self.K, self.dist, rvec_frozen, tvec_frozen, 40)

            self._draw_panels(frame, info)

            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_img.publish(out_msg)
            if info.get("t_cap"):
                self._ema("img", (time.perf_counter() - info["t_cap"]) * 1000)
        except Exception as e:
            self.get_logger().error(f"Render/publish (async): {e}")

    def _on_timer(self):
        # Lấy frame mới nhất từ FrameGrabber — nếu chưa có frame mới kể từ
        # lần xử lý trước (frame_id không đổi), bỏ qua ngay, không xử lý lại
        # frame cũ. Đây là điểm thay thế cho việc subscribe /image_raw.
        frame, fid, t_cap = self.grabber.get_latest()
        if frame is None or fid == self._last_frame_id:
            return
        self._last_frame_id = fid
        # Cho phép thread nền giải mã frame kế tiếp NGAY BÂY GIỜ, để việc đó
        # chạy song song với _process_frame() bên dưới thay vì nối đuôi.
        self.grabber.request_next()
        self._process_frame(frame, t_cap)

    def _ema(self, key, ms, a=0.2):
        old = self._lat[key]
        self._lat[key] = ms if old is None else (1 - a) * old + a * ms

    def _process_frame(self, frame: np.ndarray, t_cap: float = None):
        # ── Throttle: bỏ qua frame nếu chưa đủ thời gian từ lần inference
        # trước. Đây là cách giảm CPU usage hiệu quả nhất khi muốn giới hạn
        # tốc độ inference thấp hơn tốc độ camera thật.
        now_t = time.time()
        if self._min_interval > 0 and (now_t - self._last_infer_t) < self._min_interval:
            return
        # Lưu dt TRƯỚC khi cập nhật _last_infer_t — đây là khoảng cách thật
        # giữa lần xử lý hợp lệ trước và lần này, dùng cho Kalman 2D dt.
        dt_since_last = (now_t - self._last_infer_t) if self._last_infer_t > 0 else None
        self._last_infer_t = now_t

        # ── BOTTLENECK chính của toàn pipeline — không song song hoá được
        # vì mọi bước sau (solvePnP, Kalman, publish) phụ thuộc kết quả này.
        t_m0 = time.perf_counter()
        kpts, kv = self.model(frame)
        t_m1 = time.perf_counter()
        if t_cap:
            self._ema("wait", (t_m0 - t_cap) * 1000)
        self._ema("model", (t_m1 - t_m0) * 1000)
        valid = (kpts is not None and kv is not None
                 and np.all(kv > 0.45) and not np.any(kpts == 0.0))

        # Đếm lý do mất nhận diện để biết nên sửa ở đâu: "không thấy bút"
        # là vấn đề model/dữ liệu; "thiếu keypoint" là bút thấy được nhưng
        # 1 trong 4 điểm bị che/mờ (quy tắc đòi đủ 4 điểm); "PnP lỗi" là 4
        # điểm có nhưng hình học không giải được.
        if kpts is None:
            self._diag["no_box"] += 1
        elif not valid:
            self._diag["weak_kp"] += 1
            self._diag_weakest.append(int(np.argmin(kv)))

        x = y = z = None

        if valid:
            # dt_since_last đã được tính ở đầu callback — khoảng cách thời
            # gian thật giữa lần xử lý hợp lệ trước và lần này.
            kpts_f = kpts if self.no_filter else self.kf_2d.update(kpts, dt=dt_since_last)
            ok, rvec, tvec = cv2.solvePnP(
                PEN_3D, kpts_f.astype(np.float64),
                self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)

            if ok and tvec[2][0] > 0:
                tf = tvec if self.no_filter else self.kf.update(tvec, dt=dt_since_last)
                x, y, z = float(tf[0][0]), float(tf[1][0]), float(tf[2][0])
                self.lost = 0
                self.last_valid_pose = (kpts_f, tf, rvec)

                # Publish XYZ NGAY — đây là dữ liệu quan trọng nhất, không
                # chờ phần vẽ overlay hay encode ảnh phía dưới.
                with self._state_lock:
                    self._state = {"x": x, "y": y, "z": z,
                                   "detected": True, "fps": self._fps}

                now = self.get_clock().now().to_msg()
                pt = Point(); pt.x = x; pt.y = y; pt.z = z
                self.pub_xyz.publish(pt)

                ps = PoseStamped()
                ps.header = Header(); ps.header.stamp = now; ps.header.frame_id = "camera"
                ps.pose.position.x = x / 1000.0
                ps.pose.position.y = y / 1000.0
                ps.pose.position.z = z / 1000.0
                ps.pose.orientation.w = 1.0
                self.pub_pose.publish(ps)
                if t_cap:
                    self._ema("xyz", (time.perf_counter() - t_cap) * 1000)
            else:
                valid = False
                self._diag["pnp_fail"] += 1

        if not valid:
            self.lost += 1
            if self.lost >= self.RESET:
                self.kf.reset()
                self.kf_2d.reset()
                self.last_valid_pose = None
                with self._state_lock:
                    self._state["detected"] = False

        # FPS
        t_now = time.time()
        fps_i = 1.0 / max(t_now - self._fps_t, 1e-6)
        self._fps = 0.9 * self._fps + 0.1 * fps_i
        self._fps_t = t_now
        with self._state_lock:
            self._state["fps"] = round(self._fps, 1)

        # Toàn bộ việc VẼ overlay (circle/putText/drawFrameAxes) + encode +
        # publish ảnh debug → chuyển hết sang thread phụ (_render_and_publish).
        # Trước đây các lệnh cv2.* này chạy ĐỒNG BỘ ngay tại đây dù dữ liệu
        # XYZ/Pose đã publish xong ở trên rồi — vẫn tính vào thời gian
        # _process_frame() chạy, tức vẫn kéo FPS xuống dù không cần thiết.
        # Giờ chỉ đóng gói dữ liệu (số, không phải lệnh vẽ) rồi giao việc vẽ
        # thật cho thread khác — _process_frame() trả về ngay sau đây.
        self._det_hist.append(1 if valid else 0)
        del self._det_hist[:-30]
        draw_info = {"valid": valid, "fps": self._fps, "t_cap": t_cap,
                     "det_rate": sum(self._det_hist) / len(self._det_hist) * 100}
        if valid:
            draw_info.update(kpts=kpts_f, rvec=rvec, tvec=tf, x=x, y=y, z=z)
        else:
            # last_valid_pose có thể đã bị reset về None ở trên (khi lost
            # >= RESET) — giữ nguyên hành vi cũ: chỉ vẽ "đóng băng" khi còn
            # pose gần nhất trong ngưỡng RESET frame.
            draw_info["last_valid_pose"] = self.last_valid_pose
        self._publish_pool.submit(self._render_and_publish, frame.copy(), draw_info)

        if valid:
            self.get_logger().info(
                f"PEN  X:{x:7.1f}  Y:{y:7.1f}  Z:{z:7.1f} mm  FPS:{self._fps:.1f}")
            self._diag["ok"] += 1

        self._diag_n += 1
        if self._diag_n % 75 == 0:          # ~5 giây một lần ở 15fps
            dg, n = self._diag, sum(self._diag.values())
            names = ["Tip", "Tail", "L", "R"]
            weak = ""
            if self._diag_weakest:
                cnt = np.bincount(self._diag_weakest, minlength=4)
                weak = "  (điểm mờ nhất: " + ", ".join(
                    f"{names[i]} {cnt[i]}" for i in np.argsort(-cnt) if cnt[i]) + ")"
            print(f"🔎 [{n} frame] bắt được {dg['ok']/n*100:.0f}% | không thấy bút "
                  f"{dg['no_box']} | thiếu keypoint {dg['weak_kp']}{weak} | PnP lỗi {dg['pnp_fail']}")
            self._diag = dict.fromkeys(self._diag, 0)
            self._diag_weakest = []


# ══════════════════════════════════════════════════════════════════════════════
# 7. Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.onnx",
                        help="Mặc định best.onnx (FP32) — best_int8.onnx đã xác "
                             "nhận KHÔNG detect được bút (quantize làm mất quá "
                             "nhiều thông tin ở keypoint nhỏ như mép nắp bút).")
    parser.add_argument("--conf",  type=float, default=0.55)
    parser.add_argument("--device", default="/dev/video0",
                        help="Camera device (đường dẫn /dev/videoX hoặc số index).")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fourcc", default="MJPG",
                        help="Định dạng bắt ảnh của camera. MJPG (mặc định) nhẹ hơn "
                             "nhiều lần so với ảnh thô (YUYV) ở cùng độ phân giải, "
                             "cần thiết để chạy 720p+ trong băng thông USB Pi 4 ổn định.")
    parser.add_argument("--focus", type=int, default=None,
                        help="Khoá lấy nét thủ công (0=xa, ~100+=gần). Bỏ trống = auto.")
    parser.add_argument("--exposure", type=int, default=None,
                        help="Khoá phơi sáng thủ công. Bỏ trống = auto.")
    parser.add_argument("--threads", type=int, default=3,
                        help="Số thread ONNX Runtime dùng nội bộ. Mặc định cũ là 1 "
                             "(để nhường CPU cho usb_cam/DDS/cv_bridge của luồng cũ) — "
                             "từ Phase 1, usb_cam đã bị bỏ hẳn (FrameGrabber chạy "
                             "trong cùng tiến trình), nên không còn lý do giữ threads "
                             "thấp nữa. Đổi mặc định lên 3 (Pi 4 có 4 core, chừa 1 "
                             "core cho thread FrameGrabber + ROS2 executor). "
                             "Đánh đổi: threads=1 ~2.8 FPS, threads=2 ~4.7 FPS "
                             "(benchmark FP32 imgsz=320 trên Pi 4, đo TRƯỚC Phase 1 — "
                             "thử lại threads=3/4 để có số đo mới).")
    parser.add_argument("--target-fps", type=float, default=0.0,
                        help="Giới hạn cứng số lần inference/giây. Đặt 0 (mặc định) "
                             "để chạy full speed — quan trọng khi threads=1 vì khả "
                             "năng thật đã chỉ ~2.8 FPS, throttle thêm sẽ làm chậm "
                             "hơn nữa không cần thiết.")
    parser.add_argument("--trust-motion", type=float, default=1.0,
                        help="Hệ số tin tưởng motion model của Kalman filter "
                             "(mặc định 1.0). Giảm xuống (ví dụ 0.5) để filter tin "
                             "tưởng dự đoán vận tốc nhiều hơn, giúp chuyển động mượt "
                             "hơn giữa các lần đo thưa ở FPS thấp — đổi lại phản ứng "
                             "chậm hơn khi bút đổi hướng đột ngột. Tăng lên (ví dụ "
                             "2.0) nếu thấy quỹ đạo bị 'trễ' theo sau chuyển động thật.")
    parser.add_argument("--calib", default="calib/c920_720p.npz",
                        help="File calib camera (scripts/calibrate_camera.py). K "
                             "tự quy đổi theo độ phân giải đang chạy, nên giữ cùng "
                             "tỉ lệ khung với lúc calib (1280x720 -> 640x360 được).")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bỏ cả 2 bộ lọc Kalman, dùng thẳng kết quả từng frame. "
                             "Dùng để kiểm tra độ trễ gốc của pipeline hoặc so sánh.")
    parser.add_argument("--ping-host", default=None,
                        help="Máy cần đo ping (chỉ có trong JSON port 8081). Bỏ trống = tự "
                             "lấy IP laptop đang ssh vào Pi (biến SSH_CLIENT).")
    args, _ = parser.parse_known_args()
    ping_host = args.ping_host or (os.environ.get("SSH_CLIENT", "").split() or [None])[0]

    rclpy.init()
    node = AeroScriptVisionNode(args.model, args.conf, args.device,
                                args.width, args.height, args.fourcc,
                                args.focus, args.exposure,
                                args.threads, args.target_fps, args.trust_motion,
                                args.calib, args.no_filter, ping_host)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.grabber.stop()
        node._publish_pool.shutdown(wait=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()