#!/usr/bin/env python3
"""
run_pi4_ros.py  —  ROS2 Node chạy trên Raspberry Pi 4 sử dụng ONNX Runtime
=============================================================================
Subscribe camera → ONNX inference → solvePnP → publish kết quả.

Topics:
  Subscribe : /image_raw            (sensor_msgs/Image)  — từ usb_cam
  Publish   : /aeroscript/pen_image (sensor_msgs/Image)  — frame annotated
  Publish   : /aeroscript/pen_xyz   (geometry_msgs/Point)— tọa độ 3D (mm)
  Publish   : /aeroscript/pen_pose  (geometry_msgs/PoseStamped) — đầy đủ

Cài đặt trên Pi 4:
    pip3 install "numpy<2" opencv-python onnxruntime

Cách chạy:
    python3 run_pi4_ros.py --model best_int8.onnx
    python3 run_pi4_ros.py --model best.onnx --conf 0.5      # FP32

────────────────────────────────────────────────────────────────────────────
PHÂN TÍCH PIPELINE — bottleneck & song song hoá
────────────────────────────────────────────────────────────────────────────
Chuỗi xử lý mỗi frame trong _cb():

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

  (a) QoS BEST_EFFORT + depth=1 trên subscription camera:
      Nếu callback xử lý chậm hơn tốc độ camera publish (rất có thể, vì
      camera 30 FPS >> inference Pi 4), ROS2 với RELIABLE QoS sẽ giữ
      frame cũ trong queue chờ xử lý → frame mà callback nhận được luôn
      "trễ" tích lũy. BEST_EFFORT + depth=1 đảm bảo callback luôn nhận
      frame MỚI NHẤT có sẵn tại thời điểm rảnh, tự động bỏ frame cũ.

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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

import onnxruntime as ort


# ══════════════════════════════════════════════════════════════════════════════
# 1. ONNX Inference Engine
# ══════════════════════════════════════════════════════════════════════════════
class ONNXPoseInference:
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

    def __call__(self, bgr: np.ndarray):
        tensor, r, dw, dh = self._preprocess(bgr)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        preds = raw[0].T if raw.ndim == 3 else raw[0]
        return self._decode(preds, r, dw, dh)

    def _preprocess(self, bgr: np.ndarray):
        h0, w0 = bgr.shape[:2]
        r  = self.imgsz / max(h0, w0)
        nw, nh = int(w0 * r), int(h0 * r)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        dw, dh = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(
            bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        rgb = canvas[:, :, ::-1].astype(self._np_dtype) / 255.0
        if self.is_nhwc:
            tensor = rgb[np.newaxis]
        else:
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]
        return tensor, r, dw, dh

    def _decode(self, preds, r, dw, dh):
        # Layout: [cx,cy,w,h, conf, kp0x,kp0y,kp0v, ..., kp3v] — 17 fields, không có cls
        if preds.shape[1] < 17:
            return None, None
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
# 5. Geometry / Camera
# ══════════════════════════════════════════════════════════════════════════════
PEN_3D = np.array([
    [0.0,   0.0, 0.0],
    [0.0,  64.0, 0.0],
    [-11.5, 44.0, 0.0],
    [11.5,  44.0, 0.0],
], dtype=np.float32)

CAM_MTX = np.array([[770,0,320],[0,770,240],[0,0,1]], dtype=np.float32)
DIST    = np.zeros((4,1), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ROS2 Node
# ══════════════════════════════════════════════════════════════════════════════
class AeroScriptVisionNode(Node):
    def __init__(self, model_path: str, conf: float, cam_topic: str,
                 n_threads: int, target_fps: float = 12.0,
                 trust_motion: float = 1.0):
        super().__init__("aeroscript_vision")
        self.bridge = CvBridge()
        self.model  = ONNXPoseInference(model_path, conf_thresh=conf, n_threads=n_threads)

        # process_noise thấp hơn = tin tưởng motion model nhiều hơn giữa các
        # lần đo thưa (quan trọng ở FPS thấp ~2.8). trust_motion là hệ số
        # nhân thêm — trust_motion=1.0 dùng giá trị mặc định đã giảm sẵn so
        # với bản gốc, trust_motion<1.0 tin tưởng motion model hơn nữa.
        self.kf     = PoseKalmanFilter(process_noise=0.15 * trust_motion)
        self.kf_2d  = KeypointKalmanFilter(process_noise=3e-3 * trust_motion)
        self.lost   = 0
        self.last_valid_pose = None
        self.RESET  = 10
        self._fps_t = time.time()
        self._fps   = 0.0

        # Throttle inference rate — đây là cơ chế chính giảm CPU usage.
        # Camera có thể publish ở 15-30 FPS (ảnh nhỏ chạy rất nhanh), nhưng
        # KHÔNG có gì giới hạn _cb() chạy nhanh hơn mức cần thiết cho robot
        # arm điều khiển. Nếu target_fps=0, tắt throttle (chạy full speed
        # — hữu ích khi muốn benchmark tốc độ tối đa thực sự của model).
        self._target_fps     = target_fps
        self._min_interval   = (1.0 / target_fps) if target_fps > 0 else 0.0
        self._last_infer_t   = 0.0   # dùng chung: throttle check + dt cho Kalman 2D

        # Publishers
        self.pub_img  = self.create_publisher(Image,       "/aeroscript/pen_image", 1)
        self.pub_xyz  = self.create_publisher(Point,        "/aeroscript/pen_xyz",   1)
        self.pub_pose = self.create_publisher(PoseStamped,  "/aeroscript/pen_pose",  1)

        # Subscribe camera — BEST_EFFORT + depth=1: luôn nhận frame MỚI NHẤT,
        # tự động bỏ frame cũ khi inference chậm hơn tốc độ camera publish.
        # Tránh hiện tượng "lag dồn" do frame xếp hàng chờ xử lý.
        qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 1,
        )
        self.sub = self.create_subscription(Image, cam_topic, self._cb, qos)

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

        self.get_logger().info(f"✅ ONNX Node sẵn sàng — subscribe: {cam_topic}")
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
            return dict(self._state)

    def _publish_image_async(self, frame: np.ndarray):
        """Chạy trên thread riêng — không chặn callback chính."""
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_img.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish image (async): {e}")

    def _cb(self, msg: Image):
        # ── Throttle: bỏ qua frame nếu chưa đủ thời gian từ lần inference
        # trước. Đặt ở ĐẦU callback, TRƯỚC cv_bridge decode — vì decode
        # ảnh JPEG/raw cũng tốn CPU đáng kể, không có lý do để làm việc đó
        # cho một frame sẽ bị vứt đi ngay sau. Đây là cách giảm CPU usage
        # hiệu quả nhất: không xử lý gì cả cho phần lớn frame thừa, thay
        # vì "xử lý nhanh hơn rồi vứt" như giảm kích thước ảnh.
        now_t = time.time()
        if self._min_interval > 0 and (now_t - self._last_infer_t) < self._min_interval:
            return
        # Lưu dt TRƯỚC khi cập nhật _last_infer_t — đây là khoảng cách thật
        # giữa lần xử lý hợp lệ trước và lần này, dùng cho Kalman 2D dt.
        dt_since_last = (now_t - self._last_infer_t) if self._last_infer_t > 0 else None
        self._last_infer_t = now_t

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge: {e}"); return

        # ── BOTTLENECK chính của toàn pipeline — không song song hoá được
        # vì mọi bước sau (solvePnP, Kalman, publish) phụ thuộc kết quả này.
        kpts, kv = self.model(frame)
        valid = (kpts is not None and kv is not None
                 and np.all(kv > 0.45) and not np.any(kpts == 0.0))

        x = y = z = None

        if valid:
            # dt_since_last đã được tính ở đầu callback — khoảng cách thời
            # gian thật giữa lần xử lý hợp lệ trước và lần này.
            kpts_f = self.kf_2d.update(kpts, dt=dt_since_last)
            ok, rvec, tvec = cv2.solvePnP(
                PEN_3D, kpts_f.astype(np.float64),
                CAM_MTX, DIST, flags=cv2.SOLVEPNP_IPPE)

            if ok and tvec[2][0] > 0:
                tf = self.kf.update(tvec, dt=dt_since_last)
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

                # Vẽ overlay (chỉ phục vụ debug/giám sát — không quan trọng
                # bằng việc publish XYZ ở trên, nên đặt sau)
                labels = ["Tip", "Tail", "L", "R"]
                colors = [(0,255,0), (0,200,255), (255,100,0), (100,0,255)]
                for i, (px, py) in enumerate(kpts_f.astype(int)):
                    cv2.circle(frame, (px, py), 6, colors[i], -1)
                    cv2.putText(frame, labels[i], (px+7, py-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
                cv2.drawFrameAxes(frame, CAM_MTX, DIST, rvec, tf, 40)
                cv2.putText(frame, f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f}mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
            else:
                valid = False

        if not valid:
            self.lost += 1
            if self.lost >= self.RESET:
                self.kf.reset()
                self.kf_2d.reset()
                self.last_valid_pose = None
                with self._state_lock:
                    self._state["detected"] = False

        # Đóng băng hiển thị khi mất tracking tạm thời (< RESET frames)
        if not valid and self.last_valid_pose is not None:
            kpts_frozen, tvec_frozen, rvec_frozen = self.last_valid_pose
            for px, py in kpts_frozen.astype(int):
                cv2.circle(frame, (px, py), 6, (80,80,80), -1)
            cv2.drawFrameAxes(frame, CAM_MTX, DIST, rvec_frozen, tvec_frozen, 40)

        # FPS
        t_now = time.time()
        fps_i = 1.0 / max(t_now - self._fps_t, 1e-6)
        self._fps = 0.9 * self._fps + 0.1 * fps_i
        self._fps_t = t_now
        with self._state_lock:
            self._state["fps"] = round(self._fps, 1)

        cv2.putText(frame, f"FPS:{self._fps:.1f}",
                    (frame.shape[1]-110, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,0), 2)
        cv2.putText(frame, "TRACKING" if valid else "NO TARGET",
                    (frame.shape[1]-135, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,100) if valid else (0,0,200), 2)

        # Encode + publish ảnh debug → thread riêng, KHÔNG chặn callback.
        # frame.copy() vì frame gốc có thể bị GC/tái sử dụng sau khi _cb return.
        self._publish_pool.submit(self._publish_image_async, frame.copy())

        if valid:
            self.get_logger().info(
                f"PEN  X:{x:7.1f}  Y:{y:7.1f}  Z:{z:7.1f} mm  FPS:{self._fps:.1f}")


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
    parser.add_argument("--cam-topic", default="/image_raw")
    parser.add_argument("--threads", type=int, default=1,
                        help="Số thread ONNX Runtime dùng nội bộ (mặc định 1 — "
                             "nhường CPU cho usb_cam, web_video_server, ROS2 executor. "
                             "Đánh đổi: threads=1 ~2.8 FPS, threads=2 ~4.7 FPS "
                             "(benchmark FP32 imgsz=320 trên Pi 4).")
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
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = AeroScriptVisionNode(args.model, args.conf, args.cam_topic,
                                args.threads, args.target_fps, args.trust_motion)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_pool.shutdown(wait=False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()