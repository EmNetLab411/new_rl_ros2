#!/usr/bin/env python3
"""
Kiểm chứng giả thuyết cốt lõi của Phase 4 (PLAN.md) NGAY BÂY GIỜ, chỉ cần
camera + bút — KHÔNG cần robot/Pi: liệu detect trong 1 ô ROI nhỏ (giống
fk_roi_predictor.py sẽ cắt ra từ FK) có chính xác hơn hẳn so với quét cả
khung 1280x720 như luồng cũ hay không.

Vì chưa có robot để FK tự tính vị trí ROI, script này dùng 1 ô ROI CỐ ĐỊNH
(di chuyển bằng phím mũi tên để mô phỏng nhiều vị trí FK có thể dự đoán ra),
bạn tự đưa bút vào ô đó. So sánh trực tiếp: model chạy full-frame vs model
chỉ chạy trên đúng ô ROI đã cắt.

Chạy (từ thư mục CoVip):
    python3 scripts/test_roi_detection.py --camera-name C930e --roi-size 300

Phím:
    mũi tên   : di chuyển ô ROI (mô phỏng các vị trí FK dự đoán khác nhau)
    +/-       : tăng/giảm kích thước ô ROI
    r         : reset thống kê đếm về 0
    q / ESC   : thoát, in báo cáo tổng kết
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Tip", "Tail", "L", "R"]
COLOR_FULL = (0, 140, 255)   # cam — kết quả detect full-frame
COLOR_CROP = (0, 255, 0)     # xanh lá — kết quả detect trong ROI


class PoseONNX:
    """Giống hệt class trong pen_webcam_onnx.py, tách riêng để script này
    chạy độc lập không phụ thuộc import chéo."""

    def __init__(self, path, conf=0.55, iou=0.45, threads=4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.out_name = self.sess.get_outputs()[0].name
        self.imgsz = int(inp.shape[2])
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        self.conf, self.iou = conf, iou

    def __call__(self, bgr):
        h0, w0 = bgr.shape[:2]
        r = self.imgsz / max(h0, w0)
        nw, nh = int(w0 * r), int(h0 * r)
        dw, dh = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(bgr, (nw, nh))
        x = (canvas[:, :, ::-1].astype(self.dtype) / 255.0).transpose(2, 0, 1)[None]
        raw = self.sess.run([self.out_name], {self.in_name: x})[0]
        preds = raw[0].T
        f = preds[preds[:, 4] > self.conf]
        if not len(f):
            return None, None
        keep = self._nms(f[:, :4], f[:, 4])
        kr = f[keep[0], 5:17].reshape(4, 3)
        k = kr[:, :2].copy()
        k[:, 0] = (k[:, 0] - dw) / r
        k[:, 1] = (k[:, 1] - dh) / r
        return k.astype(np.float32), kr[:, 2]

    def _nms(self, boxes, scores):
        cx, cy, w, h = boxes.T
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(int(i))
            o = order[1:]
            inter = (np.maximum(0, np.minimum(x2[i], x2[o]) - np.maximum(x1[i], x1[o])) *
                     np.maximum(0, np.minimum(y2[i], y2[o]) - np.maximum(y1[i], y1[o])))
            order = o[inter / (areas[i] + areas[o] - inter + 1e-6) <= self.iou]
        return keep


def find_camera(name):
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        name_file, index_file = d / "name", d / "index"
        if not name_file.exists() or not index_file.exists():
            continue
        if name in name_file.read_text() and index_file.read_text().strip() == "0":
            return int(d.name.replace("video", ""))
    return None


def draw_kpts(img, kpts, confs, color, offset=(0, 0)):
    ox, oy = offset
    for (x, y), c, name in zip(kpts, confs, LABELS):
        px, py = int(x + ox), int(y + oy)
        cv2.drawMarker(img, (px, py), color, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, f"{name}:{c:.2f}", (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "pen_models" / "best.onnx"))
    ap.add_argument("--device", default=None, help="/dev/videoN thẳng, bỏ trống = tự tìm theo --camera-name")
    ap.add_argument("--camera-name", default="C930e")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--focus", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--roi-size", type=int, default=300)
    args = ap.parse_args()

    if args.device is not None:
        idx = int(args.device) if str(args.device).isdigit() else args.device
    else:
        idx = find_camera(args.camera_name)
        if idx is None:
            print(f"Không tự tìm được camera '{args.camera_name}'. Dùng --device /dev/videoN.", file=sys.stderr)
            sys.exit(1)
        print(f"Dùng camera -> /dev/video{idx}")

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    if not cap.isOpened():
        print(f"Không mở được camera /dev/video{idx}", file=sys.stderr)
        sys.exit(1)

    model = PoseONNX(args.model, conf=args.conf)
    roi_size = args.roi_size
    w, h = args.width, args.height
    cx, cy = w // 2, h // 2  # tâm ROI, di chuyển được bằng phím mũi tên
    step = 40

    n_frames = n_full_det = n_crop_det = 0
    print("Điều khiển: mũi tên di chuyển ô ROI, +/- đổi kích thước, r reset đếm, q/ESC thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        half = roi_size // 2
        x0 = int(np.clip(cx - half, 0, w - roi_size))
        y0 = int(np.clip(cy - half, 0, h - roi_size))
        x1, y1 = x0 + roi_size, y0 + roi_size
        crop = frame[y0:y1, x0:x1]

        full_k, full_c = model(frame)
        crop_k, crop_c = model(crop)

        n_frames += 1
        if full_k is not None:
            n_full_det += 1
        if crop_k is not None:
            n_crop_det += 1

        disp = frame.copy()
        cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 0), 2)
        if full_k is not None:
            draw_kpts(disp, full_k, full_c, COLOR_FULL)
        if crop_k is not None:
            draw_kpts(disp, crop_k, crop_c, COLOR_CROP, offset=(x0, y0))

        full_pct = 100 * n_full_det / n_frames
        crop_pct = 100 * n_crop_det / n_frames
        cv2.putText(disp, f"FULL-FRAME  (cam) detect: {n_full_det}/{n_frames} ({full_pct:.0f}%)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_FULL, 2)
        cv2.putText(disp, f"ROI-CROP  (xanh) detect: {n_crop_det}/{n_frames} ({crop_pct:.0f}%)",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CROP, 2)
        cv2.putText(disp, f"ROI {roi_size}x{roi_size} tai ({cx},{cy}) - dua but vao o vang",
                    (10, disp.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("test_roi_detection", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            n_frames = n_full_det = n_crop_det = 0
        elif key in (ord('+'), ord('=')):
            roi_size = min(roi_size + 20, min(w, h))
        elif key in (ord('-'), ord('_')):
            roi_size = max(roi_size - 20, 60)
        elif key == 82 or key == 0:      # up (mã phím có thể khác tuỳ hệ)
            cy = max(0, cy - step)
        elif key == 84 or key == 1:      # down
            cy = min(h, cy + step)
        elif key == 81 or key == 2:      # left
            cx = max(0, cx - step)
        elif key == 83 or key == 3:      # right
            cx = min(w, cx + step)

    cap.release()
    cv2.destroyAllWindows()

    print("\n===== KẾT QUẢ =====")
    print(f"Tổng số frame: {n_frames}")
    if n_frames:
        print(f"Detect full-frame : {n_full_det}/{n_frames} ({100*n_full_det/n_frames:.1f}%)")
        print(f"Detect trong ROI  : {n_crop_det}/{n_frames} ({100*n_crop_det/n_frames:.1f}%)")
        diff = 100 * (n_crop_det - n_full_det) / n_frames
        if diff > 5:
            print(f"-> ROI TỐT HƠN full-frame {diff:.1f} điểm % — giả thuyết Phase 4 được xác nhận, "
                  f"có thể chưa cần retrain (Phase 6) ngay.")
        elif diff < -5:
            print(f"-> ROI KÉM HƠN full-frame {-diff:.1f} điểm % — bất thường, kiểm tra lại vị trí ROI "
                  f"có thật sự bọc quanh bút không.")
        else:
            print("-> Không chênh lệch rõ rệt — cần thêm dữ liệu test ở nhiều khoảng cách/góc khác nhau "
                  "trước khi kết luận có cần retrain (Phase 6) hay không.")


if __name__ == "__main__":
    main()
