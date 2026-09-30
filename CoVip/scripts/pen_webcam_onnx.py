#!/usr/bin/env python3
"""
Detect đầu bút bằng webcam + ONNX (YOLOv8n-pose 4 keypoint) + solvePnP. Không cần ROS.

Chạy (từ thư mục DA1_EmbedLab):
    python3 scripts/pen_webcam_onnx.py                       # best.onnx, cam 0
    python3 scripts/pen_webcam_onnx.py --model pen_models/best_int8.onnx --cam 0
Phím: q/ESC thoát, s lưu ảnh.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent

# Hình học bút (mm), gốc tại ngòi: Tip, Tail, Left cap, Right cap
PEN_3D = np.array([[0, 0, 0], [0, 64, 0], [-11.5, 44, 0], [11.5, 44, 0]], dtype=np.float32)
LABELS = ["Tip", "Tail", "L", "R"]
COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (100, 0, 255)]


class PoseONNX:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "pen_models" / "best.onnx"))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fx", type=float, default=770.0,
                    help="Tiêu cự pixel (giả định 770, nên calibrate camera thật)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Hệ số làm mượt XYZ (1 = không lọc)")
    args = ap.parse_args()

    model = PoseONNX(args.model, args.conf, threads=args.threads)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera {args.cam}")

    K = np.array([[args.fx, 0, 320], [0, args.fx, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)
    smooth, fps, t_prev = None, 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kpts, kv = model(frame)
        valid = kpts is not None and np.all(kv > 0.45)
        if valid:
            for i, (px, py) in enumerate(kpts.astype(int)):
                cv2.circle(frame, (px, py), 6, COLORS[i], -1)
                cv2.putText(frame, LABELS[i], (px + 7, py - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[i], 1)
            ok_pnp, rvec, tvec = cv2.solvePnP(PEN_3D, kpts.astype(np.float64), K, dist,
                                              flags=cv2.SOLVEPNP_IPPE)
            if ok_pnp and tvec[2][0] > 0:
                t = tvec.flatten()
                smooth = t if smooth is None else args.alpha * t + (1 - args.alpha) * smooth
                cv2.drawFrameAxes(frame, K, dist, rvec, smooth.reshape(3, 1), 20)
                cv2.putText(frame, f"X:{smooth[0]:.0f} Y:{smooth[1]:.0f} Z:{smooth[2]:.0f} mm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            smooth = None
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"FPS:{fps:.1f} {'TRACKING' if valid else 'NO TARGET'}",
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 100) if valid else (0, 0, 220), 2)
        cv2.imshow("pen detect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            cv2.imwrite(str(ROOT / f"pen_capture_{int(now)}.jpg"), frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
