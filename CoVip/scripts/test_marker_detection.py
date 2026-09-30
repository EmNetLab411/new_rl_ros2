#!/usr/bin/env python3
"""
Kiểm tra marker ArUod thật (dict + id + size) có detect được đúng không —
làm được NGAY với chỉ camera + bút, KHÔNG cần robot. Nên chạy trước khi
làm Phase 3 thật (calibrate_hand_eye.py collect), tránh mất công dựng cả
robot rồi mới phát hiện marker không detect được / sai dict / sai size.

Chạy (từ thư mục CoVip, cần calib/c920_720p.npz đã có từ Phase 2):
    python3 scripts/test_marker_detection.py --marker-id 0 \\
        --marker-size-mm 10 --dict DICT_4X4_50 --camera-name C930e

Phím: q/ESC thoát.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_camera import find_camera, list_cameras  # noqa: E402
from calibrate_hand_eye import (  # noqa: E402
    load_camera_calib, _make_aruco_detector, detect_marker_pose,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--camera-name", default="C930e")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--focus", type=int, default=20)
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--marker-size-mm", type=float, required=True)
    ap.add_argument("--dict", default="DICT_4X4_50")
    args = ap.parse_args()

    K, dist = load_camera_calib()

    if args.device is not None:
        idx = int(args.device) if str(args.device).isdigit() else args.device
    else:
        idx = find_camera(args.camera_name)
        if idx is None:
            print(f"Không tự tìm được camera '{args.camera_name}'.\n{list_cameras()}", file=sys.stderr)
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

    detect_fn = _make_aruco_detector(args.dict)
    marker_size_m = args.marker_size_mm / 1000.0

    n_frames = n_det = 0
    print(f"Marker cần tìm: dict={args.dict}, id={args.marker_id}, cạnh={args.marker_size_mm}mm")
    print("Đưa marker vào khung hình. q để thoát.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        n_frames += 1
        corners, ids, _ = detect_fn(frame)
        disp = frame.copy()

        found_target = False
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(disp, corners, ids)
            ids_list = ids.flatten().tolist()
            if args.marker_id in ids_list:
                found_target = True
                n_det += 1
                T = detect_marker_pose(frame, K, dist, args.marker_id, marker_size_m, detect_fn)
                if T is not None:
                    rvec, _ = cv2.Rodrigues(T[:3, :3])
                    tvec = T[:3, 3].reshape(3, 1)
                    cv2.drawFrameAxes(disp, K, dist, rvec, tvec, marker_size_m * 0.75)
                    dist_cm = np.linalg.norm(tvec) * 100
                    cv2.putText(disp, f"id={args.marker_id} khoang cach={dist_cm:.1f}cm",
                                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pct = 100 * n_det / n_frames
        color = (0, 255, 0) if found_target else (0, 0, 255)
        cv2.putText(disp, f"Detect id={args.marker_id}: {n_det}/{n_frames} ({pct:.0f}%)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if ids is not None and args.marker_id not in ids.flatten().tolist():
            seen = ids.flatten().tolist()
            cv2.putText(disp, f"Thay ID khac: {seen} (khong phai {args.marker_id})",
                        (10, disp.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow("test_marker_detection", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nKết quả: detect được id={args.marker_id} ở {n_det}/{n_frames} frame ({100*n_det/max(n_frames,1):.1f}%)")
    if n_det == 0:
        print("KHÔNG detect được lần nào — kiểm tra lại: đúng dict chưa (DICT_4X4_50), "
              "in marker có bị mờ/thiếu viền trắng xung quanh không, khoảng cách có quá xa không "
              "(marker 10mm rất nhỏ, thử đưa lại gần camera trong khoảng 15-30cm).")


if __name__ == "__main__":
    main()
