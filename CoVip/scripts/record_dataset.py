#!/usr/bin/env python3
"""
Quay video dataset bằng Logitech C930e và bóc khung cho gán nhãn.

Quay (từ thư mục CoVip):
    python3 scripts/record_dataset.py record --tag sang_deu
    python3 scripts/record_dataset.py record --tag toi_den_ben --focus 30 --exposure 200
Phím: r bắt đầu/dừng ghi, s lưu 1 khung lẻ, q/ESC thoát.

Quay đúng bối cảnh triển khai (bút gắn trên cánh tay robot, camera cố định):
    python3 scripts/record_dataset.py record --tag robot_mounted --focus 20

Bóc khung:
    python3 scripts/record_dataset.py extract datasets/raw_videos/<file>.avi --every 6
    python3 scripts/record_dataset.py extract datasets/raw_videos/robot_mounted_*.avi \
        --every 6 --drop-blur-pct 20
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "datasets" / "raw_videos"
FRAME_DIR = ROOT / "datasets" / "frames"


def find_camera(name="C930e"):
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        if name in (d / "name").read_text() and (d / "index").read_text().strip() == "0":
            return int(d.name.replace("video", ""))
    return None


def open_camera(idx, w, h, fps, focus, exposure):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if focus >= 0:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    else:  # -1: để camera tự lấy nét, giống node trên Pi khi chạy không có --focus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


def sharpness(frame):
    """Độ nét = phương sai Laplacian; càng nhỏ càng nhoè."""
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def record(args):
    idx = args.cam if args.cam is not None else find_camera()
    if idx is None:
        raise SystemExit("Không tìm thấy C930e. Dùng --cam <index>.")
    cap = open_camera(idx, args.width, args.height, args.fps, args.focus, args.exposure)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được /dev/video{idx}")
    for _ in range(20):  # cho camera ổn định
        cap.read()
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    print(f"Camera /dev/video{idx}: {w}x{h}, focus={cap.get(cv2.CAP_PROP_FOCUS):.0f}, "
          f"autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.0f}, exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.0f}")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    writer, out_path, n_frames, t0 = None, None, 0, 0.0
    end_at = time.time() + args.seconds if args.seconds else None

    def start():
        nonlocal writer, out_path, n_frames, t0
        out_path = VIDEO_DIR / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 args.fps, (w, h))
        n_frames, t0 = 0, time.time()
        print(f"● REC {out_path}")

    def stop():
        nonlocal writer
        if writer is not None:
            writer.release()
            print(f"■ Dừng: {n_frames} khung, {time.time() - t0:.1f}s -> {out_path}")
            writer = None

    if args.headless:
        start()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is not None:
            writer.write(frame)
            n_frames += 1
        if args.headless:
            if end_at and time.time() >= end_at:
                break
            continue
        view = frame.copy()
        status = f"REC {n_frames}f {time.time() - t0:.0f}s" if writer else "STANDBY"
        cv2.putText(view, f"[{args.tag}] {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255) if writer else (0, 255, 0), 2)
        cv2.imshow("record", cv2.resize(view, (960, 540)) if w > 960 else view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            stop() if writer else start()
        elif key == ord("s"):
            p = VIDEO_DIR / f"{args.tag}_{int(time.time())}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
            print(f"Lưu khung: {p}")
        elif key in (27, ord("q")):
            break
    stop()
    cap.release()
    cv2.destroyAllWindows()


def extract(args):
    """Lấy mỗi N khung, bỏ khung gần giống khung trước và khung nhoè nhất.

    Ngưỡng nhoè lấy theo phân vị của CHÍNH video đó, không dùng số tuyệt đối:
    độ nét Laplacian phụ thuộc cảnh/ánh sáng rất mạnh (đo trên dataset cũ:
    trung vị 139 ở video này nhưng chỉ 57 ở video khác), nên một ngưỡng cố
    định sẽ hoặc không bỏ gì, hoặc bỏ sạch.
    """
    for vid in args.videos:
        vid = Path(vid)
        out = FRAME_DIR / vid.stem
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            # Chạy lại với tham số khác sẽ ghi đè và có thể xoá bớt ảnh cũ
            # trong thư mục này (ảnh nhoè bị loại), nên báo trước.
            print(f"⚠ {out} đã có ảnh — lần chạy này sẽ ghi đè theo tham số mới")
        cap = cv2.VideoCapture(str(vid))
        i = dup = 0
        last = None
        kept = []  # (đường dẫn, độ nét)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if last is not None and np.abs(small - last).mean() <= args.min_diff:
                    dup += 1
                else:
                    path = out / f"{vid.stem}_{i:06d}.jpg"
                    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
                    kept.append((path, sharpness(frame)))
                    last = small
            i += 1
        cap.release()

        blur = 0
        if args.drop_blur_pct > 0 and kept:
            vals = np.array([v for _, v in kept])
            thr = np.percentile(vals, args.drop_blur_pct)
            for path, v in kept:
                if v < thr:
                    path.unlink()
                    blur += 1
            print(f"  độ nét: p10={np.percentile(vals, 10):.0f} "
                  f"p50={np.percentile(vals, 50):.0f} p90={np.percentile(vals, 90):.0f}, "
                  f"bỏ khung dưới {thr:.0f}")
        print(f"{vid.name}: {i} khung -> giữ {len(kept) - blur} ảnh trong {out} "
              f"(bỏ {dup} trùng, {blur} nhoè)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--tag", default="session", help="Tên phiên/điều kiện, vd sang_deu, nguoc_sang")
    r.add_argument("--cam", type=int, default=None, help="Bỏ trống để tự tìm C930e")
    r.add_argument("--width", type=int, default=1280)
    r.add_argument("--height", type=int, default=720)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--focus", type=int, default=0,
                   help="Lấy nét thủ công (0=xa, ~100+=gần); -1 = tự lấy nét")
    r.add_argument("--exposure", type=int, default=None,
                   help="Phơi sáng thủ công (đơn vị 100us, vd 150); bỏ trống = tự động")
    r.add_argument("--headless", action="store_true", help="Không hiển thị (để test)")
    r.add_argument("--seconds", type=float, default=0, help="Chỉ dùng với --headless")
    r.set_defaults(fn=record)
    e = sub.add_parser("extract")
    e.add_argument("videos", nargs="+")
    e.add_argument("--every", type=int, default=6)
    e.add_argument("--min-diff", type=float, default=2.0, help="Ngưỡng khác biệt để giữ khung")
    e.add_argument("--drop-blur-pct", type=float, default=0.0,
                   help="Bỏ N%% khung nhoè nhất của mỗi video (vd 20); 0=tắt")
    e.set_defaults(fn=extract)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
