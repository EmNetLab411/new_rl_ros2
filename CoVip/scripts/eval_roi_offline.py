#!/usr/bin/env python3
"""
Đo ROI có lợi thật không — CHỈ cần laptop + bộ ảnh đã gán nhãn, KHÔNG cần
robot/Pi/camera.

Ý tưởng: khi có robot, FK sẽ cho biết bút nằm gần đâu trên ảnh để cắt ô ROI.
Ở đây chưa có robot nên lấy vị trí bút từ NHÃN, cộng thêm sai số ngẫu nhiên
(--fk-err-px) để giả lập FK + hand-eye không chính xác, rồi cắt ô quanh đó.
So với cách đang chạy trên Pi (cả khung 640x360 bóp về 192):

    - tỉ lệ detect
    - sai số điểm khớp so với nhãn (px, quy về ảnh 1280x720)
    - sai số Z sau solvePnP so với Z tính từ nhãn (mm)

Chạy (từ thư mục CoVip):
    python3 scripts/eval_roi_offline.py
    python3 scripts/eval_roi_offline.py --fk-err-px 0 20 --roi-sizes 192 320 --save-vis 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_roi_detection import PoseONNX  # noqa: E402

PEN_3D = np.array([[0.0, 0.0, 0.0], [0.0, 64.0, 0.0],
                   [-11.5, 44.0, 0.0], [11.5, 44.0, 0.0]], dtype=np.float32)
FULL_W, FULL_H = 1280, 720


def load_label(path):
    """Trả 4 điểm khớp (px ở 1280x720) hoặc None nếu thiếu/có điểm bị che."""
    rows = path.read_text().split("\n")
    if not rows or not rows[0].strip():
        return None
    v = np.array(rows[0].split(), dtype=np.float64)
    if len(v) < 17:
        return None
    k = v[5:17].reshape(4, 3)
    if np.any(k[:, 2] < 2):
        return None
    return k[:, :2] * [FULL_W, FULL_H]


def pnp_z(kpts, K, dist):
    ok, _, t = cv2.solvePnP(PEN_3D, kpts.astype(np.float64), K, dist,
                            flags=cv2.SOLVEPNP_IPPE)
    return float(t[2][0]) if ok and t[2][0] > 0 else None


def crop_at(img, cx, cy, size):
    """Cắt ô size x size quanh (cx, cy), đẩy vào trong biên ảnh."""
    h, w = img.shape[:2]
    x0 = int(np.clip(round(cx - size / 2), 0, w - size))
    y0 = int(np.clip(round(cy - size / 2), 0, h - size))
    return img[y0:y0 + size, x0:x0 + size], x0, y0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "imgsz_probe" / "pen_pose_192_sc.onnx"))
    ap.add_argument("--images", default=str(ROOT / "dataset_split" / "val" / "images"))
    ap.add_argument("--calib", default=str(ROOT / "calib" / "c930e_720p.npz"))
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--fk-err-px", type=float, nargs="+", default=[0, 20, 40],
                    help="Độ lệch chuẩn của sai số tâm ROI, px ở ảnh 640x360 "
                         "(20px ≈ 18mm khi bút cách camera 40cm)")
    ap.add_argument("--roi-sizes", type=int, nargs="+", default=[192, 256, 320],
                    help="Cạnh ô cắt, px ở ảnh 640x360")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-vis", type=int, default=0,
                    help="Lưu N ảnh so sánh vào logs/roi_eval/")
    args = ap.parse_args()

    model = PoseONNX(args.model, conf=args.conf)
    d = np.load(args.calib)
    K, dist = d["K"].astype(np.float64), d["dist"].astype(np.float64).reshape(-1, 1)
    rng = np.random.default_rng(args.seed)

    img_dir = Path(args.images)
    lab_dir = img_dir.parent / "labels"
    samples = []
    for p in sorted(img_dir.glob("*.jpg")):
        lab = lab_dir / (p.stem + ".txt")
        if not lab.exists():
            continue
        gt = load_label(lab)
        img = cv2.imread(str(p))
        if gt is None or img is None or img.shape[:2] != (FULL_H, FULL_W):
            continue
        samples.append((p, img, gt))
    print(f"Model: {Path(args.model).name} (vào {model.imgsz}px) | "
          f"{len(samples)} ảnh 1280x720 có đủ 4 điểm từ {img_dir}")

    # (tên, độ phân giải nguồn so với 720p, cạnh ô cắt ở nguồn; None = cả khung)
    modes = [("Cả khung 640x360 (đang chạy trên Pi)", 0.5, None)]
    for e in args.fk_err_px:
        for rs in args.roi_sizes:
            modes.append((f"ROI {rs}px (ở 640x360), lệch tâm {e:g}px", 0.5, rs, e))

    vis_dir = ROOT / "logs" / "roi_eval"
    rows = []
    for mode in modes:
        name, scale, size = mode[:3]
        err = mode[3] if len(mode) > 3 else 0.0
        n_det, kp_err, z_err, pen_px, cut = 0, [], [], [], 0
        for i, (p, img, gt) in enumerate(samples):
            src = img if scale == 1.0 else cv2.resize(
                img, (int(FULL_W * scale), int(FULL_H * scale)), interpolation=cv2.INTER_AREA)
            if size is None:
                inp, x0, y0 = src, 0, 0
            else:
                c = gt.mean(axis=0) * scale + rng.normal(0, err * scale / 0.5, 2)
                inp, x0, y0 = crop_at(src, c[0], c[1], size)
                g = gt * scale - [x0, y0]
                if np.any(g < 0) or np.any(g >= size):
                    cut += 1
            # Bút dài bao nhiêu px khi tới model
            pen_px.append(np.linalg.norm(gt[0] - gt[1]) * scale
                          * model.imgsz / max(inp.shape[:2]))
            k, _ = model(inp)
            if k is None:
                continue
            n_det += 1
            k_full = (k + [x0, y0]) / scale
            kp_err.append(np.linalg.norm(k_full - gt, axis=1).mean())
            z_gt, z_pr = pnp_z(gt, K, dist), pnp_z(k_full, K, dist)
            if z_gt and z_pr:
                z_err.append(abs(z_pr - z_gt))
            if i < args.save_vis:
                vis_dir.mkdir(parents=True, exist_ok=True)
                v = img.copy()
                if size is not None:
                    cv2.rectangle(v, (int(x0 / scale), int(y0 / scale)),
                                  (int((x0 + size) / scale), int((y0 + size) / scale)),
                                  (0, 255, 255), 2)
                for a, b in zip(gt, k_full):
                    cv2.circle(v, tuple(int(t) for t in a), 5, (0, 255, 0), 1)
                    cv2.drawMarker(v, tuple(int(t) for t in b), (0, 0, 255),
                                   cv2.MARKER_CROSS, 10, 1)
                cv2.imwrite(str(vis_dir / f"{p.stem[-12:]}_m{modes.index(mode)}.jpg"), v)
        n = len(samples)
        rows.append((name, 100 * n_det / n, np.median(pen_px),
                     np.median(kp_err) if kp_err else np.nan,
                     np.percentile(kp_err, 90) if kp_err else np.nan,
                     np.median(z_err) if z_err else np.nan,
                     np.percentile(z_err, 90) if z_err else np.nan, 100 * cut / n))

    print(f"\n{'Cách chạy':<44}{'detect':>8}{'bút(px)':>9}{'điểm khớp px':>16}"
          f"{'Z lệch mm':>15}{'bút lọt ô':>11}")
    print(f"{'':<44}{'%':>8}{'vào model':>9}{'giữa / 90%':>16}{'giữa / 90%':>15}{'% ảnh':>11}")
    for name, det, ppx, ke, ke90, ze, ze90, cut in rows:
        print(f"{name:<44}{det:>8.1f}{ppx:>9.0f}{ke:>9.1f} /{ke90:>5.1f}"
              f"{ze:>8.1f} /{ze90:>5.1f}{cut:>11.1f}")
    print("\nĐiểm khớp px: lệch so với nhãn, quy về ảnh 1280x720. "
          "Z lệch: so với Z tính từ chính nhãn đó.\n"
          "Bút lọt ô: % ảnh có ít nhất 1 điểm khớp nằm ngoài ô cắt.")
    if args.save_vis:
        print(f"Ảnh so sánh: {vis_dir}")


if __name__ == "__main__":
    main()
