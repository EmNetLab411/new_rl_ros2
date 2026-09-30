#!/usr/bin/env python3
"""Gán nhãn tự động ảnh bút nhỏ/nghiêng bằng model CŨ + cắt ô + xoay.

Model cũ (pen_pose_224/192) chỉ quen bút TO (>=34% khung) và ĐỨNG THẲNG
(±16°), nên chạy thẳng trên ảnh mới gần như mù (6-11% ảnh có box). Nhưng nó
không nhận nhầm nền (0 box cỡ cái thuyền trên 297 ảnh), khác bản rot_sc. Nên:
  1. cắt ảnh thành các ô vuông chồng nhau (360 và 540px) -> bút to lên
  2. xoay mỗi ô 12 góc (bước 30°) -> luôn có 1 góc bút gần đứng thẳng
  3. chạy model trên tất cả, đổi điểm ngược về toạ độ ảnh gốc
  4. chỉ nhận khi NHIỀU ô/góc cùng chỉ ra một chỗ (đồng thuận), lấy trung vị

L/R ra theo hệ của BÚT (trái/phải khi mũi bút hướng lên), đúng với PEN_3D mà
solvePnP dùng — không phải trái/phải theo ảnh.

    .venv-train/bin/python scripts/autolabel.py --dirs "nghieng_xa_*" --every 3

Ghi thẳng vào datasets/local_labels (danh sách "auto" trong _progress.json),
kèm preview để duyệt. label_local.py sẽ bỏ qua các ảnh này, --merge sẽ gộp.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import label_local as LL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREVIEW = LL.LABEL_OUT / "_preview"
ANGLES = list(range(0, 360, 30))
COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (100, 0, 255)]


def tiles(W, H, sizes=(360, 540)):
    for S in sizes:
        st = S // 2
        xs = list(range(0, W - S + 1, st)); xs += [W - S] if xs[-1] != W - S else []
        ys = list(range(0, H - S + 1, st)); ys += [H - S] if ys[-1] != H - S else []
        for y in ys:
            for x in xs:
                yield x, y, S


def candidates(model, img, imgsz, conf, sizes=(360, 540)):
    """Mọi phát hiện trên mọi (ô, góc): list (conf, min kp conf, pts 4x2 toạ độ gốc)."""
    H, W = img.shape[:2]
    crops, inv = [], []
    for x0, y0, S in tiles(W, H, sizes):
        crop = img[y0:y0 + S, x0:x0 + S]
        for a in ANGLES:
            M = cv2.getRotationMatrix2D((S / 2, S / 2), a, 1.0)
            crops.append(cv2.warpAffine(crop, M, (S, S), borderValue=(114, 114, 114)))
            Mi = cv2.invertAffineTransform(M); Mi[:, 2] += (x0, y0)
            inv.append(Mi)
    out = []
    for i in range(0, len(crops), 256):
        for r, Mi in zip(model.predict(crops[i:i + 256], imgsz=imgsz, conf=conf,
                                       half=True, verbose=False), inv[i:i + 256]):
            if not len(r.boxes):
                continue
            j = int(r.boxes.conf.argmax())
            k = r.keypoints.xy[j].cpu().numpy()
            pts = np.c_[k, np.ones(4)] @ Mi.T
            out.append((float(r.boxes.conf[j]), float(r.keypoints.conf[j].min()), pts))
    return out


def consensus(cands, min_votes):
    """Chọn cụm đông nhất: phát hiện 'đồng ý' nếu cả 4 điểm cách nhau < 25%
    chiều dài bút. Trả (pts trung vị, số phiếu, conf tốt nhất) hoặc None."""
    good = [c for c in cands if c[1] >= 0.5]
    if not good:
        return None
    best = None
    for c in sorted(good, key=lambda c: -c[0] * c[1])[:20]:
        L = max(np.linalg.norm(c[2][0] - c[2][1]), 8.0)
        agree = [g for g in good if np.all(np.linalg.norm(g[2] - c[2], axis=1) < 0.25 * L)]
        if best is None or len(agree) > len(best[1]):
            best = (c, agree)
    c, agree = best
    if len(agree) < min_votes:
        return None
    return np.median([g[2] for g in agree], axis=0), len(agree), max(g[0] for g in agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="nghieng_xa_*")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--model", default=str(ROOT / "imgsz_probe/runs/pen_pose_224/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--sizes", default="360,540", help="Cỡ ô cắt (px), vd 360,540,720")
    ap.add_argument("--only-missing", action="store_true",
                    help="Chỉ chạy ảnh CHƯA có nhãn auto (lượt 2 với tham số nới hơn)")
    ap.add_argument("--limit", type=int, default=0, help="Chỉ chạy N ảnh đầu (thử)")
    args = ap.parse_args()

    files = LL.select_files(args.dirs, args.every)
    if args.limit:
        files = files[:args.limit]
    prog = LL.load_progress(); prog.setdefault("auto", [])
    manual = set(prog["done"]) | set(prog["empty"]) | set(prog["skipped"])
    PREVIEW.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    stats = {"auto": 0, "none": 0}
    rp = LL.LABEL_OUT / "_auto_report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    for n, f in enumerate(files, 1):
        if f.stem in manual or (args.only_missing and f.stem in prog["auto"]):
            continue
        img = cv2.imread(str(f))
        sizes = tuple(int(v) for v in args.sizes.split(","))
        res = consensus(candidates(model, img, args.imgsz, args.conf, sizes), args.min_votes)
        if res is None:
            stats["none"] += 1
            if f.stem in prog["auto"]:
                prog["auto"].remove(f.stem)
                (LL.LABEL_OUT / f"{f.stem}.txt").unlink(missing_ok=True)
            continue
        pts, votes, conf = res
        LL.write_yolo_label(LL.LABEL_OUT / f"{f.stem}.txt", [tuple(p) for p in pts],
                            [False] * 4, img.shape)
        shutil.copy(f, LL.LABEL_OUT / f.name)
        if f.stem not in prog["auto"]:
            prog["auto"].append(f.stem)
        report[f.stem] = {"votes": votes, "conf": round(conf, 3), "pass": args.sizes}
        stats["auto"] += 1
        # preview: ảnh gốc thu nhỏ + ô phóng to quanh bút để soi điểm
        vis = img.copy()
        for k, p in enumerate(pts):
            cv2.circle(vis, tuple(int(v) for v in p), 4, COLORS[k], -1)
        cv2.polylines(vis, [pts[[0, 2, 1, 3]].astype(np.int32)], True, (255, 255, 255), 1)
        c = pts.mean(0); L = max(np.linalg.norm(pts[0] - pts[1]), 30)
        x0, y0 = (c - 1.2 * L).astype(int).clip(0); x1, y1 = (c + 1.2 * L).astype(int)
        zoom = cv2.resize(vis[y0:y1, x0:x1], (360, 360))
        full = cv2.resize(vis, (640, 360))
        cv2.putText(full, f"v{votes} c{conf:.2f}", (8, 24), 0, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(PREVIEW / f"{f.stem}.jpg"), np.hstack([full, zoom]))
        if n % 20 == 0:
            LL.save_progress(prog)
            print(f"  {n}/{len(files)}  auto {stats['auto']}  không chắc {stats['none']}", flush=True)
    LL.save_progress(prog)
    (LL.LABEL_OUT / "_auto_report.json").write_text(json.dumps(report, indent=0))
    print(f"XONG {len(files)} ảnh: gán tự động {stats['auto']}, bỏ (không đủ đồng thuận) {stats['none']}")


if __name__ == "__main__":
    main()
