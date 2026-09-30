#!/usr/bin/env python3
"""
Chọn ra một tập nhỏ, đa dạng nhất trong số ảnh CHƯA được prelabel_dataset.py
tự gán nhãn, để gán tay tối thiểu rồi dùng train lại model (bootstrap).

Cách chọn: gom cụm bằng k-means trên đặc trưng ảnh (thumbnail xám + vị trí/
kích thước vùng xanh của vỏ bút, để phân biệt góc/khoảng cách/tư thế), rồi
lấy 1 ảnh gần tâm cụm nhất mỗi cụm — tránh chọn trùng các khung liền kề
gần giống hệt nhau trong video.

Chạy (từ thư mục CoVip), sau khi đã chạy prelabel_dataset.py:
    python3 scripts/select_bootstrap_subset.py --k 260
"""
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "datasets" / "frames"
UPLOAD_DIR = ROOT / "datasets" / "roboflow_upload"
OUT_ZIP = UPLOAD_DIR / "bootstrap_manual_label.zip"


def already_labeled_names():
    labeled = set()
    for z in UPLOAD_DIR.glob("*.zip"):
        if z == OUT_ZIP:
            continue
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".txt") and "/labels/" in n:
                    labeled.add(Path(n).stem)
    return labeled


def blue_feature(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, 80, 40), (135, 255, 255))
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    return np.array([xs.mean() / w, ys.mean() / h, len(xs) / (h * w),
                      (xs.max() - xs.min()) / w], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=260, help="Số ảnh muốn giữ lại để gán tay")
    args = ap.parse_args()

    labeled = already_labeled_names()
    candidates = [f for f in sorted(FRAMES_DIR.glob("*/*.jpg")) if f.stem not in labeled]
    print(f"Ảnh chưa có nhãn tự động: {len(candidates)}")

    feats = []
    for f in candidates:
        img = cv2.imread(str(f))
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (24, 24)).astype(np.float32).flatten() / 255.0
        bf = blue_feature(img)
        feats.append(np.concatenate([thumb * 0.6, bf * 4.0]))  # nhấn trọng số vị trí/kích thước bút
    X = np.array(feats, dtype=np.float32)

    k = min(args.k, len(candidates))
    _, labels, centers = cv2.kmeans(
        X, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
        attempts=3, flags=cv2.KMEANS_PP_CENTERS)

    selected = []
    for c in range(k):
        idx = np.where(labels.flatten() == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(X[idx] - centers[c], axis=1)
        selected.append(candidates[idx[d.argmin()]])

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in selected:
            zf.write(f, f.name)

    by_session = {}
    for f in selected:
        by_session[f.parent.name] = by_session.get(f.parent.name, 0) + 1
    print(f"\nĐã chọn {len(selected)} ảnh đa dạng -> {OUT_ZIP}")
    for s, n in sorted(by_session.items()):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
