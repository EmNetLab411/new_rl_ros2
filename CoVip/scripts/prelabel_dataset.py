#!/usr/bin/env python3
"""
Tự động gán nhãn trước cho datasets/frames/ bằng best.onnx, chạy kiểu quét
nhiều ô nhỏ (tiling) để bù việc ảnh 1280x720 rộng hơn nhiều so với ảnh
model được train (cận cảnh). Chỉ ghi nhãn cho ảnh model tự tin (>=98%
theo benchmark), để không đưa dữ liệu sai vào dataset.

Chạy (từ thư mục CoVip):
    python3 scripts/prelabel_dataset.py

Kết quả:
    datasets/roboflow_upload/<session>.zip  — ảnh + nhãn (nếu có) + data.yaml,
    sẵn sàng kéo-thả lên Roboflow (Upload Data sẽ tự nhận annotation kèm theo).
    Ảnh chưa có nhãn vẫn nằm trong zip (không có .txt) để bạn/teammate gán tay
    phần còn lại — không cần zip riêng ảnh trắng nữa.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from pen_webcam_onnx import PoseONNX  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "datasets" / "frames"
OUT_DIR = ROOT / "datasets" / "roboflow_upload"
CONF_KEEP = 0.55  # chỉ giữ nhãn khi cả 4 điểm keypoint-conf >= ngưỡng này


def tiled_detect(model, img, tile=440, stride=300):
    h, w = img.shape[:2]
    best = None
    ys = sorted(set(range(0, max(h - tile, 0) + 1, stride)) | {max(h - tile, 0)})
    xs = sorted(set(range(0, max(w - tile, 0) + 1, stride)) | {max(w - tile, 0)})
    for y in ys:
        for x in xs:
            crop = img[y:y + tile, x:x + tile]
            k, v = model(crop)
            if k is not None:
                score = float(v.min())
                if best is None or score > best[0]:
                    kk = k.copy()
                    kk[:, 0] += x
                    kk[:, 1] += y
                    best = (score, kk, v)
    k, v = model(img)  # phòng trường hợp bút to, thấy rõ cả khung
    if k is not None and (best is None or float(v.min()) > best[0]):
        best = (float(v.min()), k, v)
    return best


def to_yolo_label(kpts, h, w, margin=1.4):
    xs, ys = kpts[:, 0], kpts[:, 1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * margin, (y1 - y0) * margin
    parts = [0, cx / w, cy / h, bw / w, bh / h]
    for x, y in kpts:
        parts += [x / w, y / h, 2]  # visibility=2: model gán, cần người soát lại phần che khuất
    return " ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts)


def main():
    model = PoseONNX(str(ROOT / "pen_models" / "best.onnx"), conf=0.25)
    sessions = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_yaml = (
        "train: train/images\n"
        "val: train/images\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [0, 1, 3, 2]\n"
        "nc: 1\n"
        "names:\n  - pen_tip\n"
    )

    total_img = total_lbl = 0
    for sess in sessions:
        files = sorted(sess.glob("*.jpg"))
        work = ROOT / "datasets" / "_prelabel_tmp" / sess.name
        (work / "train" / "images").mkdir(parents=True, exist_ok=True)
        (work / "train" / "labels").mkdir(parents=True, exist_ok=True)
        n_lbl = 0
        for f in files:
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            shutil.copy(f, work / "train" / "images" / f.name)
            r = tiled_detect(model, img)
            if r is not None and r[0] >= CONF_KEEP:
                (work / "train" / "labels" / (f.stem + ".txt")).write_text(to_yolo_label(r[1], h, w) + "\n")
                n_lbl += 1
        (work / "data.yaml").write_text(data_yaml)

        zpath = OUT_DIR / f"{sess.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in work.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(work))
        shutil.rmtree(work)

        print(f"{sess.name}: {len(files)} ảnh, {n_lbl} có nhãn tự động ({n_lbl/len(files)*100:.0f}%) -> {zpath.name}")
        total_img += len(files)
        total_lbl += n_lbl

    shutil.rmtree(ROOT / "datasets" / "_prelabel_tmp", ignore_errors=True)
    print(f"\nTổng: {total_lbl}/{total_img} ảnh ({total_lbl/total_img*100:.0f}%) đã có nhãn tự động, "
          f"còn {total_img-total_lbl} ảnh cần gán tay.")


if __name__ == "__main__":
    main()
