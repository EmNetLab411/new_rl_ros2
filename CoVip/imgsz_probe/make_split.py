"""Tạo dataset có tập validation riêng, KHÔNG rò rỉ dữ liệu, từ bản export
Roboflow gốc (vốn chỉ có train, và dùng chính train làm val).

Hai điểm quan trọng:
  1. 3 bản augment của cùng 1 ảnh gốc PHẢI nằm cùng một phía (train hoặc val).
     Nếu tách ra 2 phía, val sẽ chấm điểm trên ảnh gần như giống hệt ảnh đã
     học -> điểm ảo cao, không phát hiện được học vẹt. Gom nhóm bằng
     (tiền tố tên, nội dung nhãn) vì augment chỉ đổi sáng/nhiễu, không đổi
     hình học nên nhãn của 3 bản giống hệt nhau.
  2. Tập val chỉ lấy 1 bản mỗi nhóm — chấm 3 bản gần giống nhau không cho
     thêm thông tin gì, chỉ tốn thời gian.

Dùng symlink, không nhân đôi 2445 ảnh trên đĩa.
"""
import os, re, random, shutil
from collections import defaultdict
from pathlib import Path

SRC = Path("/home/ducanh/new_rl_ros2/CoVip/COVIP_training.v4i.yolov8")
DST = Path("/home/ducanh/new_rl_ros2/CoVip/dataset_split")
VAL_FRAC, SEED = 0.15, 0

grp = defaultdict(list)
for p in sorted((SRC / "train/labels").glob("*.txt")):
    stem = p.stem
    pre = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)
    grp[(pre, p.read_text().strip())].append(stem)

# tách riêng nhóm CÓ bút và KHÔNG bút rồi chia theo cùng tỉ lệ, để val giữ
# đúng cơ cấu như train (không bị lệch toàn ảnh có bút hoặc toàn ảnh trống)
pen   = sorted(k for k in grp if k[1])
empty = sorted(k for k in grp if not k[1])
rng = random.Random(SEED)
rng.shuffle(pen); rng.shuffle(empty)

val_keys, train_keys = [], []
for bucket in (pen, empty):
    n = round(len(bucket) * VAL_FRAC)
    val_keys += bucket[:n]; train_keys += bucket[n:]

if DST.exists(): shutil.rmtree(DST)
for split in ("train", "val"):
    (DST / split / "images").mkdir(parents=True)
    (DST / split / "labels").mkdir(parents=True)

def link(stems, split):
    for s in stems:
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            src = (SRC / "train" / sub / (s + ext)).resolve()
            (DST / split / sub / (s + ext)).symlink_to(src)

n_tr = n_va = 0
for k in train_keys:
    link(grp[k], "train"); n_tr += len(grp[k])
for k in val_keys:
    link(grp[k][:1], "val"); n_va += 1   # chỉ 1 bản/nhóm cho val

# flip_idx [0,1,3,2]: khi lật ngang ảnh, keypoint L (mép trái) và R (mép phải)
# PHẢI đổi chỗ cho nhau. Bản gốc để [0,1,2,3] (không đổi) -> dạy sai L/R ở
# ~50% số frame vì Ultralytics bật fliplr=0.5 mặc định.
(DST / "data.yaml").write_text(f"""path: {DST}
train: train/images
val: val/images

kpt_shape: [4, 3]
flip_idx: [0, 1, 3, 2]

nc: 1
names:
  - pen_tip
""")

print(f"nhóm ảnh gốc : {len(grp)}  (có bút {len(pen)}, không bút {len(empty)})")
print(f"  -> train: {len(train_keys)} nhóm = {n_tr} ảnh (đủ 3 bản augment)")
print(f"  -> val  : {len(val_keys)} nhóm = {n_va} ảnh (1 bản/nhóm)")
print(f"data.yaml: {DST/'data.yaml'}   flip_idx ĐÃ SỬA thành [0,1,3,2]")
