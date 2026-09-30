"""Export yolov8n-pose COCO gốc ở nhiều imgsz -> .tflite để ĐO TỐC ĐỘ trên Pi.

Tốc độ suy luận chỉ phụ thuộc kiến trúc + imgsz, KHÔNG phụ thuộc trọng số đã
train. Nên đo bằng model COCO gốc cho biết ngay imgsz=224 nhanh cỡ nào, mà
không phải train trước rồi mới biết có đáng không.

Khác biệt duy nhất: model COCO có 17 keypoint thay vì 4 -> chỉ khác vài kênh ở
lớp cuối, chênh lệch tốc độ <2%, vẫn đại diện tốt.
"""
import shutil, sys
from pathlib import Path
from ultralytics import YOLO

OUT = Path(__file__).parent
SIZES = [320, 256, 224, 192]

for sz in SIZES:
    dst = OUT / f"probe_{sz}_float32.tflite"
    if dst.exists():
        print(f"[bỏ qua] đã có {dst.name}"); continue
    print(f"\n{'='*60}\n>>> EXPORT imgsz={sz}\n{'='*60}", flush=True)
    try:
        YOLO("yolov8n-pose.pt").export(format="tflite", imgsz=sz, int8=False)
    except Exception as e:
        print(f"!!! LỖI imgsz={sz}: {e}", file=sys.stderr); continue
    src = next(Path("yolov8n-pose_saved_model").glob("*_float32.tflite"), None)
    if src is None:
        print(f"!!! không tìm thấy file .tflite cho imgsz={sz}", file=sys.stderr); continue
    shutil.move(str(src), dst)
    shutil.rmtree("yolov8n-pose_saved_model", ignore_errors=True)
    print(f"--> {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")

print("\n=== KẾT QUẢ ===")
for f in sorted(OUT.glob("probe_*_float32.tflite")):
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
