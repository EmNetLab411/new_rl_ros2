"""Export yolov8n-pose COCO ở nhiều imgsz sang ONNX để ĐO TỐC ĐỘ trên Pi.

Dùng ONNX thay TFLite vì đường export TFLite của ultralytics 8.4.x đã chuyển
sang ai-edge-torch, đòi torch>=2.11 (venv đang 2.9.1) -> lỗi. ONNX vẫn cho
biết TỈ LỆ tăng tốc giữa các imgsz, đủ để quyết định có đáng train hay không;
việc chuyển sang TFLite chỉ cần làm 1 lần cho model cuối cùng.
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

OUT = Path(__file__).parent
for sz in (320, 256, 224, 192):
    dst = OUT / f"probe_{sz}.onnx"
    if dst.exists():
        print(f"[bỏ qua] {dst.name}"); continue
    print(f"\n>>> imgsz={sz}", flush=True)
    p = YOLO("yolov8n-pose.pt").export(format="onnx", imgsz=sz, simplify=True)
    shutil.move(str(p), dst)
    print(f"--> {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")

print("\n=== XONG ===")
for f in sorted(OUT.glob("probe_*.onnx")):
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")
