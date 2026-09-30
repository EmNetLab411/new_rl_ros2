#!/usr/bin/env python3
"""Train lại yolov8n-pose ở imgsz nhỏ hơn để đạt 15-20fps trên Pi 4.

Lý do train lại (xem PLAN.md mục 5b/5c): model hiện tại chạy imgsz=320, riêng
lệnh invoke() đã tốn 116ms trong điều kiện lý tưởng -> trần cứng ~8fps trên
CPU Pi 4. Giảm xuống 224 cắt FLOPs còn ~49%, là đường duy nhất tới 15fps mà
không cần mua thêm phần cứng.

Dataset dùng bản ĐÃ TÁCH VAL của imgsz_probe/make_split.py, KHÔNG dùng thẳng
bản Roboflow gốc, vì bản gốc có 2 lỗi:
  - không có tập val (val trỏ vào chính train) -> không đo được học vẹt
  - flip_idx [0,1,2,3] sai: khi lật ngang ảnh, keypoint L (mép trái) và R
    (mép phải) phải đổi chỗ, nếu không ~50% frame bị dạy sai L/R

Chạy:
    ../.venv-train/bin/python train_pose.py --imgsz 224
    ../.venv-train/bin/python train_pose.py --imgsz 224 --epochs 200
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
DATA = HERE.parent / "dataset_split" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=40,
                    help="Dừng sớm nếu val không cải thiện sau ngần này epoch.")
    ap.add_argument("--degrees", type=float, default=0.0,
                    help="Xoay ảnh ngẫu nhiên ±độ khi train. Dataset hiện có 100%% bút "
                         "đứng thẳng (±16°) nên model mất nhận diện khi bút nghiêng/"
                         "nằm ngang; đặt 180 để model học mọi hướng.")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Phóng/thu ảnh ngẫu nhiên ±tỉ lệ (0.5 = mặc định Ultralytics). "
                         "Bút trong dataset luôn to (cao >=34%% khung); tăng lên 0.8-0.9 "
                         "để model học cả bút ở xa, nhỏ trong khung.")
    ap.add_argument("--model", default="yolov8n-pose.pt",
                    help="Trọng số khởi đầu — mặc định bản COCO pretrain, "
                         "hội tụ nhanh hơn nhiều so với train từ số ngẫu nhiên.")
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Chưa có {DATA} — chạy make_split.py trước.")

    dev = "0" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("⚠️  KHÔNG thấy GPU — train trên CPU sẽ mất nhiều giờ.\n"
              "    Cài driver rồi chạy lại:  sudo ubuntu-drivers autoinstall && sudo reboot\n"
              "    (xem PLAN.md mục 5c)\n")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # Hậu tố tên để không ghi đè model cũ khi thử augment khác
    suffix = ("_rot" if args.degrees > 0 else "") + ("_sc" if args.scale != 0.5 else "")
    name = f"pen_pose_{args.imgsz}{suffix}"
    model = YOLO(args.model)
    model.train(
        data=str(DATA), imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, patience=args.patience, device=dev,
        project=str(HERE / "runs"), name=name, exist_ok=True,
        # fliplr an toàn vì data.yaml đã có flip_idx [0,1,3,2] đổi chỗ L<->R
        fliplr=0.5, degrees=args.degrees, scale=args.scale,
    )

    best = HERE / "runs" / name / "weights" / "best.pt"
    print(f"\n=== Train xong: {best} ===")

    # ONNX TRƯỚC — đường này chắc chắn chạy được, đảm bảo luôn có file đem lên
    # Pi dù TFLite có hỏng. Trọng số best.pt vẫn giữ nguyên nên export lại lúc
    # nào cũng được.
    onnx_out = HERE / f"{name}.onnx"
    shutil.move(str(YOLO(str(best)).export(format="onnx", imgsz=args.imgsz,
                                           simplify=True)), onnx_out)
    print(f"=== ONNX: {onnx_out.name} ({onnx_out.stat().st_size/1e6:.1f} MB) ===")

    # TFLite float32 — nhanh hơn ONNX ~30% trên Pi 4 nên vẫn đáng có, nhưng
    # ultralytics 8.4.x đã chuyển sang ai-edge-torch đòi torch>=2.11 nên có
    # thể lỗi. KHÔNG dùng float16 (Cortex-A72 không có phần cứng fp16, chạy
    # CHẬM HƠN fp32: 190ms vs 116ms) và KHÔNG dùng int8 (đã xác nhận làm mất
    # hẳn khả năng detect bút).
    tfl_out = None
    try:
        YOLO(str(best)).export(format="tflite", imgsz=args.imgsz, int8=False)
        src = next(Path(".").rglob("*_float32.tflite"), None)
        if src:
            tfl_out = HERE / f"{name}_float32.tflite"
            shutil.copy(src, tfl_out)
            print(f"=== TFLite: {tfl_out.name} ({tfl_out.stat().st_size/1e6:.1f} MB) ===")
    except Exception as e:
        print(f"\n⚠️  Export TFLite hỏng (đã biết trước, xem PLAN.md mục 5c): {e}\n"
              f"    Không sao — dùng bản ONNX ở trên, chậm hơn ~30%.\n"
              f"    Muốn có TFLite: nâng torch>=2.11 hoặc hạ ultralytics về 8.3.x.")

    use = (tfl_out or onnx_out).name
    print(f"\nThêm vào deploy_to_pi.sh rồi chạy trên Pi:\n"
          f"  python3 run_pi4_ros2.py --model {use} --device /dev/video0 \\\n"
          f"      --width 640 --height 360 --fourcc MJPG --conf 0.55 --threads 2")


if __name__ == "__main__":
    main()
