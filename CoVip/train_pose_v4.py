"""
train_pose_v4.py
================
Huấn luyện YOLOv8n-pose trên bộ dữ liệu bút, export ONNX FP32,
sau đó lượng tử hoá (quantize) sang INT8 bằng ONNX Runtime —
để deploy xuống Raspberry Pi 4 với độ trễ inference thấp nhất.

Yêu cầu môi trường (laptop Ubuntu / RTX 3050):
    pip install ultralytics torch torchvision onnx onnxruntime

Cách chạy:
    python train_pose_v4.py
    python train_pose_v4.py --skip-train     # Chỉ export (đã train rồi)
    python train_pose_v4.py --fp32-only       # Chỉ xuất FP32, không quantize INT8

Lưu ý quan trọng về INT8:
    Khác với --half (FP16, chỉ cast kiểu dữ liệu), INT8 cần "calibration" —
    chạy thử model trên một tập ảnh thật để đo phân phối giá trị từng layer,
    rồi mới tính được scale/zero-point lượng tử hoá đúng. Vì vậy quy trình
    ở đây là 2 bước: export ONNX FP32 trước → rồi quantize_static() riêng,
    thay vì 1 lệnh export(int8=True) như TFLite.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO


# ── Cấu hình đường dẫn ─────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
DATA_YAML    = SCRIPT_DIR / "COVIP_training.v4i.yolov8" / "data.yaml"
PROJECT_DIR  = SCRIPT_DIR / "runs" / "pose" / "Aero_Models"
RUN_NAME     = "pen_pose_v4-3"
BEST_WEIGHTS = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
IMGSZ        = 320

# ── Siêu tham số huấn luyện ────────────────────────────────────────────────
TRAIN_CFG = dict(
    epochs   = 150,
    patience = 30,
    imgsz    = IMGSZ,
    batch    = 16,
    verbose  = True,
)


def select_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU phát hiện: {name}  ({vram:.1f} GB VRAM)")
        return "0"
    print("⚠️  Không tìm thấy GPU — huấn luyện trên CPU (rất chậm!)")
    print("💡 Gợi ý: giảm epochs xuống 50–100 để thử nghiệm trước.")
    return "cpu"


def train(device: str) -> Path:
    """Huấn luyện và trả về đường dẫn best.pt thực tế."""
    if not DATA_YAML.exists():
        sys.exit(f"❌ Không tìm thấy data.yaml: {DATA_YAML}")

    print(f"\n📂 Dataset : {DATA_YAML}")
    print(f"📦 Project : {PROJECT_DIR}/{RUN_NAME}")
    print(f"⚙️  Config  : epochs={TRAIN_CFG['epochs']}, imgsz={TRAIN_CFG['imgsz']}, batch={TRAIN_CFG['batch']}\n")

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data    = str(DATA_YAML),
        device  = device,
        project = str(PROJECT_DIR),
        name    = RUN_NAME,
        **TRAIN_CFG,
    )

    best = None
    try:
        best = Path(model.trainer.best)
        if not best.exists():
            best = None
    except AttributeError:
        pass

    if best is None:
        try:
            save_dir = Path(model.trainer.save_dir)
            best = save_dir / "weights" / "best.pt"
        except AttributeError:
            pass

    if best is None or not best.exists():
        candidates = sorted(PROJECT_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            best = candidates[-1]

    if best is None or not best.exists():
        sys.exit(f"❌ Không tìm thấy best.pt trong {PROJECT_DIR}!")

    print(f"\n✅ Huấn luyện hoàn tất!")
    print(f"   Run folder : {best.parent.parent}")
    print(f"   Weights    : {best}")
    return best


def export_onnx_fp32(weights: Path) -> Path:
    """
    Bước 1: Export PyTorch (.pt) → ONNX FP32.
    Đây là file trung gian — luôn cần có trước khi quantize INT8,
    vì lượng tử hoá tĩnh (static quantization) của ONNX Runtime
    chỉ nhận đầu vào là graph ONNX FP32 chuẩn.
    """
    print(f"\n⏳ [1/2] Export ONNX FP32 (file trung gian cho quantize)...")
    print(f"   Weights nguồn: {weights}\n")

    model = YOLO(str(weights))
    try:
        export_path = model.export(
            format   = "onnx",
            half     = False,     # FP32 — bắt buộc cho bước calibration sau
            imgsz    = IMGSZ,
            simplify = True,
            nms      = False,
            opset    = 12,         # opset 12 tương thích tốt với onnxruntime trên ARM
        )
    except ModuleNotFoundError as e:
        print(f"\n❌ Thiếu thư viện: {e}")
        print("👉 Cài đặt: pip install onnx onnxslim")
        sys.exit(1)

    print(f"✅ FP32 ONNX: {export_path}")
    return Path(export_path)


def _collect_calibration_images(data_yaml: Path, n_images: int = 100) -> list[Path]:
    """
    Lấy ngẫu nhiên n_images ảnh từ tập train làm dữ liệu calibration.
    100 ảnh là đủ cho hầu hết model nhỏ như yolov8n-pose — nhiều hơn
    không cải thiện đáng kể nhưng làm calibration chậm hơn tuyến tính.
    """
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    base = data_yaml.parent
    train_rel = cfg.get("train", "train/images")
    train_dir = (base / train_rel).resolve()
    if not train_dir.exists():
        # data.yaml có thể chỉ trỏ tới thư mục images trực tiếp
        train_dir = base / "train" / "images"

    all_imgs = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    if not all_imgs:
        sys.exit(f"❌ Không tìm thấy ảnh calibration trong: {train_dir}")

    random.seed(42)
    chosen = random.sample(all_imgs, min(n_images, len(all_imgs)))
    print(f"   Đã chọn {len(chosen)} ảnh calibration từ {train_dir}")
    return chosen


class _PenCalibrationReader:
    """
    CalibrationDataReader cho onnxruntime.quantization.quantize_static.
    Tiền xử lý từng ảnh giống hệt _preprocess trong ONNXPoseInference
    (letterbox + normalize) để calibration phản ánh đúng input thực tế.
    """
    def __init__(self, image_paths: list, imgsz: int, input_name: str):
        self._paths      = image_paths
        self._imgsz      = imgsz
        self._input_name = input_name
        self._idx        = 0

    def get_next(self):
        if self._idx >= len(self._paths):
            return None
        path = self._paths[self._idx]
        self._idx += 1

        bgr = cv2.imread(str(path))
        if bgr is None:
            return self.get_next()  # ảnh lỗi → bỏ qua, lấy ảnh kế tiếp

        h0, w0 = bgr.shape[:2]
        r  = self._imgsz / max(h0, w0)
        nw, nh = int(w0 * r), int(h0 * r)
        canvas = np.full((self._imgsz, self._imgsz, 3), 114, np.uint8)
        dw, dh = (self._imgsz - nw) // 2, (self._imgsz - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(bgr, (nw, nh))

        rgb    = canvas[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # NCHW — khớp export opset 12

        return {self._input_name: tensor}

    def rewind(self):
        self._idx = 0


def quantize_to_int8(fp32_onnx_path: Path, data_yaml: Path) -> Path:
    """
    Bước 2: Lượng tử hoá tĩnh FP32 ONNX → INT8 ONNX.
    Dùng QDQ (Quantize-DeQuantize) format — tương thích rộng với
    onnxruntime trên ARM, kể cả các bản cũ hơn thường có trên Pi OS.
    """
    try:
        from onnxruntime.quantization import (
            quantize_static, QuantType, QuantFormat, CalibrationMethod
        )
    except ImportError:
        sys.exit("❌ Thiếu thư viện: pip install onnxruntime")

    print(f"\n⏳ [2/2] Quantize INT8 (QDQ format)...")

    calib_images = _collect_calibration_images(data_yaml, n_images=100)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess  # chỉ cần lấy tên input, giải phóng ngay

    reader = _PenCalibrationReader(calib_images, IMGSZ, input_name)

    int8_path = fp32_onnx_path.with_name(fp32_onnx_path.stem + "_int8.onnx")

    quantize_static(
        model_input        = str(fp32_onnx_path),
        model_output       = str(int8_path),
        calibration_data_reader = reader,
        quant_format        = QuantFormat.QDQ,
        activation_type      = QuantType.QInt8,
        weight_type          = QuantType.QInt8,
        calibrate_method     = CalibrationMethod.MinMax,
        per_channel          = False,   # per-tensor — nhanh hơn trên CPU ARM, đủ chính xác cho model nhỏ
    )

    fp32_size = fp32_onnx_path.stat().st_size / 1e6
    int8_size = int8_path.stat().st_size / 1e6
    print(f"\n✅ Quantize hoàn tất!")
    print(f"   FP32 : {fp32_size:.1f} MB  →  INT8 : {int8_size:.1f} MB  "
          f"(giảm {(1 - int8_size/fp32_size)*100:.0f}%)")
    print(f"   File : {int8_path}")
    return int8_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Bỏ qua bước train, dùng best.pt sẵn có")
    parser.add_argument("--fp32-only", action="store_true",
                        help="Chỉ export FP32 ONNX, không quantize INT8")
    parser.add_argument("--n-calib", type=int, default=100,
                        help="Số ảnh dùng calibration INT8 (mặc định 100)")
    args = parser.parse_args()

    if args.skip_train:
        if not BEST_WEIGHTS.exists():
            sys.exit(f"❌ --skip-train nhưng không tìm thấy: {BEST_WEIGHTS}")
        print(f"⏭️  Bỏ qua train, dùng: {BEST_WEIGHTS}")
        weights = BEST_WEIGHTS
    else:
        device  = select_device()
        weights = train(device)

    fp32_path = export_onnx_fp32(weights)

    if args.fp32_only:
        print(f"\n📋 Copy sang Pi 4:")
        print(f"   scp {fp32_path} pi@<PI_IP>:~/aeroscript/")
        return

    int8_path = quantize_to_int8(fp32_path, DATA_YAML)

    print()
    print("─" * 60)
    print("📋 BƯỚC TIẾP THEO — copy sang Raspberry Pi 4:")
    print(f"   scp {int8_path} pi@<PI_IP>:~/aeroscript/")
    print(f"   scp run_pi4_ros2.py        pi@<PI_IP>:~/aeroscript/")
    print("─" * 60)
    print("\n⚠️  Nếu INT8 detect kém hơn rõ rệt so với FP32 (hay xảy ra với")
    print("   model rất nhỏ), dùng lại FP32 — trên Pi 4 chênh lệch tốc độ")
    print("   FP32 vs INT8 thường chỉ ~20-30%, không lớn như TFLite.")


if __name__ == "__main__":
    main()