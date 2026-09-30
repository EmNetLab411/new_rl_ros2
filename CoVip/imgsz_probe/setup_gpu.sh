#!/usr/bin/env bash
# setup_gpu.sh — Chạy SAU KHI đã cài driver NVIDIA và khởi động lại máy.
#
# Việc cần làm: venv .venv-train đang có torch bản CPU (ultralytics tự kéo về
# lúc cài), nên dù driver đã chạy vẫn không dùng được RTX 3060. Script này
# thay bằng bản CUDA rồi kiểm tra lại.
#
# Chạy:
#   ./CoVip/imgsz_probe/setup_gpu.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HERE}/../.venv-train/bin/python"
PIP="${HERE}/../.venv-train/bin/pip"

echo "== 1. Kiểm tra driver =="
if ! command -v nvidia-smi >/dev/null; then
    echo "❌ Không có nvidia-smi. Driver chưa cài xong hoặc chưa khởi động lại máy."
    echo "   Chạy: sudo ubuntu-drivers autoinstall && sudo reboot"
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "❌ nvidia-smi lỗi — nhiều khả năng CHƯA KHỞI ĐỘNG LẠI sau khi cài driver."
    echo "   Chạy: sudo reboot"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo

echo "== 2. torch trong venv hiện tại =="
CUR="$(${PY} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "chưa có")"
echo "   ${CUR}"

if [[ "${CUR}" == *"+cpu"* || "${CUR}" == "chưa có" ]]; then
    echo
    echo "== 3. Cài torch bản CUDA (tải ~2.5GB, mất vài phút) =="
    # cu128 khớp với dòng torch 2.9.x mà ultralytics 8.4.x dùng; driver 595 thừa mới để chạy.
    ${PIP} install --force-reinstall torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "   -> đã là bản CUDA, bỏ qua bước cài lại."
fi

echo
echo "== 4. Kiểm tra lại =="
${PY} - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}   cuda.is_available() = {ok}")
if ok:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n✅ SẴN SÀNG TRAIN. Chạy:")
    print("   cd /home/ducanh/new_rl_ros2/CoVip/imgsz_probe")
    print("   ../.venv-train/bin/python train_pose.py --imgsz 224")
else:
    raise SystemExit("❌ Vẫn không thấy GPU — xem PLAN.md mục 5c.")
EOF
