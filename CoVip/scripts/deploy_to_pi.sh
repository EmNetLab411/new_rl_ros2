#!/usr/bin/env bash
# deploy_to_pi.sh — Đưa code từ máy dev lên Raspberry Pi qua scp.
#
# Dùng LẠI cho MỌI lần port trong tương lai: khi có file mới cần đưa lên Pi,
# chỉ cần thêm 1 cặp dòng vào LOCAL_FILES/REMOTE_FILES bên dưới rồi chạy
# lại script này — không cần nhớ lại từng lệnh scp/mkdir như trước.
#
# ✅ ĐÃ XÁC MINH (2026-09-23, qua `ls ~` thật trên Pi): thư mục chạy vision
# trên Pi tên là "~/aeroscript" (PHẲNG — best.onnx, run_pi4_ros2.py nằm
# thẳng trong đó, KHÔNG có thư mục con "pen_models/" như máy dev), và
# "~/ros2_ws" là workspace ROS2 riêng, NẰM CÙNG CẤP "~/aeroscript" dưới
# thư mục home (không phải "~/new_rl_ros2/CoVip" như máy dev đặt tên).
# Vì 2 phía đặt tên khác nhau (CoVip <-> aeroscript), mỗi file cần khai báo
# RIÊNG đường dẫn nguồn (máy dev) và đích (Pi) — xem LOCAL_FILES/REMOTE_FILES.
#
# Chạy (từ bất kỳ đâu, không cần cd vào CoVip trước):
#   ./CoVip/scripts/deploy_to_pi.sh
#
# Đổi Pi khác (IP/user khác) mà KHÔNG cần sửa file này:
#   PI_HOST=other_user@10.0.0.5 ./CoVip/scripts/deploy_to_pi.sh
#
# Đổi thư mục home trên Pi (mặc định "~", tức /home/<user-trong-PI_HOST>):
#   PI_HOME=/home/piros2 ./CoVip/scripts/deploy_to_pi.sh
#
# Chỉ in ra việc sẽ làm, không copy thật (kiểm trước khi chạy thật):
#   DRY_RUN=1 ./CoVip/scripts/deploy_to_pi.sh

set -euo pipefail

PI_HOST="${PI_HOST:-piros2@192.168.50.1}"
PI_HOME="${PI_HOME:-~}"
DRY_RUN="${DRY_RUN:-0}"

# Thư mục gốc trên máy dev (cha của CoVip/ và ros2_ws/) — tự tính từ vị trí
# script này (không hardcode đường dẫn tuyệt đối), để chạy đúng dù repo được
# clone ở đâu, trên máy nào.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ─────────────────────────────────────────────────────────────────────────
# Danh sách file cần đưa lên Pi — THÊM 1 CẶP DÒNG (cùng chỉ số) VÀO ĐÂY khi
# có file mới. LOCAL_FILES tính từ LOCAL_ROOT (máy dev, "CoVip/..." hoặc
# "ros2_ws/..."). REMOTE_FILES tính từ PI_HOME (Pi, "aeroscript/..." hoặc
# "ros2_ws/..." — ĐÃ đổi tên CoVip -> aeroscript cho khớp Pi thật).
# ─────────────────────────────────────────────────────────────────────────
LOCAL_FILES=(
    "CoVip/run_pi4_ros2.py"
    "CoVip/scripts/calibrate_camera.py"
    "CoVip/scripts/calibrate_hand_eye.py"
    "CoVip/scripts/fk_roi_predictor.py"
    "CoVip/scripts/arm_models.py"
    "CoVip/scripts/newarm_bringup.py"
    "CoVip/scripts/arm_io.py"
    "CoVip/scripts/board_detect.py"
    "CoVip/scripts/goto_board_center.py"
    "CoVip/scripts/publish_camera_info.py"
    "CoVip/scripts/benchmark_tflite.py"
    "CoVip/scripts/inspect_tflite.py"
    "CoVip/scripts/benchmark_onnx.py"
    "CoVip/scripts/monitor_resources.py"
    "CoVip/scripts/analyze_run.py"
    "CoVip/scripts/run_compare.sh"
    # Model bút train lại ở imgsz nhỏ hơn (PLAN.md mục 5c) — dataset_split,
    # yolov8n-pose, flip_idx đã sửa. .onnx luôn dùng được; .tflite là NCHW
    # (khác best_float32.tflite cũ là NHWC) — run_pi4_ros2.py tự nhận layout.
    "CoVip/imgsz_probe/pen_pose_224.onnx"
    "CoVip/imgsz_probe/pen_pose_192.onnx"
    "CoVip/imgsz_probe/runs/pen_pose_224/weights/best.tflite"
    "CoVip/imgsz_probe/runs/pen_pose_192/weights/best.tflite"
    # Train lại 30/9 với +591 ảnh mới (bút nghiêng/nằm ngang/chúc xuống/xa,
    # nền có thuyền xanh): thấy bút 99% khung test so với 7% của bản 192 cũ.
    "CoVip/imgsz_probe/pen_pose_192_sc.onnx"
    "CoVip/imgsz_probe/runs/pen_pose_192_sc/weights/best.tflite"
    # Calib camera — run_pi4_ros2.py đọc file này (--calib), thiếu thì XYZ sai.
    # Calib lại 2026-10-07 (C930e, 50 ảnh, fx 771.5 ở 720p). File cũ
    # c920_720p.npz SAI tiêu cự ~18% do lỗi tinh chỉnh góc bàn cờ — không dùng.
    "CoVip/calib/c930e_720p.npz"
    "ros2_ws/src/visual_servoing/scripts/rl/fk_ik_utils.py"
    "ros2_ws/src/visual_servoing/scripts/rl/fk_newarm.py"
)
REMOTE_FILES=(
    "aeroscript/run_pi4_ros2.py"
    "aeroscript/scripts/calibrate_camera.py"
    "aeroscript/scripts/calibrate_hand_eye.py"
    "aeroscript/scripts/fk_roi_predictor.py"
    "aeroscript/scripts/arm_models.py"
    "aeroscript/scripts/newarm_bringup.py"
    "aeroscript/scripts/arm_io.py"
    "aeroscript/scripts/board_detect.py"
    "aeroscript/scripts/goto_board_center.py"
    "aeroscript/scripts/publish_camera_info.py"
    "aeroscript/scripts/benchmark_tflite.py"
    "aeroscript/scripts/inspect_tflite.py"
    "aeroscript/scripts/benchmark_onnx.py"
    "aeroscript/scripts/monitor_resources.py"
    "aeroscript/scripts/analyze_run.py"
    "aeroscript/scripts/run_compare.sh"
    "aeroscript/pen_pose_224.onnx"
    "aeroscript/pen_pose_192.onnx"
    "aeroscript/pen_pose_224.tflite"
    "aeroscript/pen_pose_192.tflite"
    "aeroscript/pen_pose_192_sc.onnx"
    "aeroscript/pen_pose_192_sc.tflite"
    "aeroscript/calib/c930e_720p.npz"
    "ros2_ws/src/visual_servoing/scripts/rl/fk_ik_utils.py"
    "ros2_ws/src/visual_servoing/scripts/rl/fk_newarm.py"
)

# Mở SẴN 1 kết nối SSH rồi cho mọi lệnh ssh/scp bên dưới dùng chung
# (ControlMaster) — chỉ hỏi mật khẩu đúng 1 lần thay vì mỗi file 1 lần.
# Muốn không phải nhập mật khẩu nữa: chạy 1 lần `ssh-copy-id ${PI_HOST}`.
CTRL_SOCK="${TMPDIR:-/tmp}/deploy-pi-%C"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CTRL_SOCK}" -o ControlPersist=120)
if [ "${DRY_RUN}" != "1" ]; then
    echo "Kết nối ${PI_HOST} (nhập mật khẩu 1 lần)..."
    ssh "${SSH_OPTS[@]}" -fN "${PI_HOST}"
    trap 'ssh "${SSH_OPTS[@]}" -O exit "${PI_HOST}" 2>/dev/null || true' EXIT
fi

echo "== Deploy lên Pi =="
echo "Nguồn (máy dev) : ${LOCAL_ROOT}"
echo "Đích            : ${PI_HOST}:${PI_HOME}/{aeroscript,ros2_ws}"
[ "${DRY_RUN}" = "1" ] && echo "(DRY RUN — chỉ in ra, không copy thật)"
echo

# 1) Tạo trước toàn bộ thư mục đích cần thiết trên Pi (scp không tự tạo thư mục)
REMOTE_DIRS=()
for rf in "${REMOTE_FILES[@]}"; do
    REMOTE_DIRS+=("${PI_HOME}/$(dirname "${rf}")")
done
UNIQUE_DIRS=($(printf "%s\n" "${REMOTE_DIRS[@]}" | sort -u))

echo "-- Tạo thư mục đích trên Pi (nếu chưa có) --"
if [ "${DRY_RUN}" = "1" ]; then
    echo "  ssh ${PI_HOST} \"mkdir -p ${UNIQUE_DIRS[*]}\""
else
    ssh "${SSH_OPTS[@]}" "${PI_HOST}" "mkdir -p ${UNIQUE_DIRS[*]}"
fi

# 2) Copy từng file — báo rõ file nào thiếu ở máy dev, không dừng cả script
echo
echo "-- Copy file --"
MISSING=()
for i in "${!LOCAL_FILES[@]}"; do
    lf="${LOCAL_FILES[$i]}"
    rf="${REMOTE_FILES[$i]}"
    src="${LOCAL_ROOT}/${lf}"
    dst="${PI_HOST}:${PI_HOME}/${rf}"
    if [ ! -f "${src}" ]; then
        echo "  [BỎ QUA] không thấy ${src}"
        MISSING+=("${lf}")
        continue
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  scp ${src} ${dst}"
    else
        echo "  ${lf} -> ${rf}"
        scp -q "${SSH_OPTS[@]}" "${src}" "${dst}"
    fi
done

echo
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "XONG NHƯNG THIẾU: ${#MISSING[@]}/${#LOCAL_FILES[@]} file không tìm thấy trên máy dev (xem log ở trên)."
    exit 1
fi
echo "XONG — đã đưa ${#LOCAL_FILES[@]} file lên ${PI_HOST}:${PI_HOME}"
