#!/usr/bin/env bash
# Chạy liên tiếp nhiều cấu hình của run_pi4_ros2.py, mỗi cấu hình ghi 1 file
# log, rồi in bảng so sánh (tốc độ, độ trễ, CPU, RAM, nhiệt, sai số so với bảng).
#
#   scripts/run_compare.sh                 # 4 lượt: 360 360b 720 720b, mỗi lượt 60 giây
#   scripts/run_compare.sh 90 360 360b     # chỉ 360p không bảng / có bảng, mỗi lượt 90 giây
#   scripts/run_compare.sh 60 360b 720b    # có bảng: 360p so với 720p
#
# Tên lượt: 360 | 720 = độ phân giải bắt ảnh; thêm "b" = có dò bảng (--board).
# Đổi mặc định bằng biến môi trường, vd:
#   MODEL=pen_pose_192_sc.tflite DEVICE=/dev/video0 EXTRA="--pen-dims-mm 43 29.5 15.5" scripts/run_compare.sh
#
# Trong MỖI lượt hãy làm cùng một việc để so sánh công bằng: đưa bút vào khung
# hình; ở lượt có bảng thì chạm mũi bút vào dấu + và 4 góc vùng vẽ, mỗi điểm
# giữ yên 2-3 giây. Stream xem như mọi lần (web_video_server chạy ở cửa sổ khác).
set -u
cd "$(dirname "$0")/.."

SECS="${1:-60}"; shift 2>/dev/null || true
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(360 360b 720 720b)
MODEL="${MODEL:-pen_pose_192_sc.tflite}"
DEVICE="${DEVICE:-/dev/video0}"
CONF="${CONF:-0.3}"
THREADS="${THREADS:-2}"
EXTRA="${EXTRA:-}"
STAMP="cmp$(date +%H%M%S)"

v4l2-ctl -d "$DEVICE" --set-ctrl=brightness=160,contrast=128,gain=120 2>/dev/null \
    || echo "(không đặt được độ sáng/gain cho $DEVICE — bỏ qua)"

i=0
for r in "${RUNS[@]}"; do
    i=$((i + 1))
    case "$r" in
        360*) W=640;  H=360 ;;
        720*) W=1280; H=720 ;;
        *) echo "Không hiểu tên lượt '$r' (dùng 360, 360b, 720, 720b)"; exit 1 ;;
    esac
    BOARD=""; NAME="${r%b}p_khongbang"
    case "$r" in *b) BOARD="--board"; NAME="${r%b}p_bang" ;; esac
    echo
    echo "════ Lượt $i/${#RUNS[@]}: ${W}x${H} ${BOARD:+có dò bảng}${BOARD:-không dò bảng} — ${SECS} giây ════"
    [ -n "$BOARD" ] && echo "     Chạm mũi bút vào dấu + và 4 góc vùng vẽ, mỗi điểm giữ yên 2-3 giây."
    echo "     Bắt đầu sau 5 giây..."; sleep 5
    # shellcheck disable=SC2086
    python3 -u run_pi4_ros2.py --model "$MODEL" --device "$DEVICE" --width $W --height $H \
        --fourcc MJPG --conf "$CONF" --threads "$THREADS" $BOARD --duration "$SECS" \
        --run-tag "${STAMP}_${NAME}" $EXTRA 2>&1 | grep -E "🎯|🔎|❌|⚠️|📝|Calib|🖊️"
done

echo
echo "════ So sánh ════"
python3 scripts/analyze_run.py logs/run_*_"${STAMP}"_*.csv
