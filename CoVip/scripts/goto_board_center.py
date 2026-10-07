#!/usr/bin/env python3
"""
Bài test: tay tự đưa đầu bút tới TRƯỚC TÂM BẢNG VẼ (dấu +), dừng cách mặt
bảng --standoff-mm theo pháp tuyến (mặc định 20mm, không chạm bảng).

Cần đang chạy:
  - driver servo (wicom_roboarm, servos_newarm.yaml) — hoặc cầu nối mô phỏng
  - run_pi4_ros2.py --board   (phát /aeroscript/board_pose; và /aeroscript/pen_xyz nếu có bút)
  - calib/T_cam_to_base.npy   (calibrate_hand_eye.py collect hoặc collect-tip)

Hai chế độ:
  --tip-source pen   VÒNG KÍN: camera đo cả bảng lẫn đầu bút, đi từng bước nhỏ
                     theo sai lệch còn lại. Sai số servo và T_cam_to_base được
                     bù; T_cam_to_base chỉ cần gần đúng.
  --tip-source none  VÒNG HỞ (khi camera không thấy bút, hoặc để so sánh):
                     tin hoàn toàn vào T_cam_to_base + FK + tool_offset. Sai số
                     cuối = tổng sai số của cả ba; đo bằng thước để biết.

    python3 scripts/goto_board_center.py --dry-run            # chỉ tính, không gửi lệnh
    python3 scripts/goto_board_center.py --tip-source pen
    python3 scripts/goto_board_center.py --tip-source none --standoff-mm 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_models import add_arm_args, arm_from_args  # noqa: E402
from board_detect import pose_to_T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tip-source", choices=("pen", "none"), default="pen")
    ap.add_argument("--standoff-mm", type=float, default=20.0, help="dừng cách mặt bảng bao xa")
    ap.add_argument("--offset-mm", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"),
                    help="điểm đích lệch khỏi tâm bảng (mm; x sang phải, y lên, nhìn vào bảng)")
    ap.add_argument("--tol-mm", type=float, default=3.0, help="[pen] dừng khi sai lệch nhỏ hơn mức này")
    ap.add_argument("--gain", type=float, default=0.6, help="[pen] phần sai lệch sửa mỗi bước")
    ap.add_argument("--max-step-mm", type=float, default=25.0, help="[pen] bước sửa lớn nhất")
    ap.add_argument("--max-iters", type=int, default=25)
    ap.add_argument("--speed", type=float, default=25.0, help="tốc độ khớp tối đa (độ/s)")
    ap.add_argument("--dwell", type=float, default=1.0, help="giây đứng yên lấy mẫu mỗi lần đo")
    ap.add_argument("--hand-eye", default=str(ROOT / "calib" / "T_cam_to_base.npy"))
    ap.add_argument("--board-topic", default="/aeroscript/board_pose")
    ap.add_argument("--tip-topic", default="/aeroscript/pen_xyz")
    ap.add_argument("--dry-run", action="store_true", help="in điểm đích + lệnh servo, không di chuyển")
    ap.add_argument("--no-home", action="store_true", help="xong không đưa tay về home")
    add_arm_args(ap)
    args = ap.parse_args()
    arm = arm_from_args(args)
    if arm.name != "newarm":
        raise SystemExit("Bài test này chỉ hỗ trợ --arm newarm.")
    if not Path(args.hand_eye).exists():
        raise SystemExit(f"Chưa có {args.hand_eye} — chạy calibrate_hand_eye.py (collect hoặc collect-tip) trước.")
    T_cb = np.load(args.hand_eye)          # camera -> base_link

    import rclpy
    from geometry_msgs.msg import Point, PoseStamped
    from arm_io import ArmIO

    rclpy.init()
    node = rclpy.create_node("goto_board_center")
    io = ArmIO(node, arm, speed_deg_s=args.speed)
    boards, tips = [], []
    node.create_subscription(PoseStamped, args.board_topic,
                             lambda m: boards.append(pose_to_T(m.pose.position, m.pose.orientation)), 10)
    node.create_subscription(Point, args.tip_topic, lambda m: tips.append((m.x, m.y, m.z)), 10)

    def finish(code):
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return code

    def measure_tip():
        """Đầu bút trong hệ camera (m), trung vị trong args.dwell giây; None nếu mất bút."""
        tips.clear()
        io.spin(args.dwell)
        if len(tips) < 4:
            return None
        return np.median(np.array(tips), axis=0) / 1000.0

    if not io.wait_ready():
        print("Không có /pca9685_servo/joint_states — driver (hoặc cầu nối mô phỏng) chưa chạy.")
        return finish(1)

    # ── 1. Bảng: lấy trung vị vị trí + pháp tuyến trong 2 giây ──
    io.spin(2.0)
    if len(boards) < 3:
        print(f"Không nhận được {args.board_topic} — camera có thấy ít nhất 2 marker của bảng không? "
              "run_pi4_ros2.py đã chạy với --board chưa?")
        return finish(1)
    Tb = np.array(boards)
    centre_cam = np.median(Tb[:, :3, 3], axis=0)
    x_cam, y_cam, n_cam = (np.median(Tb[:, :3, i], axis=0) for i in range(3))
    n_cam /= np.linalg.norm(n_cam)
    if n_cam @ centre_cam > 0:              # pháp tuyến phải hướng về phía camera/tay
        n_cam = -n_cam
    target_cam = (centre_cam + n_cam * args.standoff_mm / 1000.0
                  + x_cam * args.offset_mm[0] / 1000.0 + y_cam * args.offset_mm[1] / 1000.0)
    R_cb, t_cb = T_cb[:3, :3], T_cb[:3, 3]
    target_base = R_cb @ target_cam + t_cb
    n_base = R_cb @ n_cam
    print(f"Bảng ({len(boards)} mẫu): tâm trong hệ camera {np.round(centre_cam*1000, 0)} mm")
    print(f"Điểm đích (cách bảng {args.standoff_mm:.0f}mm): base_link {np.round(target_base*1000, 1)} mm")

    # ── 2. Đường tới: lùi thêm 40mm theo pháp tuyến rồi mới áp vào ──
    q_pre = arm.ik_tip(target_base + n_base * 0.040)
    q_goal = arm.ik_tip(target_base, q_prev=q_pre) if q_pre is not None else None
    if q_goal is None:
        print("Điểm đích NGOÀI tầm với / giới hạn khớp với cấu hình servo + tool_offset hiện tại.")
        print("  Kiểm vị trí đặt bảng (newarm_draw_region.py) hoặc T_cam_to_base.")
        return finish(1)
    for name, q in (("điểm lùi", q_pre), ("điểm đích", q_goal)):
        print(f"  {name}: q(độ)={np.round(np.degrees(q), 1)}  lệnh servo={np.round(arm.servo_degs_from_q(q), 1)}")
    if args.dry_run:
        print("--dry-run: không gửi lệnh.")
        return finish(0)

    io.move_to(arm.servo_degs_from_q(q_pre))
    io.move_to(arm.servo_degs_from_q(q_goal))
    q_cur = list(q_goal)

    # ── 3. Vòng kín bằng đầu bút camera đo ──
    code = 0
    if args.tip_source == "pen":
        history = []
        for it in range(1, args.max_iters + 1):
            tip_cam = measure_tip()
            if tip_cam is None:
                print(f"  [{it}] camera không thấy bút — dừng tại đây (kiểm khung hình/ánh sáng).")
                code = 1
                break
            err_base = R_cb @ (target_cam - tip_cam)
            e = float(np.linalg.norm(err_base))
            depth = float((tip_cam - centre_cam) @ n_cam)        # khoảng cách tới mặt bảng
            history.append(e)
            print(f"  [{it}] sai lệch {e*1000:5.1f} mm  (từng trục base {np.round(err_base*1000, 1)}), "
                  f"đầu bút cách mặt bảng {depth*1000:.1f} mm")
            if e * 1000 <= args.tol_mm:
                print(f"ĐẠT sau {it} lần đo: sai lệch {e*1000:.1f} mm <= {args.tol_mm:.1f} mm.")
                break
            step = err_base * args.gain
            if np.linalg.norm(step) * 1000 > args.max_step_mm:
                step *= args.max_step_mm / (np.linalg.norm(step) * 1000)
            q_next = arm.ik_tip(np.array(arm.tip_point(q_cur)) + step, q_prev=q_cur)
            if q_next is None:
                print("  bước sửa đưa tay ra ngoài tầm với / giới hạn khớp — dừng.")
                code = 1
                break
            io.move_to(arm.servo_degs_from_q(q_next), settle_sec=0.5)
            q_cur = list(q_next)
        else:
            print(f"CHƯA ĐẠT sau {args.max_iters} lần: sai lệch còn {history[-1]*1000:.1f} mm.")
            code = 1
    else:
        fk = np.array(arm.tip_point(q_cur))
        print(f"Vòng hở: đã tới điểm đích theo FK ({np.round(fk*1000, 1)} mm). "
              f"Đo bằng thước khoảng cách đầu công cụ tới dấu + (mong đợi {args.standoff_mm:.0f} mm, "
              "thẳng trước dấu +).")
        tip_cam = measure_tip()
        if tip_cam is not None:
            err = R_cb @ (target_cam - tip_cam)
            print(f"  (camera vẫn thấy bút: sai lệch đo được {np.linalg.norm(err)*1000:.1f} mm)")

    if not args.no_home:
        input_ok = True
        try:
            input("Enter để đưa tay về home (Ctrl+C để giữ nguyên tư thế)... ") if sys.stdin.isatty() else None
        except KeyboardInterrupt:
            input_ok = False
        if input_ok:
            io.move_to(arm.servo_degs_from_q(q_pre))
            io.move_to(arm.servo_degs_from_q([0.0, 0.0, 0.0, 0.0]))
    return finish(code)


if __name__ == "__main__":
    sys.exit(main())
