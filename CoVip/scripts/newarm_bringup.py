#!/usr/bin/env python3
"""
Bring-up cánh tay mới (newarm, thiết kế cuối newarm_final) TRƯỚC khi làm
việc với camera: đưa servo về home, xác nhận chiều quay, kiểm FK bằng thước.
Kết quả hiệu chỉnh ghi vào newarm_servo_calib.json (cạnh fk_newarm.py) —
fk_newarm tự nạp file đó, nên các script vision dùng đúng số mà không sửa code.

Cần driver đang chạy (trừ --dry-run / show / set):
    ros2 launch wicom_roboarm wicom_roboarm.launch.py servo_config:=servos_newarm.yaml

Thứ tự dùng:
    # 0. Xem cấu hình hiện tại + giới hạn khớp suy ra
    python3 scripts/newarm_bringup.py show

    # 1. LÚC LẮP: đưa servo về góc lệnh lắp ráp rồi mới gắn sừng/khâu.
    #    Mặc định 90° cả 4 khớp, tay gắn THẲNG XUỐNG. Khuỷu lắp lệch để có
    #    dải [-30°,150°]: gắn cẳng tay thẳng hàng bắp tay ở lệnh 30°:
    python3 scripts/newarm_bringup.py set --joint elbow --home 30
    python3 scripts/newarm_bringup.py home

    # 2. Xác nhận chiều quay từng khớp (tự ghi inverted vào file calib)
    python3 scripts/newarm_bringup.py directions

    # 3. Kiểm FK bằng thước ở vài tư thế (đạt: lệch < 10mm)
    python3 scripts/newarm_bringup.py fk-check

    # Đo lại chiều dài hộp bút -> đầu bút (m, âm = xuống):
    python3 scripts/newarm_bringup.py set --tool-offset 0 0 -0.058

Thêm --dry-run vào home/directions/fk-check để chỉ in lệnh, không cần ROS.

QUY ƯỚC ĐỨNG NHÌN (dùng cho mọi mô tả chiều bên dưới): đứng đối diện tay
sao cho BÁNH RĂNG SERVO KHUỶU nằm bên TRÁI bạn (sừng servo vai bên PHẢI).
Khi đó: +X = sang phải, +Z = lên, -Y = VỀ PHÍA BẠN ("phía trước" của tay).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

_FK_UTILS_DIR = (Path(__file__).resolve().parent.parent.parent
                 / "ros2_ws" / "src" / "visual_servoing" / "scripts" / "rl")
sys.path.insert(0, str(_FK_UTILS_DIR))
import fk_newarm as F  # noqa: E402

COMMAND_TOPIC = "/pca9685_servo/command"
ENABLE_SERVICE = "/pca9685_servo/enable"
JOG_DEG = 15.0

# Chuyển động MONG ĐỢI khi góc khớp URDF tăng (+), theo quy ước đứng nhìn ở trên
EXPECTED_MOTION = {
    "base": "nhìn từ TRÊN xuống: cả tay quay CÙNG chiều kim đồng hồ",
    "shoulder": "cả tay vung VỀ PHÍA BẠN (đầu bút đi về phía bạn và lên)",
    "elbow": "cẳng tay gập VỀ PHÍA BẠN (cùng phía với vai)",
    "wrist_roll": "nhìn từ khuỷu dọc xuống đầu bút: hộp bút quay CÙNG chiều kim đồng hồ",
}

# Tư thế kiểm FK (độ, góc khớp URDF [base, shoulder, elbow, wrist_roll])
FK_CHECK_POSES_DEG = [
    (0, 30, 0, 0),
    (0, 30, 30, 0),
    (0, 45, 45, 0),
    (0, 60, 30, 0),
    (30, 45, 45, 0),
    (-30, 30, 60, 0),
]


def load_calib_file():
    p = Path(F.CALIB_PATH)
    return json.loads(p.read_text()) if p.exists() else {}


def save_calib_file(data):
    Path(F.CALIB_PATH).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Đã ghi {F.CALIB_PATH}")


def q_deg_to_servo(q_deg):
    return [F.q_to_servo_deg(n, math.radians(d)) for n, d in zip(F.JOINT_NAMES, q_deg)]


def check_in_window(q_deg):
    """Tên các khớp mà q_deg nằm ngoài cửa sổ servo 0..180."""
    bad = []
    for n, d, lo, hi in zip(F.JOINT_NAMES, q_deg, F.JOINT_LIMITS_LOW, F.JOINT_LIMITS_HIGH):
        if not (math.degrees(lo) - 1e-6 <= d <= math.degrees(hi) + 1e-6):
            bad.append(n)
    return bad


class Robot:
    """Gửi lệnh servo qua driver wicom_roboarm. dry_run: chỉ in."""

    def __init__(self, dry_run):
        self.dry_run = dry_run
        if dry_run:
            return
        import rclpy
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger
        self._rclpy, self._JointState = rclpy, JointState
        rclpy.init()
        self.node = rclpy.create_node("newarm_bringup")
        self.pub = self.node.create_publisher(JointState, COMMAND_TOPIC, 10)
        cli = self.node.create_client(Trigger, ENABLE_SERVICE)
        if not cli.wait_for_service(timeout_sec=3.0):
            raise SystemExit(f"Không thấy {ENABLE_SERVICE} — đã chạy driver wicom_roboarm chưa?")
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        print("Đã enable servo.")

    def send(self, servo_degs, hold_sec=2.0):
        txt = ", ".join(f"{n}={d:.1f}" for n, d in zip(F.JOINT_NAMES, servo_degs))
        print(f"  -> lệnh servo (độ): {txt}")
        if self.dry_run:
            return
        msg = self._JointState()
        msg.name = list(F.JOINT_NAMES)
        msg.position = [float(d) for d in servo_degs]
        t_end = time.time() + hold_sec
        while time.time() < t_end:      # lặp lại để không dính command_timeout của driver
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.pub.publish(msg)
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            time.sleep(0.1)

    def close(self):
        if not self.dry_run:
            self.node.destroy_node()
            self._rclpy.shutdown()


def cmd_show(_args):
    print(f"File hiệu chỉnh: {F.CALIB_PATH} ({'ĐÃ nạp' if F.CALIB_LOADED else 'chưa có — đang dùng mặc định'})")
    print("Khớp        servo      home  đảo   dải góc khớp (độ, 0 = tay thẳng xuống)")
    for n, lo, hi in zip(F.JOINT_NAMES, F.JOINT_LIMITS_LOW, F.JOINT_LIMITS_HIGH):
        s = F.SERVO_SPECS[n]
        print(f"  {n:10s} {s['model']:10s} {s['home_deg']:5.1f} {str(s['inverted']):5s} "
              f"[{math.degrees(lo):+6.1f}, {math.degrees(hi):+6.1f}]")
    print(f"TOOL_OFFSET (gốc hộp bút -> đầu bút, m): {F.TOOL_OFFSET}")
    x, y, z = F.fk_tip([0, 0, 0, 0])
    print(f"Đầu bút ở home (base_link, m): ({x:+.4f}, {y:+.4f}, {z:+.4f})")
    return 0


def cmd_set(args):
    data = load_calib_file()
    if args.joint:
        over = data.setdefault("servos", {}).setdefault(args.joint, {})
        if args.home is not None:
            over["home_deg"] = args.home
        if args.inverted is not None:
            over["inverted"] = args.inverted == "true"
        if args.range is not None:
            over["range_deg"] = args.range
    if args.tool_offset:
        data["tool_offset"] = args.tool_offset
    if not data:
        print("Không có gì để ghi (dùng --joint ... hoặc --tool-offset ...).")
        return 1
    save_calib_file(data)
    return 0


def cmd_home(args):
    robot = Robot(args.dry_run)
    print("Đưa cả 4 servo về góc home (q=0). Tay phải TREO THẲNG XUỐNG, cẳng tay thẳng hàng bắp tay.")
    robot.send(q_deg_to_servo([0, 0, 0, 0]), hold_sec=args.hold)
    robot.close()
    return 0


def cmd_directions(args):
    print(__doc__.split("QUY ƯỚC ĐỨNG NHÌN")[1].join(["QUY ƯỚC ĐỨNG NHÌN", ""]).strip())
    robot = Robot(args.dry_run)
    data = load_calib_file()
    changed = False
    for i, name in enumerate(F.JOINT_NAMES):
        q = [0.0, 0.0, 0.0, 0.0]
        # nhích theo chiều + nếu cửa sổ servo cho phép, không thì chiều -
        sign = 1.0 if math.degrees(F.JOINT_LIMITS_HIGH[i]) >= JOG_DEG else -1.0
        q[i] = sign * JOG_DEG
        print(f"\n[{name}] về home rồi nhích {q[i]:+.0f}° (góc khớp URDF)")
        robot.send(q_deg_to_servo([0, 0, 0, 0]), hold_sec=args.hold)
        robot.send(q_deg_to_servo(q), hold_sec=args.hold)
        exp = EXPECTED_MOTION[name] + ("" if sign > 0 else "  — NHƯNG lần này nhích chiều ÂM nên phải NGƯỢC LẠI")
        print(f"  Mong đợi: {exp}")
        if args.dry_run:
            continue
        ans = input("  Tay có chuyển động đúng như mong đợi không? [y = đúng / n = ngược / s = bỏ qua]: ").strip().lower()
        if ans == "n":
            cur = F.SERVO_SPECS[name]["inverted"]
            data.setdefault("servos", {}).setdefault(name, {})["inverted"] = not cur
            changed = True
            print(f"  -> ghi inverted={not cur} cho {name}")
    robot.send(q_deg_to_servo([0, 0, 0, 0]), hold_sec=args.hold)
    robot.close()
    if changed:
        save_calib_file(data)
        print("Chạy lại `directions` một lần nữa để xác nhận cả 4 khớp đều đúng chiều.")
    elif not args.dry_run:
        print("\nCả 4 khớp đúng chiều (hoặc đã bỏ qua) — không cần đổi gì.")
    return 0


def cmd_fk_check(args):
    home = F.fk_tip([0, 0, 0, 0])
    print("Đánh dấu điểm ngay dưới đầu bút ở tư thế home (dây dọi/ê-ke). Mọi số đo dưới đây")
    print("tính TỪ điểm home đó, theo quy ước đứng nhìn (bánh răng servo khuỷu bên TRÁI bạn):")
    print("  phía_bạn = đầu bút tiến về phía bạn | phải = sang phải bạn | lên = cao hơn home  (mm)\n")
    robot = Robot(args.dry_run)
    robot.send(q_deg_to_servo([0, 0, 0, 0]), hold_sec=args.hold)
    errs = []
    for k, q_deg in enumerate(FK_CHECK_POSES_DEG, 1):
        bad = check_in_window(q_deg)
        if bad:
            print(f"[{k}] q={q_deg}: BỎ QUA — {bad} ngoài cửa sổ servo hiện tại")
            continue
        p = F.fk_tip([math.radians(d) for d in q_deg])
        exp = ((home[1] - p[1]) * 1000, (p[0] - home[0]) * 1000, (p[2] - home[2]) * 1000)
        print(f"[{k}] q(độ)={q_deg}")
        robot.send(q_deg_to_servo(q_deg), hold_sec=args.hold)
        print(f"  FK mong đợi: phía_bạn={exp[0]:+7.1f}  phải={exp[1]:+7.1f}  lên={exp[2]:+7.1f}  mm")
        if args.dry_run:
            continue
        raw = input("  Nhập số đo thật 'phía_bạn phải lên' (mm), Enter để bỏ qua: ").split()
        if len(raw) == 3:
            try:
                m = [float(v) for v in raw]
            except ValueError:
                print("  không đọc được số, bỏ qua tư thế này.")
                continue
            e = math.sqrt(sum((a - b) ** 2 for a, b in zip(m, exp)))
            errs.append(e)
            print(f"  lệch {e:.1f} mm  (từng trục: {m[0]-exp[0]:+.1f}, {m[1]-exp[1]:+.1f}, {m[2]-exp[2]:+.1f})")
    robot.send(q_deg_to_servo([0, 0, 0, 0]), hold_sec=args.hold)
    robot.close()
    if errs:
        worst = max(errs)
        print(f"\n{len(errs)} tư thế đã đo: lệch trung bình {sum(errs)/len(errs):.1f} mm, lớn nhất {worst:.1f} mm")
        print("ĐẠT (<10mm) — sang Phase 3 hand-eye được." if worst < 10 else
              "CHƯA ĐẠT (>=10mm) — kiểm lại home/chiều quay/TOOL_OFFSET; lệch đều theo 1 khớp thường là sai home của khớp đó.")
        return 0 if worst < 10 else 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="In cấu hình servo + giới hạn khớp hiện tại")
    s = sub.add_parser("set", help="Ghi số hiệu chỉnh vào newarm_servo_calib.json")
    s.add_argument("--joint", choices=F.JOINT_NAMES)
    s.add_argument("--home", type=float, help="lệnh servo (độ) ứng với q=0")
    s.add_argument("--inverted", choices=("true", "false"))
    s.add_argument("--range", type=float, help="tổng hành trình servo (độ), mặc định 180")
    s.add_argument("--tool-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="vector đầu ra J4 (gốc hopbut_1) -> đầu bút (m); số CAD: 0 0 -0.0625 "
                        "— đo lại trên tay thật")
    for name, helptext in (("home", "Đưa servo về home (dùng lúc lắp ráp)"),
                           ("directions", "Xác nhận chiều quay từng khớp"),
                           ("fk-check", "Kiểm FK bằng thước")):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--dry-run", action="store_true", help="chỉ in lệnh, không cần ROS/robot")
        c.add_argument("--hold", type=float, default=2.0, help="giây giữ mỗi lệnh")
    args = ap.parse_args()
    return {"show": cmd_show, "set": cmd_set, "home": cmd_home,
            "directions": cmd_directions, "fk-check": cmd_fk_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
