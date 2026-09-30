"""
Chọn mô hình cánh tay cho các script vision (fk_roi_predictor, calibrate_hand_eye).

    assarm  (mặc định) — tay mới, 5 khớp base/shoulder/elbow/wrist_roll/pen
            (bút gắn trên servo J5 thay gripper), FK trong ros2_ws/.../rl/fk_assarm.py
    old4dof — tay 6-DOF cũ khoá wrist_roll/pen, FK fk_4dof trong fk_ik_utils.py

Mỗi Arm cung cấp:
    joint_names                 tên khớp đọc từ /pca9685_servo/joint_states
    q_from_servo_degs(degs)     độ lệnh servo -> góc URDF (rad)
    hand_eye_matrix(q)          4x4 của phần thân cứng gắn marker (Phase 3)
    tip_point(q)                (x,y,z) đầu bút trong base_link (Phase 4)
"""
import sys
from pathlib import Path

import numpy as np

_FK_UTILS_DIR = (Path(__file__).resolve().parent.parent.parent
                 / "ros2_ws" / "src" / "visual_servoing" / "scripts" / "rl")
sys.path.insert(0, str(_FK_UTILS_DIR))

ARM_CHOICES = ("assarm", "old4dof")


class AssArm:
    name = "assarm"

    def __init__(self, tool_offset=None):
        import fk_assarm
        self._fk = fk_assarm
        self.joint_names = fk_assarm.JOINT_NAMES + (fk_assarm.J5_NAME,)
        # None -> tính theo góc J5 thật (tool_offset_j5); đặt số -> ghi đè cố định
        self.tool_offset = tuple(tool_offset) if tool_offset is not None else None

    def q_from_servo_degs(self, degs):
        return [self._fk.servo_deg_to_q(n, d) for n, d in zip(self.joint_names, degs)]

    def hand_eye_matrix(self, q):
        # Khung giá bút (sau J5) — marker gắn trên giá bút. Không phụ thuộc
        # vị trí đầu bút: Tsai-Lenz tự triệt tiêu offset cố định marker<->giá.
        return np.array(self._fk.fk_pen_matrix(q[:4], q[4]))

    def tip_point(self, q):
        tool = self.tool_offset if self.tool_offset is not None else self._fk.tool_offset_j5(q[4])
        return self._fk.fk_tip(q[:4], tool)


class Old4Dof:
    name = "old4dof"
    joint_names = ("base", "shoulder", "elbow", "wrist_pitch")
    PI_HOME_DEG = 90.0

    def __init__(self, wrist_roll_deg=90.0, pen_deg=90.0):
        import fk_ik_utils
        self._fk = fk_ik_utils
        self.wrist_roll_deg = wrist_roll_deg
        self.pen_deg = pen_deg

    def q_from_servo_degs(self, degs):
        return [np.radians(d - self.PI_HOME_DEG) for d in degs]

    def hand_eye_matrix(self, q):
        return np.array(self._fk.fk_4dof_matrix(
            q, raw=True, wrist_roll_deg=self.wrist_roll_deg, pen_deg=self.pen_deg))

    def tip_point(self, q):
        return self._fk.fk_4dof(
            q, raw=True, wrist_roll_deg=self.wrist_roll_deg, pen_deg=self.pen_deg)


def make_arm(name, wrist_roll_deg=90.0, pen_deg=90.0, tool_offset=None):
    if name == "assarm":
        return AssArm(tool_offset=tool_offset)
    if name == "old4dof":
        return Old4Dof(wrist_roll_deg=wrist_roll_deg, pen_deg=pen_deg)
    raise ValueError(f"arm không hợp lệ: {name} (chọn {ARM_CHOICES})")


def joint_states_to_servo_degs(names, positions, order):
    """Đọc sensor_msgs/JointState -> list độ servo theo đúng thứ tự `order`.
    |giá trị| < 6.3 thì coi là rad (logic phòng thủ giống control_backends.py)."""
    lookup = dict(zip(names, positions))
    missing = [n for n in order if n not in lookup]
    if missing:
        raise ValueError(f"joint_states thiếu các khớp: {missing}")
    out = []
    for n in order:
        v = float(lookup[n])
        if abs(v) < 6.3:
            v = float(np.degrees(v))
        out.append(v)
    return out


def add_arm_args(ap):
    ap.add_argument("--arm", choices=ARM_CHOICES, default="assarm",
                    help="Mô hình cánh tay (mặc định assarm = tay mới)")
    ap.add_argument("--wrist-roll-deg", type=float, default=90.0,
                    help="[old4dof] góc khoá servo wrist_roll")
    ap.add_argument("--pen-deg", type=float, default=90.0,
                    help="[old4dof] góc khoá servo pen")
    ap.add_argument("--tool-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="[assarm] vector Plate_1 -> đầu bút (m), mặc định fk_assarm.TOOL_OFFSET")


def arm_from_args(args):
    return make_arm(args.arm, wrist_roll_deg=args.wrist_roll_deg,
                    pen_deg=args.pen_deg, tool_offset=args.tool_offset)
