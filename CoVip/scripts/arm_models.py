"""
Chọn mô hình cánh tay cho các script vision (fk_roi_predictor, calibrate_hand_eye).

    newarm  (mặc định) — tay mới (thiết kế cuối newarm_final), 4 khớp
            base/shoulder/elbow/wrist_roll, bút gắn cứng đồng trục J4,
            FK trong ros2_ws/.../rl/fk_newarm.py
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

ARM_CHOICES = ("newarm", "old4dof")


class NewArm:
    name = "newarm"

    def __init__(self, tool_offset=None):
        import fk_newarm
        self._fk = fk_newarm
        self.joint_names = fk_newarm.JOINT_NAMES
        self.tool_offset = tuple(tool_offset) if tool_offset is not None else fk_newarm.TOOL_OFFSET

    def q_from_servo_degs(self, degs):
        return self._fk.servo_degs_to_q(degs)

    def hand_eye_matrix(self, q):
        # Khung hộp bút hopbut_1 (sau J4) — marker dán trên hộp bút/đĩa gắn
        # bút. Không phụ thuộc TOOL_OFFSET: Tsai-Lenz tự triệt tiêu offset cố
        # định marker<->hộp bút.
        return np.array(self._fk.fk_flange_matrix(q))

    def tip_point(self, q):
        return self._fk.fk_tip(q, self.tool_offset)

    def servo_degs_from_q(self, q):
        return [self._fk.q_to_servo_deg(n, v) for n, v in zip(self.joint_names, q)]

    def ik_tip(self, target, q_prev=None, branch=(-1, -1)):
        """IK vị trí đầu công cụ (m, base_link) -> q (rad) hoặc None.
        Có q_prev: nghiệm gần q_prev nhất (bám liên tục, không nhảy nhánh).
        Không có: nhánh vẽ `branch` (tay vươn về phía trước, khuỷu gập ra sau)."""
        if q_prev is not None:
            return self._fk.ik_tip_nearest(tuple(target), q_prev, tool_offset=self.tool_offset)
        for q, b in self._fk.ik_tip_branches(tuple(target), tool_offset=self.tool_offset):
            if b == branch:
                return q
        return None


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
    if name == "newarm":
        return NewArm(tool_offset=tool_offset)
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
    ap.add_argument("--arm", choices=ARM_CHOICES, default="newarm",
                    help="Mô hình cánh tay (mặc định newarm = tay mới)")
    ap.add_argument("--wrist-roll-deg", type=float, default=90.0,
                    help="[old4dof] góc khoá servo wrist_roll")
    ap.add_argument("--pen-deg", type=float, default=90.0,
                    help="[old4dof] góc khoá servo pen")
    ap.add_argument("--tool-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="[newarm] vector hopbut_1 -> đầu bút (m), mặc định fk_newarm.TOOL_OFFSET")


def arm_from_args(args):
    return make_arm(args.arm, wrist_roll_deg=args.wrist_roll_deg,
                    pen_deg=args.pen_deg, tool_offset=args.tool_offset)
