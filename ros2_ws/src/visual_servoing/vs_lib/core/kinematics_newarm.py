"""
IK cho cánh tay mới (newarm, thiết kế cuối newarm_final) — bản thay thế
KinematicsSolver (core/kinematics.py, tay cũ) cho executor vẽ. CHƯA nối vào
drawing_executor_ros2.py; chờ tích hợp sau khi có T_cam_to_base thật (Phase 3).

Khác KinematicsSolver cũ:
  - Đầu vào là điểm đầu bút trong base_link của fk_newarm, đơn vị MÉT (cũ: cm,
    hệ IK riêng + offset tay chỉnh trong robot_config.yaml).
  - KHÔNG có tham số tilt: tay mới không có wrist pitch, hướng bút do IK vị
    trí quyết định (xem pen_tilt_from_normal_deg để giám sát).
  - Trả 4 góc LỆNH servo [base, shoulder, elbow, wrist_roll] (độ, 0-180) đã
    áp home/chiều quay từ fk_newarm.SERVO_SPECS (+ newarm_servo_calib.json)
    — không cần sign_*/offset_* của robot_config.yaml nữa.
  - Giữ nhánh IK giữa các lần gọi (bám liên tục không nhảy nhánh).

    solver = NewArmKinematicsSolver()
    degs = solver.solve_ik(x, y, z)          # None nếu không với tới
    python3 kinematics_newarm.py             # tự kiểm
"""
import math
import os
import sys

_RL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "rl")
sys.path.insert(0, os.path.abspath(_RL_DIR))
import fk_newarm as F  # noqa: E402


class NewArmKinematicsSolver:
    JOINT_NAMES = F.JOINT_NAMES

    def __init__(self, wrist_roll_deg=0.0, front=-1, elbow=-1, max_step_deg=None):
        """
        wrist_roll_deg: góc khớp J4 giữ cố định (độ, URDF) — bút đồng trục J4 nên
            không ảnh hưởng đầu bút; chọn sao cho marker/hộp bút quay về camera.
        front, elbow: nhánh IK dùng cho lần giải ĐẦU TIÊN. (-1, -1) = tay vươn
            về -Y (phía trước), khuỷu gập cùng phía — nhánh vẽ lên bảng.
        max_step_deg: nếu đặt, từ chối nghiệm nhảy quá ngần này độ so với lần
            trước (chống quật tay khi vision nhiễu); None = không chặn.
        """
        self.q4 = math.radians(wrist_roll_deg)
        self.branch = (front, elbow)
        self.max_step = None if max_step_deg is None else math.radians(max_step_deg)
        self.q_prev = None

    def reset(self):
        self.q_prev = None

    def solve_q(self, x, y, z):
        """Điểm đầu bút (m, base_link) -> [q1..q4] (rad, URDF) hoặc None."""
        target = (x, y, z)
        if self.q_prev is None:
            cands = [q for q, b in F.ik_tip_branches(target, q4=self.q4) if b == self.branch]
            q = cands[0] if cands else None
        else:
            q = F.ik_tip_nearest(target, self.q_prev)
            if q is not None and self.max_step is not None:
                if max(abs(a - b) for a, b in zip(q, self.q_prev)) > self.max_step:
                    return None
        if q is not None:
            self.q_prev = q
        return q

    def solve_ik(self, x, y, z):
        """Điểm đầu bút (m, base_link) -> [base, shoulder, elbow, wrist_roll]
        độ lệnh servo, hoặc None nếu không với tới / vượt giới hạn."""
        q = self.solve_q(x, y, z)
        if q is None:
            return None
        return [F.q_to_servo_deg(n, v) for n, v in zip(self.JOINT_NAMES, q)]

    @staticmethod
    def pen_tilt_from_normal_deg(q, normal):
        """Góc (độ) giữa hướng bút và pháp tuyến hướng VÀO bảng (vector đơn vị,
        base_link). Bút lò xo: nên < ~45°."""
        v = F.pen_direction(q)
        c = sum(a * b for a, b in zip(v, normal))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    @staticmethod
    def fk(servo_degs):
        """Độ lệnh servo -> vị trí đầu bút (m, base_link)."""
        return F.fk_tip(F.servo_degs_to_q(servo_degs))


def _self_test():
    s = NewArmKinematicsSolver()
    # vẽ hình vuông 10cm trên bảng thẳng đứng cách trục J1 32cm, tâm z=0.30
    y_plane = F._J1_WORLD[1] - 0.32
    ok, worst_err, worst_tilt, worst_step, prev = True, 0.0, 0.0, 0.0, None
    corners = [(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05), (-0.05, -0.05)]
    n = 0
    for (u0, v0), (u1, v1) in zip(corners[:-1], corners[1:]):
        for i in range(50):
            t = i / 50
            p = (u0 + (u1 - u0) * t, y_plane, 0.30 + v0 + (v1 - v0) * t)
            degs = s.solve_ik(*p)
            n += 1
            if degs is None:
                ok = False
                continue
            worst_err = max(worst_err, math.dist(s.fk(degs), p))
            worst_tilt = max(worst_tilt, s.pen_tilt_from_normal_deg(s.q_prev, (0.0, -1.0, 0.0)))
            if prev is not None:
                worst_step = max(worst_step, max(abs(a - b) for a, b in zip(degs, prev)))
            prev = degs
    print(f"[self-test] {n} điểm trên hình vuông 10cm: sai số FK(IK) max {worst_err*1000:.2e} mm, "
          f"bước servo max {worst_step:.2f}°/điểm, góc bút-pháp tuyến max {worst_tilt:.1f}°")
    far = s.solve_ik(0.0, -1.0, 0.3)
    print(f"[self-test] điểm ngoài tầm với -> {far}")
    ok = ok and worst_err < 1e-6 and worst_step < 5.0 and far is None
    print("[self-test] " + ("ĐẠT" if ok else "KHÔNG ĐẠT"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
