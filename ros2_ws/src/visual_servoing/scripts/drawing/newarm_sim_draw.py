#!/usr/bin/env python3
"""
Vẽ thử một hình vuông trên bảng ArUco bằng tay mới (newarm) trong Gazebo.

Luồng: /vision/board_pose (vision_aruco_detector) -> BoardTransform (TF camera
-> base_link) -> điểm trên bảng trong base_link -> IK giải tích fk_newarm ->
/arm_controller/joint_trajectory. Trong lúc chạy đọc TF base_link->pen_tip để
đo sai số bám và phát /drawing/pen_position cho gazebo_drawing_visualizer.

    ros2 launch visual_servoing newarm_sim.launch.py
    ros2 run visual_servoing newarm_sim_draw
    ros2 run visual_servoing newarm_sim_draw --ros-args -p side_m:=0.08 -p board_source:=nominal
"""
import math
import sys

import numpy as np
import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rl import fk_newarm as F
from rl.board_transform import BoardTransform

DRAW_BRANCH = (-1, -1)
# Bảng danh định trong base_link (khớp scripts/rl/newarm_make_sim.py): tâm + hệ
# trục bảng (x sang phải, y lên, z = pháp tuyến hướng về tay)
NOMINAL_BOARD = np.array([
    [-1.0, 0.0, 0.0, -0.015],
    [0.0, 0.0, 1.0, F._J1_WORLD[1] - 0.225],
    [0.0, 1.0, 0.0, F._J1_WORLD[2] - 0.163],
    [0.0, 0.0, 0.0, 1.0],
])


class NewarmSimDraw(Node):
    def __init__(self):
        super().__init__('newarm_sim_draw', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
        self.declare_parameter('side_m', 0.10)
        self.declare_parameter('lift_m', 0.02)
        self.declare_parameter('step_m', 0.005)
        self.declare_parameter('draw_speed_m_s', 0.02)
        self.declare_parameter('board_source', 'vision')       # vision | nominal
        self.declare_parameter('vision_timeout_s', 20.0)
        self.declare_parameter('joint_limits_deg', [-90.0, 90.0, -90.0, 90.0, -30.0, 150.0, -90.0, 90.0])
        self.side = float(self.get_parameter('side_m').value)
        self.lift = float(self.get_parameter('lift_m').value)
        self.step = float(self.get_parameter('step_m').value)
        self.speed = float(self.get_parameter('draw_speed_m_s').value)
        self.board_source = self.get_parameter('board_source').value
        self.vision_timeout = float(self.get_parameter('vision_timeout_s').value)
        lim = [math.radians(v) for v in self.get_parameter('joint_limits_deg').value]
        self.q_lo, self.q_hi = lim[0::2], lim[1::2]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.board_tf = BoardTransform(self.tf_buffer, lock_transform=True, use_ideal_rotation=False)
        self.q_now = None
        self.moving = True
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.create_subscription(PoseStamped, '/vision/board_pose', self._on_board, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.pen_pub = self.create_publisher(Point, '/drawing/pen_position', 10)

    def _on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in F.JOINT_NAMES):
            self.q_now = [pos[n] for n in F.JOINT_NAMES]
            self.moving = (len(msg.velocity) != len(msg.name)
                           or max(abs(v) for v in msg.velocity) > 1e-4)

    def _on_board(self, msg):
        self.board_tf.update_from_pose(msg)

    def _spin_until(self, cond, timeout_s):
        """Quay node tới khi cond() đúng hoặc hết timeout (giờ sim)."""
        t0 = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = self.get_clock().now().nanoseconds * 1e-9
            if now == 0.0:
                continue
            t0 = now if t0 is None else t0
            if cond():
                return True
            if now - t0 > timeout_s:
                return False
        return False

    def _tip_tf(self):
        try:
            tf = self.tf_buffer.lookup_transform('base_link', 'pen_tip', rclpy.time.Time())
        except Exception:
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    def _ik(self, p, q_prev):
        for q, branch in F.ik_tip_branches(tuple(p), q4=q_prev[3], check_limits=False):
            if branch == DRAW_BRANCH:
                if all(lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in zip(q, self.q_lo, self.q_hi)):
                    return q
                return None
        return None

    def run(self):
        log = self.get_logger()
        if not self._spin_until(lambda: self.q_now is not None, 30.0):
            log.error('Không nhận được /joint_states — đã chạy newarm_sim.launch.py chưa?')
            return 1

        T_board = NOMINAL_BOARD
        if self.board_source == 'vision':
            if self._spin_until(lambda: self.board_tf.is_ready, self.vision_timeout):
                T_board = self.board_tf.T_combined
                d = (T_board[:3, 3] - NOMINAL_BOARD[:3, 3]) * 1000
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(T_board[:3, 2] @ NOMINAL_BOARD[:3, 2])))))
                log.info(f'Bảng từ vision: tâm {np.round(T_board[:3, 3], 4)} (base_link), '
                         f'lệch so với danh định {np.round(d, 1)} mm, pháp tuyến lệch {ang:.2f}°')
            else:
                log.error('Không có /vision/board_pose (camera không thấy đủ marker?). '
                          'Chạy lại với -p board_source:=nominal để bỏ qua vision.')
                return 1
        else:
            log.info(f'Dùng bảng danh định, tâm {np.round(T_board[:3, 3], 4)} (base_link)')

        # Quỹ đạo trong hệ bảng: (x, y, z, đang_vẽ)
        h = self.side / 2
        corners = [(-h, h), (h, h), (h, -h), (-h, -h), (-h, h)]
        path = [(corners[0][0], corners[0][1], self.lift, False),
                (corners[0][0], corners[0][1], 0.0, True)]     # hạ bút rồi dừng 1s
        for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
            n = max(1, int(round(math.hypot(x1 - x0, y1 - y0) / self.step)))
            start = 0 if (x0, y0) == corners[0] else 1
            path += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 0.0, True) for i in range(start, n + 1)]
        path.append((corners[-1][0], corners[-1][1], self.lift, False))

        q_prev = [0.0, 0.0, 0.0, self.q_now[3]]
        qs, pts_base = [], []
        for x, y, z, _ in path:
            p = (T_board @ np.array([x, y, z, 1.0]))[:3]
            q = self._ik(p, q_prev)
            if q is None:
                log.error(f'Điểm bảng ({x*100:.1f}, {y*100:.1f}, {z*100:.1f}) cm ngoài tầm với / giới hạn khớp')
                return 1
            qs.append(q)
            pts_base.append(p)
            q_prev = q

        # Gói thành 1 JointTrajectory: tới điểm đầu 3s, vẽ theo tốc độ đặt, về home 3s
        traj = JointTrajectory()
        traj.joint_names = list(F.JOINT_NAMES)
        t, times = 3.0, []
        for i, q in enumerate(qs):
            if i > 0:
                dist = float(np.linalg.norm(pts_base[i] - pts_base[i - 1]))
                t += 1.0 if dist < 1e-9 else max(0.05, dist / self.speed)
            times.append(t)
        for q, tt in zip(qs + [[0.0, 0.0, 0.0, 0.0]], times + [t + 3.0]):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.time_from_start = DurationMsg(sec=int(tt), nanosec=int((tt % 1) * 1e9))
            traj.points.append(pt)
        self._spin_until(lambda: self.traj_pub.get_subscription_count() > 0, 10.0)
        self.traj_pub.publish(traj)
        t_start = self.get_clock().now().nanoseconds * 1e-9
        log.info(f'Gửi {len(traj.points)} điểm, vẽ vuông {self.side*100:.0f}cm, tổng {t + 3.0:.1f}s (giờ sim)')

        # Theo dõi: sai số so với đường danh định, tại các mốc thời gian đang vẽ
        T_inv = np.linalg.inv(T_board)
        draw_t0, draw_t1 = times[2], times[-2]
        err_plane, err_path, err_fk, err_true = [], [], [], []
        T_nom_inv = np.linalg.inv(NOMINAL_BOARD)
        static_poses = set()

        def sample():
            now = self.get_clock().now().nanoseconds * 1e-9 - t_start
            tip = self._tip_tf()
            if tip is None:
                return now > t + 4.0
            self.pen_pub.publish(Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])))
            # TF và joint_states lệch pha nhau vài chục ms -> chỉ so khi tay đứng yên
            if self.q_now is not None and not self.moving and now > 1.0:
                err_fk.append(float(np.linalg.norm(tip - np.array(F.fk_tip(self.q_now)))))
                static_poses.add(tuple(round(math.degrees(v)) for v in self.q_now))
            if draw_t0 <= now <= draw_t1:
                b = T_inv @ np.append(tip, 1.0)
                err_plane.append(abs(float(b[2])))
                err_true.append(float((T_nom_inv @ np.append(tip, 1.0))[2]))
                # khoảng cách tới chu vi hình vuông trong mặt bảng
                err_path.append(abs(max(abs(float(b[0])), abs(float(b[1]))) - h))
            return now > t + 4.0

        self._spin_until(sample, t + 30.0)
        if not err_path:
            log.error('Không lấy được mẫu TF nào trong lúc vẽ')
            return 1
        home_err = float(np.abs(np.array(self.q_now)).max()) if self.q_now else float('nan')
        log.info(f'KẾT QUẢ ({len(err_path)} mẫu lúc vẽ): '
                 f'lệch khỏi cạnh hình vuông TB {np.mean(err_path)*1000:.2f} / max {np.max(err_path)*1000:.2f} mm; '
                 f'lệch khỏi mặt bảng TB {np.mean(err_plane)*1000:.2f} / max {np.max(err_plane)*1000:.2f} mm')
        if self.board_source == 'vision':
            log.info(f'So với mặt bảng THẬT trong world (danh định): đầu bút cách mặt bảng '
                     f'{np.min(err_true)*1000:+.1f} .. {np.max(err_true)*1000:+.1f} mm '
                     f'(+ = chưa chạm, - = lún vào) — đây là sai số ước lượng bảng của vision')
        if err_fk:
            log.info(f'TF pen_tip (URDF Gazebo) so với fk_newarm(joint_states) lúc đứng yên, '
                     f'{len(static_poses)} tư thế {sorted(static_poses)}°: max {np.max(err_fk)*1000:.3f} mm')
        log.info(f'Về home lệch {math.degrees(home_err):.2f}°')
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = NewarmSimDraw()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
