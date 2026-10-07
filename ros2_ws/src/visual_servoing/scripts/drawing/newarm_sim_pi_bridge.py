#!/usr/bin/env python3
"""
Cầu nối giả lập giao diện của Pi trên mô phỏng Gazebo tay mới — để chạy thử
NGUYÊN các script CoVip (calibrate_hand_eye.py collect-tip, goto_board_center.py,
newarm_bringup.py) trước khi có robot thật.

    Thật (Pi)                           Mô phỏng (node này)
    /pca9685_servo/command  (độ)   ->   /arm_controller/joint_trajectory (rad)
    /pca9685_servo/joint_states    <-   /joint_states  (đổi sang rad của góc LỆNH servo, như driver)
    /pca9685_servo/enable          <-   Trigger giả, luôn thành công
    /aeroscript/pen_xyz   (mm)     <-   TF camera_optical_link -> pen_tip + nhiễu Gauss
    /aeroscript/board_pose         <-   /vision/board_pose_raw (vision_aruco_detector)

Chạy với cùng file hiệu chỉnh servo như các script CoVip:
    export NEWARM_CALIB=$(ros2 pkg prefix visual_servoing)/share/visual_servoing/config/newarm_servo_calib.sim.json
    ros2 run visual_servoing newarm_sim_pi_bridge
"""
import math

import numpy as np
import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rl import fk_newarm as F


class NewarmSimPiBridge(Node):
    def __init__(self):
        super().__init__('newarm_sim_pi_bridge', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
        self.declare_parameter('pen_noise_mm', 2.0)       # nhiễu mỗi trục của XYZ bút giả lập
        self.declare_parameter('pen_bias_mm', [0.0, 0.0, 0.0])   # lệch hệ thống (thử độ bền vòng kín)
        self.declare_parameter('pen_rate_hz', 15.0)
        self.declare_parameter('servo_offset_deg', [0.0, 0.0, 0.0, 0.0])   # servo lắp lệch home (thử độ bền)
        self.declare_parameter('move_time_s', 0.15)
        self.noise = float(self.get_parameter('pen_noise_mm').value)
        self.bias = np.array(self.get_parameter('pen_bias_mm').value, float)
        self.servo_off = [math.radians(v) for v in self.get_parameter('servo_offset_deg').value]
        self.move_time = float(self.get_parameter('move_time_s').value)
        self.rng = np.random.default_rng(0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.traj_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.state_pub = self.create_publisher(JointState, '/pca9685_servo/joint_states', 10)
        self.pen_pub = self.create_publisher(Point, '/aeroscript/pen_xyz', 1)
        self.board_pub = self.create_publisher(PoseStamped, '/aeroscript/board_pose', 1)
        self.create_subscription(JointState, '/pca9685_servo/command', self._on_command, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.create_subscription(PoseStamped, '/vision/board_pose_raw', self.board_pub.publish, 10)
        self.create_service(Trigger, '/pca9685_servo/enable', self._on_enable)
        self.cmd_deg = None
        import threading
        import time
        self._time = time
        # nhịp theo giờ THẬT (script CoVip chạy giờ thật)
        threading.Thread(target=self._pen_loop, daemon=True).start()
        self.get_logger().info(
            f'Cầu nối Pi giả lập sẵn sàng: nhiễu bút {self.noise}mm, lệch {self.bias.tolist()}mm, '
            f'servo lệch {self.get_parameter("servo_offset_deg").value}°, calib {F.CALIB_PATH} '
            f'({"đã nạp" if F.CALIB_LOADED else "KHÔNG có — dùng mặc định"})')

    def _on_enable(self, _req, resp):
        resp.success, resp.message = True, 'sim'
        return resp

    def _on_command(self, msg):
        cmd = dict(zip(msg.name, msg.position))
        if not all(n in cmd for n in F.JOINT_NAMES):
            return
        self.cmd_deg = [float(cmd[n]) for n in F.JOINT_NAMES]
        traj = JointTrajectory()
        traj.joint_names = list(F.JOINT_NAMES)
        pt = JointTrajectoryPoint()
        # servo_offset: tay "thật" (Gazebo) lệch khỏi góc mà phần mềm tưởng
        pt.positions = [F.servo_deg_to_q(n, d) + o for n, d, o in zip(F.JOINT_NAMES, self.cmd_deg, self.servo_off)]
        pt.time_from_start = DurationMsg(sec=0, nanosec=int(self.move_time * 1e9))
        traj.points.append(pt)
        self.traj_pub.publish(traj)

    def _on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if not all(n in pos for n in F.JOINT_NAMES):
            return
        out = JointState()
        out.header = msg.header
        out.name = list(F.JOINT_NAMES)
        # driver thật báo lại góc LỆNH (không có phản hồi vị trí), đơn vị rad
        degs = self.cmd_deg or [F.q_to_servo_deg(n, pos[n]) for n in F.JOINT_NAMES]
        out.position = [math.radians(d) for d in degs]
        self.state_pub.publish(out)

    def _pen_loop(self):
        period = 1.0 / float(self.get_parameter('pen_rate_hz').value)
        while rclpy.ok():
            self._time.sleep(period)
            try:
                tf = self.tf_buffer.lookup_transform('camera_optical_link', 'pen_tip', rclpy.time.Time())
            except Exception:
                continue
            t = tf.transform.translation
            p = np.array([t.x, t.y, t.z]) * 1000.0 + self.bias + self.rng.normal(0.0, self.noise, 3)
            if p[2] <= 0:
                continue
            self.pen_pub.publish(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))


def main(args=None):
    rclpy.init(args=args)
    node = NewarmSimPiBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
