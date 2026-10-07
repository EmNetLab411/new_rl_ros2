"""
Vào/ra dùng chung cho các script chạy tay mới qua driver wicom_roboarm (hoặc
qua cầu nối mô phỏng newarm_sim_pi_bridge — cùng topic, cùng đơn vị):

    /pca9685_servo/joint_states   driver phát (rad của góc LỆNH servo)
    /pca9685_servo/command        JointState, độ lệnh servo 0..180
    /pca9685_servo/enable         Trigger
"""
import math
import time

import numpy as np

from arm_models import joint_states_to_servo_degs

COMMAND_TOPIC = "/pca9685_servo/command"
STATE_TOPIC = "/pca9685_servo/joint_states"
ENABLE_SERVICE = "/pca9685_servo/enable"


class ArmIO:
    def __init__(self, node, arm, speed_deg_s=25.0, rate_hz=10.0):
        import rclpy
        from sensor_msgs.msg import JointState
        self._rclpy, self._JointState = rclpy, JointState
        self.node, self.arm = node, arm
        self.speed, self.dt = float(speed_deg_s), 1.0 / rate_hz
        self.servo_degs = None          # trạng thái driver báo về
        self.cmd = None                 # lệnh gần nhất đã gửi
        node.create_subscription(JointState, STATE_TOPIC, self._on_state, 10)
        self._pub = node.create_publisher(JointState, COMMAND_TOPIC, 10)

    def _on_state(self, msg):
        try:
            self.servo_degs = joint_states_to_servo_degs(msg.name, msg.position, self.arm.joint_names)
        except ValueError:
            pass

    def spin(self, sec):
        t_end = time.time() + sec
        while time.time() < t_end:
            self._rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, t_end - time.time())))

    def wait_ready(self, timeout=10.0):
        """Chờ joint_states rồi enable. False nếu driver không chạy."""
        from std_srvs.srv import Trigger
        t0 = time.time()
        while self.servo_degs is None and time.time() - t0 < timeout:
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.servo_degs is None:
            return False
        cli = self.node.create_client(Trigger, ENABLE_SERVICE)
        if cli.wait_for_service(timeout_sec=3.0):
            fut = cli.call_async(Trigger.Request())
            self._rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        self.cmd = list(self.servo_degs)
        return True

    def _send(self, degs):
        msg = self._JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.arm.joint_names)
        msg.position = [float(d) for d in degs]
        self._pub.publish(msg)
        self.cmd = list(degs)

    def move_to(self, servo_degs, settle_sec=0.8):
        """Đi tới lệnh servo đích theo đường thẳng trong không gian khớp, khớp
        nhanh nhất không quá speed_deg_s. Trả khi đã tới + chờ settle_sec."""
        start = np.array(self.cmd if self.cmd is not None else self.servo_degs, float)
        goal = np.clip(np.array(servo_degs, float), 0.0, 180.0)
        n = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / (self.speed * self.dt))))
        for i in range(1, n + 1):
            self._send(start + (goal - start) * i / n)
            self.spin(self.dt)
        # nhắc lại lệnh trong lúc chờ để không dính command_timeout của driver
        t_end = time.time() + settle_sec
        while time.time() < t_end:
            self._send(goal)
            self.spin(self.dt)

    def q(self):
        """Góc khớp URDF (rad) theo LỆNH gần nhất (servo không có phản hồi vị trí)."""
        return self.arm.q_from_servo_degs(self.cmd if self.cmd is not None else self.servo_degs)
