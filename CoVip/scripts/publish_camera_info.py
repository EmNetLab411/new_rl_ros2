#!/usr/bin/env python3
"""
Publish /camera_info từ file hiệu chuẩn calib/c930e_720p.npz (Phase 2).

CẦN THIẾT trước khi chạy vision_aruco_detector (Phase 3) — nếu không có
node này, vision_aruco_detector sẽ ÂM THẦM dùng ma trận K giả định cho
camera mô phỏng Gazebo 640x480 (sai hoàn toàn với C920 720p thật), không
báo lỗi gì, khiến kết quả hand-eye calibration sai mà không biết.

Chạy (trên Pi, cùng lúc với vision_aruco_detector):
    python3 scripts/publish_camera_info.py
"""
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

ROOT = Path(__file__).resolve().parent.parent


def main():
    calib_path = ROOT / "calib" / "c930e_720p.npz"
    if not calib_path.exists():
        print(f"Chưa có {calib_path} — chạy scripts/calibrate_camera.py trước (Phase 2).",
              file=sys.stderr)
        sys.exit(1)

    data = np.load(calib_path)
    K, dist, image_size = data["K"], data["dist"], data["image_size"]
    w, h = int(image_size[0]), int(image_size[1])

    rclpy.init()
    node = Node("publish_camera_info")
    pub = node.create_publisher(CameraInfo, "/camera_info", 10)

    msg = CameraInfo()
    msg.width, msg.height = w, h
    msg.k = [float(v) for v in K.flatten()]
    msg.d = [float(v) for v in dist.flatten()]
    msg.distortion_model = "plumb_bob"
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [K[0, 0], 0.0, K[0, 2], 0.0,
             0.0, K[1, 1], K[1, 2], 0.0,
             0.0, 0.0, 1.0, 0.0]

    node.get_logger().info(f"Publish /camera_info từ {calib_path} ({w}x{h}, fx={K[0,0]:.1f}) ở 10Hz.")

    def tick():
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)

    node.create_timer(0.1, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
