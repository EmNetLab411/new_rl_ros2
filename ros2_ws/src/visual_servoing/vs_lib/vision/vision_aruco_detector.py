#!/usr/bin/env python3
"""
Vision ArUco Detector for Visual Servoing
Adapted from Visual_Servoing/nodes/vision_node_ros2.py

Detects ArUco markers (IDs 0-3) and publishes board pose.
Works with Gazebo camera simulation.
"""

import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import time


# ArUco Board Configuration (matching Visual_Servoing)
BOARD_SIZE_M = 0.120       # 120mm board
OFFSET = 0.048             # Marker offset from center
MARKER_SIZE = 0.020        # 20mm markers
HALF_MARKER = MARKER_SIZE / 2


def get_marker_corners_3d(center_x, center_y, half_size):
    """Get 3D coordinates of marker corners in board frame."""
    return np.array([
        [center_x - half_size, center_y + half_size, 0],
        [center_x + half_size, center_y + half_size, 0],
        [center_x + half_size, center_y - half_size, 0],
        [center_x - half_size, center_y - half_size, 0]
    ], dtype=np.float32)


# Board marker positions (IDs 0-3 at corners)
# Standard ArUco Board Layout:
# 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
# Using standard 2D coords: X-right, Y-up (relative to board center)
BOARD_CONFIG_3D = {
    0: get_marker_corners_3d(-OFFSET,  OFFSET, HALF_MARKER),  # Top-left
    1: get_marker_corners_3d( OFFSET,  OFFSET, HALF_MARKER),  # Top-right
    2: get_marker_corners_3d( OFFSET, -OFFSET, HALF_MARKER),  # Bottom-right
    3: get_marker_corners_3d(-OFFSET, -OFFSET, HALF_MARKER)   # Bottom-left
}


class VisionArucoDetector(Node):
    """ROS2 node for ArUco marker detection and board pose estimation."""
    
    def __init__(self):
        super().__init__('vision_aruco_detector')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('show_gui', False)
        self.declare_parameter('board_pose_topic', '/vision/board_pose')
        self.declare_parameter('raw_board_pose_topic', '/vision/board_pose_raw')
        self.declare_parameter('board_detected_topic', '/vision/board_detected')
        self.declare_parameter('board_tracking_mode', 'static_locked')
        self.declare_parameter('stable_frame_count', 30)
        self.declare_parameter('publish_raw_board_pose', True)
        self.declare_parameter('use_pose_cache', True)
        self.declare_parameter('cache_timeout', 1.0)
        # Board geometry (defaults = original 120mm sim board). The printed
        # newarm workspace board uses offset 0.075 / size 0.030 — see
        # config/vision_board_newarm.yaml.
        self.declare_parameter('board_marker_offset_m', OFFSET)
        self.declare_parameter('board_marker_size_m', MARKER_SIZE)
        self.board_marker_offset = float(self.get_parameter('board_marker_offset_m').value)
        self.board_marker_size = float(self.get_parameter('board_marker_size_m').value)
        _o, _h = self.board_marker_offset, self.board_marker_size / 2
        self.board_config = {
            0: get_marker_corners_3d(-_o,  _o, _h),  # Top-left
            1: get_marker_corners_3d( _o,  _o, _h),  # Top-right
            2: get_marker_corners_3d( _o, -_o, _h),  # Bottom-right
            3: get_marker_corners_3d(-_o, -_o, _h),  # Bottom-left
        }
        
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.show_gui = self.get_parameter('show_gui').value
        self.board_pose_topic = self.get_parameter('board_pose_topic').value
        self.raw_board_pose_topic = self.get_parameter('raw_board_pose_topic').value
        self.board_detected_topic = self.get_parameter('board_detected_topic').value
        self.board_tracking_mode = self.get_parameter('board_tracking_mode').value
        self.publish_raw_board_pose = self.get_parameter('publish_raw_board_pose').value
        self.use_pose_cache = self.get_parameter('use_pose_cache').value
        self.buffer_size = max(1, int(self.get_parameter('stable_frame_count').value))
        self.cache_timeout = float(self.get_parameter('cache_timeout').value)
        
        if self.board_tracking_mode not in {'static_locked', 'continuous'}:
            self.get_logger().warn(
                f"[VisionAruco] Unknown board_tracking_mode='{self.board_tracking_mode}', "
                "falling back to 'static_locked'"
            )
            self.board_tracking_mode = 'static_locked'
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # ArUco detector setup - handle different OpenCV versions
        # Using DICT_4X4_1000 to match user's marker files (4x4_1000-*.svg)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        # OpenCV 4.7+ uses DetectorParameters(), older uses DetectorParameters_create()
        try:
            self.aruco_params = aruco.DetectorParameters()
            self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = aruco.DetectorParameters_create()
            self.use_new_api = False
        
        # Camera intrinsics (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Default camera matrix for Gazebo (640x480, 80° FOV)
        fx = 554.254  # Approximate for 80° FOV
        fy = 554.254
        cx = 320.0
        cy = 240.0
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        
        # Publishers
        self.board_pose_pub = self.create_publisher(
            PoseStamped, self.board_pose_topic, 10)
        self.raw_board_pose_pub = None
        if self.publish_raw_board_pose:
            self.raw_board_pose_pub = self.create_publisher(
                PoseStamped, self.raw_board_pose_topic, 10)
        self.board_detected_pub = self.create_publisher(
            Bool, self.board_detected_topic, 10)
        
        # State
        self.last_board_pose = None
        self.last_detection_time = 0.0
        self.detection_count = 0
        self.board_locked = False
        self.locked_pose = None
        self.pose_buffer = []  # To average multiple detections for stable lock
        
        self.get_logger().info(f"[VisionAruco] Started, listening on {image_topic}")
        self.get_logger().info(
            f"[VisionAruco] Mode={self.board_tracking_mode}, "
            f"effective_pose_topic={self.board_pose_topic}, "
            f"raw_pose_topic={self.raw_board_pose_topic if self.publish_raw_board_pose else 'disabled'}"
        )
        self.get_logger().info(
            f"[VisionAruco] Board markers: offset ±{self.board_marker_offset*1000:.1f}mm, "
            f"size {self.board_marker_size*1000:.1f}mm, "
            f"Pose cache: {self.cache_timeout}s"
        )

    @staticmethod
    def _copy_pose(source_pose: PoseStamped) -> PoseStamped:
        """Create a detached PoseStamped copy so cached/locked poses are stable."""
        copied_pose = PoseStamped()
        copied_pose.header.stamp = source_pose.header.stamp
        copied_pose.header.frame_id = source_pose.header.frame_id
        copied_pose.pose.position.x = source_pose.pose.position.x
        copied_pose.pose.position.y = source_pose.pose.position.y
        copied_pose.pose.position.z = source_pose.pose.position.z
        copied_pose.pose.orientation.x = source_pose.pose.orientation.x
        copied_pose.pose.orientation.y = source_pose.pose.orientation.y
        copied_pose.pose.orientation.z = source_pose.pose.orientation.z
        copied_pose.pose.orientation.w = source_pose.pose.orientation.w
        return copied_pose

    def _build_locked_pose(self) -> PoseStamped:
        """Average the buffered detections into one stable effective pose."""
        avg_pos = np.mean([
            [p.pose.position.x, p.pose.position.y, p.pose.position.z]
            for p in self.pose_buffer
        ], axis=0)

        avg_q = np.mean([
            [p.pose.orientation.x, p.pose.orientation.y,
             p.pose.orientation.z, p.pose.orientation.w]
            for p in self.pose_buffer
        ], axis=0)
        avg_q /= np.linalg.norm(avg_q)

        locked_pose = PoseStamped()
        locked_pose.header.frame_id = self.pose_buffer[-1].header.frame_id
        locked_pose.pose.position.x = float(avg_pos[0])
        locked_pose.pose.position.y = float(avg_pos[1])
        locked_pose.pose.position.z = float(avg_pos[2])
        locked_pose.pose.orientation.x = float(avg_q[0])
        locked_pose.pose.orientation.y = float(avg_q[1])
        locked_pose.pose.orientation.z = float(avg_q[2])
        locked_pose.pose.orientation.w = float(avg_q[3])
        return locked_pose
    
    def camera_info_callback(self, msg: CameraInfo):
        """Update camera intrinsics from camera_info topic (only once)."""
        if msg.k[0] > 0:
            new_fx = msg.k[0]
            # Only update and log if fx changed significantly (first real update)
            if self.camera_matrix is None or abs(self.camera_matrix[0, 0] - new_fx) > 1.0:
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                self.dist_coeffs = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)
                self.get_logger().info(f"[VisionAruco] Camera matrix updated: fx={new_fx:.1f}")
    
    def image_callback(self, msg: Image):
        """Process camera image for ArUco detection."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # Detect ArUco markers - use correct API based on version
        if self.use_new_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(cv_image)
        else:
            corners, ids, rejected = aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        
        detected = Bool()
        detected.data = False
        
        if ids is not None and len(ids) >= 2:
            # Collect detected markers that match our board
            object_points = []
            image_points = []
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.board_config:
                    obj_pts = self.board_config[marker_id]
                    img_pts = corners[i][0]
                    object_points.append(obj_pts)
                    image_points.append(img_pts)
            
            if len(object_points) >= 2:
                # Stack points for solvePnP
                object_points = np.vstack(object_points).astype(np.float32)
                image_points = np.vstack(image_points).astype(np.float32)
                
                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points,
                    self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Convert rotation vector to matrix then to quaternion using scipy
                    R_mat, _ = cv2.Rodrigues(rvec)
                    quat = R_scipy.from_matrix(R_mat).as_quat() # [x, y, z, w]
                    
                    # Store in temporary pose
                    current_pose = PoseStamped()
                    current_pose.header.stamp = self.get_clock().now().to_msg()
                    current_pose.header.frame_id = 'camera_optical_link'
                    current_pose.pose.position.x = float(tvec[0])
                    current_pose.pose.position.y = float(tvec[1])
                    current_pose.pose.position.z = float(tvec[2])
                    current_pose.pose.orientation.x = float(quat[0])
                    current_pose.pose.orientation.y = float(quat[1])
                    current_pose.pose.orientation.z = float(quat[2])
                    current_pose.pose.orientation.w = float(quat[3])

                    if self.raw_board_pose_pub is not None:
                        self.raw_board_pose_pub.publish(self._copy_pose(current_pose))

                    effective_pose = None
                    if self.board_tracking_mode == 'static_locked':
                        if not self.board_locked:
                            self.pose_buffer.append(self._copy_pose(current_pose))
                            if len(self.pose_buffer) > self.buffer_size:
                                self.pose_buffer.pop(0)

                            if len(self.pose_buffer) >= self.buffer_size:
                                self.locked_pose = self._build_locked_pose()
                                self.board_locked = True
                                self.get_logger().info(
                                    f"✨ Board pose LOCKED after {self.buffer_size} samples"
                                )

                        if self.board_locked and self.locked_pose is not None:
                            effective_pose = self._copy_pose(self.locked_pose)
                    else:
                        effective_pose = current_pose

                    if effective_pose is not None:
                        effective_pose.header.stamp = self.get_clock().now().to_msg()
                        self.board_pose_pub.publish(effective_pose)
                        self.last_board_pose = self._copy_pose(effective_pose)
                        self.last_detection_time = time.time()
                        detected.data = True
                        
                        if self.detection_count == 0 or self.detection_count % 3000 == 0:
                            self.get_logger().info(
                                f"[VisionAruco] Board pose at "
                                f"({effective_pose.pose.position.x:.3f}, "
                                f"{effective_pose.pose.position.y:.3f}, "
                                f"{effective_pose.pose.position.z:.3f})m "
                                f"[frame #{self.detection_count}]"
                            )
                    
                    self.detection_count += 1
        
        # Use cached pose when detection fails (handles robot occlusion)
        if self.use_pose_cache and not detected.data and self.last_board_pose is not None:
            elapsed = time.time() - self.last_detection_time
            if elapsed < self.cache_timeout:
                # Publish cached pose with updated timestamp
                cached_msg = self._copy_pose(self.last_board_pose)
                cached_msg.header.stamp = self.get_clock().now().to_msg()
                self.board_pose_pub.publish(cached_msg)
                detected.data = True  # Still "detected" via cache
        
        self.board_detected_pub.publish(detected)
        
        # GUI visualization
        if self.show_gui:
            vis_image = cv_image.copy()
            if ids is not None:
                aruco.drawDetectedMarkers(vis_image, corners, ids)
            cv2.imshow('ArUco Detection', vis_image)
            cv2.waitKey(1)
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])


def main(args=None):
    rclpy.init(args=args)
    node = VisionArucoDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_gui:
            cv2.destroyAllWindows()
        node.destroy_node()
        # Only shutdown if not already shut down
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
