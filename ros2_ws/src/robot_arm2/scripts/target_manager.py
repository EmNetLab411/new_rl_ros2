#!/usr/bin/env python3
"""
Target Manager Node for Gazebo Fortress
Uses 'ign service' CLI to teleport target sphere visually

The target sphere is already spawned in the world file.
This node just teleports it when receiving /target_position messages.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import subprocess


class TargetManager(Node):
    """Teleports visual target sphere in Gazebo using ign CLI"""
    
    def __init__(self):
        super().__init__('target_manager')
        
        self.current_position = [0.0, 0.225, 0.22]  # Center of +Y workspace
        self.world_name = "rl_training_world"
        
        # Subscribe to target position updates from RL environment
        self.target_sub = self.create_subscription(
            Point,
            '/target_position',
            self._target_callback,
            10
        )
        
        self.get_logger().info("🎯 TargetManager ready. Subscribing to /target_position")
    
    def _teleport_sphere(self, x: float, y: float, z: float) -> bool:
        """Teleport sphere to new position using ign service"""
        try:
            cmd = [
                'gz', 'service',
                '-s', f'/world/{self.world_name}/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'name: "target_sphere" position: {{x: {x}, y: {y}, z: {z}}} orientation: {{w: 1.0}}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                self.current_position = [x, y, z]
                self.get_logger().info(f"🎯 Target: ({x:.3f}, {y:.3f}, {z:.3f})")
                return True
                
        except subprocess.TimeoutExpired:
            self.get_logger().debug("Teleport timed out")
        except Exception as e:
            self.get_logger().debug(f"Error: {e}")
        
        return False
    
    def _target_callback(self, msg: Point):
        """Callback when new target position received"""
        self._teleport_sphere(msg.x, msg.y, msg.z)


def main():
    rclpy.init()
    node = TargetManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
