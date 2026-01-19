#!/usr/bin/env python3
"""
Drawing Environment for RL Training

Extends the base RLEnvironment to support multi-waypoint drawing tasks.
The robot must reach a sequence of waypoints to draw a shape.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
from typing import Tuple, Optional, List
from geometry_msgs.msg import Point

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.rl_environment import RLEnvironment
from drawing.shape_generator import ShapeGenerator, Shape


class DrawingEnvironment(RLEnvironment):
    """
    RL Environment for drawing shapes by following waypoint sequences.
    """
    
    def __init__(self, 
                 max_episode_steps=300,
                 waypoint_tolerance=0.01,
                 shape_type='triangle',
                 shape_size=0.10,
                 y_plane=0.20,
                 randomize_shape=False):
        """
        Initialize Drawing Environment.
        
        Args:
            max_episode_steps: Max steps per episode
            waypoint_tolerance: Distance to consider waypoint reached (1cm)
            shape_type: 'triangle', 'square', 'line', or 'random_triangle'
            shape_size: Size of shape in meters
            y_plane: Y coordinate of drawing plane
            randomize_shape: Whether to randomize shape position each episode
        """
        super().__init__(max_episode_steps=max_episode_steps, goal_tolerance=waypoint_tolerance)
        
        self.get_logger().info("✏️ Initializing Drawing Environment...")
        
        self.waypoint_tolerance = waypoint_tolerance
        self.shape_type = shape_type
        self.shape_size = shape_size
        self.randomize_shape = randomize_shape
        
        self.shape_generator = ShapeGenerator(
            y_plane=y_plane, x_center=0.0, z_center=0.25, default_size=shape_size
        )
        
        self.current_shape: Optional[Shape] = None
        self.waypoints: np.ndarray = np.array([])
        self.waypoint_index = 0
        self.total_waypoints = 0
        self.waypoints_reached = 0
        self.line_points: List[np.ndarray] = []
        
        # Publisher for pen position
        self.pen_position_pub = self.create_publisher(Point, '/drawing/pen_position', 10)
        
        # Service client for line reset
        from std_srvs.srv import Empty
        self.reset_line_client = self.create_client(Empty, '/drawing/reset_line')
        
        # Observation space: 18D
        # [joints(6), EE(3), target(3), dist(3), dist3d(1), progress(1), remaining(1)]
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.pi/2]*6 +              # joint positions
                [0.0, 0.0, 0.0] +           # EE position (+Y workspace)
                [0.0, 0.0, 0.0] +           # target position (+Y workspace)
                [-1.0]*3 +                   # distance components
                [0.0, 0.0, 0.0]             # dist3d, progress, remaining
            ),
            high=np.array(
                [np.pi/2]*6 +               # joint positions
                [0.50, 0.50, 0.60] +        # EE position (+Y workspace)
                [0.50, 0.50, 0.60] +        # target position (+Y workspace)
                [1.0]*3 +                    # distance components
                [1.0, 1.0, 30.0]            # dist3d, progress, remaining
            ),
            dtype=np.float32
        )
        
        self.get_logger().info(f"📊 Drawing: shape={shape_type}, size={shape_size*100:.0f}cm")
        self.get_logger().info(f"📊 State: 18D (6 joints + 12 other)")
        self.get_logger().info("✅ Drawing Environment ready!")
    
    def _generate_shape(self) -> Shape:
        """Generate the target shape."""
        if self.shape_type == 'triangle':
            return self.shape_generator.equilateral_triangle(size=self.shape_size)
        elif self.shape_type == 'dense_triangle':
            # Continuous trajectory with 10 points per edge = 30 waypoints
            return self.shape_generator.dense_triangle(size=self.shape_size, points_per_edge=10)
        elif self.shape_type == 'square':
            return self.shape_generator.square(size=self.shape_size)
        elif self.shape_type == 'line':
            return self.shape_generator.line(length=self.shape_size)
        elif self.shape_type == 'random_triangle':
            return self.shape_generator.random_triangle(min_size=0.05, max_size=self.shape_size)
        else:
            return self.shape_generator.equilateral_triangle(size=self.shape_size)
    
    def get_state(self) -> Optional[np.ndarray]:
        """Get current state including waypoint progress."""
        if not self.data_ready:
            return None
        
        try:
            if self.waypoint_index < len(self.waypoints):
                target = self.waypoints[self.waypoint_index]
                self.target_x, self.target_y, self.target_z = target[0], target[1], target[2]
            
            dist_x = self.target_x - self.robot_x
            dist_y = self.target_y - self.robot_y
            dist_z = self.target_z - self.robot_z
            dist_3d = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
            
            progress = self.waypoint_index / max(1, self.total_waypoints)
            remaining = float(self.total_waypoints - self.waypoint_index)
            
            # 18D state vector (no velocities)
            state = np.array([
                # Joint positions (6)
                *self.joint_positions,
                # End-effector position (3)
                self.robot_x, self.robot_y, self.robot_z,
                # Target waypoint position (3)
                self.target_x, self.target_y, self.target_z,
                # Distance vector (3)
                dist_x, dist_y, dist_z,
                # Distance, progress, remaining (3)
                dist_3d, progress, remaining
            ], dtype=np.float32)  # Total: 18D
            
            return state
        except Exception as e:
            self.get_logger().error(f"State error: {e}")
            return None
    
    def reset_environment(self) -> Optional[np.ndarray]:
        """Reset for new drawing episode."""
        self.get_logger().info("🔄 Resetting Drawing Environment...")
        self.current_step = 0
        
        self.current_shape = self._generate_shape()
        self.waypoints = self.current_shape.waypoints
        self.total_waypoints = len(self.waypoints)
        self.waypoint_index = 0
        self.waypoints_reached = 0
        self.line_points = []
        
        self.get_logger().info(f"   Shape: {self.current_shape.name} ({self.total_waypoints} waypoints)")
        
        # Reset line visualization
        self._reset_line_visualization()
        
        # Move to home
        self._move_to_joint_positions(np.zeros(6), duration=2.0)
        time.sleep(0.5)
        
        # Set first waypoint
        if len(self.waypoints) > 0:
            self.target_x, self.target_y, self.target_z = self.waypoints[0]
        
        time.sleep(0.2)
        self.get_logger().info(f"✅ Drawing reset! Shape: {self.current_shape.name}")
        return self.get_state()
    
    def _reset_line_visualization(self):
        """Call line visualizer reset service."""
        if self.reset_line_client.wait_for_service(timeout_sec=0.5):
            from std_srvs.srv import Empty
            self.reset_line_client.call_async(Empty.Request())
    
    def _publish_pen_position(self):
        """Publish pen position for line visualization."""
        msg = Point(x=self.robot_x, y=self.robot_y, z=self.robot_z)
        self.pen_position_pub.publish(msg)
        self.line_points.append(np.array([self.robot_x, self.robot_y, self.robot_z]))
    
    def step(self, action: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """Execute one step."""
        self.current_step += 1
        
        state_before = self.get_state()
        if state_before is None:
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # 18D state: dist_3d is at index 15
        dist_before = state_before[15]
        
        target_joints = np.clip(action, self.joint_limits_low, self.joint_limits_high)
        self._move_to_joint_positions(target_joints, duration=0.8)  # Faster movement
        time.sleep(0.1)  # Reduced delay for faster line drawing
        
        self._publish_pen_position()
        
        next_state = self.get_state()
        if next_state is None:
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # 18D state: dist_3d is at index 15
        dist_after = next_state[15]
        reward, done = self._calculate_drawing_reward(dist_after, dist_before)
        
        # Ground collision check
        if self.robot_z <= 0.05:
            reward = -50.0
            done = True
            self._move_to_joint_positions(np.zeros(6), duration=1.0)
            time.sleep(0.5)
        
        if self.current_step >= self.max_episode_steps:
            done = True
        
        info = {
            'distance': dist_after,
            'waypoint_index': self.waypoint_index,
            'total_waypoints': self.total_waypoints,
            'waypoints_reached': self.waypoints_reached,
            'shape_complete': self.waypoint_index >= self.total_waypoints,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_drawing_reward(self, dist_after: float, dist_before: float) -> Tuple[float, bool]:
        """Calculate reward with waypoint advancement."""
        done = False
        
        if dist_after < self.waypoint_tolerance:
            self.waypoints_reached += 1
            self.waypoint_index += 1
            
            if self.waypoint_index >= self.total_waypoints:
                reward = 50.0
                done = True
                self.get_logger().info(f"🎨 SHAPE COMPLETE!")
            else:
                reward = 5.0
                next_wp = self.waypoints[self.waypoint_index]
                self.target_x, self.target_y, self.target_z = next_wp
                self.get_logger().info(f"✓ Waypoint {self.waypoint_index}/{self.total_waypoints}")
        else:
            improvement = dist_before - dist_after
            dist_reward = -5.0 * dist_after
            improve_reward = 5.0 * improvement if improvement >= 0 else 10.0 * improvement
            reward = np.clip(dist_reward + improve_reward - 0.1, -10.0, 10.0)
        
        return reward, done


def main(args=None):
    rclpy.init(args=args)
    
    try:
        env = DrawingEnvironment(shape_type='triangle', shape_size=0.10)
        shape = env._generate_shape()
        print(f"\nShape: {shape.name}, {shape.num_waypoints} waypoints")
        for i, wp in enumerate(shape.waypoints):
            print(f"  P{i+1}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
        rclpy.spin(env)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
