#!/usr/bin/env python3
"""
ROS2 Humble RL Environment for 6-DOF Robot Arm
Adapted from ROS1 Noetic main_rl_environment_noetic.py

This provides:
1. State space: end-effector position + 6 joint states + target position + distances
2. Action space: 2D target position (Y, Z) on drawing surface
3. Reward calculation: distance-based with goal achievement
4. Episode management with reset and step functions
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
import numpy as np
import random
import time
from typing import Tuple, Optional

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Pose, Quaternion
from gazebo_msgs.msg import ModelStates
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration

# TF2 for end-effector tracking
import tf2_ros
from tf2_ros import TransformException

# Gym for RL spaces
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


# ============================================================================
# WORKSPACE CONFIGURATION - 3D Workspace for Target Randomization
# ============================================================================

# 3D workspace parameters (based on detailed FK analysis - 50,000 samples)
# Robot is at origin, targets are in FRONT of robot (-Y direction)
#
# From FK exploration (5-95 percentile for safety):
#   Reachable X: [-0.24, +0.24] → using ±24cm (symmetric)
#   Reachable Y: [-0.25, +0.22] → using -35cm to -5cm (only forward, extended)
#   Reachable Z: [0.06, 0.42] → using 8-40cm (avoid ground, safe ceiling)
#
# Key positions:
#   Home: X=-0.003, Y=-0.022, Z=0.488 (high above)
#   J2=45°: Y=-0.28 (forward reach)
#   J2=90°: Y=-0.39 (max forward, low Z)
#
# Note: Robot faces -Y direction (toward drawing surface)

SURFACE_X_MIN = -0.24  # -24cm (symmetric, 5-95% safe)
SURFACE_X_MAX = 0.24   # +24cm (symmetric)
SURFACE_Y_MIN = -0.35  # -35cm (max forward reach)
SURFACE_Y_MAX = -0.05  # -5cm (close to robot but not too close)
SURFACE_Z_MIN = 0.08   # 8cm (avoid ground collision)
SURFACE_Z_MAX = 0.40   # 40cm (within reachable height)

# Target sphere radius (for border margin calculation)
TARGET_RADIUS = 0.01  # 1cm radius

# Workspace boundaries for target spawning (with 1cm margin from borders)
# This ensures the 1cm radius target sphere stays fully within the workspace
WORKSPACE_BOUNDS = {
    'x_min': SURFACE_X_MIN + TARGET_RADIUS,  # -23cm
    'x_max': SURFACE_X_MAX - TARGET_RADIUS,  # +23cm
    'y_min': SURFACE_Y_MIN + TARGET_RADIUS,  # -34cm
    'y_max': SURFACE_Y_MAX - TARGET_RADIUS,  # -6cm
    'z_min': SURFACE_Z_MIN + TARGET_RADIUS,  # 9cm
    'z_max': SURFACE_Z_MAX - TARGET_RADIUS   # 39cm
}


# ============================================================================
# RL ENVIRONMENT CLASS
# ============================================================================

class RLEnvironment(Node):
    """
    ROS2 RL Environment for 6-DOF Robot Arm
    
    Provides Gym-compatible interface for reinforcement learning training.
    """
    
    def __init__(self, max_episode_steps=200, goal_tolerance=0.01):
        """
        Initialize RL Environment
        
        Args:
            max_episode_steps: Maximum steps per episode (default: 200)
            goal_tolerance: Distance threshold for goal achievement (default: 1cm = sphere radius)
        """
        super().__init__('rl_environment')
        
        self.get_logger().info("🤖 Initializing RL Environment for 6-DOF Robot...")
        
        # Configuration
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.current_step = 0
        
        # Robot state variables (6-DOF)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.joint_positions = [0.0] * 6
        self.joint_velocities = [0.0] * 6
        
        # Target sphere state (initial position at center of workspace)
        self.target_x = 0.0
        self.target_y = (WORKSPACE_BOUNDS['y_min'] + WORKSPACE_BOUNDS['y_max']) / 2  # Center of Y workspace
        self.target_z = 0.30  # Center of Z workspace
        
        # State readiness flag
        self.data_ready = False
        
        # Joint limits: All joints ±90° with home at 0°
        self.joint_limits_low = np.array([
            -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2
        ])
        self.joint_limits_high = np.array([
            np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2
        ])
        
        # IK success tracking (legacy, not used with direct joint control)
        self.last_ik_success = 1.0
        
        # RL Spaces (Gym-compatible)
        # ACTION SPACE: 6D ABSOLUTE joint angles (radians) - Direct joint control!
        # Agent outputs target joint positions, constrained by joint limits ±90°
        # This allows the robot to move freely to any configuration in 1 step
        self.action_space = spaces.Box(
            low=self.joint_limits_low,   # -π/2 for all joints
            high=self.joint_limits_high,  # +π/2 for all joints
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: 16D state for 6-DOF direct joint control
        # [joints(6), robot_xyz(3), target_xyz(3), dist_xyz(3), dist_3d(1)]
        # NOTE: key_velocities removed - not useful for learning
        self.observation_space = spaces.Box(
            low=np.array([
                -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2,  # joint limits min
                -0.30, -0.50, 0.0,                                  # robot_xyz min
                -0.30, -0.50, 0.0,                                  # target_xyz min
                -0.60, -0.60, -0.60,                                # dist_xyz min
                0.0                                                  # dist_3d min
            ]),
            high=np.array([
                np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2,  # joint limits max
                0.30, 0.0, 0.50,                                    # robot_xyz max
                0.30, 0.0, 0.50,                                    # target_xyz max
                0.60, 0.60, 0.60,                                    # dist_xyz max
                1.0                                                  # dist_3d max
            ]),
            dtype=np.float32
        )
        
        self.get_logger().info(f"📊 Action space: 6D absolute joint angles (±90°)")
        self.get_logger().info(f"📊 Observation space: 16D state")
        
        # Target sphere state (static sphere in world file)
        self.target_spawned = True
        
        # Initialize ROS2 interfaces
        self._setup_tf_listener()
        self._setup_action_clients()
        self._setup_service_clients()
        self._setup_subscribers()
        
        self.get_logger().info("✅ RL Environment initialized!")
    
    def _setup_tf_listener(self):
        """Initialize TF2 listener for end-effector position tracking"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_logger().info("✅ TF2 listener initialized")
    
    def _setup_action_clients(self):
        """Initialize action client for robot trajectory control"""
        self.get_logger().info("⏳ Connecting to trajectory action server...")
        
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("❌ Trajectory action server not available!")
            raise Exception("Trajectory action server timeout")
        
        self.get_logger().info("✅ Trajectory action server connected!")
    
    def _setup_service_clients(self):
        """Initialize publishers for target teleportation"""
        self.get_logger().info("⏳ Setting up publishers...")
        
        # Publisher for target position (target_manager subscribes and teleports sphere)
        self.target_position_pub = self.create_publisher(
            Point,
            '/target_position',
            10
        )
        
        self.get_logger().info("✅ Publishers created")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers for robot and environment state"""
        self.get_logger().info("⏳ Setting up state subscribers...")
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        # Subscribe to model states (target sphere)
        self.model_state_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._model_state_callback,
            10
        )
        
        self.get_logger().info("✅ State subscribers initialized!")
    
    def _joint_state_callback(self, msg: JointState):
        """Update joint positions and velocities for 6-DOF robot"""
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
        positions = [0.0] * 6
        velocities = [0.0] * 6
        found_all = True
        
        for idx, joint_name in enumerate(joint_names):
            if joint_name in msg.name:
                jidx = msg.name.index(joint_name)
                try:
                    positions[idx] = msg.position[jidx]
                    velocities[idx] = msg.velocity[jidx] if len(msg.velocity) > jidx else 0.0
                except Exception as e:
                    self.get_logger().warn(f"Error reading joint {joint_name}: {e}", throttle_duration_sec=5.0)
                    found_all = False
            else:
                found_all = False
        
        self.joint_positions = positions
        self.joint_velocities = velocities
        
        if found_all:
            self.data_ready = True
        
        # Update end-effector position
        self._update_end_effector_position()
    
    def _model_state_callback(self, msg: ModelStates):
        """Update target sphere position"""
        try:
            if 'my_sphere' in msg.name:
                sphere_index = msg.name.index('my_sphere')
                sphere_pose = msg.pose[sphere_index]
                
                self.target_x = sphere_pose.position.x
                self.target_y = sphere_pose.position.y
                self.target_z = sphere_pose.position.z
                
                if len(self.joint_positions) == 6:
                    self.data_ready = True
        except Exception as e:
            self.get_logger().warn(f"Error processing model states: {e}", throttle_duration_sec=5.0)
    
    def _update_end_effector_position(self):
        """
        Update end-effector position using TF2
        
        Reads transform from base_link to End-effector_1 (pen tip)
        Uses short timeout since transform should be immediately available
        """
        try:
            # Look up transform with short timeout (transform is always available)
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'End-effector_1',
                rclpy.time.Time(),  # Get latest
                timeout=rclpy.duration.Duration(seconds=0, nanoseconds=100000000)  # 0.1s timeout
            )
            
            # Extract position
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
            self.robot_z = transform.transform.translation.z
            
        except Exception as e:
            # TF not available - log occasionally
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=5.0
            )
    
    # NOTE: Target sphere spawning is now handled by target_manager.py node
    # This node uses Ignition Transport to spawn and teleport the visual sphere
    
    def get_state(self) -> Optional[np.ndarray]:
        """
        Get current environment state for RL agent
        
        State vector for 6-DOF robot (23 elements):
        - Joint positions (6): [joint1, ..., joint6]
        - End-effector position (3): [robot_x, robot_y, robot_z]
        - Target position (3): [target_x, target_y, target_z]
        - Distance to target (3): [dist_x, dist_y, dist_z]
        - Euclidean distance (1): [dist_3d]
        - IK success flag (1): [ik_success]
        - Joint velocities (6): [vel1, ..., vel6]
        
        Returns:
            numpy array of state (23D) or None if not ready
        """
        if not self.data_ready:
            return None
        
        try:
            # Calculate distances
            dist_x = self.target_x - self.robot_x
            dist_y = self.target_y - self.robot_y
            dist_z = self.target_z - self.robot_z
            dist_3d = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
            
            state = np.array([
                # Joint positions (6)
                *self.joint_positions,
                # End-effector position (3)
                self.robot_x, self.robot_y, self.robot_z,
                # Target position (3)
                self.target_x, self.target_y, self.target_z,
                # Distance vector (3)
                dist_x, dist_y, dist_z,
                # Euclidean distance (1)
                dist_3d
            ], dtype=np.float32)  # Total: 16D
            
            return state
            
        except Exception as e:
            self.get_logger().error(f"Error creating state vector: {e}")
            return None
    
    def reset_environment(self) -> Optional[np.ndarray]:
        """
        Reset environment for new episode
        
        1. Move robot to home position [0,0,0,0,0,0]
        2. Randomize target sphere position
        3. Wait for robot to settle
        4. Return initial state
        
        Returns:
            Initial state observation (18D)
        """
        self.get_logger().info("🔄 Resetting environment...")
        self.current_step = 0
        
        # 1. Move robot to home position
        home_joints = np.zeros(6)
        self.get_logger().info("   Moving to home position...")
        success = self._move_to_joint_positions(home_joints, duration=2.0)
        
        if not success:
            self.get_logger().warn("⚠️ Failed to reach home position")
        
        # Wait for robot to settle
        time.sleep(0.5)
        
        # 2. Randomize target sphere position
        self._randomize_target()
        
        # 3. Wait for state to update
        time.sleep(0.2)
        
        self.get_logger().info(f"✅ Environment reset! Target: ({self.target_y:.3f}, {self.target_z:.3f})")
        
        return self.get_state()
    
    def _randomize_target(self):
        """Randomize target sphere position within 3D workspace"""
        # Random X, Y, Z within workspace bounds
        self.target_x = random.uniform(WORKSPACE_BOUNDS['x_min'], WORKSPACE_BOUNDS['x_max'])
        self.target_y = random.uniform(WORKSPACE_BOUNDS['y_min'], WORKSPACE_BOUNDS['y_max'])
        self.target_z = random.uniform(WORKSPACE_BOUNDS['z_min'], WORKSPACE_BOUNDS['z_max'])
        
        self.get_logger().info(f"   Target: X={self.target_x:.3f}, Y={self.target_y:.3f}, Z={self.target_z:.3f}")
        
        # Publish target position to target_manager node (teleports visual sphere)
        try:
            target_msg = Point()
            target_msg.x = self.target_x
            target_msg.y = self.target_y
            target_msg.z = self.target_z
            self.target_position_pub.publish(target_msg)
        except Exception as e:
            self.get_logger().debug(f"   Could not publish target position: {e}")
    
    def step(self, action: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """
        Execute one environment step using DIRECT JOINT CONTROL
        
        Args:
            action: 6D ABSOLUTE joint angles (radians) - target joint positions
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Get state before action
        state_before = self.get_state()
        if state_before is None:
            self.get_logger().error("State not available before action!")
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # Calculate distance before (dist_3d is at index 15 in 18D state)
        dist_before = state_before[15]
        
        # ABSOLUTE JOINT CONTROL: Action IS the target joint positions (not delta!)
        target_joints = np.array(action)
        
        # Clip to joint limits (±90°)
        target_joints = np.clip(target_joints, self.joint_limits_low, self.joint_limits_high)
        
        # Execute movement - robot moves directly to target in single trajectory
        success = self._move_to_joint_positions(target_joints, duration=1.0)
        
        # Wait for movement to complete and state to update
        time.sleep(0.3)
        
        # Get state after action
        next_state = self.get_state()
        
        if next_state is None:
            self.get_logger().error("State not available after action!")
            return None, -10.0, True, {'error': 'state_unavailable'}
        
        # Calculate reward
        dist_after = next_state[15]  # dist_3d
        reward, done = self._calculate_reward(dist_after, dist_before)
        
        # Check for ground collision (Z < 0.01m) - AUTO RECOVERY
        if self.robot_z < 0.01:
            reward = -50.0
            done = True
            self.get_logger().warn(f"⚠️ Ground collision! Z={self.robot_z:.3f}m - Resetting to home...")
            # AUTO-RESET: Move robot to home position to prevent getting stuck
            home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self._move_to_joint_positions(home_position, duration=1.0)
            time.sleep(0.5)  # Wait for recovery
        
        # Check episode termination
        if self.current_step >= self.max_episode_steps:
            done = True
            self.get_logger().info(f"Episode ended: max steps reached ({self.max_episode_steps})")
        
        # Info dict
        info = {
            'distance': dist_after,
            'success': success,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, dist_after: float, dist_before: float) -> Tuple[float, bool]:
        """
        Calculate reward based on distance to goal
        
        NEW Reward Structure:
        - Goal reached (dist < 1cm): +10.0, episode done
        - Distance penalty: -5.0 * distance (encourages staying close)
        - Improvement bonus: +10.0 * improvement (closer = reward)
        - Moving away penalty: 2× (10.0 * improvement * 2.0)
        - Step penalty: -0.5
        - Clipped to [-10, +10] to prevent explosion
        
        Args:
            dist_after: Distance to goal after action
            dist_before: Distance to goal before action
        
        Returns:
            Tuple of (reward, done)
        """
        done = False
        
        # Goal reached (1cm = radius of target sphere)
        if dist_after < self.goal_tolerance:  # 0.01m = 1cm
            reward = 10.0
            done = True
            self.get_logger().info(f"🎯 Goal reached! Distance: {dist_after*1000:.1f}mm")
        else:
            # Distance penalty (negative, proportional to distance)
            dist_reward = -5.0 * dist_after
            
            # Improvement reward (asymmetric: 2× penalty for moving away)
            improvement = dist_before - dist_after
            if improvement >= 0:
                # Getting closer - positive reward
                improve_reward = 10.0 * improvement
            else:
                # Moving away - 2× penalty (harsher punishment)
                improve_reward = 10.0 * improvement * 2.0
            
            # Step penalty
            step_penalty = 0.5
            
            # Combine all components
            reward = dist_reward + improve_reward - step_penalty
            
            # Clip to prevent reward explosion
            reward = np.clip(reward, -10.0, 10.0)
        
        return reward, done
    
    def _execute_target_action(self, target_x: float, target_z: float) -> Tuple[bool, bool, float]:
        """
        Execute target-based action with IK
        
        Args:
            target_x: Target X position in meters
            target_z: Target Z position in meters
        
        Returns:
            Tuple of (execution_success, ik_success, ik_error)
        """
        # Compute IK for target position (uses all 6 joints)
        from .fk_ik_utils import constrained_ik_6dof
        
        try:
            joint_angles, ik_success, ik_error = constrained_ik_6dof(
                target_y=self.target_y,  # Use current target Y
                target_z=target_z,
                target_x=target_x,
                initial_guess=self.joint_positions,  # Warm start from current position
                tolerance=0.005  # 5mm tolerance
            )
            
            # Update IK success flag for state
            self.last_ik_success = 1.0 if ik_success else 0.0
            
            # CRITICAL: Move robot even if IK "failed" (error > tolerance)
            # The IK solution is still the best we can do, so execute it
            if ik_error < 0.2:  # Only reject if error is huge (>20cm)
                execution_success = self._move_to_joint_positions(joint_angles, duration=1.0)
                if not ik_success:
                    self.get_logger().warn(f"IK error {ik_error*1000:.1f}mm > tolerance, but moving anyway")
                return execution_success, ik_success, ik_error
            else:
                self.get_logger().error(f"IK error too large: {ik_error*1000:.1f}mm - not moving")
                return False, False, ik_error
                
        except Exception as e:
            self.get_logger().error(f"IK solver error: {e}")
            return False, False, float('inf')
    
    def _move_to_joint_positions(self, target_positions: np.ndarray, duration: float = 0.5) -> bool:
        """
        Move robot to specified joint positions
        
        Args:
            joint_angles: Target joint angles [6] in radians
            duration: Trajectory duration in seconds
        
        Returns:
            True if movement successful
        """
        if len(target_positions) != 6:
            self.get_logger().error(f"Expected 6 joint angles, got {len(target_positions)}")
            return False
        
        # Clip to joint limits
        target_positions = np.clip(target_positions, self.joint_limits_low, self.joint_limits_high)
        
        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.velocities = [0.0] * 6
        # Set duration to 0.5 seconds for fast training
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5 seconds
        
        goal_msg.trajectory.points = [point]
        
        # Send goal and wait
        try:
            self.get_logger().info(f"Sending trajectory: {np.degrees(target_positions).astype(int)}°")
            
            send_goal_future = self.trajectory_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=2.0)
            
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Goal rejected by action server")
                return False
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 2.0)
            
            result = result_future.result()
            if result:
                # Wait for robot to settle
                time.sleep(0.2)
                return True
            else:
                return False
                
        except Exception as e:
            self.get_logger().error(f"Trajectory execution error: {e}")
            return False


def main(args=None):
    """Test the RL environment"""
    rclpy.init(args=args)
    
    try:
        env = RLEnvironment()
        
        # Spin to process callbacks
        rclpy.spin(env)
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
