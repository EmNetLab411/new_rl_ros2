#!/usr/bin/env python3
"""
Deploy Trained RL Model on Raspberry Pi - ROS2 Humble

Run trained ONNX/TFLite model on Raspberry Pi to control the real 6DOF robot arm.
Uses direct joint control (no IK) with the pi_servo_interface node.

Hardware Requirements:
    - Raspberry Pi (3B+/4/5)
    - PCA9685 servo driver + 6 servos
    - Robot arm assembled

Software Requirements:
    - ROS2 Humble
    - onnxruntime: pip3 install onnxruntime
    - pi_servo_interface running

Usage:
    # Start servo interface first:
    ros2 run robot_arm2 pi_servo_interface
    
    # Then run deployment (ONNX format):
    python3 deploy_on_pi.py --model actor_sac_best.onnx
    
    # With custom target position:
    python3 deploy_on_pi.py --model actor_sac_best.onnx --target 0.15 -0.05 0.30

Model Specs:
    - State: 18D (joints(6), robot_xyz(3), target_xyz(3), dist_xyz(3), dist_3d(1), vel(2))
    - Action: 6D (joint angle deltas, ±0.1 rad)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
import numpy as np
import argparse
import time
import csv
from datetime import datetime
import os
import sys

# Add parent directory for FK imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'rl'))

# Import TFLite Runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("❌ TFLite Runtime not found - install with: pip3 install tflite-runtime")

# Try to import FK utilities
try:
    from fk_ik_utils import forward_kinematics
    FK_AVAILABLE = True
except ImportError:
    FK_AVAILABLE = False
    print("⚠️  FK utilities not found - using simplified FK")


# ============================================================================
# CONFIGURATION
# ============================================================================

# State/Action dimensions (must match training)
STATE_DIM = 18
ACTION_DIM = 6

# Action scaling (joint angle delta per step)
MAX_JOINT_DELTA = 0.1  # radians (~5.7°)

# Control rate
CONTROL_RATE_HZ = 10.0

# Joint limits (radians) - must match training
JOINT_LIMITS = np.array([
    [-3.14159, 3.14159],  # Joint 1: ±180°
    [-1.5708, 1.5708],    # Joint 2: ±90°
    [-1.5708, 1.5708],    # Joint 3: ±90°
    [-1.5708, 1.5708],    # Joint 4: ±90°
    [-3.14159, 3.14159],  # Joint 5: ±180°
    [-3.14159, 3.14159],  # Joint 6: ±180°
])

# Workspace bounds (meters) - 3D workspace
WORKSPACE_X = (0.03, 0.27)   # 3-27cm
WORKSPACE_Y = (-0.12, 0.12)  # ±12cm
WORKSPACE_Z = (0.18, 0.42)   # 18-42cm

# Default target position
DEFAULT_TARGET = np.array([0.15, 0.0, 0.30])  # Center of workspace

# Goal threshold for success
GOAL_THRESHOLD = 0.01  # 1cm


# ============================================================================
# SIMPLE FK (if fk_ik_utils not available)
# ============================================================================

def simple_forward_kinematics(joint_angles):
    """
    Simplified FK for 6DOF arm (approximate)
    Returns end-effector position [x, y, z]
    
    Note: This is a placeholder - use actual FK from fk_ik_utils if available
    """
    # Link lengths (approximate, adjust to your robot)
    L1 = 0.08   # Base to shoulder
    L2 = 0.10   # Shoulder to elbow
    L3 = 0.08   # Elbow to wrist
    L4 = 0.05   # Wrist to end-effector
    
    j1, j2, j3, j4, j5, j6 = joint_angles
    
    # Simple planar approximation
    r = L2 * np.cos(j2) + L3 * np.cos(j2 + j3) + L4 * np.cos(j2 + j3 + j4)
    z = L1 + L2 * np.sin(j2) + L3 * np.sin(j2 + j3) + L4 * np.sin(j2 + j3 + j4)
    
    x = r * np.cos(j1)
    y = r * np.sin(j1)
    
    return np.array([x, y, z])


# Use available FK
if FK_AVAILABLE:
    def get_ee_position(joint_angles):
        return np.array(forward_kinematics(joint_angles))
else:
    get_ee_position = simple_forward_kinematics


# ============================================================================
# RL DEPLOYMENT NODE
# ============================================================================

class RLDeploymentNode(Node):
    """ROS2 node for deploying trained RL model on real robot"""
    
    def __init__(self, model_path: str, log_performance: bool = True):
        super().__init__('rl_deployment_node')
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("🤖 RL Model Deployment - 6DOF Robot Arm")
        self.get_logger().info("=" * 70)
        
        self.model_path = model_path
        self.log_performance = log_performance
        
        # Load TFLite model
        if not TFLITE_AVAILABLE:
            raise RuntimeError("TFLite runtime not available! Install with: pip3 install tflite-runtime")
        
        if not model_path.endswith('.tflite'):
            model_path = model_path + '.tflite'
        
        self._load_tflite_model(model_path)
        
        # Current state
        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.joint_velocities = np.zeros(6, dtype=np.float32)
        self.target_position = DEFAULT_TARGET.copy()
        self.enabled = False
        
        # Setup ROS2 interfaces
        self._setup_ros2_interfaces()
    
    def _load_tflite_model(self, model_path: str):
        """Load TFLite model"""
        if not TFLITE_AVAILABLE:
            raise RuntimeError("TFLite runtime not available!")
        
        self.get_logger().info(f"\n📦 Loading TFLite model: {model_path}")
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.get_logger().info(f"✅ TFLite model loaded")
        self.get_logger().info(f"   Input: {self.input_details['shape']}")
        self.get_logger().info(f"   Output: {self.output_details['shape']}")
        
        # Verify dimensions
        if self.input_details['shape'][1] != STATE_DIM:
            self.get_logger().warn(f"Model expects state dim {self.input_details['shape'][1]}, expected {STATE_DIM}")
        if self.output_details['shape'][1] != ACTION_DIM:
            self.get_logger().warn(f"Model expects action dim {self.output_details['shape'][1]}, expected {ACTION_DIM}")
        
        self.model_type = 'tflite'
    
    def _setup_ros2_interfaces(self):
        """Setup ROS2 subscribers, publishers, and service clients"""
        # Subscriber for joint states from servo interface
        self.joint_sub = self.create_subscription(
            JointState,
            '/servo/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher for servo commands
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/servo/command',
            10
        )
        
        # Service clients
        self.enable_client = self.create_client(Trigger, '/servo/enable')
        self.home_client = self.create_client(Trigger, '/servo/home')
        
        # Wait for servo interface
        self.get_logger().info("\n⏳ Waiting for servo interface...")
        if not self.enable_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("⚠️  Servo interface not found - running without hardware")
        else:
            self.get_logger().info("✅ Servo interface connected")
        
        # Performance logging
        self.performance_log = []
        self.log_header = [
            'timestamp_ms', 'step', 'episode',
            'distance_cm', 'inference_ms',
            'ee_x', 'ee_y', 'ee_z',
            'target_x', 'target_y', 'target_z',
            'j1', 'j2', 'j3', 'j4', 'j5', 'j6',
            'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
            'reward'
        ]
        
        self.get_logger().info("✅ Deployment node ready!")
    
    def joint_state_callback(self, msg: JointState):
        """Update joint positions from servo interface"""
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6], dtype=np.float32)
        if len(msg.velocity) >= 6:
            self.joint_velocities = np.array(msg.velocity[:6], dtype=np.float32)
    
    def get_state(self) -> np.ndarray:
        """
        Construct 18D state vector matching training format
        
        State: [joints(6), robot_xyz(3), target_xyz(3), dist_xyz(3), dist_3d(1), vel(2)]
        """
        # Get current EE position
        ee_pos = get_ee_position(self.joint_positions)
        
        # Distances
        dist_xyz = ee_pos - self.target_position
        dist_3d = np.linalg.norm(dist_xyz)
        
        # Key velocities (joints 1 and 2 for now)
        key_velocities = self.joint_velocities[:2]
        
        # Construct state
        state = np.concatenate([
            self.joint_positions,      # 6: joint angles
            ee_pos,                    # 3: end-effector XYZ
            self.target_position,      # 3: target XYZ
            dist_xyz,                  # 3: distance XYZ
            [dist_3d],                 # 1: distance 3D
            key_velocities             # 2: key velocities
        ]).astype(np.float32)
        
        return state
    
    def run_inference(self, state: np.ndarray) -> tuple:
        """
        Run TFLite model inference
        
        Returns:
            (action, latency_ms)
        """
        start = time.perf_counter()
        
        input_data = np.expand_dims(state, axis=0).astype(np.float32)
        
        # TFLite inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        action = self.interpreter.get_tensor(self.output_details['index'])[0]
        
        latency_ms = (time.perf_counter() - start) * 1000
        return action, latency_ms
    
    def apply_action(self, action: np.ndarray):
        """Apply joint delta action to robot"""
        # Scale action (model outputs in [-1, 1])
        joint_deltas = action * MAX_JOINT_DELTA
        
        # Compute new positions
        new_positions = self.joint_positions + joint_deltas
        
        # Apply joint limits
        for i in range(6):
            new_positions[i] = np.clip(new_positions[i], JOINT_LIMITS[i, 0], JOINT_LIMITS[i, 1])
        
        # Publish command
        msg = Float64MultiArray()
        msg.data = new_positions.tolist()
        self.cmd_pub.publish(msg)
        
        return new_positions
    
    def enable_servos(self):
        """Enable servos via service call"""
        if self.enable_client.service_is_ready():
            future = self.enable_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is not None:
                self.get_logger().info(f"✅ {future.result().message}")
                self.enabled = True
            return True
        return False
    
    def go_home(self):
        """Move to home position via service call"""
        if self.home_client.service_is_ready():
            future = self.home_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is not None:
                self.get_logger().info(f"🏠 {future.result().message}")
            return True
        return False
    
    def run(self, target_position: np.ndarray = None, 
            num_episodes: int = 1, 
            max_steps: int = 200):
        """Main deployment loop"""
        
        if target_position is not None:
            self.target_position = target_position
        
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("🚀 Starting RL Deployment")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Target: {self.target_position}")
        self.get_logger().info(f"Episodes: {num_episodes}")
        self.get_logger().info(f"Max steps: {max_steps}")
        self.get_logger().info(f"Control rate: {CONTROL_RATE_HZ} Hz")
        self.get_logger().info("Press Ctrl+C to stop")
        self.get_logger().info("=" * 70 + "\n")
        
        # Enable servos and go home
        self.enable_servos()
        time.sleep(0.5)
        self.go_home()
        time.sleep(2.0)  # Wait for home position
        
        rate = self.create_rate(CONTROL_RATE_HZ)
        total_steps = 0
        
        try:
            for episode in range(num_episodes):
                self.get_logger().info(f"\n📍 Episode {episode + 1}/{num_episodes}")
                
                episode_reward = 0.0
                episode_start = time.time()
                
                # Reset to home at start of episode
                self.go_home()
                time.sleep(1.0)
                
                for step in range(max_steps):
                    # Check for shutdown
                    if not rclpy.ok():
                        break
                    
                    step_start = time.perf_counter()
                    total_steps += 1
                    
                    # Get state
                    state = self.get_state()
                    
                    # Run inference
                    action, inference_ms = self.run_inference(state)
                    
                    # Apply action
                    new_positions = self.apply_action(action)
                    
                    # Wait for servo movement
                    time.sleep(0.1)
                    
                    # Get new EE position and calculate reward
                    ee_pos = get_ee_position(self.joint_positions)
                    distance = np.linalg.norm(ee_pos - self.target_position)
                    reward = -distance
                    episode_reward += reward
                    
                    # Print status
                    self.get_logger().info(
                        f"Step {step+1:3d} | "
                        f"Dist: {distance*100:5.1f}cm | "
                        f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                        f"Inf: {inference_ms:.1f}ms | "
                        f"R: {reward:.3f}"
                    )
                    
                    # Log performance
                    if self.log_performance:
                        self.performance_log.append([
                            int(time.time() * 1000),
                            step + 1,
                            episode + 1,
                            distance * 100,
                            inference_ms,
                            ee_pos[0], ee_pos[1], ee_pos[2],
                            self.target_position[0], self.target_position[1], self.target_position[2],
                            *self.joint_positions,
                            *action,
                            reward
                        ])
                    
                    # Check success
                    if distance < GOAL_THRESHOLD:
                        self.get_logger().info(f"🎯 TARGET REACHED! Distance: {distance*100:.1f}cm")
                        break
                    
                    rate.sleep()
                
                # Episode summary
                episode_time = time.time() - episode_start
                avg_reward = episode_reward / (step + 1)
                self.get_logger().info(
                    f"\n📊 Episode {episode+1} Summary:\n"
                    f"   Steps: {step + 1}\n"
                    f"   Time: {episode_time:.1f}s\n"
                    f"   Total reward: {episode_reward:.2f}\n"
                    f"   Avg reward: {avg_reward:.3f}\n"
                    f"   Final distance: {distance*100:.1f}cm"
                )
        
        except KeyboardInterrupt:
            self.get_logger().info("\n⚠️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Save logs and cleanup"""
        self.get_logger().info("\n🛑 Shutting down...")
        
        # Save performance log
        if self.log_performance and self.performance_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"deployment_log_{timestamp}.csv"
            
            try:
                with open(log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.log_header)
                    writer.writerows(self.performance_log)
                
                self.get_logger().info(f"💾 Log saved: {log_file}")
                
                # Print summary stats
                distances = [row[3] for row in self.performance_log]
                self.get_logger().info(f"   Samples: {len(self.performance_log)}")
                self.get_logger().info(f"   Min distance: {min(distances):.1f}cm")
                self.get_logger().info(f"   Avg distance: {np.mean(distances):.1f}cm")
                
            except Exception as e:
                self.get_logger().error(f"Failed to save log: {e}")
        
        self.get_logger().info("👋 Deployment complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Deploy trained RL model on Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 deploy_on_pi.py --model actor_sac_best.tflite
  python3 deploy_on_pi.py --model actor_sac_best.tflite --target 0.15 0.05 0.30
  python3 deploy_on_pi.py --model actor_sac_best.tflite --episodes 5 --steps 100
        """
    )
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to TFLite model')
    parser.add_argument('--target', type=float, nargs=3, default=None,
                        help='Target position (x y z) in meters')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes (default: 1)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Max steps per episode (default: 200)')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable performance logging')
    
    args = parser.parse_args()
    
    # Parse target
    target = None
    if args.target:
        target = np.array(args.target, dtype=np.float32)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        node = RLDeploymentNode(
            model_path=args.model,
            log_performance=not args.no_log
        )
        
        node.run(
            target_position=target,
            num_episodes=args.episodes,
            max_steps=args.steps
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main())
