#!/usr/bin/env python3
"""
Deploy Drawing Model on Raspberry Pi

Traces a 10-waypoint equilateral triangle using trained SAC + Neural IK.
Designed for drawing task deployment with waypoint-by-waypoint progress.

Usage:
    python3 deploy_drawing_on_pi.py
    python3 deploy_drawing_on_pi.py --model actor_drawing.tflite --ik neural_ik.tflite
"""

import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime

# =============================================================================
# DRAWING CONFIGURATION (matches training)
# =============================================================================

# State dimensions for DRAWING task
STATE_DIM = 18  # joints(6) + EE(3) + target(3) + dist(3) + dist3d(1) + progress(1) + remaining(1)
ACTION_DIM = 3  # SAC outputs 3D delta direction
JOINT_DIM = 6   # Output joint angles

# Waypoint configuration
POINTS_PER_EDGE = 3   # 3 points per edge
TOTAL_WAYPOINTS = 10  # 3 edges × 3 + 1 return

# Triangle parameters (matches training)
SHAPE_SIZE = 0.15     # 15cm triangle
Y_PLANE = 0.20        # 20cm from ground
WAYPOINT_TOLERANCE = 0.005  # 0.5cm tolerance

# Control parameters
CONTROL_RATE_HZ = 5.0
STEP_SIZE = 0.15      # Max 15cm per step

# Joint limits (radians) - ±90°
JOINT_LIMITS = np.array([
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708]
])


def generate_triangle_waypoints():
    """Generate 10-waypoint equilateral triangle (matches training)"""
    height = SHAPE_SIZE * np.sqrt(3) / 2
    cx, cz = 0.0, 0.25  # Center
    
    # Vertices - START FROM TOP (apex)
    p1 = np.array([cx, Y_PLANE, cz + 2*height/3])          # Top (apex/START)
    p2 = np.array([cx - SHAPE_SIZE/2, Y_PLANE, cz - height/3])  # Bottom-left
    p3 = np.array([cx + SHAPE_SIZE/2, Y_PLANE, cz - height/3])  # Bottom-right
    
    corners = [p1, p2, p3, p1]  # TOP→BL→BR→TOP
    
    waypoints = []
    for edge in range(3):
        start = corners[edge]
        end = corners[edge + 1]
        for t in np.linspace(0, 1, POINTS_PER_EDGE, endpoint=False):
            point = start + t * (end - start)
            waypoints.append(point)
    
    # Return to start
    waypoints.append(p1)
    
    return np.array(waypoints)


# =============================================================================
# SIMPLE FORWARD KINEMATICS
# =============================================================================

def forward_kinematics(joints):
    """Simple FK for 6DOF arm - returns EE position [x, y, z]"""
    # Link lengths (approximate)
    L1 = 0.05   # Base to shoulder
    L2 = 0.10   # Shoulder to elbow
    L3 = 0.10   # Elbow to wrist1
    L4 = 0.08   # Wrist1 to wrist2
    L5 = 0.05   # Wrist2 to EE
    
    j1, j2, j3, j4, j5, j6 = joints
    
    # Simplified calculation
    r = L2 * np.cos(j2) + L3 * np.cos(j2 + j3) + L4 * np.cos(j2 + j3 + j4)
    
    x = r * np.sin(j1)
    y = L1 + L2 * np.sin(j2) + L3 * np.sin(j2 + j3) + L4 * np.sin(j2 + j3 + j4) + L5
    z = r * np.cos(j1) + 0.25
    
    return np.array([x, y, z])


# =============================================================================
# DRAWING DEPLOYMENT
# =============================================================================

class DrawingDeployment:
    """Deploy trained drawing model on real robot"""
    
    def __init__(self, actor_path, neural_ik_path=None, use_ros=True):
        self.actor_path = actor_path
        self.neural_ik_path = neural_ik_path
        self.use_ros = use_ros
        
        # Model interpreters
        self.actor_interpreter = None
        self.nik_interpreter = None
        
        # State
        self.joint_positions = np.zeros(6)
        self.current_waypoint_idx = 0
        self.waypoints = generate_triangle_waypoints()
        self.waypoints_reached = 0
        
        # Logging
        self.step_log = []
        
        print("="*70)
        print("🎨 DRAWING DEPLOYMENT")
        print("="*70)
        print(f"   Triangle: {TOTAL_WAYPOINTS} waypoints ({POINTS_PER_EDGE} per edge)")
        print(f"   Size: {SHAPE_SIZE*100:.0f}cm | Tolerance: {WAYPOINT_TOLERANCE*100:.1f}cm")
        print("="*70)
    
    def load_models(self):
        """Load TFLite models"""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        
        print("\n📦 Loading models...")
        
        # Load Actor
        if os.path.exists(self.actor_path):
            self.actor_interpreter = tflite.Interpreter(model_path=self.actor_path)
            self.actor_interpreter.allocate_tensors()
            print(f"✅ Actor: {self.actor_path}")
        else:
            print(f"❌ Actor not found: {self.actor_path}")
            return False
        
        # Load Neural IK
        if self.neural_ik_path and os.path.exists(self.neural_ik_path):
            self.nik_interpreter = tflite.Interpreter(model_path=self.neural_ik_path)
            self.nik_interpreter.allocate_tensors()
            print(f"✅ Neural IK: {self.neural_ik_path}")
        else:
            print("⚠️  Neural IK not loaded - will need external IK")
        
        return True
    
    def setup_ros(self):
        """Setup ROS2 interfaces (if using ROS)"""
        if not self.use_ros:
            print("⚠️  Running in simulation mode (no ROS)")
            return True
        
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float32MultiArray
            
            rclpy.init()
            self.node = rclpy.create_node('drawing_deployment')
            
            # Joint state subscriber
            self.joint_sub = self.node.create_subscription(
                JointState, '/joint_states', self._joint_callback, 10
            )
            
            # Joint command publisher
            self.joint_pub = self.node.create_publisher(
                Float32MultiArray, '/joint_commands', 10
            )
            
            print("✅ ROS2 interfaces ready")
            return True
            
        except Exception as e:
            print(f"❌ ROS2 setup failed: {e}")
            self.use_ros = False
            return False
    
    def _joint_callback(self, msg):
        """Update joint positions from servo feedback"""
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6])
    
    def get_state(self):
        """Construct 18D state vector (matches training)"""
        # Current EE position
        ee_pos = forward_kinematics(self.joint_positions)
        
        # Current target waypoint
        target = self.waypoints[self.current_waypoint_idx]
        
        # Distance components
        dist_xyz = target - ee_pos
        dist_3d = np.linalg.norm(dist_xyz)
        
        # Progress
        progress = self.waypoints_reached / TOTAL_WAYPOINTS
        remaining = TOTAL_WAYPOINTS - self.waypoints_reached
        
        # Build state: [joints(6) + EE(3) + target(3) + dist(3) + dist3d(1) + progress(1) + remaining(1)]
        state = np.concatenate([
            self.joint_positions,    # 6
            ee_pos,                  # 3
            target,                  # 3
            dist_xyz,                # 3
            [dist_3d],               # 1
            [progress],              # 1
            [remaining]              # 1
        ])
        
        return state.astype(np.float32)
    
    def run_actor(self, state):
        """Run SAC actor inference"""
        input_details = self.actor_interpreter.get_input_details()
        output_details = self.actor_interpreter.get_output_details()
        
        self.actor_interpreter.set_tensor(input_details[0]['index'], state.reshape(1, -1))
        self.actor_interpreter.invoke()
        
        action = self.actor_interpreter.get_tensor(output_details[0]['index'])[0]
        return action
    
    def run_neural_ik(self, target_xyz):
        """Run Neural IK to get joint angles"""
        if self.nik_interpreter is None:
            return None
        
        input_details = self.nik_interpreter.get_input_details()
        output_details = self.nik_interpreter.get_output_details()
        
        self.nik_interpreter.set_tensor(input_details[0]['index'], 
                                         target_xyz.astype(np.float32).reshape(1, -1))
        self.nik_interpreter.invoke()
        
        joints = self.nik_interpreter.get_tensor(output_details[0]['index'])[0]
        return joints
    
    def compute_target_position(self, action, ee_pos, waypoint):
        """Compute target XYZ from action (matches training logic)"""
        direction = waypoint - ee_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.001:
            direction_norm = direction / distance
            move_amount = (action[0] + 1) / 2 * STEP_SIZE
            fine_adjust = action[1:3] * 0.02
            
            delta = direction_norm * move_amount
            target_xyz = ee_pos + delta
            target_xyz[0] += fine_adjust[0]
            target_xyz[2] += fine_adjust[1] if len(fine_adjust) > 1 else 0
        else:
            target_xyz = waypoint
        
        # Clamp to safe bounds
        target_xyz = np.clip(target_xyz, [-0.15, 0.10, 0.05], [0.15, 0.30, 0.45])
        
        return target_xyz
    
    def send_joints(self, joints):
        """Send joint commands to robot"""
        # Clip to limits
        joints = np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        
        if self.use_ros:
            from std_msgs.msg import Float32MultiArray
            msg = Float32MultiArray()
            msg.data = joints.tolist()
            self.joint_pub.publish(msg)
        else:
            # Simulation: just update internal state
            self.joint_positions = joints
        
        return joints
    
    def check_waypoint_reached(self, ee_pos):
        """Check if current waypoint is reached"""
        target = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(ee_pos - target)
        
        if distance <= WAYPOINT_TOLERANCE:
            return True, distance
        return False, distance
    
    def run(self, max_steps=100):
        """Main drawing loop"""
        print("\n" + "="*70)
        print("🎨 STARTING TRIANGLE TRACING")
        print("="*70)
        print(f"\nWaypoints: {TOTAL_WAYPOINTS}")
        for i, wp in enumerate(self.waypoints):
            print(f"  WP{i+1}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]")
        print()
        
        step = 0
        start_time = time.time()
        
        while step < max_steps and self.waypoints_reached < TOTAL_WAYPOINTS:
            step += 1
            
            # Get current state
            state = self.get_state()
            ee_pos = forward_kinematics(self.joint_positions)
            waypoint = self.waypoints[self.current_waypoint_idx]
            
            # Run SAC actor
            action = self.run_actor(state)
            
            # Compute target position
            target_xyz = self.compute_target_position(action, ee_pos, waypoint)
            
            # Run Neural IK
            joints = self.run_neural_ik(target_xyz)
            
            if joints is not None:
                # Send to robot
                self.send_joints(joints)
            
            # Wait for movement
            time.sleep(1.0 / CONTROL_RATE_HZ)
            
            # Spin ROS if available
            if self.use_ros:
                import rclpy
                rclpy.spin_once(self.node, timeout_sec=0.1)
            
            # Check waypoint
            ee_after = forward_kinematics(self.joint_positions)
            reached, distance = self.check_waypoint_reached(ee_after)
            
            # Log
            log_entry = {
                'step': step,
                'ee': ee_after.tolist(),
                'target': waypoint.tolist(),
                'distance_cm': distance * 100,
                'waypoint_idx': self.current_waypoint_idx,
                'waypoints_reached': self.waypoints_reached,
                'joints_deg': np.degrees(self.joint_positions).tolist()
            }
            self.step_log.append(log_entry)
            
            # Print progress
            wp_status = f"WP{self.current_waypoint_idx + 1}/{TOTAL_WAYPOINTS}"
            print(f"Step {step:3d} | EE: [{ee_after[0]:.3f}, {ee_after[1]:.3f}, {ee_after[2]:.3f}] | "
                  f"Dist: {distance*100:.2f}cm | {wp_status}")
            
            if reached:
                self.waypoints_reached += 1
                print(f"  ✅ WAYPOINT {self.current_waypoint_idx + 1} REACHED!")
                
                if self.waypoints_reached >= TOTAL_WAYPOINTS:
                    print("\n🎉🎉🎉 TRIANGLE COMPLETE! 🎉🎉🎉")
                    break
                
                self.current_waypoint_idx += 1
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("📊 DEPLOYMENT SUMMARY")
        print("="*70)
        print(f"   Steps: {step}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Waypoints: {self.waypoints_reached}/{TOTAL_WAYPOINTS}")
        print(f"   Success: {'✅ YES' if self.waypoints_reached >= TOTAL_WAYPOINTS else '❌ NO'}")
        
        # Save log
        self.save_log()
        
        return self.waypoints_reached >= TOTAL_WAYPOINTS
    
    def save_log(self):
        """Save step log to file"""
        import json
        
        log_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f'deployment_log_{timestamp}.jsonl')
        
        with open(log_path, 'w') as f:
            for entry in self.step_log:
                f.write(json.dumps(entry) + '\n')
        
        print(f"\n📝 Log saved: {log_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.use_ros:
            import rclpy
            self.node.destroy_node()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Deploy Drawing Model on Pi')
    parser.add_argument('--model', '-m', type=str, default='actor_drawing.tflite',
                        help='Path to SAC actor TFLite model')
    parser.add_argument('--ik', type=str, default='neural_ik.tflite',
                        help='Path to Neural IK TFLite model')
    parser.add_argument('--no-ros', action='store_true', help='Run without ROS')
    parser.add_argument('--steps', type=int, default=100, help='Max steps')
    
    args = parser.parse_args()
    
    # Find models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    actor_path = args.model
    if not os.path.exists(actor_path):
        actor_path = os.path.join(script_dir, args.model)
    if not os.path.exists(actor_path):
        actor_path = os.path.join(script_dir, '..', 'checkpoints', 'sac_drawing', args.model)
    
    ik_path = args.ik
    if not os.path.exists(ik_path):
        ik_path = os.path.join(script_dir, args.ik)
    if not os.path.exists(ik_path):
        ik_path = os.path.join(script_dir, '..', 'checkpoints', args.ik)
    
    # Create deployment
    deployer = DrawingDeployment(
        actor_path=actor_path,
        neural_ik_path=ik_path if os.path.exists(ik_path) else None,
        use_ros=not args.no_ros
    )
    
    # Load models
    if not deployer.load_models():
        print("❌ Failed to load models")
        return 1
    
    # Setup ROS
    if not args.no_ros:
        deployer.setup_ros()
    
    try:
        # Run deployment
        success = deployer.run(max_steps=args.steps)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        success = False
    finally:
        deployer.cleanup()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
