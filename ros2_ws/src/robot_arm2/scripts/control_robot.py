#!/usr/bin/env python3
"""
Manual Robot Control Script for 6-DOF Robot Arm

This script allows manual control of the robot by entering joint angles directly.
Shows before/after end-effector positions and distances like the training script.
Uses DIRECT trajectory control (single command) for instant movement.

Usage:
    python3 control_robot.py
    
Commands:
    - Enter 6 joint angles in DEGREES (space-separated)
    - 'home' or 'h' - Move to home position [0,0,0,0,0,0]
    - 'reset' or 'r' - Reset and randomize target
    - 'fk <angles>' - Show FK prediction only (no movement)
    - 'q' or 'quit' - Exit
"""

import rclpy
import numpy as np
import time
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.rl_environment import RLEnvironment, WORKSPACE_BOUNDS
from rl.fk_ik_utils import fk


def manual_control():
    """Interactive manual control mode"""
    
    print("=" * 70)
    print("🎮 MANUAL ROBOT CONTROL - 6DOF Robot Arm")
    print("=" * 70)
    print()
    print("📝 Commands:")
    print("  • Enter 6 joint angles in DEGREES (space-separated)")
    print("    Example: 0 30 -20 0 15 0")
    print("  • 'home' or 'h'  - Move to home position [0,0,0,0,0,0]")
    print("  • 'reset' or 'r' - Reset environment and randomize target")
    print("  • 'fk <angles>'  - Show FK prediction only (no movement)")
    print("  • 'ws'           - Show workspace bounds")
    print("  • 'q' or 'quit'  - Exit")
    print()
    print("📐 Joint limits: ±90° for all joints")
    print("=" * 70)
    
    # Show workspace
    print("\n📦 WORKSPACE BOUNDS:")
    print(f"   X: [{WORKSPACE_BOUNDS['x_min']*100:.0f}cm, {WORKSPACE_BOUNDS['x_max']*100:.0f}cm]")
    print(f"   Y: [{WORKSPACE_BOUNDS['y_min']*100:.0f}cm, {WORKSPACE_BOUNDS['y_max']*100:.0f}cm]")
    print(f"   Z: [{WORKSPACE_BOUNDS['z_min']*100:.0f}cm, {WORKSPACE_BOUNDS['z_max']*100:.0f}cm]")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create environment
        print("\n📦 Creating RL environment...")
        env = RLEnvironment(max_episode_steps=100, goal_tolerance=0.01)
        
        # Wait for initialization
        print("   Waiting for robot to initialize...")
        time.sleep(2.0)
        for _ in range(20):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        # Initial reset
        print("   Resetting to home position...")
        state = env.reset_environment()
        for _ in range(10):
            rclpy.spin_once(env, timeout_sec=0.1)
        
        print("✅ Ready for manual control!")
        print()
        
        while True:
            try:
                # Update state from callbacks
                for _ in range(5):
                    rclpy.spin_once(env, timeout_sec=0.05)
                
                # Get current state
                current_state = env.get_state()
                
                if current_state is not None:
                    # Extract positions from state
                    joints_current = current_state[:6]
                    ee_current = current_state[6:9]
                    target_pos = current_state[9:12]
                    distance = current_state[15]
                    
                    print("\n" + "-" * 50)
                    print("📍 CURRENT STATE:")
                    print(f"   Joints (deg): [{', '.join([f'{np.degrees(j):.1f}' for j in joints_current])}]")
                    print(f"   End-effector: X={ee_current[0]:.4f}, Y={ee_current[1]:.4f}, Z={ee_current[2]:.4f}")
                    print(f"   Target:       X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")
                    print(f"   Distance:     {distance*100:.2f}cm")
                    if distance < 0.01:
                        print("   🎯 GOAL REACHED!")
                    print("-" * 50)
                
                # Get user input
                user_input = input("\n🎮 Enter command: ").strip().lower()
                
                if not user_input:
                    continue
                
                # Parse commands
                if user_input in ['q', 'quit', 'exit']:
                    print("👋 Exiting manual control mode...")
                    break
                
                elif user_input in ['h', 'home']:
                    print("\n🏠 Moving to home position [0,0,0,0,0,0]...")
                    target_angles = np.zeros(6)
                    env._move_to_joint_positions(target_angles, duration=1.5)
                    time.sleep(0.5)
                    for _ in range(10):
                        rclpy.spin_once(env, timeout_sec=0.1)
                    print("✅ At home position")
                    continue
                
                elif user_input in ['r', 'reset']:
                    print("\n🔄 Resetting environment...")
                    state = env.reset_environment()
                    for _ in range(10):
                        rclpy.spin_once(env, timeout_sec=0.1)
                    print("✅ Environment reset with new target")
                    continue
                
                elif user_input == 'ws':
                    print("\n📦 WORKSPACE BOUNDS:")
                    print(f"   X: [{WORKSPACE_BOUNDS['x_min']*100:.0f}cm, {WORKSPACE_BOUNDS['x_max']*100:.0f}cm]")
                    print(f"   Y: [{WORKSPACE_BOUNDS['y_min']*100:.0f}cm, {WORKSPACE_BOUNDS['y_max']*100:.0f}cm]")
                    print(f"   Z: [{WORKSPACE_BOUNDS['z_min']*100:.0f}cm, {WORKSPACE_BOUNDS['z_max']*100:.0f}cm]")
                    continue
                
                elif user_input.startswith('fk '):
                    # FK prediction only
                    try:
                        angles_str = user_input[3:].split()
                        if len(angles_str) != 6:
                            print("❌ Please enter exactly 6 angles")
                            continue
                        angles_deg = np.array([float(a) for a in angles_str])
                        angles_rad = np.radians(angles_deg)
                        
                        x, y, z = fk(angles_rad)
                        print(f"\n🧮 FK Prediction for {angles_deg}°:")
                        print(f"   End-effector: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
                        
                        # Check if in workspace
                        in_ws = (WORKSPACE_BOUNDS['x_min'] <= x <= WORKSPACE_BOUNDS['x_max'] and
                                 WORKSPACE_BOUNDS['y_min'] <= y <= WORKSPACE_BOUNDS['y_max'] and
                                 WORKSPACE_BOUNDS['z_min'] <= z <= WORKSPACE_BOUNDS['z_max'])
                        print(f"   In workspace: {'✅ YES' if in_ws else '❌ NO'}")
                        
                        if current_state is not None:
                            target_pos = current_state[9:12]
                            dist = np.sqrt((x-target_pos[0])**2 + (y-target_pos[1])**2 + (z-target_pos[2])**2)
                            print(f"   Distance to target: {dist*100:.2f}cm")
                    except ValueError:
                        print("❌ Invalid angles! Enter numbers only.")
                    continue
                
                # Parse joint angles
                try:
                    angles_str = user_input.split()
                    if len(angles_str) != 6:
                        print("❌ Please enter exactly 6 joint angles (in degrees)")
                        print("   Example: 0 30 -20 0 15 0")
                        continue
                    
                    target_angles_deg = np.array([float(a) for a in angles_str])
                    target_angles_rad = np.radians(target_angles_deg)
                    
                    # Check limits
                    if np.any(np.abs(target_angles_deg) > 90):
                        print("⚠️ Warning: Some angles exceed ±90° limit, will be clipped")
                        target_angles_rad = np.clip(target_angles_rad, -np.pi/2, np.pi/2)
                        target_angles_deg = np.degrees(target_angles_rad)
                    
                except ValueError:
                    print("❌ Invalid input! Please enter 6 numbers.")
                    continue
                
                # Get state BEFORE movement
                state_before = env.get_state()
                if state_before is None:
                    print("❌ Could not get state")
                    continue
                
                ee_before = state_before[6:9]
                joints_before = state_before[:6]
                target_pos = state_before[9:12]
                dist_before = state_before[15]
                
                # FK prediction
                pred_x, pred_y, pred_z = fk(target_angles_rad)
                pred_dist = np.sqrt((pred_x-target_pos[0])**2 + (pred_y-target_pos[1])**2 + (pred_z-target_pos[2])**2)
                
                print(f"\n{'='*60}")
                print("📋 MOVEMENT PLAN")
                print(f"{'='*60}")
                print(f"Target joints (deg): [{', '.join([f'{a:.1f}' for a in target_angles_deg])}]")
                print(f"\n🧮 FK Prediction:")
                print(f"   Predicted EE: X={pred_x:.4f}, Y={pred_y:.4f}, Z={pred_z:.4f}")
                print(f"   Predicted distance to target: {pred_dist*100:.2f}cm")
                
                print(f"\n📍 BEFORE:")
                print(f"   Current joints: [{', '.join([f'{np.degrees(j):.1f}' for j in joints_before])}]°")
                print(f"   End-effector: X={ee_before[0]:.4f}, Y={ee_before[1]:.4f}, Z={ee_before[2]:.4f}")
                print(f"   Distance to target: {dist_before*100:.2f}cm")
                
                # Execute movement in SINGLE STEP using direct trajectory control
                print(f"\n⏳ Executing movement (single trajectory)...")
                
                success = env._move_to_joint_positions(target_angles_rad, duration=1.5)
                
                # Wait for movement to complete and update state
                time.sleep(0.5)
                for _ in range(10):
                    rclpy.spin_once(env, timeout_sec=0.1)
                
                # Get state AFTER movement
                state_after = env.get_state()
                if state_after is None:
                    print("❌ Could not get state after movement")
                    continue
                
                ee_after = state_after[6:9]
                joints_after = state_after[:6]
                dist_after = state_after[15]
                
                # Calculate metrics
                ee_movement = np.linalg.norm(ee_after - ee_before)
                joint_movement = np.linalg.norm(np.array(joints_after) - np.array(joints_before))
                dist_improvement = dist_before - dist_after
                
                # Calculate reward manually (same formula as environment)
                if dist_after < 0.01:
                    reward = 10.0
                else:
                    improvement = dist_before - dist_after
                    reward = improvement * 10.0 - 0.5
                
                print(f"\n📍 AFTER:")
                print(f"   Final joints: [{', '.join([f'{np.degrees(j):.1f}' for j in joints_after])}]°")
                print(f"   End-effector: X={ee_after[0]:.4f}, Y={ee_after[1]:.4f}, Z={ee_after[2]:.4f}")
                print(f"   Distance to target: {dist_after*100:.2f}cm")
                
                print(f"\n📏 MOVEMENT SUMMARY:")
                print(f"   Trajectory: {'✅ Success' if success else '❌ Failed'}")
                print(f"   EE moved: {ee_movement*100:.2f}cm")
                print(f"   Joints moved: {np.degrees(joint_movement):.1f}°")
                print(f"   Distance change: {dist_improvement*100:+.2f}cm")
                print(f"   Reward (if training): {reward:.3f}")
                
                # Goal check
                if dist_after < 0.01:
                    print(f"\n🎉🎉🎉 GOAL REACHED! Distance: {dist_after*1000:.1f}mm 🎉🎉🎉")
                elif dist_improvement > 0:
                    print(f"\n✅ Getting closer! ({dist_improvement*100:.2f}cm improvement)")
                else:
                    print(f"\n⚠️ Moving away from target ({-dist_improvement*100:.2f}cm)")
                
                print(f"{'='*60}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        print("\n🛑 Cleaning up...")
        env.destroy_node()
        rclpy.shutdown()
        print("✅ Done!")


if __name__ == '__main__':
    manual_control()
