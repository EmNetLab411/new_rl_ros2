#!/usr/bin/env python3
"""
Workspace Exploration Script for 6-DOF Robot

This script samples joint configurations and computes the reachable workspace
using Forward Kinematics. Outputs:
1. Text summary of workspace bounds
2. 3D scatter plot of reachable positions
3. Heatmap of reachability density

Usage:
    python3 explore_workspace.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Try to import FK
try:
    from fk_ik_utils import forward_kinematics
    FK_AVAILABLE = True
except ImportError:
    FK_AVAILABLE = False
    print("⚠️  Could not import forward_kinematics, using simplified FK")


def simplified_fk(joints):
    """
    Simplified FK for 6-DOF robot (approximation)
    Based on typical 6DOF arm geometry
    """
    # Link lengths (approximate - adjust based on your URDF)
    L1 = 0.10  # Base to shoulder
    L2 = 0.15  # Shoulder to elbow  
    L3 = 0.15  # Elbow to wrist
    L4 = 0.10  # Wrist to end-effector
    
    j1, j2, j3, j4, j5, j6 = joints
    
    # Simplified calculation (not exact, but gives workspace idea)
    r = (L2 * np.cos(j2) + L3 * np.cos(j2 + j3) + L4 * np.cos(j2 + j3 + j4))
    
    x = r * np.cos(j1)
    y = r * np.sin(j1)
    z = L1 + L2 * np.sin(j2) + L3 * np.sin(j2 + j3) + L4 * np.sin(j2 + j3 + j4)
    
    return x, y, z


def explore_workspace(num_samples=10000, joint_limits_deg=90):
    """
    Sample joint space and compute reachable workspace
    
    Args:
        num_samples: Number of random joint configurations to sample
        joint_limits_deg: Joint limits in degrees (±this value)
    """
    print("=" * 70)
    print("🔍 6-DOF Robot Workspace Exploration")
    print("=" * 70)
    
    # Joint limits (radians)
    joint_limit = np.radians(joint_limits_deg)
    
    print(f"\n📐 Joint limits: ±{joint_limits_deg}° (±{joint_limit:.3f} rad)")
    print(f"📊 Sampling {num_samples:,} random configurations...")
    
    # Storage for positions
    positions = []
    valid_count = 0
    
    # Random sampling
    np.random.seed(42)  # Reproducible
    
    for i in range(num_samples):
        # Random joint angles within limits
        joints = np.random.uniform(-joint_limit, joint_limit, 6)
        
        try:
            if FK_AVAILABLE:
                x, y, z = forward_kinematics(joints)
            else:
                x, y, z = simplified_fk(joints)
            
            # Skip invalid positions (NaN, very far, underground)
            if np.isfinite([x, y, z]).all() and z > 0:
                positions.append([x, y, z])
                valid_count += 1
        except Exception as e:
            continue
        
        if (i + 1) % 2000 == 0:
            print(f"   Processed {i+1:,} samples... ({valid_count:,} valid)")
    
    positions = np.array(positions)
    print(f"\n✅ Found {len(positions):,} valid positions")
    
    # Calculate bounds
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    
    print("\n" + "=" * 70)
    print("📊 WORKSPACE BOUNDS (in meters)")
    print("=" * 70)
    print(f"  X: [{x_min:.4f}, {x_max:.4f}] → Range: {(x_max-x_min):.4f}m ({(x_max-x_min)*100:.1f}cm)")
    print(f"  Y: [{y_min:.4f}, {y_max:.4f}] → Range: {(y_max-y_min):.4f}m ({(y_max-y_min)*100:.1f}cm)")
    print(f"  Z: [{z_min:.4f}, {z_max:.4f}] → Range: {(z_max-z_min):.4f}m ({(z_max-z_min)*100:.1f}cm)")
    
    # Recommended conservative workspace (80% of range)
    margin = 0.1  # 10% margin on each side
    
    rec_x_min = x_min + margin * (x_max - x_min)
    rec_x_max = x_max - margin * (x_max - x_min)
    rec_y_min = y_min + margin * (y_max - y_min)
    rec_y_max = y_max - margin * (y_max - y_min)
    rec_z_min = z_min + margin * (z_max - z_min)
    rec_z_max = z_max - margin * (z_max - z_min)
    
    print("\n" + "=" * 70)
    print("💡 RECOMMENDED WORKSPACE (with 10% safety margin)")
    print("=" * 70)
    print(f"  X: [{rec_x_min:.4f}, {rec_x_max:.4f}]")
    print(f"  Y: [{rec_y_min:.4f}, {rec_y_max:.4f}]")
    print(f"  Z: [{rec_z_min:.4f}, {rec_z_max:.4f}]")
    
    # Python code for copy-paste
    print("\n" + "=" * 70)
    print("📝 CODE TO COPY (for rl_environment.py)")
    print("=" * 70)
    print(f"""
# Workspace boundaries from FK exploration
WORKSPACE_BOUNDS = {{
    'x_min': {rec_x_min:.4f},
    'x_max': {rec_x_max:.4f},
    'y_min': {rec_y_min:.4f},
    'y_max': {rec_y_max:.4f},
    'z_min': {rec_z_min:.4f},
    'z_max': {rec_z_max:.4f}
}}
""")
    
    # Test specific home position
    print("\n" + "=" * 70)
    print("🏠 KEY POSITIONS")
    print("=" * 70)
    
    test_configs = [
        ("Home [0,0,0,0,0,0]", [0, 0, 0, 0, 0, 0]),
        ("All +45°", [0.785] * 6),
        ("All -45°", [-0.785] * 6),
        ("J1=90°", [1.57, 0, 0, 0, 0, 0]),
        ("J2=45°", [0, 0.785, 0, 0, 0, 0]),
        ("J2=-45°", [0, -0.785, 0, 0, 0, 0]),
    ]
    
    for name, joints in test_configs:
        try:
            if FK_AVAILABLE:
                x, y, z = forward_kinematics(joints)
            else:
                x, y, z = simplified_fk(joints)
            print(f"  {name:20s} → X={x:+.4f}, Y={y:+.4f}, Z={z:.4f}")
        except Exception as e:
            print(f"  {name:20s} → Error: {e}")
    
    # Visualization
    print("\n" + "=" * 70)
    print("📈 Generating visualization...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Subsample for plotting (too many points is slow)
    plot_indices = np.random.choice(len(positions), min(5000, len(positions)), replace=False)
    plot_positions = positions[plot_indices]
    
    ax1.scatter(plot_positions[:, 0], plot_positions[:, 1], plot_positions[:, 2],
                c=plot_positions[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Reachable Workspace')
    
    # XY plane projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], cmap='viridis', s=1, alpha=0.3)
    ax2.axhline(y=rec_y_min, color='r', linestyle='--', label='Recommended Y bounds')
    ax2.axhline(y=rec_y_max, color='r', linestyle='--')
    ax2.axvline(x=rec_x_min, color='b', linestyle='--', label='Recommended X bounds')
    ax2.axvline(x=rec_x_max, color='b', linestyle='--')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane (Top View)')
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')
    
    # XZ plane projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(positions[:, 0], positions[:, 2], c=positions[:, 1], cmap='viridis', s=1, alpha=0.3)
    ax3.axhline(y=rec_z_min, color='r', linestyle='--', label='Recommended Z bounds')
    ax3.axhline(y=rec_z_max, color='r', linestyle='--')
    ax3.axvline(x=rec_x_min, color='b', linestyle='--', label='Recommended X bounds')
    ax3.axvline(x=rec_x_max, color='b', linestyle='--')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Plane (Side View)')
    ax3.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(script_dir, 'workspace_exploration.png')
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()
    
    return positions, (rec_x_min, rec_x_max, rec_y_min, rec_y_max, rec_z_min, rec_z_max)


if __name__ == '__main__':
    explore_workspace(num_samples=20000)
