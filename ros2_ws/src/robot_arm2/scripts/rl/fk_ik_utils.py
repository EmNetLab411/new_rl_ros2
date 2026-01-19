"""
Forward and Inverse Kinematics for 6-DOF Robot Arm
Uses actual URDF joint transforms for accurate FK
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional

# Joint transforms from URDF
JOINT_TRANSFORMS = [
    # Joint 1: Base_1 → Link1_1 (continuous, Z-axis)
    {'xyz': np.array([0.0, 0.0, 0.068502]), 'axis': np.array([0, 0, 1])},
    # Joint 2: Link1_1 → Link2_1 (continuous, -X-axis after flip)
    {'xyz': np.array([0.041821, -0.019984, 0.053522]), 'axis': np.array([-1, 0, 0])},
    # Joint 3: Link2_1 → Link3_1 (continuous, +X-axis after flip)
    {'xyz': np.array([-0.075886, -7e-06, 0.116723]), 'axis': np.array([1, 0, 0])},
    # Joint 4: Link3_1 → Link4_1 (continuous, Y-axis negative)
    {'xyz': np.array([0.032204, 0.031535, 0.062164]), 'axis': np.array([0, -1, 0])},
    # Joint 5: Link4_1 → Link5_1 (continuous, +X-axis after flip)
    {'xyz': np.array([-0.032579, -0.0331, 0.077214]), 'axis': np.array([1, 0, 0])},
    # Joint 6: Link5_1 → Link6_1 (continuous, Y-axis negative)
    {'xyz': np.array([0.0316, 0.0153, 0.0638]), 'axis': np.array([0, -1, 0])},
]

END_EFFECTOR_OFFSET = np.array([0.00007, -0.016091, 0.046444])


JOINT_LIMITS_LOW = np.array([-np.pi/2] * 6)
JOINT_LIMITS_HIGH = np.array([np.pi/2] * 6)


def fk(joint_angles):
    """Forward Kinematics using URDF transforms"""
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
    
    def rot_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def rot_y(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def rot_x(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    pos = np.array([0.0, 0.0, 0.0])
    R = np.eye(3)
    
    # Joint 1: Z-axis
    pos = pos + JOINT_TRANSFORMS[0]['xyz']
    R = R @ rot_z(joint_angles[0])
    
    # Joint 2: -X-axis (flipped for +Y workspace)
    pos = pos + R @ JOINT_TRANSFORMS[1]['xyz']
    R = R @ rot_x(-joint_angles[1])
    
    # Joint 3: X-axis
    pos = pos + R @ JOINT_TRANSFORMS[2]['xyz']
    R = R @ rot_x(joint_angles[2])
    
    # Joint 4: -Y-axis
    pos = pos + R @ JOINT_TRANSFORMS[3]['xyz']
    R = R @ rot_y(-joint_angles[3])
    
    # Joint 5: +X-axis (flipped for +Y workspace)
    pos = pos + R @ JOINT_TRANSFORMS[4]['xyz']
    R = R @ rot_x(joint_angles[4])
    
    # Joint 6: -Y-axis
    pos = pos + R @ JOINT_TRANSFORMS[5]['xyz']
    R = R @ rot_y(-joint_angles[5])
    
    # End-effector offset
    pos = pos + R @ END_EFFECTOR_OFFSET
    
    return (pos[0], pos[1], pos[2])
