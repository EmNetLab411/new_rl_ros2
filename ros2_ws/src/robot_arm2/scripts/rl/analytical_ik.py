#!/usr/bin/env python3
"""
Robust Analytical Inverse Kinematics for 6-DOF Robot Arm

This module provides multiple IK methods for reliable joint angle computation:
1. Damped Least Squares (DLS) Jacobian - iterative, handles singularities
2. Multi-start optimization - tries multiple initial configurations
3. Hybrid approach - combines methods for best results

Usage:
    from analytical_ik import AnalyticalIK
    
    ik = AnalyticalIK()
    joints, success, error = ik.solve([0.1, -0.2, 0.3])
    
    # Or with current position for smooth motion:
    joints, success, error = ik.solve([0.1, -0.2, 0.3], current_joints=current)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, List
import time

# Import FK from existing module
try:
    from .fk_ik_utils import fk, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH, JOINT_TRANSFORMS, END_EFFECTOR_OFFSET
except ImportError:
    from fk_ik_utils import fk, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH, JOINT_TRANSFORMS, END_EFFECTOR_OFFSET


class AnalyticalIK:
    """
    Robust 6-DOF Inverse Kinematics Solver
    
    Uses multiple strategies to find reliable IK solutions:
    1. DLS Jacobian (fast, iterative)
    2. Multi-start optimization (reliable, slower)
    3. Hybrid approach (best of both)
    """
    
    def __init__(self, 
                 position_tolerance: float = 0.002,  # 2mm default
                 max_iterations: int = 100,
                 damping: float = 0.05):
        """
        Initialize IK solver.
        
        Args:
            position_tolerance: Max acceptable position error (meters)
            max_iterations: Max iterations for iterative methods
            damping: Damping factor for DLS (higher = more stable, less accurate)
        """
        self.position_tolerance = position_tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        
        # Pre-compute some useful configurations for multi-start
        self.seed_configurations = self._generate_seed_configurations()
    
    def _generate_seed_configurations(self, n_seeds: int = 8) -> List[np.ndarray]:
        """Generate diverse initial configurations for multi-start optimization"""
        seeds = [np.zeros(6)]  # Home position
        
        # Add some typical arm poses
        typical_poses = [
            [0, 0.5, 0.5, 0, 0, 0],      # Forward reach
            [0, 0.5, -0.5, 0, 0, 0],     # Forward different elbow
            [0, -0.5, 0.5, 0, 0, 0],     # Back reach
            [0.5, 0.5, 0.5, 0, 0, 0],    # Rotated forward
            [-0.5, 0.5, 0.5, 0, 0, 0],   # Rotated other way
            [0, 1.0, 0, 0, 0, 0],        # Arm down
            [0, 0, 1.0, 0, 0, 0],        # Different arm config
        ]
        
        for pose in typical_poses:
            seeds.append(np.array(pose))
        
        # Add random configurations
        np.random.seed(42)  # Reproducible
        for _ in range(n_seeds - len(seeds)):
            random_joints = np.random.uniform(
                JOINT_LIMITS_LOW * 0.7,  # Stay away from limits
                JOINT_LIMITS_HIGH * 0.7
            )
            seeds.append(random_joints)
        
        return seeds
    
    def compute_jacobian(self, joints: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute numerical Jacobian matrix (3x6) at current joint configuration.
        
        J[i,j] = ∂position_i / ∂joint_j
        
        Args:
            joints: Current joint angles [6]
            epsilon: Perturbation for numerical differentiation
            
        Returns:
            Jacobian matrix (3x6)
        """
        J = np.zeros((3, 6))
        base_pos = np.array(fk(joints))
        
        for j in range(6):
            joints_plus = joints.copy()
            joints_plus[j] += epsilon
            pos_plus = np.array(fk(joints_plus))
            J[:, j] = (pos_plus - base_pos) / epsilon
        
        return J
    
    def solve_dls(self, 
                  target_position: np.ndarray,
                  initial_joints: Optional[np.ndarray] = None,
                  max_iters: Optional[int] = None) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK using Damped Least Squares (DLS) Jacobian method.
        
        This is the Levenberg-Marquardt approach applied to IK:
        Δθ = J^T (J J^T + λ²I)^{-1} Δx
        
        Advantages:
        - Fast (typically 5-20 iterations)
        - Handles singularities gracefully
        - Smooth motion when starting from current pose
        
        Args:
            target_position: [x, y, z] target position
            initial_joints: Starting joint configuration (default: zeros)
            max_iters: Override max iterations
            
        Returns:
            (joint_angles, success, final_error)
        """
        max_iters = max_iters or self.max_iterations
        joints = np.array(initial_joints) if initial_joints is not None else np.zeros(6)
        joints = np.clip(joints, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        
        target = np.array(target_position)
        lambda_sq = self.damping ** 2
        
        for iteration in range(max_iters):
            # Current end-effector position
            current_pos = np.array(fk(joints))
            error_vec = target - current_pos
            error_norm = np.linalg.norm(error_vec)
            
            # Check convergence
            if error_norm < self.position_tolerance:
                return joints.astype(np.float32), True, error_norm
            
            # Compute Jacobian
            J = self.compute_jacobian(joints)
            
            # DLS solution: Δθ = J^T (J J^T + λ²I)^{-1} Δx
            # This is more numerically stable than pseudo-inverse
            JJT = J @ J.T
            damped_JJT = JJT + lambda_sq * np.eye(3)
            
            try:
                delta_x_solved = np.linalg.solve(damped_JJT, error_vec)
                delta_joints = J.T @ delta_x_solved
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if matrix is singular
                J_pinv = np.linalg.pinv(J)
                delta_joints = J_pinv @ error_vec
            
            # Adaptive step size (reduce if error increases)
            step_size = 1.0
            for _ in range(5):  # Line search
                new_joints = joints + step_size * delta_joints
                new_joints = np.clip(new_joints, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
                new_error = np.linalg.norm(target - np.array(fk(new_joints)))
                
                if new_error < error_norm:
                    joints = new_joints
                    break
                step_size *= 0.5
            else:
                # No improvement - might be stuck
                joints = new_joints
        
        # Final check
        final_error = np.linalg.norm(target - np.array(fk(joints)))
        success = final_error < self.position_tolerance
        
        return joints.astype(np.float32), success, final_error
    
    def solve_optimization(self,
                           target_position: np.ndarray,
                           initial_joints: Optional[np.ndarray] = None,
                           method: str = 'SLSQP') -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK using scipy optimization.
        
        Args:
            target_position: [x, y, z] target position
            initial_joints: Starting configuration
            method: Optimization method ('SLSQP', 'L-BFGS-B', 'trust-constr')
            
        Returns:
            (joint_angles, success, final_error)
        """
        target = np.array(target_position)
        x0 = np.array(initial_joints) if initial_joints is not None else np.zeros(6)
        x0 = np.clip(x0, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        
        def objective(joints):
            pos = np.array(fk(joints))
            pos_error = np.sum((pos - target) ** 2)
            # Small regularization to prefer minimal joint motion
            joint_reg = 0.001 * np.sum(joints ** 2)
            return pos_error + joint_reg
        
        def position_error(joints):
            pos = np.array(fk(joints))
            return np.linalg.norm(pos - target)
        
        bounds = list(zip(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH))
        
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'ftol': 1e-8,
                'disp': False
            }
        )
        
        joints = result.x
        final_error = position_error(joints)
        success = final_error < self.position_tolerance
        
        return joints.astype(np.float32), success, final_error
    
    def solve_multistart(self,
                         target_position: np.ndarray,
                         current_joints: Optional[np.ndarray] = None,
                         n_starts: int = 5) -> Tuple[np.ndarray, bool, float]:
        """
        Multi-start optimization for better global convergence.
        
        Tries multiple initial configurations and returns the best solution.
        If current_joints is provided, it gets priority.
        
        Args:
            target_position: [x, y, z] target position
            current_joints: Current joint configuration (preferred start)
            n_starts: Number of different starting points
            
        Returns:
            (joint_angles, success, final_error)
        """
        target = np.array(target_position)
        best_joints = None
        best_error = float('inf')
        best_success = False
        
        # Build list of starting configurations
        starts = []
        if current_joints is not None:
            starts.append(np.array(current_joints))
        
        # Add seed configurations
        for seed in self.seed_configurations[:n_starts]:
            starts.append(seed)
        
        # Try each starting configuration
        for x0 in starts:
            try:
                joints, success, error = self.solve_dls(target, x0, max_iters=50)
                
                # Prefer solutions close to current pose for smooth motion
                if current_joints is not None:
                    joint_distance = np.linalg.norm(joints - current_joints)
                    # Penalize large joint jumps
                    effective_error = error + 0.1 * joint_distance
                else:
                    effective_error = error
                
                if error < best_error:
                    best_joints = joints
                    best_error = error
                    best_success = success
                    
                    # Early exit if good enough
                    if success and error < self.position_tolerance * 0.5:
                        break
                        
            except Exception:
                continue
        
        if best_joints is None:
            # All starts failed - return home position
            return np.zeros(6, dtype=np.float32), False, float('inf')
        
        return best_joints, best_success, best_error
    
    def solve(self,
              target_position: np.ndarray,
              current_joints: Optional[np.ndarray] = None,
              prefer_smooth: bool = True) -> Tuple[np.ndarray, bool, float]:
        """
        Main IK solver - hybrid approach for best results.
        
        Strategy:
        1. Try DLS from current pose (fast, smooth motion)
        2. If fails, try multi-start optimization (more reliable)
        3. Return best solution found
        
        Args:
            target_position: [x, y, z] target end-effector position
            current_joints: Current joint configuration (for smooth motion)
            prefer_smooth: If True, prefer solutions close to current pose
            
        Returns:
            Tuple of (joint_angles, success, position_error)
        """
        target = np.array(target_position)
        
        # Strategy 1: Fast DLS from current pose
        if current_joints is not None:
            joints, success, error = self.solve_dls(target, current_joints)
            if success:
                return joints, success, error
        
        # Strategy 2: DLS from home
        joints, success, error = self.solve_dls(target, np.zeros(6))
        if success:
            return joints, success, error
        
        # Strategy 3: Multi-start if DLS failed
        joints, success, error = self.solve_multistart(target, current_joints, n_starts=5)
        
        return joints, success, error
    
    def check_reachability(self, target_position: np.ndarray) -> Tuple[bool, float]:
        """
        Quick check if a target is likely reachable.
        
        Args:
            target_position: [x, y, z] position to check
            
        Returns:
            (is_likely_reachable, estimated_error)
        """
        _, success, error = self.solve(target_position, prefer_smooth=False)
        return success, error


# ============================================================================
# CONVENIENCE FUNCTIONS (drop-in replacement for old IK)
# ============================================================================

# Global solver instance
_solver = None

def get_solver() -> AnalyticalIK:
    """Get or create global solver instance"""
    global _solver
    if _solver is None:
        _solver = AnalyticalIK()
    return _solver


def analytical_ik_6dof(
    target_x: float,
    target_y: float, 
    target_z: float,
    initial_guess: Optional[np.ndarray] = None,
    tolerance: float = 0.002
) -> Tuple[np.ndarray, bool, float]:
    """
    Drop-in replacement for constrained_ik_6dof.
    
    Args:
        target_x, target_y, target_z: Target position
        initial_guess: Starting joint configuration
        tolerance: Position tolerance (meters)
        
    Returns:
        (joint_angles, success, error)
    """
    solver = get_solver()
    solver.position_tolerance = tolerance
    
    target = np.array([target_x, target_y, target_z])
    return solver.solve(target, current_joints=initial_guess)


# ============================================================================
# TESTING
# ============================================================================

def test_ik_solver():
    """Test the IK solver with various targets"""
    print("=" * 60)
    print("🧪 Testing Analytical IK Solver")
    print("=" * 60)
    
    solver = AnalyticalIK(position_tolerance=0.002)  # 2mm
    
    # Generate test targets from FK
    test_cases = []
    np.random.seed(42)
    
    for i in range(20):
        # Random joint configuration
        joints = np.random.uniform(JOINT_LIMITS_LOW * 0.8, JOINT_LIMITS_HIGH * 0.8)
        pos = fk(joints)
        test_cases.append((pos, joints))
    
    # Run tests
    successes = 0
    errors = []
    times = []
    
    print(f"\n{'Target (x,y,z)':<30} {'Error (mm)':<12} {'Success':<10} {'Time (ms)':<10}")
    print("-" * 62)
    
    for target_pos, original_joints in test_cases:
        start = time.time()
        solved_joints, success, error = solver.solve(target_pos)
        elapsed = (time.time() - start) * 1000
        
        times.append(elapsed)
        errors.append(error * 1000)  # Convert to mm
        
        if success:
            successes += 1
        
        # Verify FK matches
        solved_pos = fk(solved_joints)
        verify_error = np.linalg.norm(np.array(target_pos) - np.array(solved_pos))
        
        status = "✅" if success else "❌"
        print(f"({target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:.3f})  "
              f"{error*1000:>8.2f}     {status:<10} {elapsed:>6.1f}")
    
    print("-" * 62)
    print(f"\n📊 Results:")
    print(f"   Success rate: {successes}/{len(test_cases)} ({100*successes/len(test_cases):.1f}%)")
    print(f"   Mean error:   {np.mean(errors):.2f} mm")
    print(f"   Max error:    {np.max(errors):.2f} mm")
    print(f"   Mean time:    {np.mean(times):.1f} ms")
    print(f"   Max time:     {np.max(times):.1f} ms")
    
    # Test with current_joints for smooth motion
    print("\n🔄 Testing smooth motion (with current_joints)...")
    
    # Simulate trajectory
    current = np.zeros(6)
    trajectory_errors = []
    
    for target_pos, _ in test_cases[:5]:
        joints, success, error = solver.solve(target_pos, current_joints=current)
        if success:
            joint_jump = np.max(np.abs(joints - current))
            trajectory_errors.append(error * 1000)
            print(f"   Jump: {np.degrees(joint_jump):>5.1f}° | Error: {error*1000:.2f}mm")
            current = joints
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_ik_solver()
