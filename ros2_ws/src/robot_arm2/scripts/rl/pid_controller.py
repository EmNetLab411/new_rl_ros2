"""
PID Controller for Residual RL

Simple 3D PID controller that provides baseline actions.
SAC learns small corrections on top of PID output.

Usage in Residual RL:
    final_action = pid_action + alpha * sac_correction
"""

import numpy as np


class PIDController:
    """
    3D PID controller for end-effector position control.
    
    Computes desired movement direction based on error between
    current position and target position.
    """
    
    def __init__(self, Kp=2.0, Ki=0.0, Kd=0.1, output_limit=0.5):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain (main driver)
            Ki: Integral gain (for steady-state error, usually 0)
            Kd: Derivative gain (for damping oscillations)
            output_limit: Maximum output magnitude per axis
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limit = output_limit
        
        # State variables
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
    
    def compute(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Compute control output from position error.
        
        Args:
            current_pos: Current end-effector position [x, y, z]
            target_pos: Target position [x, y, z]
        
        Returns:
            control: Desired delta position [dx, dy, dz] in meters
        """
        error = target_pos - current_pos
        
        # Proportional term (main driver - move toward target)
        P = self.Kp * error
        
        # Integral term (accumulates error over time)
        self.integral += error
        I = self.Ki * self.integral
        
        # Derivative term (dampens oscillations)
        D = self.Kd * (error - self.prev_error)
        self.prev_error = error.copy()
        
        # Combine PID terms
        control = P + I + D
        
        # Clip output to limit
        control = np.clip(control, -self.output_limit, self.output_limit)
        
        return control
    
    def compute_normalized(self, current_pos: np.ndarray, target_pos: np.ndarray,
                          pos_min: np.ndarray, pos_max: np.ndarray) -> np.ndarray:
        """
        Compute normalized control output in [-1, 1] range.
        
        Useful when action space is normalized (like SAC's output).
        
        Args:
            current_pos: Current EE position [x, y, z]
            target_pos: Target position [x, y, z]
            pos_min: Minimum position bounds
            pos_max: Maximum position bounds
        
        Returns:
            control: Normalized action in [-1, 1] range
        """
        # Compute raw PID output
        raw_output = self.compute(current_pos, target_pos)
        
        # Convert delta to absolute target position
        target_from_pid = current_pos + raw_output
        
        # Normalize to [-1, 1]
        normalized = 2.0 * (target_from_pid - pos_min) / (pos_max - pos_min) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def reset(self):
        """Reset controller state for new episode."""
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)


# Default PID gains (tuned for ~12cm workspace)
DEFAULT_KP = 2.0  # Strong proportional for quick response
DEFAULT_KI = 0.0  # No integral (avoids windup issues)
DEFAULT_KD = 0.1  # Small derivative for damping
