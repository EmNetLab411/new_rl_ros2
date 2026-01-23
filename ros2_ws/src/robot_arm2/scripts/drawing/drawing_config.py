#!/usr/bin/env python3
"""
Drawing Training Configuration

Central configuration file for drawing training parameters.
Change POINTS_PER_EDGE to scale waypoint density.
"""

# =============================================================================
# WAYPOINT CONFIGURATION
# =============================================================================

# Number of waypoints per edge of the triangle
# Total waypoints = POINTS_PER_EDGE * 3 + 1 (for return to start)
# Examples:
#   POINTS_PER_EDGE = 1  → 4 waypoints (3 corners + 1 return)
#   POINTS_PER_EDGE = 3  → 10 waypoints (9 + 1 return)
#   POINTS_PER_EDGE = 4  → 13 waypoints
#   POINTS_PER_EDGE = 10 → 31 waypoints
POINTS_PER_EDGE = 3  # Currently: 10 waypoints total

# Computed total waypoints (3 edges × points_per_edge + 1 return)
TOTAL_WAYPOINTS = POINTS_PER_EDGE * 3 + 1

# Shape type - always use 'triangle' (equilateral_triangle with points_per_edge)
SHAPE_TYPE = 'triangle'

# =============================================================================
# SHAPE PARAMETERS
# =============================================================================

# Triangle size (side length in meters)
SHAPE_SIZE = 0.15  # 15cm sides (matches train_robot.py)

# Y-plane (height above ground)
Y_PLANE = 0.20  # 20cm above ground

# Triangle center position (X, Y, Z) in meters
TRIANGLE_CENTER = (0.0, 0.20, 0.25)  # Center at X=0, Y=0.20m (height), Z=0.25m

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Waypoint tolerance (distance threshold to consider waypoint reached)
WAYPOINT_TOLERANCE = 0.01  # 0.5cm tolerance

# Max steps per episode
DEFAULT_MAX_STEPS = 100
MIN_MAX_STEPS = 5  # Minimum for any configuration

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_waypoint_info():
    """Get human-readable waypoint configuration info."""
    return f"{TOTAL_WAYPOINTS} waypoints ({POINTS_PER_EDGE} per edge)"

def validate_config():
    """Validate configuration parameters."""
    assert POINTS_PER_EDGE >= 1, "POINTS_PER_EDGE must be >= 1"
    assert SHAPE_SIZE > 0, "SHAPE_SIZE must be positive"
    assert Y_PLANE > 0, "Y_PLANE must be positive"
    assert WAYPOINT_TOLERANCE > 0, "WAYPOINT_TOLERANCE must be positive"
    print(f"✅ Drawing config validated: {get_waypoint_info()}")

# Auto-validate on import
validate_config()
