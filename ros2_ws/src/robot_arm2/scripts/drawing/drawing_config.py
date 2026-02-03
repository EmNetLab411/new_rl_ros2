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
POINTS_PER_EDGE = 7  # 10 waypoints total

# Shape type
SHAPE_TYPE = 'triangle'

# Computed total waypoints
if SHAPE_TYPE == 'square':
    # 4 edges * points + 1 return
    TOTAL_WAYPOINTS = POINTS_PER_EDGE * 4 + 1  
else:
    # 3 edges * points + 1 return (triangle)
    TOTAL_WAYPOINTS = POINTS_PER_EDGE * 3 + 1

# =============================================================================
# SHAPE PARAMETERS
# =============================================================================

# Square size (side length in meters)
SHAPE_SIZE = 0.15  # 15cm sides

# Y-plane (height above ground)
Y_PLANE = 0.20  # 20cm above ground

# Shape center position (X, Y, Z) in meters
# Positioned to fit within workspace Z limit (0.16m - 0.28m)
# Square Height 10cm. Half=5cm. 
# Top = 0.22 + 0.05 = 0.27m (< 0.28m)
# Bottom = 0.22 - 0.05 = 0.17m (> 0.16m)
TRIANGLE_CENTER = (0.0, 0.20, 0.25)  # X=0, Y=0.20m, Z=0.25m

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Waypoint tolerance (distance threshold to consider waypoint reached)
WAYPOINT_TOLERANCE = 0.01  # 1cm tolerance

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
if __name__ != "__main__":
    validate_config()
