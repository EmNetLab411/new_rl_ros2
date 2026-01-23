#!/usr/bin/env python3
"""
Gazebo Drawing Visualizer for Robot Arm

Spawns drawing shapes and pen lines directly in Gazebo using Ignition Transport.
Unlike RViz markers, these are actual 3D entities visible in Gazebo.

Features:
- Spawn triangle/polygon outline as visual marker
- Draw pen path as the robot moves (green line)
- Reset lines on episode reset
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_srvs.srv import Empty
import subprocess
import numpy as np
from typing import List, Tuple
import time
# Add scripts directory to path for imports
# When installed, __file__ is in install/robot_arm2/lib/robot_arm2/
# We need to point to src/robot_arm2/scripts/
import sys
import os

# Find the workspace root and add scripts directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Try to find the workspace by looking for 'install' in path
if 'install' in script_dir:
    # Running from installed location
    ws_root = script_dir.split('install')[0]
    scripts_dir = os.path.join(ws_root, 'src', 'robot_arm2', 'scripts')
else:
    # Running from source
    scripts_dir = os.path.dirname(script_dir)

if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import drawing config for waypoint settings
from drawing.drawing_config import POINTS_PER_EDGE, SHAPE_SIZE, TRIANGLE_CENTER


class GazeboDrawingVisualizer(Node):
    """
    Visualizes drawings in Gazebo using spawned visual entities.
    """
    
    def __init__(self):
        super().__init__('gazebo_drawing_visualizer')
        
        self.get_logger().info("🎨 Gazebo Drawing Visualizer starting...")
        
        self.world_name = "rl_training_world"
        
        # Pen path tracking
        self.pen_points: List[np.ndarray] = []
        self.last_point = None
        self.min_distance = 0.005  # 5mm between points
        self.line_segment_id = 0
        self.spawned_segments: List[str] = []
        
        # Triangle outline tracking
        self.triangle_spawned = False
        self.triangle_segments: List[str] = []
        
        # Line color (bright green)
        self.line_color = (0.0, 1.0, 0.0, 1.0)  # RGBA
        self.target_color = (1.0, 0.5, 0.0, 0.8)  # Orange for target shape
        
        # Subscriber for pen position
        self.position_sub = self.create_subscription(
            Point, '/drawing/pen_position', self.position_callback, 10
        )
        
        # Subscriber for triangle waypoints
        self.shape_sub = self.create_subscription(
            Point, '/drawing/spawn_triangle', self.spawn_triangle_callback, 10
        )
        
        # Service to reset
        self.reset_srv = self.create_service(
            Empty, '/drawing/reset_line', self.reset_callback
        )
        
        # Publisher for triangle spawn trigger
        self.triangle_pub = self.create_publisher(Point, '/drawing/triangle_center', 10)
        
        # Spawn initial triangle after a delay
        self.create_timer(2.0, self.spawn_default_triangle)
        self.default_triangle_spawned = False
        
        self.get_logger().info("✅ Gazebo Drawing Visualizer ready!")
        self.get_logger().info("   Pen path: /drawing/pen_position")
        self.get_logger().info("   Reset:    /drawing/reset_line")
    
    def spawn_default_triangle(self):
        """Spawn default dense triangle waypoints on startup"""
        if self.default_triangle_spawned:
            return
        self.default_triangle_spawned = True
        
        # Use config values for waypoint spawning
        self.spawn_dense_waypoints(
            center=TRIANGLE_CENTER,
            size=SHAPE_SIZE,
            points_per_edge=POINTS_PER_EDGE  # From drawing_config.py
        )
    
    def spawn_dense_waypoints(self, center: Tuple[float, float, float], 
                               size: float = 0.15, points_per_edge: int = 10):
        """
        Spawn dense waypoint spheres along triangle edges.
        
        Instead of thick cylinder outline, spawns small spheres at each
        waypoint to show the individual targets.
        
        Args:
            center: (x, y, z) center of triangle
            size: Side length in meters
            points_per_edge: Number of waypoints per edge (default: 10)
        """
        total_points = points_per_edge * 3
        self.get_logger().info(f"🔺 Spawning {total_points + 1} waypoint targets (includes return)...")
        
        # Clear old targets
        self._delete_triangle()
        
        # Calculate triangle corners - START FROM TOP (apex)
        height = size * np.sqrt(3) / 2
        cx, cy, cz = center
        
        p1 = np.array([cx, cy, cz + 2*height/3])          # Top (apex/START)
        p2 = np.array([cx - size/2, cy, cz - height/3])   # Bottom-left
        p3 = np.array([cx + size/2, cy, cz - height/3])   # Bottom-right
        
        corners = [p1, p2, p3, p1]  # TOP→Bottom-left→Bottom-right→TOP
        
        # Generate and spawn waypoint spheres
        waypoint_id = 0
        sphere_radius = 0.005  # 5mm radius = 1cm diameter per waypoint
        
        for edge in range(3):
            start = corners[edge]
            end = corners[edge + 1]
            
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                point = start + t * (end - start)
                name = f"waypoint_{waypoint_id}"
                
                # Spawn small sphere at waypoint
                self._spawn_sphere(name, point, sphere_radius, self.target_color)
                self.triangle_segments.append(name)
                waypoint_id += 1
        
        # Spawn return-to-start waypoint (same as P1 but different color to indicate end)
        name = f"waypoint_{waypoint_id}"
        self._spawn_sphere(name, p1, sphere_radius, (0.0, 1.0, 0.0, 0.8))  # Green for end
        self.triangle_segments.append(name)
        waypoint_id += 1
        
        self.triangle_spawned = True
        self.get_logger().info(f"✅ Spawned {waypoint_id} waypoint targets (5mm spheres)")
    
    def _spawn_sphere(self, name: str, position: np.ndarray,
                      radius: float, color: Tuple[float, float, float, float]):
        """Spawn a sphere at given position."""
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        try:
            cmd = [
                'ign', 'service',
                '-s', f'/world/{self.world_name}/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}" '
                         f'pose: {{position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}}}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                self.spawned_segments.append(name)
                
        except Exception as e:
            self.get_logger().debug(f"Spawn sphere error: {e}")
    
    def spawn_triangle_shape(self, center: Tuple[float, float, float], size: float = 0.10):
        """
        Legacy: Spawn triangle outline with cylinders.
        Now redirects to spawn_dense_waypoints.
        """
        self.spawn_dense_waypoints(center, size, points_per_edge=10)
    
    def _spawn_line_segment(self, name: str, p1: np.ndarray, p2: np.ndarray, 
                            color: Tuple[float, float, float, float], 
                            radius: float = 0.002):
        """
        Spawn a line segment as a cylinder in Gazebo.
        
        Args:
            name: Unique name for the segment
            p1, p2: Start and end points
            color: RGBA color tuple
            radius: Line thickness (radius of cylinder)
        """
        # Calculate cylinder parameters
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 0.001:  # Too short
            return
        
        # Midpoint for position
        mid = (p1 + p2) / 2
        
        # Calculate rotation to align cylinder with line
        # Cylinder default axis is Z, we need to rotate to direction
        direction_norm = direction / length
        
        # Simplified: Use axis-angle to quaternion
        # Default Z axis
        z_axis = np.array([0, 0, 1])
        
        # Cross product for rotation axis
        axis = np.cross(z_axis, direction_norm)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Parallel or anti-parallel to Z
            if direction_norm[2] > 0:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                qw, qx, qy, qz = 0.0, 1.0, 0.0, 0.0
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
            
            # Axis-angle to quaternion
            qw = np.cos(angle / 2)
            qx = axis[0] * np.sin(angle / 2)
            qy = axis[1] * np.sin(angle / 2)
            qz = axis[2] * np.sin(angle / 2)
        
        # Create SDF for cylinder
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        # Spawn using ign service
        try:
            cmd = [
                'ign', 'service',
                '-s', f'/world/{self.world_name}/create',
                '--reqtype', 'ignition.msgs.EntityFactory',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}" '
                         f'pose: {{position: {{x: {mid[0]}, y: {mid[1]}, z: {mid[2]}}}, '
                         f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}}}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                self.spawned_segments.append(name)
            else:
                self.get_logger().debug(f"Spawn failed: {result.stderr}")
                
        except Exception as e:
            self.get_logger().debug(f"Spawn error: {e}")
    
    def _delete_entity(self, name: str):
        """Delete entity from Gazebo"""
        try:
            cmd = [
                'ign', 'service',
                '-s', f'/world/{self.world_name}/remove',
                '--reqtype', 'ignition.msgs.Entity',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '1000',
                '--req', f'name: "{name}" type: MODEL'
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        except Exception as e:
            self.get_logger().debug(f"Delete error: {e}")
    
    def _delete_triangle(self):
        """Delete triangle segments"""
        for name in self.triangle_segments:
            self._delete_entity(name)
        self.triangle_segments = []
        self.triangle_spawned = False
    
    def add_pen_point(self, position: np.ndarray):
        """Add point to pen path and draw line segment"""
        if self.last_point is not None:
            dist = np.linalg.norm(position - self.last_point)
            if dist < self.min_distance:
                return
            
            # Spawn line segment from last point to current (1mm radius = pen tip)
            segment_name = f"pen_line_{self.line_segment_id}"
            self._spawn_line_segment(
                segment_name, 
                self.last_point, 
                position, 
                self.line_color,
                0.001  # 1mm radius = 2mm diameter line
            )
            self.line_segment_id += 1
        
        self.pen_points.append(position.copy())
        self.last_point = position.copy()
    
    def reset(self):
        """Clear pen path"""
        self.get_logger().info("🔄 Resetting pen path...")
        
        # Delete all pen line segments
        for name in self.spawned_segments:
            if name.startswith("pen_line_"):
                self._delete_entity(name)
        
        self.spawned_segments = [s for s in self.spawned_segments if not s.startswith("pen_line_")]
        self.pen_points = []
        self.last_point = None
        self.line_segment_id = 0
        
        self.get_logger().info("✅ Pen path reset")
    
    def position_callback(self, msg: Point):
        """Handle pen position updates"""
        position = np.array([msg.x, msg.y, msg.z])
        self.add_pen_point(position)
    
    def spawn_triangle_callback(self, msg: Point):
        """Spawn triangle at specified center"""
        self.spawn_triangle_shape(center=(msg.x, msg.y, msg.z))
    
    def reset_callback(self, request, response):
        """Service callback to reset"""
        self.reset()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GazeboDrawingVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
