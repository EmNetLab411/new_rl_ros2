#!/usr/bin/env python3
"""
Shape Generator for Drawing Tasks

Generates waypoint sequences for various polygon shapes.
Used by the drawing RL environment to define drawing objectives.

Shapes are generated on a flat Y-plane (vertical drawing surface).
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Shape:
    """A drawable shape defined by waypoints"""
    name: str
    waypoints: np.ndarray  # (N, 3) array of [x, y, z] coordinates
    closed: bool = True    # Whether to return to start point
    
    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)
    
    def get_waypoint(self, index: int) -> np.ndarray:
        """Get waypoint at index (wraps around if closed)"""
        if self.closed:
            return self.waypoints[index % len(self.waypoints)]
        else:
            return self.waypoints[min(index, len(self.waypoints) - 1)]


class ShapeGenerator:
    """
    Generates drawable shapes as waypoint sequences.
    
    All shapes are generated on a flat Y-plane (Y = constant).
    This simulates a vertical drawing surface in front of the robot.
    """
    
    def __init__(self, 
                 y_plane: float = 0.20,
                 x_center: float = 0.0,
                 z_center: float = 0.25,
                 default_size: float = 0.10):
        """
        Initialize shape generator.
        
        Args:
            y_plane: Y coordinate of drawing surface (constant)
            x_center: X center of shapes
            z_center: Z center of shapes  
            default_size: Default size for shapes (meters)
        """
        self.y_plane = y_plane
        self.x_center = x_center
        self.z_center = z_center
        self.default_size = default_size
    
    def equilateral_triangle(self, 
                             size: float = None,
                             center: Tuple[float, float] = None,
                             points_per_edge: int = 1) -> Shape:
        """
        Generate equilateral triangle waypoints.
        
        Triangle is oriented with apex at top, base at bottom.
        
        Args:
            size: Side length in meters (default: default_size)
            center: (x, z) center position (default: use generator defaults)
            points_per_edge: Points per edge (1=corners only, 3=10 total, etc.)
            
        Returns:
            Shape object with (points_per_edge * 3 + 1) waypoints
            - points_per_edge=1: 4 waypoints (3 corners + 1 return)
            - points_per_edge=3: 10 waypoints (9 + 1 return)
        """
        size = size or self.default_size
        cx = center[0] if center else self.x_center
        cz = center[1] if center else self.z_center
        
        # Equilateral triangle geometry
        # Height = size * sqrt(3) / 2
        height = size * np.sqrt(3) / 2
        
        # Vertices - START FROM TOP (apex)
        #     P1 (apex/start)
        #    /  \
        #   /    \
        #  P2----P3
        
        p1 = np.array([cx,          self.y_plane, cz + 2*height/3])  # Top (apex/START)
        p2 = np.array([cx - size/2, self.y_plane, cz - height/3])    # Bottom-left
        p3 = np.array([cx + size/2, self.y_plane, cz - height/3])    # Bottom-right
        
        corners = [p1, p2, p3, p1]  # TOP→Bottom-left→Bottom-right→TOP
        
        # Generate waypoints
        if points_per_edge == 1:
            # Original behavior: just corners + return
            waypoints = np.array([p1, p2, p3, p1])
        else:
            # Interpolate points along each edge
            waypoints = []
            for i in range(3):  # 3 edges
                start = corners[i]
                end = corners[i + 1]
                # Exclude endpoint to avoid duplicates
                for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                    point = start + t * (end - start)
                    waypoints.append(point)
            # Add return to start
            waypoints.append(p1)
            waypoints = np.array(waypoints)
        
        total_wp = len(waypoints)
        return Shape(
            name=f"equilateral_triangle_{total_wp}wp",
            waypoints=waypoints,
            closed=True
        )
    
    def dense_triangle(self, 
                       size: float = None,
                       center: Tuple[float, float] = None,
                       points_per_edge: int = 10) -> Shape:
        """
        Generate equilateral triangle with dense waypoints along edges.
        
        This creates a continuous trajectory for smooth drawing,
        with many intermediate points between corners.
        
        Args:
            size: Side length in meters
            center: (x, z) center position
            points_per_edge: Number of waypoints per edge (default: 10)
            
        Returns:
            Shape with (points_per_edge * 3) waypoints for smooth trajectory
        """
        size = size or self.default_size
        cx = center[0] if center else self.x_center
        cz = center[1] if center else self.z_center
        
        # Triangle corners
        height = size * np.sqrt(3) / 2
        p1 = np.array([cx - size/2, self.y_plane, cz - height/3])  # Bottom-left
        p2 = np.array([cx,          self.y_plane, cz + 2*height/3])  # Top (apex)
        p3 = np.array([cx + size/2, self.y_plane, cz - height/3])  # Bottom-right
        
        corners = [p1, p2, p3, p1]  # Close the triangle
        
        # Generate dense points along each edge
        waypoints = []
        for i in range(3):  # 3 edges
            start = corners[i]
            end = corners[i + 1]
            
            # Interpolate points along edge (excluding end point to avoid duplicates)
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                point = start + t * (end - start)
                waypoints.append(point)
        
        # Add return to start point to complete the shape
        waypoints.append(p1)
        
        waypoints = np.array(waypoints)
        
        return Shape(
            name=f"dense_triangle_{points_per_edge}pp_edge",
            waypoints=waypoints,
            closed=True
        )
    
    def square(self, size: float = None, center: Tuple[float, float] = None) -> Shape:
        """Generate square waypoints (4 corners + return to start)"""
        size = size or self.default_size
        cx = center[0] if center else self.x_center
        cz = center[1] if center else self.z_center
        
        half = size / 2
        
        #  P4----P3
        #  |      |
        #  |      |
        #  P1----P2
        
        p1 = np.array([cx - half, self.y_plane, cz - half])  # Bottom-left
        p2 = np.array([cx + half, self.y_plane, cz - half])  # Bottom-right
        p3 = np.array([cx + half, self.y_plane, cz + half])  # Top-right
        p4 = np.array([cx - half, self.y_plane, cz + half])  # Top-left
        
        waypoints = np.array([p1, p2, p3, p4, p1])
        
        return Shape(name="square", waypoints=waypoints, closed=True)
    
    def line(self, 
             start: Tuple[float, float] = None,
             end: Tuple[float, float] = None,
             length: float = None) -> Shape:
        """
        Generate simple line (2 waypoints).
        
        Args:
            start: (x, z) start position
            end: (x, z) end position  
            length: If start/end not given, create horizontal line of this length
        """
        if start and end:
            p1 = np.array([start[0], self.y_plane, start[1]])
            p2 = np.array([end[0], self.y_plane, end[1]])
        else:
            length = length or self.default_size
            p1 = np.array([self.x_center - length/2, self.y_plane, self.z_center])
            p2 = np.array([self.x_center + length/2, self.y_plane, self.z_center])
        
        return Shape(name="line", waypoints=np.array([p1, p2]), closed=False)
    
    def polygon(self, n_sides: int, size: float = None) -> Shape:
        """Generate regular polygon with n sides"""
        size = size or self.default_size
        radius = size / (2 * np.sin(np.pi / n_sides))  # Circumradius
        
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_sides, endpoint=False)
        
        waypoints = []
        for angle in angles:
            x = self.x_center + radius * np.cos(angle)
            z = self.z_center + radius * np.sin(angle)
            waypoints.append([x, self.y_plane, z])
        
        # Close the shape
        waypoints.append(waypoints[0])
        
        return Shape(
            name=f"polygon_{n_sides}",
            waypoints=np.array(waypoints),
            closed=True
        )
    
    def random_triangle(self, 
                        workspace_x: Tuple[float, float] = (-0.15, 0.15),
                        workspace_z: Tuple[float, float] = (0.15, 0.35),
                        min_size: float = 0.05,
                        max_size: float = 0.15) -> Shape:
        """
        Generate random equilateral triangle within workspace.
        
        Args:
            workspace_x: (min, max) X bounds
            workspace_z: (min, max) Z bounds
            min_size: Minimum triangle size
            max_size: Maximum triangle size
        """
        size = np.random.uniform(min_size, max_size)
        height = size * np.sqrt(3) / 2
        
        # Random center that keeps triangle in bounds
        margin_x = size / 2 + 0.01
        margin_z = height / 2 + 0.01
        
        cx = np.random.uniform(workspace_x[0] + margin_x, workspace_x[1] - margin_x)
        cz = np.random.uniform(workspace_z[0] + margin_z, workspace_z[1] - margin_z)
        
        return self.equilateral_triangle(size=size, center=(cx, cz))


def test_shape_generator():
    """Test shape generation"""
    print("=" * 60)
    print("Testing Shape Generator")
    print("=" * 60)
    
    gen = ShapeGenerator(y_plane=0.20, x_center=0.0, z_center=0.25)
    
    # Test equilateral triangle
    triangle = gen.equilateral_triangle(size=0.10)
    print(f"\n{triangle.name}:")
    print(f"  Waypoints: {triangle.num_waypoints}")
    for i, wp in enumerate(triangle.waypoints):
        print(f"  P{i+1}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
    
    # Verify equilateral
    d12 = np.linalg.norm(triangle.waypoints[1] - triangle.waypoints[0])
    d23 = np.linalg.norm(triangle.waypoints[2] - triangle.waypoints[1])
    d31 = np.linalg.norm(triangle.waypoints[0] - triangle.waypoints[2])
    print(f"  Side lengths: {d12:.4f}, {d23:.4f}, {d31:.4f}")
    print(f"  Equilateral: {np.allclose([d12, d23, d31], d12)}")
    
    # Test square
    square = gen.square(size=0.08)
    print(f"\n{square.name}:")
    print(f"  Waypoints: {square.num_waypoints}")
    
    # Test random triangle
    np.random.seed(42)
    rand_tri = gen.random_triangle()
    print(f"\nrandom_triangle:")
    print(f"  Waypoints: {rand_tri.num_waypoints}")
    for i, wp in enumerate(rand_tri.waypoints):
        print(f"  P{i+1}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")
    
    print("\n" + "=" * 60)
    print("Shape generator OK!")


if __name__ == '__main__':
    test_shape_generator()
