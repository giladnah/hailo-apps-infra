"""
Game board system for Freezing Phantoms.

Manages obstacles, boundaries, and collision detection for the game board.
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


@dataclass
class Obstacle:
    """Represents a game board obstacle."""

    obstacle_id: str
    obstacle_type: str  # "rectangle", "circle", "polygon"
    position: Tuple[float, float]  # (x, y) center or top-left
    size: Tuple[float, float]  # (width, height) or (radius, 0) for circles
    color: Tuple[int, int, int] = (255, 0, 0)  # RGB color
    rotation: float = 0.0  # Rotation in radians
    points: Optional[List[Tuple[float, float]]] = None  # For polygon obstacles


@dataclass
class Boundary:
    """Represents a game board boundary."""

    boundary_id: str
    boundary_type: str  # "wall", "finish_line", "start_line"
    start: Tuple[float, float]
    end: Tuple[float, float]
    color: Tuple[int, int, int] = (255, 255, 0)  # RGB color
    thickness: int = 5


class BoardSystem:
    """
    Manages game board with obstacles, boundaries, and collision detection.

    Attributes:
        width (int): Board width in pixels.
        height (int): Board height in pixels.
        obstacles (List[Obstacle]): List of board obstacles.
        boundaries (List[Boundary]): List of board boundaries.
    """

    def __init__(self, config: dict):
        """
        Initialize board system.

        Args:
            config (dict): Board configuration from game config.
        """
        self.width = config.get("width", 1920)
        self.height = config.get("height", 1080)
        self.obstacles = self._load_obstacles(config.get("obstacles", []))
        self.boundaries = self._load_boundaries(config.get("boundaries", []))

        hailo_logger.info(
            "Board system initialized: %dx%d, %d obstacles, %d boundaries",
            self.width, self.height, len(self.obstacles), len(self.boundaries)
        )

    def _load_obstacles(self, obstacles_config: List[dict]) -> List[Obstacle]:
        """Load obstacles from configuration."""
        obstacles = []

        for i, obs_config in enumerate(obstacles_config):
            obstacle_id = obs_config.get("id", f"obstacle_{i}")
            obstacle_type = obs_config.get("type", "rectangle")
            position = tuple(obs_config.get("position", [0, 0]))
            size = tuple(obs_config.get("size", [100, 100]))
            color = tuple(obs_config.get("color", [255, 0, 0]))
            rotation = obs_config.get("rotation", 0.0)
            points = obs_config.get("points", None)

            obstacle = Obstacle(
                obstacle_id=obstacle_id,
                obstacle_type=obstacle_type,
                position=position,
                size=size,
                color=color,
                rotation=rotation,
                points=points
            )
            obstacles.append(obstacle)

        return obstacles

    def _load_boundaries(self, boundaries_config: List[dict]) -> List[Boundary]:
        """Load boundaries from configuration."""
        boundaries = []

        for i, bound_config in enumerate(boundaries_config):
            boundary_id = bound_config.get("id", f"boundary_{i}")
            boundary_type = bound_config.get("type", "wall")
            start = tuple(bound_config.get("start", [0, 0]))
            end = tuple(bound_config.get("end", [100, 100]))
            color = tuple(bound_config.get("color", [255, 255, 0]))
            thickness = bound_config.get("thickness", 5)

            boundary = Boundary(
                boundary_id=boundary_id,
                boundary_type=boundary_type,
                start=start,
                end=end,
                color=color,
                thickness=thickness
            )
            boundaries.append(boundary)

        return boundaries

    def check_collision(self, position: Tuple[float, float], radius: float = 10.0) -> bool:
        """
        Check if a position collides with any obstacles or boundaries.

        Args:
            position (Tuple[float, float]): Position to check (x, y).
            radius (float): Collision radius around position.

        Returns:
            bool: True if collision detected, False otherwise.
        """
        x, y = position

        # Check boundary collisions
        for boundary in self.boundaries:
            if self._check_boundary_collision(position, radius, boundary):
                hailo_logger.debug("Collision with boundary: %s", boundary.boundary_id)
                return True

        # Check obstacle collisions
        for obstacle in self.obstacles:
            if self._check_obstacle_collision(position, radius, obstacle):
                hailo_logger.debug("Collision with obstacle: %s", obstacle.obstacle_id)
                return True

        return False

    def _check_boundary_collision(
        self,
        position: Tuple[float, float],
        radius: float,
        boundary: Boundary
    ) -> bool:
        """Check collision with a specific boundary."""
        x, y = position
        x1, y1 = boundary.start
        x2, y2 = boundary.end

        # Calculate distance from point to line segment
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            # Point to point distance
            dist = math.sqrt(A * A + B * B)
        else:
            # Project point onto line segment
            param = dot / len_sq
            if param < 0:
                # Closest to start point
                dist = math.sqrt(A * A + B * B)
            elif param > 1:
                # Closest to end point
                dist = math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            else:
                # Closest to line segment
                xx = x1 + param * C
                yy = y1 + param * D
                dist = math.sqrt((x - xx) ** 2 + (y - yy) ** 2)

        return dist <= radius + boundary.thickness / 2

    def _check_obstacle_collision(
        self,
        position: Tuple[float, float],
        radius: float,
        obstacle: Obstacle
    ) -> bool:
        """Check collision with a specific obstacle."""
        pos_x, pos_y = position
        obs_x, obs_y = obstacle.position

        if obstacle.obstacle_type == "rectangle":
            width, height = obstacle.size
            # Check if point is inside rectangle
            return (abs(pos_x - obs_x) <= width / 2 + radius and
                    abs(pos_y - obs_y) <= height / 2 + radius)

        elif obstacle.obstacle_type == "circle":
            radius_obs = obstacle.size[0]
            # Check if point is inside circle
            dist = math.sqrt((pos_x - obs_x) ** 2 + (pos_y - obs_y) ** 2)
            return dist <= radius_obs + radius

        elif obstacle.obstacle_type == "polygon" and obstacle.points:
            # Check if point is inside polygon using ray casting
            return self._point_in_polygon(position, obstacle.points, radius)

        return False

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
        radius: float  # noqa: ARG002
    ) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_valid_moves(
        self,
        current_pos: Tuple[float, float],
        move_distance: float = 20.0
    ) -> List[Tuple[float, float]]:
        """
        Get valid movement directions from current position.

        Args:
            current_pos (Tuple[float, float]): Current position (x, y).
            move_distance (float): Distance to check for valid moves.

        Returns:
            List[Tuple[float, float]]: List of valid move positions.
        """
        valid_moves = []
        directions = [
            (0, -move_distance),      # Up
            (move_distance, 0),       # Right
            (0, move_distance),       # Down
            (-move_distance, 0),      # Left
            (move_distance * 0.7, -move_distance * 0.7),  # Up-right
            (move_distance * 0.7, move_distance * 0.7),   # Down-right
            (-move_distance * 0.7, move_distance * 0.7),  # Down-left
            (-move_distance * 0.7, -move_distance * 0.7), # Up-left
        ]

        for dx, dy in directions:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Check if position is within board bounds
            if (0 <= new_pos[0] <= self.width and
                0 <= new_pos[1] <= self.height):

                # Check for collisions
                if not self.check_collision(new_pos):
                    valid_moves.append(new_pos)

        return valid_moves

    def is_position_valid(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position is valid (within bounds and no collisions).

        Args:
            position (Tuple[float, float]): Position to check (x, y).

        Returns:
            bool: True if position is valid, False otherwise.
        """
        x, y = position

        # Check bounds
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False

        # Check collisions
        return not self.check_collision(position)

    def get_obstacles_in_region(
        self,
        center: Tuple[float, float],
        radius: float
    ) -> List[Obstacle]:
        """
        Get obstacles within a region.

        Args:
            center (Tuple[float, float]): Center of region (x, y).
            radius (float): Region radius.

        Returns:
            List[Obstacle]: Obstacles within the region.
        """
        obstacles_in_region = []
        cx, cy = center

        for obstacle in self.obstacles:
            obs_x, obs_y = obstacle.position

            # Calculate distance to obstacle center
            if obstacle.obstacle_type == "circle":
                obs_radius = obstacle.size[0]
                dist = math.sqrt((cx - obs_x) ** 2 + (cy - obs_y) ** 2)
                if dist <= radius + obs_radius:
                    obstacles_in_region.append(obstacle)

            elif obstacle.obstacle_type == "rectangle":
                width, height = obstacle.size
                # Check if regions overlap
                if (abs(cx - obs_x) <= radius + width / 2 and
                    abs(cy - obs_y) <= radius + height / 2):
                    obstacles_in_region.append(obstacle)

        return obstacles_in_region

    def render_debug_overlay(self, canvas: np.ndarray) -> np.ndarray:
        """
        Render debug overlay showing obstacles and boundaries.

        Args:
            canvas (np.ndarray): Canvas to render on.

        Returns:
            np.ndarray: Canvas with debug overlay.
        """
        debug_canvas = canvas.copy()

        # Render obstacles
        for obstacle in self.obstacles:
            if obstacle.obstacle_type == "rectangle":
                width, height = obstacle.size
                x1 = int(obstacle.position[0] - width / 2)
                y1 = int(obstacle.position[1] - height / 2)
                x2 = int(obstacle.position[0] + width / 2)
                y2 = int(obstacle.position[1] + height / 2)

                cv2.rectangle(debug_canvas, (x1, y1), (x2, y2), obstacle.color, 2)

            elif obstacle.obstacle_type == "circle":
                radius = int(obstacle.size[0])
                center = (int(obstacle.position[0]), int(obstacle.position[1]))
                cv2.circle(debug_canvas, center, radius, obstacle.color, 2)

        # Render boundaries
        for boundary in self.boundaries:
            start = (int(boundary.start[0]), int(boundary.start[1]))
            end = (int(boundary.end[0]), int(boundary.end[1]))
            cv2.line(debug_canvas, start, end, boundary.color, boundary.thickness)

        return debug_canvas
