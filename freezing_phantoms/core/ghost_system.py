"""
Ghost system for Freezing Phantoms game.

Implements moving ghosts with different AI behaviors.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.core.board_system import BoardSystem

hailo_logger = get_logger(__name__)


class GhostType(Enum):
    """Ghost behavior types."""
    PATROL = "patrol"      # Moves in predefined pattern
    CHASE = "chase"        # Follows players when in range
    GUARD = "guard"        # Blocks specific areas
    WANDER = "wander"      # Random movement


@dataclass
class GhostState:
    """Current state of a ghost."""
    position: Tuple[float, float]
    target_position: Optional[Tuple[float, float]]
    velocity: Tuple[float, float]
    last_update_time: float
    current_patrol_index: int = 0
    is_chasing: bool = False
    chase_target_id: Optional[int] = None


class Ghost:
    """
    Individual ghost with AI behavior.

    Attributes:
        ghost_id (str): Unique ghost identifier.
        ghost_type (GhostType): Type of ghost behavior.
        start_position (Tuple[float, float]): Initial position.
        speed (float): Movement speed in pixels per second.
        detection_range (float): Range for detecting players.
        patrol_points (List[Tuple[float, float]]): Patrol waypoints.
        color (Tuple[int, int, int]): Ghost color (RGB).
        size (float): Ghost size for rendering.
    """

    def __init__(
        self,
        ghost_id: str,
        ghost_type: GhostType,
        start_position: Tuple[float, float],
        speed: float = 50.0,
        detection_range: float = 200.0,
        patrol_points: Optional[List[Tuple[float, float]]] = None,
        color: Tuple[int, int, int] = (0, 0, 255),
        size: float = 30.0
    ):
        """
        Initialize ghost.

        Args:
            ghost_id (str): Unique ghost identifier.
            ghost_type (GhostType): Type of ghost behavior.
            start_position (Tuple[float, float]): Initial position (x, y).
            speed (float): Movement speed in pixels per second.
            detection_range (float): Range for detecting players.
            patrol_points (List[Tuple[float, float]], optional): Patrol waypoints.
            color (Tuple[int, int, int]): Ghost color (RGB).
            size (float): Ghost size for rendering.
        """
        self.ghost_id = ghost_id
        self.ghost_type = ghost_type
        self.start_position = start_position
        self.speed = speed
        self.detection_range = detection_range
        self.patrol_points = patrol_points or []
        self.color = color
        self.size = size

        # Initialize state
        self.state = GhostState(
            position=start_position,
            target_position=None,
            velocity=(0.0, 0.0),
            last_update_time=time.time()
        )

        hailo_logger.debug(
            "Ghost created: %s, type=%s, pos=(%.1f, %.1f)",
            ghost_id, ghost_type.value, start_position[0], start_position[1]
        )

    def update(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        board: BoardSystem,
        dt: float
    ) -> None:
        """
        Update ghost position and behavior.

        Args:
            player_positions (Dict[int, Tuple[float, float]]): Current player positions.
            board (BoardSystem): Game board for collision detection.
            dt (float): Delta time since last update.
        """
        current_time = time.time()

        # Update based on ghost type
        if self.ghost_type == GhostType.PATROL:
            self._update_patrol_behavior(board, dt)
        elif self.ghost_type == GhostType.CHASE:
            self._update_chase_behavior(player_positions, board, dt)
        elif self.ghost_type == GhostType.GUARD:
            self._update_guard_behavior(player_positions, board, dt)
        elif self.ghost_type == GhostType.WANDER:
            self._update_wander_behavior(board, dt)

        # Update position based on velocity
        self._update_position(board, dt)

        self.state.last_update_time = current_time

    def _update_patrol_behavior(self, board: BoardSystem, dt: float) -> None:  # noqa: ARG002
        """Update patrol ghost behavior."""
        if not self.patrol_points:
            return

        # If no target or reached target, get next patrol point
        if (self.state.target_position is None or
            self._distance_to_target() < 10.0):

            self.state.current_patrol_index = (self.state.current_patrol_index + 1) % len(self.patrol_points)
            self.state.target_position = self.patrol_points[self.state.current_patrol_index]

        # Move towards target
        self._move_towards_target()

    def _update_chase_behavior(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        board: BoardSystem,
        dt: float
    ) -> None:
        """Update chase ghost behavior."""
        # Find closest player within detection range
        closest_player = None
        closest_distance = float('inf')

        for player_id, player_pos in player_positions.items():
            distance = self._distance_to_position(player_pos)
            if distance < self.detection_range and distance < closest_distance:
                closest_player = player_id
                closest_distance = distance

        if closest_player is not None:
            # Chase the closest player
            self.state.is_chasing = True
            self.state.chase_target_id = closest_player
            self.state.target_position = player_positions[closest_player]
            self._move_towards_target()
        else:
            # No players in range, return to patrol or wander
            self.state.is_chasing = False
            self.state.chase_target_id = None
            if self.patrol_points:
                self._update_patrol_behavior(board, dt)
            else:
                self._update_wander_behavior(board, dt)

    def _update_guard_behavior(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        board: BoardSystem,
        dt: float
    ) -> None:
        """Update guard ghost behavior."""
        # Guard ghosts stay near their start position
        # They only move to intercept players that get too close

        # Find players within guard range
        guard_range = self.detection_range * 1.5
        threatening_players = []

        for player_id, player_pos in player_positions.items():
            distance = self._distance_to_position(player_pos)
            if distance < guard_range:
                threatening_players.append((player_id, player_pos, distance))

        if threatening_players:
            # Move towards the closest threatening player
            threatening_players.sort(key=lambda x: x[2])
            closest_player_id, closest_player_pos, _ = threatening_players[0]

            self.state.target_position = closest_player_pos
            self.state.is_chasing = True
            self.state.chase_target_id = closest_player_id
            self._move_towards_target()
        else:
            # Return to guard position
            self.state.is_chasing = False
            self.state.chase_target_id = None
            self.state.target_position = self.start_position
            self._move_towards_target()

    def _update_wander_behavior(self, board: BoardSystem, dt: float) -> None:  # noqa: ARG002
        """Update wander ghost behavior."""
        # If no target or reached target, pick a random direction
        if (self.state.target_position is None or
            self._distance_to_target() < 20.0):

            # Pick a random position within board bounds
            margin = 100
            x = random.uniform(margin, board.width - margin)
            y = random.uniform(margin, board.height - margin)
            self.state.target_position = (x, y)

        self._move_towards_target()

    def _move_towards_target(self) -> None:
        """Move towards the current target position."""
        if self.state.target_position is None:
            return

        target_x, target_y = self.state.target_position
        current_x, current_y = self.state.position

        # Calculate direction vector
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 0:
            # Normalize direction and apply speed
            self.state.velocity = (
                (dx / distance) * self.speed,
                (dy / distance) * self.speed
            )
        else:
            self.state.velocity = (0.0, 0.0)

    def _update_position(self, board: BoardSystem, dt: float) -> None:  # noqa: ARG002
        """Update ghost position based on velocity."""
        if not self.state.velocity or (self.state.velocity[0] == 0 and self.state.velocity[1] == 0):
            return

        # Calculate new position
        vx, vy = self.state.velocity
        new_x = self.state.position[0] + vx * dt
        new_y = self.state.position[1] + vy * dt

        new_position = (new_x, new_y)

        # Check for collisions with board
        if board.is_position_valid(new_position):
            self.state.position = new_position
        else:
            # Hit an obstacle, try to find a new direction
            self._handle_collision(board)

    def _handle_collision(self, board: BoardSystem) -> None:
        """Handle collision with board obstacles."""
        # Try to find a valid direction
        current_pos = self.state.position
        valid_moves = board.get_valid_moves(current_pos, 30.0)

        if valid_moves:
            # Pick a random valid direction
            new_target = random.choice(valid_moves)
            self.state.target_position = new_target
            self._move_towards_target()
        else:
            # No valid moves, stop
            self.state.velocity = (0.0, 0.0)
            self.state.target_position = None

    def _distance_to_target(self) -> float:
        """Calculate distance to current target."""
        if self.state.target_position is None:
            return float('inf')

        return self._distance_to_position(self.state.target_position)

    def _distance_to_position(self, position: Tuple[float, float]) -> float:
        """Calculate distance to a position."""
        dx = self.state.position[0] - position[0]
        dy = self.state.position[1] - position[1]
        return math.sqrt(dx * dx + dy * dy)

    def render(self, canvas: np.ndarray, homography_transform_func) -> np.ndarray:
        """
        Render ghost on canvas.

        Args:
            canvas (np.ndarray): Canvas to render on.
            homography_transform_func: Function to transform coordinates.

        Returns:
            np.ndarray: Canvas with ghost rendered.
        """
        # Transform ghost position to projector coordinates
        projector_pos = homography_transform_func(self.state.position[0], self.state.position[1])
        if projector_pos is None:
            return canvas

        x, y = projector_pos
        size = int(self.size)

        # Render ghost as a circle with a simple face
        cv2.circle(canvas, (int(x), int(y)), size, self.color, -1)

        # Add simple face
        eye_size = size // 4
        cv2.circle(canvas, (int(x - size//3), int(y - size//3)), eye_size, (255, 255, 255), -1)
        cv2.circle(canvas, (int(x + size//3), int(y - size//3)), eye_size, (255, 255, 255), -1)

        # Add pupil
        pupil_size = eye_size // 2
        cv2.circle(canvas, (int(x - size//3), int(y - size//3)), pupil_size, (0, 0, 0), -1)
        cv2.circle(canvas, (int(x + size//3), int(y - size//3)), pupil_size, (0, 0, 0), -1)

        # Add mouth
        mouth_y = y + size // 3
        cv2.ellipse(canvas, (int(x), int(mouth_y)), (size//3, size//6), 0, 0, 180, (255, 255, 255), 2)

        return canvas


class GhostManager:
    """
    Manages all ghosts in the game.

    Attributes:
        ghosts (List[Ghost]): List of all ghosts.
        board (BoardSystem): Game board for collision detection.
    """

    def __init__(self, config: dict, board: BoardSystem):
        """
        Initialize ghost manager.

        Args:
            config (dict): Ghost configuration from game config.
            board (BoardSystem): Game board for collision detection.
        """
        self.ghosts = []
        self.board = board

        # Load ghosts from configuration
        ghosts_config = config.get("ghosts", [])
        for i, ghost_config in enumerate(ghosts_config):
            ghost_id = ghost_config.get("id", f"ghost_{i}")
            ghost_type = GhostType(ghost_config.get("type", "patrol"))
            start_pos = tuple(ghost_config.get("start_pos", [100, 100]))
            speed = ghost_config.get("speed", 50.0)
            detection_range = ghost_config.get("detection_range", 200.0)
            patrol_points = ghost_config.get("patrol_points", [])
            color = tuple(ghost_config.get("color", [0, 0, 255]))
            size = ghost_config.get("size", 30.0)

            ghost = Ghost(
                ghost_id=ghost_id,
                ghost_type=ghost_type,
                start_position=start_pos,
                speed=speed,
                detection_range=detection_range,
                patrol_points=patrol_points,
                color=color,
                size=size
            )

            self.ghosts.append(ghost)

        hailo_logger.info("Ghost manager initialized with %d ghosts", len(self.ghosts))

    def update(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        dt: float
    ) -> None:
        """
        Update all ghosts.

        Args:
            player_positions (Dict[int, Tuple[float, float]]): Current player positions.
            dt (float): Delta time since last update.
        """
        for ghost in self.ghosts:
            ghost.update(player_positions, self.board, dt)

    def render(self, canvas: np.ndarray, homography_transform_func) -> np.ndarray:
        """
        Render all ghosts.

        Args:
            canvas (np.ndarray): Canvas to render on.
            homography_transform_func: Function to transform coordinates.

        Returns:
            np.ndarray: Canvas with all ghosts rendered.
        """
        for ghost in self.ghosts:
            canvas = ghost.render(canvas, homography_transform_func)

        return canvas

    def get_ghosts_in_range(
        self,
        position: Tuple[float, float],
        range_distance: float
    ) -> List[Ghost]:
        """
        Get ghosts within a range of a position.

        Args:
            position (Tuple[float, float]): Position to check around.
            range_distance (float): Range distance.

        Returns:
            List[Ghost]: Ghosts within range.
        """
        ghosts_in_range = []

        for ghost in self.ghosts:
            distance = math.sqrt(
                (position[0] - ghost.state.position[0]) ** 2 +
                (position[1] - ghost.state.position[1]) ** 2
            )

            if distance <= range_distance:
                ghosts_in_range.append(ghost)

        return ghosts_in_range
