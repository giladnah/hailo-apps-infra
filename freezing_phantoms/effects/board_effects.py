"""
Board effects for Freezing Phantoms game.

Renders game board elements including obstacles, boundaries, and debug overlays.
"""

import cv2
import numpy as np
from typing import List, Optional

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.core.board_system import BoardSystem
from freezing_phantoms.core.ghost_system import GhostManager

hailo_logger = get_logger(__name__)


class BoardEffects:
    """
    Renders board elements and effects.

    Attributes:
        board (BoardSystem): Game board system.
        ghost_manager (GhostManager): Ghost management system.
        debug_mode (bool): Whether to show debug overlays.
    """

    def __init__(self, board: BoardSystem, ghost_manager: GhostManager, debug_mode: bool = False):
        """
        Initialize board effects.

        Args:
            board (BoardSystem): Game board system.
            ghost_manager (GhostManager): Ghost management system.
            debug_mode (bool): Whether to show debug overlays.
        """
        self.board = board
        self.ghost_manager = ghost_manager
        self.debug_mode = debug_mode

        hailo_logger.debug("BoardEffects initialized with debug_mode=%s", debug_mode)

    def render_board(
        self,
        canvas: np.ndarray,
        homography_transform_func,
        player_data: Optional[dict] = None,
        frame_width: int = 1280,  # noqa: ARG002
        frame_height: int = 720  # noqa: ARG002
    ) -> np.ndarray:
        """
        Render complete board with all elements.

        Args:
            canvas (np.ndarray): Canvas to render on.
            homography_transform_func: Function to transform camera to projector coordinates.
            player_data (Optional[Dict]): Current player data.
            frame_width (int): Camera frame width.
            frame_height (int): Camera frame height.

        Returns:
            np.ndarray: Canvas with board elements rendered.
        """
        # Render obstacles
        canvas = self._render_obstacles(canvas, homography_transform_func)

        # Render boundaries
        canvas = self._render_boundaries(canvas, homography_transform_func)

        # Render ghosts
        canvas = self.ghost_manager.render(canvas, homography_transform_func)

        # Render debug overlays if enabled
        if self.debug_mode:
            canvas = self._render_debug_overlay(canvas, homography_transform_func, player_data)

        return canvas

    def _render_obstacles(self, canvas: np.ndarray, homography_transform_func) -> np.ndarray:
        """Render board obstacles."""
        for obstacle in self.board.obstacles:
            canvas = self._render_obstacle(canvas, obstacle, homography_transform_func)

        return canvas

    def _render_obstacle(self, canvas: np.ndarray, obstacle, homography_transform_func) -> np.ndarray:
        """Render a single obstacle."""
        # Transform obstacle position to projector coordinates
        projector_pos = homography_transform_func(obstacle.position[0], obstacle.position[1])
        if projector_pos is None:
            return canvas

        x, y = projector_pos

        if obstacle.obstacle_type == "rectangle":
            width, height = obstacle.size
            # Scale size based on homography (approximate)
            scaled_width = int(width * 0.8)  # Adjust scaling as needed
            scaled_height = int(height * 0.8)

            x1 = int(x - scaled_width // 2)
            y1 = int(y - scaled_height // 2)
            x2 = int(x + scaled_width // 2)
            y2 = int(y + scaled_height // 2)

            # Draw filled rectangle with border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), obstacle.color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)

        elif obstacle.obstacle_type == "circle":
            radius = int(obstacle.size[0] * 0.8)  # Scale radius
            center = (int(x), int(y))

            # Draw filled circle with border
            cv2.circle(canvas, center, radius, obstacle.color, -1)
            cv2.circle(canvas, center, radius, (0, 0, 0), 2)

        elif obstacle.obstacle_type == "polygon" and obstacle.points:
            # Transform polygon points
            polygon_points = []
            for point in obstacle.points:
                transformed_point = homography_transform_func(point)
                if transformed_point is not None:
                    polygon_points.append((int(transformed_point[0]), int(transformed_point[1])))

            if len(polygon_points) >= 3:
                # Draw filled polygon
                cv2.fillPoly(canvas, [np.array(polygon_points)], obstacle.color)
                cv2.polylines(canvas, [np.array(polygon_points)], True, (0, 0, 0), 2)

        return canvas

    def _render_boundaries(self, canvas: np.ndarray, homography_transform_func) -> np.ndarray:
        """Render board boundaries."""
        for boundary in self.board.boundaries:
            canvas = self._render_boundary(canvas, boundary, homography_transform_func)

        return canvas

    def _render_boundary(self, canvas: np.ndarray, boundary, homography_transform_func) -> np.ndarray:
        """Render a single boundary."""
        # Transform boundary points to projector coordinates
        start_pos = homography_transform_func(boundary.start[0], boundary.start[1])
        end_pos = homography_transform_func(boundary.end[0], boundary.end[1])

        if start_pos is None or end_pos is None:
            return canvas

        start = (int(start_pos[0]), int(start_pos[1]))
        end = (int(end_pos[0]), int(end_pos[1]))

        # Draw boundary line
        cv2.line(canvas, start, end, boundary.color, boundary.thickness)

        # Add boundary type label
        if boundary.boundary_type == "finish_line":
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            cv2.putText(
                canvas, "FINISH", (mid_x - 30, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, boundary.color, 2
            )
        elif boundary.boundary_type == "start_line":
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            cv2.putText(
                canvas, "START", (mid_x - 25, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, boundary.color, 2
            )

        return canvas

    def _render_debug_overlay(
        self,
        canvas: np.ndarray,
        homography_transform_func,
        player_data: Optional[dict]
    ) -> np.ndarray:
        """Render debug overlay with leg positions and board elements."""
        debug_canvas = canvas.copy()

        # Render leg positions if player data is available
        if player_data and "players" in player_data:
            debug_canvas = self._render_leg_positions(debug_canvas, player_data["players"], homography_transform_func)

        # Render collision zones
        debug_canvas = self._render_collision_zones(debug_canvas, homography_transform_func)

        # Render ghost paths
        debug_canvas = self._render_ghost_paths(debug_canvas, homography_transform_func)

        # Add debug info text
        debug_canvas = self._render_debug_info(debug_canvas, player_data)

        return debug_canvas

    def _render_leg_positions(
        self,
        canvas: np.ndarray,
        players: List,
        homography_transform_func
    ) -> np.ndarray:
        """Render leg position markers for all players."""
        for player in players:
            if hasattr(player, 'leg_position') and player.leg_position is not None:
                # Transform leg position to projector coordinates
                projector_pos = homography_transform_func(player.leg_position[0], player.leg_position[1])
                if projector_pos is not None:
                    x, y = projector_pos

                    # Draw leg position marker
                    cv2.circle(canvas, (int(x), int(y)), 15, (0, 255, 0), -1)
                    cv2.circle(canvas, (int(x), int(y)), 15, (0, 0, 0), 2)

                    # Add player ID
                    cv2.putText(
                        canvas, f"P{player.track_id}", (int(x) - 10, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )

                    # Add confidence score
                    if hasattr(player, 'leg_confidence'):
                        conf_text = f"{player.leg_confidence:.2f}"
                        cv2.putText(
                            canvas, conf_text, (int(x) - 15, int(y) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                        )

        return canvas

    def _render_collision_zones(self, canvas: np.ndarray, homography_transform_func) -> np.ndarray:
        """Render collision detection zones around obstacles."""
        for obstacle in self.board.obstacles:
            # Transform obstacle position
            projector_pos = homography_transform_func(obstacle.position[0], obstacle.position[1])
            if projector_pos is None:
                continue

            x, y = projector_pos

            if obstacle.obstacle_type == "rectangle":
                width, height = obstacle.size
                scaled_width = int(width * 0.8)
                scaled_height = int(height * 0.8)

                # Draw collision zone (slightly larger than obstacle)
                x1 = int(x - scaled_width // 2 - 10)
                y1 = int(y - scaled_height // 2 - 10)
                x2 = int(x + scaled_width // 2 + 10)
                y2 = int(y + scaled_height // 2 + 10)

                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1)

            elif obstacle.obstacle_type == "circle":
                radius = int(obstacle.size[0] * 0.8)
                center = (int(x), int(y))

                # Draw collision zone
                cv2.circle(canvas, center, radius + 10, (255, 255, 0), 1)

        return canvas

    def _render_ghost_paths(self, canvas: np.ndarray, homography_transform_func) -> np.ndarray:
        """Render ghost movement paths and targets."""
        for ghost in self.ghost_manager.ghosts:
            # Transform ghost position
            ghost_pos = homography_transform_func(ghost.state.position[0], ghost.state.position[1])
            if ghost_pos is None:
                continue

            x, y = ghost_pos

            # Draw ghost path to target
            if ghost.state.target_position is not None:
                target_pos = homography_transform_func(ghost.state.target_position[0], ghost.state.target_position[1])
                if target_pos is not None:
                    target_x, target_y = target_pos
                    cv2.line(canvas, (int(x), int(y)), (int(target_x), int(target_y)), (255, 0, 255), 1)

                    # Draw target marker
                    cv2.circle(canvas, (int(target_x), int(target_y)), 5, (255, 0, 255), -1)

            # Draw detection range for chase ghosts
            if ghost.ghost_type.value == "chase":
                detection_radius = int(ghost.detection_range * 0.8)  # Scale for display
                cv2.circle(canvas, (int(x), int(y)), detection_radius, (255, 100, 100), 1)

        return canvas

    def _render_debug_info(self, canvas: np.ndarray, player_data: Optional[dict]) -> np.ndarray:
        """Render debug information text."""
        # Add debug mode indicator
        cv2.putText(
            canvas, "DEBUG MODE", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        # Add board info
        cv2.putText(
            canvas, f"Board: {self.board.width}x{self.board.height}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        cv2.putText(
            canvas, f"Obstacles: {len(self.board.obstacles)}", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        cv2.putText(
            canvas, f"Ghosts: {len(self.ghost_manager.ghosts)}", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Add player info
        if player_data and "players" in player_data:
            players_with_legs = sum(1 for p in player_data["players"]
                                  if hasattr(p, 'leg_position') and p.leg_position is not None)
            cv2.putText(
                canvas, f"Players with legs: {players_with_legs}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return canvas
