"""
Projection renderer for Freezing Phantoms game.

Renders game graphics using homography-based coordinate transformation.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.core.game_state import GameState
from freezing_phantoms.core.tile_grid import TileGrid, TileState
from freezing_phantoms.effects.board_effects import BoardEffects

hailo_logger = get_logger(__name__)


class ProjectionRenderer:
    """
    Renders game visuals to projector using camera-projector homography.

    Attributes:
        homography_matrix (np.ndarray): 3x3 homography matrix for coordinate transformation.
        projector_resolution (Tuple[int, int]): Projector resolution (width, height).
        window_name (str): OpenCV window name.
    """

    def __init__(
        self,
        homography_matrix: np.ndarray,
        projector_resolution: Tuple[int, int],
        display_index: int = 1,
        tile_grid: Optional[TileGrid] = None,
        board_effects: Optional[BoardEffects] = None,
        movement_detector: Optional["MovementDetector"] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize projection renderer.

        Args:
            homography_matrix (np.ndarray): Homography matrix (3x3).
            projector_resolution (Tuple[int, int]): Projector resolution (width, height).
            display_index (int): Display index for projector (0=primary, 1=secondary).
            tile_grid (Optional[TileGrid]): Tile grid system for game board.
            board_effects (Optional[BoardEffects]): Board visual effects.
            movement_detector (Optional[MovementDetector]): Movement detector for visualization.
            config (Optional[Dict]): Game configuration dictionary for ghost settings.
        """
        self.H = homography_matrix
        self.projector_resolution = projector_resolution
        self.display_index = display_index
        self.window_name = "Freezing Phantoms - Projector"
        self.tile_grid = tile_grid
        self.board_effects = board_effects
        self.movement_detector = movement_detector
        self.config = config or {}

        # Animation frame counter
        self.frame_counter = 0

        # Lava step effect
        self.lava_step_effect_time = 0
        self.lava_step_duration = 0.5  # Half second flash

        # Load ghost image
        self.ghost_image = None
        self.ghost_config = self.config.get("ghost", {})
        if self.ghost_config.get("enabled", True):
            self._load_ghost_image()

        hailo_logger.info(
            "ProjectionRenderer initialized: %s, display=%d", projector_resolution, display_index
        )

        # Create fullscreen window
        self._create_window()

    def _create_window(self) -> None:
        """Create fullscreen OpenCV window for projector."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        # Move to projector display (HDMI-1 at position 0,0)
        if self.display_index > 0:
            cv2.moveWindow(self.window_name, 0, 0)

        hailo_logger.info("Created fullscreen window: %s", self.window_name)

    def _load_ghost_image(self) -> None:
        """Load ghost image from assets folder."""
        from pathlib import Path

        # Get ghost image path
        ghost_path = self.ghost_config.get("image_path", "assets/ghost.png")

        # Resolve relative path from freezing_phantoms directory
        base_dir = Path(__file__).parent.parent  # Go up from core/ to freezing_phantoms/
        full_path = base_dir / ghost_path

        if not full_path.exists():
            hailo_logger.error("Ghost image not found at: %s", full_path)
            return

        # Load image with alpha channel (transparency)
        self.ghost_image = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)  # noqa: PGH003

        if self.ghost_image is None:
            hailo_logger.error("Failed to load ghost image from: %s", full_path)
            return

        # Resize ghost to configured size
        ghost_size = self.ghost_config.get("size", 150)
        self.ghost_image = cv2.resize(
            self.ghost_image,
            (ghost_size, ghost_size),
            interpolation=cv2.INTER_AREA  # noqa: PGH003
        )

        hailo_logger.info("Loaded ghost image: %dx%d from %s",
                         ghost_size, ghost_size, full_path)

    def transform_point(self, cam_x: float, cam_y: float) -> Tuple[int, int]:
        """
        Transform camera coordinates to projector coordinates.

        Args:
            cam_x (float): Camera x coordinate.
            cam_y (float): Camera y coordinate.

        Returns:
            Tuple[int, int]: Projector (x, y) coordinates.
        """
        # Create point in homogeneous coordinates
        point = np.array([[[cam_x, cam_y]]], dtype=np.float32)

        # Apply homography transformation
        transformed = cv2.perspectiveTransform(point, self.H)

        proj_x = int(transformed[0][0][0])
        proj_y = int(transformed[0][0][1])

        return proj_x, proj_y

    def transform_bbox(
        self, bbox: Dict[str, float], frame_width: int, frame_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Transform bounding box from camera to projector space.

        Args:
            bbox (Dict[str, float]): Bounding box (xmin, ymin, width, height).
            frame_width (int): Camera frame width.
            frame_height (int): Camera frame height.

        Returns:
            Tuple[int, int, int, int]: Projector (x, y, width, height).
        """
        # Convert normalized bbox to pixel coordinates
        x_min = int(bbox["xmin"] * frame_width)
        y_min = int(bbox["ymin"] * frame_height)
        x_max = int((bbox["xmin"] + bbox["width"]) * frame_width)
        y_max = int((bbox["ymin"] + bbox["height"]) * frame_height)

        # Transform corners
        top_left = self.transform_point(x_min, y_min)
        bottom_right = self.transform_point(x_max, y_max)

        # Calculate projector bbox (handle negative dimensions with abs)
        proj_x = min(top_left[0], bottom_right[0])
        proj_y = min(top_left[1], bottom_right[1])
        proj_width = abs(bottom_right[0] - top_left[0])
        proj_height = abs(bottom_right[1] - top_left[1])

        return proj_x, proj_y, proj_width, proj_height

    def _render_upside_down_text(
        self,
        canvas: np.ndarray,
        text: str,
        x: Optional[int],
        y: int,
        color: Tuple[int, int, int],
        font_scale: float = 2.0,
        thickness: int = 4,
        centered: bool = False,
    ) -> np.ndarray:
        """
        Render text upside down (rotated 180 degrees) for player viewing.

        Args:
            canvas (np.ndarray): Canvas to render on.
            text (str): Text to render.
            x (Optional[int]): X coordinate (bottom-left of text), None if centered.
            y (int): Y coordinate (bottom-left of text).
            color (Tuple[int, int, int]): Text color (BGR).
            font_scale (float): Font scale.
            thickness (int): Text thickness.
            centered (bool): If True, center text horizontally (x is ignored).

        Returns:
            np.ndarray: Canvas with upside-down text rendered.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text dimensions
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # Calculate x position if centered
        if centered:
            x = (canvas.shape[1] - text_width) // 2

        # Create a temporary image for the text
        text_img = np.zeros((text_height + 10, text_width + 10, 3), dtype=np.uint8)

        # Render text on temporary image
        cv2.putText(
            text_img,
            text,
            (5, text_height + 2),
            font,
            font_scale,
            color,
            thickness,
        )

        # Rotate 180 degrees
        text_img_rotated = cv2.rotate(text_img, cv2.ROTATE_180)

        # Calculate position to place the rotated text
        y_start = y - text_height - 10
        y_end = y
        x_end = x + text_width + 10

        # Ensure coordinates are within bounds
        y_start = max(0, y_start)
        y_end = min(canvas.shape[0], y_end)
        x_end = min(canvas.shape[1], x_end)

        # Overlay the rotated text onto the canvas
        roi_height = y_end - y_start
        roi_width = x_end - x
        if roi_height > 0 and roi_width > 0:
            text_img_rotated_cropped = text_img_rotated[:roi_height, :roi_width]
            # Blend text (only where text exists)
            mask = cv2.cvtColor(text_img_rotated_cropped, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            roi = canvas[y_start:y_end, x:x_end]
            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(text_img_rotated_cropped, text_img_rotated_cropped, mask=mask)
            canvas[y_start:y_end, x:x_end] = cv2.add(bg, fg)

        return canvas

    def trigger_lava_step_effect(self) -> None:
        """Trigger visual effect for stepping on lava."""
        self.lava_step_effect_time = time.time()
        hailo_logger.debug("Lava step visual effect triggered")

    def render_frame(
        self,
        state: GameState,
        player_data: Optional[Dict],
        caught_players: List[int] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        violation_type: Optional[str] = None,
        game_stats: Optional[Dict] = None,
        current_hearts: int = 3,
        current_score: int = 0,
        ghost_position: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Render projection frame based on game state.

        Args:
            state (GameState): Current game state.
            player_data (Optional[Dict]): Current player data.
            caught_players (List[int], optional): List of caught player track IDs.
            frame_width (int): Camera frame width.
            frame_height (int): Camera frame height.
            violation_type (Optional[str]): Type of violation for penalty state.
            game_stats (Optional[Dict]): Game statistics for win screen.
            current_hearts (int): Current number of hearts remaining.
            current_score (int): Current score.
            ghost_position (Optional[Tuple[float, float]]): Ghost (x, y) position in projector coordinates.

        Returns:
            np.ndarray: Rendered projection image.
        """
        self.frame_counter += 1

        # Update board effects frame counter
        if self.board_effects:
            self.board_effects.update()

        # Create black canvas
        canvas = np.zeros(
            (self.projector_resolution[1], self.projector_resolution[0], 3),
            dtype=np.uint8,
        )

        if state == GameState.COUNTDOWN:
            canvas = self._render_countdown(canvas)
        elif state == GameState.GREEN_LIGHT:
            canvas = self._render_green_light(canvas, player_data, frame_width, frame_height)
        elif state == GameState.TRANSITION:
            canvas = self._render_transition(canvas, player_data, frame_width, frame_height)
        elif state == GameState.RED_LIGHT:
            canvas = self._render_red_light(canvas, player_data, frame_width, frame_height, ghost_position)
        elif state == GameState.PENALTY:
            canvas = self._render_penalty(
                canvas, player_data, caught_players, frame_width, frame_height, violation_type
            )
        elif state == GameState.WIN:
            canvas = self._render_win(canvas, game_stats=game_stats)
        elif state == GameState.GAME_OVER:
            canvas = self._render_game_over(canvas)

        # Render single UI bar at BOTTOM (closest to projector/player) LAST so it overlays everything
        # SPATIAL LAYOUT: Top pixels = far from projector (start), Bottom pixels = near projector (goal)
        # Contains: hearts (left), announcement (center), score (right) - all upside down
        canvas = self._render_ui_bar_bottom(canvas, state, current_hearts, current_score)

        return canvas

    def _render_tile_grid(self, canvas: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Render the tile grid on canvas with high intensity.

        Args:
            canvas (np.ndarray): Canvas to render on.
            alpha (float): Tile opacity (0-1). Default 1.0 for full intensity.

        Returns:
            np.ndarray: Canvas with tile grid rendered.
        """
        if not self.tile_grid:
            return canvas

        # Render lava background if available
        if self.board_effects:
            canvas = self.board_effects.render_lava_background(canvas, alpha=0.2)

        # Check if lava step effect is active
        lava_flash = False
        if time.time() - self.lava_step_effect_time < self.lava_step_duration:
            lava_flash = True

        # Render each tile at high intensity
        for tile in self.tile_grid.get_all_tiles():
            # Determine tile color based on state (bright, high intensity)
            if tile.state == TileState.GREEN:
                color = (0, 255, 0)  # Bright green - safe
            else:  # RED
                # Flash red tiles bright white when lava is stepped on
                if lava_flash:
                    color = (255, 255, 255)  # White flash
                else:
                    color = (0, 0, 255)  # Bright red - lava

            # Draw tile with high opacity for visibility
            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (tile.proj_x, tile.proj_y),
                (tile.proj_x + tile.proj_width, tile.proj_y + tile.proj_height),
                color,
                -1,
            )
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            # Draw tile border (skip bottom tiles to avoid yellow edges from UI bar overlap)
            # Only draw borders for tiles not in the bottom row
            if tile.row < (self.tile_grid.rows - 1):  # Skip last row
                cv2.rectangle(
                    canvas,
                    (tile.proj_x, tile.proj_y),
                    (tile.proj_x + tile.proj_width, tile.proj_y + tile.proj_height),
                    (50, 50, 50),
                    2,
                )

        # Highlight goal row
        if self.tile_grid:
            for col in range(self.tile_grid.cols):
                tile = self.tile_grid.tiles.get((self.tile_grid.goal_row, col))
                if tile:
                    # Draw gold border for goal tiles
                    cv2.rectangle(
                        canvas,
                        (tile.proj_x, tile.proj_y),
                        (tile.proj_x + tile.proj_width, tile.proj_y + tile.proj_height),
                        (0, 215, 255),  # Gold
                        4,
                    )

        return canvas

    def _render_countdown(self, canvas: np.ndarray) -> np.ndarray:
        """
        Render countdown state.

        Args:
            canvas (np.ndarray): Canvas to render on.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Render tile grid at high intensity
        canvas = self._render_tile_grid(canvas, alpha=0.95)

        # Display "GET READY!" text at bottom, upside down
        text = "GET READY!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 6

        # Pulsating effect
        pulse = int(abs(np.sin(self.frame_counter * 0.1)) * 100 + 155)
        color = (pulse, pulse, 0)  # Cyan

        # Render text upside down at bottom
        y_pos = canvas.shape[0] - 100  # 100px from bottom
        canvas = self._render_upside_down_text(
            canvas, text, None, y_pos, color, font_scale, thickness, centered=True
        )

        # Highlight starting position
        if self.tile_grid:
            player_tile = self.tile_grid.get_player_tile()
            center_x, center_y = player_tile.get_center()
            cv2.circle(canvas, (center_x, center_y), 30, (255, 255, 255), 3)

        return canvas

    def _render_green_light(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        frame_width: int,  # noqa: ARG002
        frame_height: int,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Render green light state (players can move).

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (Optional[Dict]): Player data.
            frame_width (int): Frame width.
            frame_height (int): Frame height.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Render tile grid at full intensity
        canvas = self._render_tile_grid(canvas, alpha=1.0)

        # Render player position indicator
        if player_data and "players" in player_data and self.tile_grid:
            for player in player_data["players"]:
                # Get leg position and transform to projector space
                if hasattr(player, 'leg_position') and player.leg_position:
                    leg_x, leg_y = player.leg_position
                    proj_x, proj_y = self.transform_point(leg_x, leg_y)

                    # Update tile grid with player position
                    self.tile_grid.get_player_current_tile((proj_x, proj_y))

                    # Draw player indicator
                    if self.board_effects:
                        canvas = self.board_effects.render_player_indicator(
                            canvas, proj_x, proj_y, radius=25, color=(255, 255, 255)
                        )
                    else:
                        # Fallback: simple circle
                        cv2.circle(canvas, (proj_x, proj_y), 25, (255, 255, 255), 3)
                        cv2.circle(canvas, (proj_x, proj_y), 5, (255, 255, 255), -1)

        return canvas

    def _render_transition(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        frame_width: int,  # noqa: ARG002
        frame_height: int,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Render transition state (flashing between green and red).

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (Optional[Dict]): Player data.
            frame_width (int): Frame width.
            frame_height (int): Frame height.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Flash effect - alternate between green and dark every 0.25s
        flash_on = (int(self.frame_counter / 15) % 2) == 0  # ~0.25s at 60fps

        if flash_on:
            # Bright flash - render normal green light
            canvas = self._render_tile_grid(canvas, alpha=1.0)
        else:
            # Dark - black out
            canvas[:, :] = (0, 0, 0)  # Black
            canvas = self._render_tile_grid(canvas, alpha=0.2)

        # Render player position indicator (always visible)
        if player_data and "players" in player_data and self.tile_grid:
            for player in player_data["players"]:
                # Get leg position and transform to projector space
                if hasattr(player, 'leg_position') and player.leg_position:
                    leg_x, leg_y = player.leg_position
                    proj_x, proj_y = self.transform_point(leg_x, leg_y)

                    # Update tile grid with player position
                    self.tile_grid.get_player_current_tile((proj_x, proj_y))

                    # Render player indicator
                    if self.board_effects:
                        canvas = self.board_effects.render_player_indicator(
                            canvas, proj_x, proj_y, color=(255, 255, 0)  # Yellow for transition
                        )

        return canvas

    def _render_red_light(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        frame_width: int,  # noqa: ARG002
        frame_height: int,  # noqa: ARG002
        ghost_position: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Render red light state (players must freeze).

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (Optional[Dict]): Player data.
            frame_width (int): Frame width.
            frame_height (int): Frame height.
            ghost_position (Optional[Tuple[float, float]]): Ghost position if active.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Red tint overlay
        canvas[:, :] = (10, 0, 30)  # Dark red background

        # Render tile grid at full intensity during freeze
        canvas = self._render_tile_grid(canvas, alpha=1.0)

        # Draw scanning laser effect
        laser_y = int((self.frame_counter * 5) % canvas.shape[0])
        cv2.line(canvas, (0, laser_y), (canvas.shape[1], laser_y), (0, 0, 255), 3)

        # Draw smoothed keypoints for debugging
        if player_data and "players" in player_data and self.movement_detector:
            canvas = self._draw_smoothed_keypoints(canvas, player_data)

        # Render player position with frozen indicator
        if player_data and "players" in player_data:
            for player in player_data["players"]:
                # Get leg position
                if hasattr(player, 'leg_position') and player.leg_position:
                    leg_x, leg_y = player.leg_position
                    proj_x, proj_y = self.transform_point(leg_x, leg_y)

                    # Draw frozen player indicator
                    cv2.circle(canvas, (proj_x, proj_y), 30, (100, 200, 255), 3)
                    cv2.circle(canvas, (proj_x, proj_y), 5, (100, 200, 255), -1)

        # Render ghost if active
        if ghost_position is not None:
            canvas = self._render_ghost(canvas, ghost_position)

        # Add "FREEZE!" text at bottom, upside down
        text = "FREEZE!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4

        y_pos = canvas.shape[0] - 80  # 80px from bottom
        canvas = self._render_upside_down_text(
            canvas, text, None, y_pos, (0, 0, 255), font_scale, thickness, centered=True
        )

        return canvas

    def _render_penalty(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        caught_players: List[int],  # noqa: ARG002
        frame_width: int,  # noqa: ARG002
        frame_height: int,  # noqa: ARG002
        violation_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render penalty state (highlight caught players).

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (Optional[Dict]): Player data.
            caught_players (List[int]): Caught player track IDs.
            frame_width (int): Frame width.
            frame_height (int): Frame height.
            violation_type (Optional[str]): Type of violation ("movement" or "tile").

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Dim red background
        canvas[:, :] = (20, 0, 40)

        # Render tile grid at high intensity (keep visible during penalty)
        canvas = self._render_tile_grid(canvas, alpha=0.9)

        # Determine message based on violation type
        if violation_type == "tile":
            message = "STEPPED ON LAVA!"
        else:
            message = "MOVED!"

        # Display message at bottom, upside down
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4

        # Pulsating red color
        pulse = int(abs(np.sin(self.frame_counter * 0.2)) * 100 + 155)

        y_pos = canvas.shape[0] - 80  # 80px from bottom
        canvas = self._render_upside_down_text(
            canvas, message, None, y_pos, (0, 0, pulse), font_scale, thickness, centered=True
        )

        # Show player indicator
        if player_data and "players" in player_data:
            for player in player_data["players"]:
                if hasattr(player, 'leg_position') and player.leg_position:
                    leg_x, leg_y = player.leg_position
                    proj_x, proj_y = self.transform_point(leg_x, leg_y)

                    # Pulsating red spotlight
                    overlay = canvas.copy()
                    cv2.circle(overlay, (proj_x, proj_y), 60, (0, 0, pulse), -1)
                    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

                    # Draw X marker
                    size = 30
                    cv2.line(canvas, (proj_x - size, proj_y - size), (proj_x + size, proj_y + size), (255, 255, 255), 4)
                    cv2.line(canvas, (proj_x - size, proj_y + size), (proj_x + size, proj_y - size), (255, 255, 255), 4)

        return canvas

    def _render_win(self, canvas: np.ndarray, game_stats: Optional[Dict] = None) -> np.ndarray:
        """
        Render win state with score display.

        Args:
            canvas (np.ndarray): Canvas to render on.
            game_stats (Optional[Dict]): Game statistics including score, time, penalties.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Render tile grid at high intensity
        canvas = self._render_tile_grid(canvas, alpha=0.95)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Pulsating gold color for main text
        pulse = int(abs(np.sin(self.frame_counter * 0.15)) * 100 + 155)
        gold_color = (0, pulse, 255)  # Gold

        # Victory text - normal orientation since player is at bottom (close to camera)
        text = "YOU WIN!"
        font_scale = 4
        thickness = 8

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = canvas.shape[0] - 400  # Higher up to make room for score

        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, gold_color, thickness)

        # Display score and stats if available
        if game_stats:
            score_y = text_y + 80
            stats_font_scale = 1.2
            stats_thickness = 3
            line_spacing = 50

            # Performance rating
            rating = game_stats.get("rating", "")
            rating_color = (100, 255, 100) if rating in ["PERFECT!", "EXCELLENT"] else (200, 200, 200)
            rating_size = cv2.getTextSize(rating, font, 2.0, 4)[0]
            rating_x = (canvas.shape[1] - rating_size[0]) // 2
            cv2.putText(canvas, rating, (rating_x, score_y), font, 2.0, rating_color, 4)

            score_y += line_spacing + 20

            # Score
            score = game_stats.get("score", 0)
            score_text = f"SCORE: {score}"
            score_size = cv2.getTextSize(score_text, font, stats_font_scale, stats_thickness)[0]
            score_x = (canvas.shape[1] - score_size[0]) // 2
            cv2.putText(canvas, score_text, (score_x, score_y), font, stats_font_scale, (255, 255, 255), stats_thickness)

            score_y += line_spacing

            # Time
            time_val = game_stats.get("time", 0)
            time_text = f"Time: {time_val:.1f}s"
            time_size = cv2.getTextSize(time_text, font, stats_font_scale, stats_thickness)[0]
            time_x = (canvas.shape[1] - time_size[0]) // 2
            cv2.putText(canvas, time_text, (time_x, score_y), font, stats_font_scale, (200, 200, 200), stats_thickness)

            score_y += line_spacing

            # Penalties
            total_penalties = game_stats.get("total_penalties", 0)
            movement_penalties = game_stats.get("movement_penalties", 0)
            lava_penalties = game_stats.get("lava_penalties", 0)

            penalty_text = f"Penalties: {total_penalties}"
            penalty_size = cv2.getTextSize(penalty_text, font, stats_font_scale, stats_thickness)[0]
            penalty_x = (canvas.shape[1] - penalty_size[0]) // 2
            penalty_color = (100, 255, 100) if total_penalties == 0 else (200, 200, 200)
            cv2.putText(canvas, penalty_text, (penalty_x, score_y), font, stats_font_scale, penalty_color, stats_thickness)

            # Penalty breakdown (smaller text)
            if total_penalties > 0:
                score_y += 40
                breakdown = f"(Movement: {movement_penalties}, Lava: {lava_penalties})"
                breakdown_scale = 0.8
                breakdown_size = cv2.getTextSize(breakdown, font, breakdown_scale, 2)[0]
                breakdown_x = (canvas.shape[1] - breakdown_size[0]) // 2
                cv2.putText(canvas, breakdown, (breakdown_x, score_y), font, breakdown_scale, (150, 150, 150), 2)

        # Subtitle
        subtitle = "Press R to restart"
        sub_font_scale = 1.5
        sub_thickness = 3
        sub_size = cv2.getTextSize(subtitle, font, sub_font_scale, sub_thickness)[0]
        sub_x = (canvas.shape[1] - sub_size[0]) // 2
        sub_y = canvas.shape[0] - 50  # Near bottom

        cv2.putText(
            canvas, subtitle, (sub_x, sub_y), font, sub_font_scale, (200, 200, 200), sub_thickness
        )

        return canvas

    def _render_ui_bar_bottom(
        self, canvas: np.ndarray, state: GameState, current_hearts: int, current_score: int
    ) -> np.ndarray:
        """
        Render single UI bar at BOTTOM (closest to projector/player) with all game information,
        all facing the player (upside down).

        SPATIAL LAYOUT:
        - Top of image (row 0) = FAR from projector = Player starts here
        - Bottom of image (row 5) = NEAR projector = Goal, where UI bar is
        - Player stands beyond bottom edge, looking down at floor projection

        Layout: Hearts (left) | Announcement (center) | Score (right)

        Args:
            canvas (np.ndarray): Canvas to render on.
            state (GameState): Current game state.
            current_hearts (int): Number of hearts remaining.
            current_score (int): Current score.

        Returns:
            np.ndarray: Canvas with UI bar rendered.
        """
        height = canvas.shape[0]
        width = canvas.shape[1]
        bar_height = 100  # Single bar height
        bar_y_start = height - bar_height  # Position at BOTTOM

        # Semi-transparent dark bar at bottom
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, bar_y_start), (width, height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # === HEARTS (left side, upside down) ===
        hearts_y = bar_y_start + 50  # Centered in bar
        heart_spacing = 50
        hearts_start_x = 50  # Left side

        for i in range(5):  # Show 5 heart slots
            heart_x = hearts_start_x + (i * heart_spacing)
            if i < current_hearts:
                # Filled heart (red) - upside down
                cv2.circle(canvas, (heart_x - 10, hearts_y), 12, (0, 0, 255), -1)
                cv2.circle(canvas, (heart_x + 10, hearts_y), 12, (0, 0, 255), -1)
                pts = np.array(
                    [
                        [heart_x - 22, hearts_y],
                        [heart_x, hearts_y - 25],  # Flipped: negative Y for upside down
                        [heart_x + 22, hearts_y],
                    ],
                    np.int32,
                )
                cv2.fillPoly(canvas, [pts], (0, 0, 255))
            else:
                # Empty heart (gray outline) - upside down
                cv2.circle(canvas, (heart_x - 10, hearts_y), 12, (100, 100, 100), 2)
                cv2.circle(canvas, (heart_x + 10, hearts_y), 12, (100, 100, 100), 2)
                pts = np.array(
                    [
                        [heart_x - 22, hearts_y],
                        [heart_x, hearts_y - 25],  # Flipped: negative Y for upside down
                        [heart_x + 22, hearts_y],
                    ],
                    np.int32,
                )
                cv2.polylines(canvas, [pts], False, (100, 100, 100), 2)

        # === STATE MESSAGE (center, upside down) ===
        if state == GameState.GREEN_LIGHT:
            message = "GO!"
            color = (0, 255, 0)
        elif state == GameState.RED_LIGHT:
            message = "FREEZE!"
            color = (0, 0, 255)
        elif state == GameState.TRANSITION:
            message = "GET READY..."
            color = (255, 255, 0)
        elif state == GameState.PENALTY:
            message = "PENALTY!"
            color = (255, 0, 255)
        elif state == GameState.COUNTDOWN:
            message = "Stand on Green Tile"
            color = (0, 255, 255)
        else:
            message = ""
            color = (255, 255, 255)

        if message:
            # Render upside down text in center
            canvas = self._render_upside_down_text(
                canvas, message, None, bar_y_start + 70, color, 1.5, 3, centered=True
            )

        # === SCORE (right side, upside down) ===
        score_text = f"Score: {current_score}"
        score_size = cv2.getTextSize(score_text, font, 1.2, 3)[0]
        score_x = width - score_size[0] - 50
        canvas = self._render_upside_down_text(
            canvas, score_text, score_x, bar_y_start + 60, (255, 255, 255), 1.2, 3, centered=False
        )

        return canvas

    def _draw_smoothed_keypoints(self, canvas: np.ndarray, player_data: Dict) -> np.ndarray:
        """
        Draw smoothed keypoint positions for debugging.
        Shows the positions after smoothing (before velocity calculation).

        Args:
            canvas (np.ndarray): Canvas to draw on.
            player_data (Dict): Player data with pose information.

        Returns:
            np.ndarray: Canvas with smoothed keypoints drawn.
        """
        if not self.movement_detector or not player_data or "players" not in player_data:
            return canvas

        for player in player_data["players"]:
            track_id = player.track_id

            # Get smoothed pose from movement detector
            smoothed_pose = self.movement_detector.get_smoothed_pose(track_id)
            if not smoothed_pose:
                continue

            # Draw each smoothed keypoint
            for keypoint_name, (x, y) in smoothed_pose.items():
                # Transform to projector coordinates
                proj_point = cv2.perspectiveTransform(
                    np.array([[[x, y]]], dtype=np.float32),
                    self.H
                )[0][0]
                proj_x, proj_y = int(proj_point[0]), int(proj_point[1])

                # Draw smoothed keypoint (cyan circle)
                cv2.circle(canvas, (proj_x, proj_y), 8, (255, 255, 0), -1)  # Cyan filled
                cv2.circle(canvas, (proj_x, proj_y), 8, (255, 255, 255), 2)  # White border

                # Draw keypoint label
                label_y = proj_y - 12
                cv2.putText(canvas, keypoint_name[:5], (proj_x - 15, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return canvas

    def _render_ghost(self, canvas: np.ndarray, ghost_position: Tuple[float, float]) -> np.ndarray:
        """
        Render ghost sprite at specified position with alpha blending.

        Args:
            canvas (np.ndarray): Canvas to render on.
            ghost_position (Tuple[float, float]): Ghost (x, y) position in projector coordinates.

        Returns:
            np.ndarray: Canvas with ghost rendered.
        """
        if self.ghost_image is None:
            return canvas

        ghost_x, ghost_y = int(ghost_position[0]), int(ghost_position[1])
        ghost_h, ghost_w = self.ghost_image.shape[:2]

        # Calculate top-left position (center the ghost on the target position)
        x1 = ghost_x - ghost_w // 2
        y1 = ghost_y - ghost_h // 2
        x2 = x1 + ghost_w
        y2 = y1 + ghost_h

        # Clip to canvas bounds
        canvas_h, canvas_w = canvas.shape[:2]

        # Calculate visible region
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = ghost_w - max(0, x2 - canvas_w)
        src_y2 = ghost_h - max(0, y2 - canvas_h)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(canvas_w, x2)
        dst_y2 = min(canvas_h, y2)

        # Check if ghost is visible
        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2 or src_x1 >= src_x2 or src_y1 >= src_y2:
            return canvas

        # Extract ghost region
        ghost_region = self.ghost_image[src_y1:src_y2, src_x1:src_x2]

        # Check if ghost has alpha channel
        if ghost_region.shape[2] == 4:
            # Extract RGB and alpha channels
            ghost_rgb = ghost_region[:, :, :3]
            ghost_alpha = ghost_region[:, :, 3] / 255.0

            # Apply configured opacity
            opacity = self.ghost_config.get("opacity", 0.8)
            ghost_alpha = ghost_alpha * opacity

            # Expand alpha to 3 channels for blending
            ghost_alpha_3ch = np.dstack([ghost_alpha] * 3)

            # Extract canvas region
            canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]

            # Alpha blend
            blended = (ghost_rgb * ghost_alpha_3ch + canvas_region * (1 - ghost_alpha_3ch)).astype(np.uint8)

            # Place blended region back
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = blended
        else:
            # No alpha channel, just overlay
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = ghost_region

        return canvas

    def _render_game_over(self, canvas: np.ndarray) -> np.ndarray:
        """
        Render game over state.

        Args:
            canvas (np.ndarray): Canvas to render on.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Dim background
        canvas[:, :] = (20, 0, 40)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Game Over text
        text = "GAME OVER"
        font_scale = 3
        thickness = 6
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = canvas.shape[0] // 2 - 50

        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

        # Out of hearts message
        subtitle = "Out of Hearts!"
        sub_scale = 1.5
        sub_thickness = 3
        sub_size = cv2.getTextSize(subtitle, font, sub_scale, sub_thickness)[0]
        sub_x = (canvas.shape[1] - sub_size[0]) // 2
        sub_y = text_y + 80

        cv2.putText(canvas, subtitle, (sub_x, sub_y), font, sub_scale, (200, 200, 200), sub_thickness)

        # Press R to restart
        restart_text = "Press R to Restart"
        restart_scale = 1.2
        restart_thickness = 2
        restart_size = cv2.getTextSize(restart_text, font, restart_scale, restart_thickness)[0]
        restart_x = (canvas.shape[1] - restart_size[0]) // 2
        restart_y = canvas.shape[0] - 100

        cv2.putText(
            canvas, restart_text, (restart_x, restart_y), font, restart_scale, (150, 150, 150), restart_thickness
        )

        return canvas

    def _draw_particle_swirls(self, canvas: np.ndarray, color: Tuple[int, int, int]) -> None:
        """
        Draw animated particle swirls for ambiance.

        Args:
            canvas (np.ndarray): Canvas to draw on.
            color (Tuple[int, int, int]): Particle color.
        """
        # Draw a few animated particles
        num_particles = 20
        for i in range(num_particles):
            angle = (self.frame_counter * 0.05 + i * (360 / num_particles)) % 360
            radius_base = min(canvas.shape[:2]) // 4

            x = int(canvas.shape[1] // 2 + radius_base * np.cos(np.radians(angle)))
            y = int(canvas.shape[0] // 2 + radius_base * np.sin(np.radians(angle)))

            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (x, y), 5, color, -1)

    def display(self, frame: np.ndarray) -> None:
        """
        Display frame on projector window.

        Args:
            frame (np.ndarray): Frame to display.
        """
        cv2.imshow(self.window_name, frame)

    def render_simple_pose_landmarks(
        self,
        player_data: Optional[Dict],
        frame_width: int,  # noqa: ARG002
        frame_height: int,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Render only pose landmarks projected through homography (for calibration testing).

        Args:
            player_data: Player data dictionary.
            frame_width: Camera frame width.
            frame_height: Camera frame height.

        Returns:
            np.ndarray: Rendered frame with pose landmarks.
        """
        # Create blank canvas
        canvas = np.zeros((self.projector_resolution[1], self.projector_resolution[0], 3), dtype=np.uint8)

        if not player_data or "players" not in player_data:
            return canvas

        # Draw pose keypoints for each player
        for player in player_data["players"]:
            pose_keypoints = player.pose_keypoints
            pose_confidences = player.pose_keypoint_confidences
            track_id = player.track_id

            # Define color for this player
            colors = [
                (0, 255, 0),    # Green
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (255, 128, 0),  # Orange
                (0, 128, 255),  # Light blue
            ]
            color = colors[track_id % len(colors)]

            # Draw each keypoint
            for keypoint_name, (cam_x, cam_y) in pose_keypoints.items():
                confidence = pose_confidences.get(keypoint_name, 0.0)

                # Only draw keypoints with sufficient confidence
                if confidence < 0.3:
                    continue

                # Transform to projector coordinates
                proj_x, proj_y = self.transform_point(cam_x, cam_y)

                # Check if point is within bounds
                if 0 <= proj_x < self.projector_resolution[0] and 0 <= proj_y < self.projector_resolution[1]:
                    # Draw circle for keypoint
                    radius = int(5 + confidence * 10)  # Size based on confidence
                    cv2.circle(canvas, (proj_x, proj_y), radius, color, -1)

                    # Draw small label with keypoint name
                    cv2.putText(
                        canvas,
                        keypoint_name[:4],  # First 4 chars
                        (proj_x + 5, proj_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1
                    )

            # Draw leg position if available
            if hasattr(player, 'leg_position') and player.leg_position:
                leg_x, leg_y = player.leg_position
                proj_x, proj_y = self.transform_point(leg_x, leg_y)

                if 0 <= proj_x < self.projector_resolution[0] and 0 <= proj_y < self.projector_resolution[1]:
                    # Draw larger circle for leg position
                    cv2.circle(canvas, (proj_x, proj_y), 15, (255, 255, 255), 2)
                    cv2.putText(
                        canvas,
                        f"Player {track_id}",
                        (proj_x - 30, proj_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

        # Draw info text
        cv2.putText(
            canvas,
            "Pose Landmarks Projection Test",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.putText(
            canvas,
            f"Players detected: {len(player_data.get('players', []))}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        cv2.putText(
            canvas,
            "Press ESC to exit",
            (20, self.projector_resolution[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        return canvas

    def cleanup(self) -> None:
        """Clean up renderer resources."""
        cv2.destroyWindow(self.window_name)
        hailo_logger.info("Projection renderer cleaned up")

