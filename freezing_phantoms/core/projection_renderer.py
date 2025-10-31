"""
Projection renderer for Freezing Phantoms game.

Renders game graphics using homography-based coordinate transformation.
"""

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
    ):
        """
        Initialize projection renderer.

        Args:
            homography_matrix (np.ndarray): Homography matrix (3x3).
            projector_resolution (Tuple[int, int]): Projector resolution (width, height).
            display_index (int): Display index for projector (0=primary, 1=secondary).
            tile_grid (Optional[TileGrid]): Tile grid system for game board.
            board_effects (Optional[BoardEffects]): Board visual effects.
        """
        self.H = homography_matrix
        self.projector_resolution = projector_resolution
        self.display_index = display_index
        self.window_name = "Freezing Phantoms - Projector"
        self.tile_grid = tile_grid
        self.board_effects = board_effects

        # Animation frame counter
        self.frame_counter = 0

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
        font: int,
        font_scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        y_offset: int = 100,
    ) -> np.ndarray:
        """
        Render text upside down at the bottom of the canvas.

        Args:
            canvas (np.ndarray): Canvas to render on.
            text (str): Text to render.
            font (int): Font type.
            font_scale (float): Font scale.
            color (Tuple[int, int, int]): Text color.
            thickness (int): Text thickness.
            y_offset (int): Offset from bottom edge.

        Returns:
            np.ndarray: Canvas with text rendered.
        """
        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        # Create temporary image for text
        text_img = np.zeros((text_height + 20, text_width + 20, 3), dtype=np.uint8)

        # Draw text on temporary image
        cv2.putText(
            text_img,
            text,
            (10, text_height + 5),
            font,
            font_scale,
            color,
            thickness,
        )

        # Rotate 180 degrees
        text_img_rotated = cv2.rotate(text_img, cv2.ROTATE_180)

        # Calculate position at bottom center
        img_h, img_w = text_img_rotated.shape[:2]
        x_pos = (canvas.shape[1] - img_w) // 2
        y_pos = canvas.shape[0] - y_offset - img_h

        # Ensure position is within bounds
        if y_pos < 0:
            y_pos = 0
        if x_pos < 0:
            x_pos = 0

        # Calculate actual region to paste
        end_y = min(y_pos + img_h, canvas.shape[0])
        end_x = min(x_pos + img_w, canvas.shape[1])
        actual_h = end_y - y_pos
        actual_w = end_x - x_pos

        # Overlay text on canvas where it's not black
        if actual_h > 0 and actual_w > 0:
            text_region = text_img_rotated[:actual_h, :actual_w]
            mask = np.any(text_region > 0, axis=2)
            canvas[y_pos:end_y, x_pos:end_x][mask] = text_region[mask]

        return canvas

    def render_frame(
        self,
        state: GameState,
        player_data: Optional[Dict],
        caught_players: List[int] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        violation_type: Optional[str] = None,
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
        elif state == GameState.RED_LIGHT:
            canvas = self._render_red_light(canvas, player_data, frame_width, frame_height)
        elif state == GameState.PENALTY:
            canvas = self._render_penalty(
                canvas, player_data, caught_players, frame_width, frame_height, violation_type
            )
        elif state == GameState.WIN:
            canvas = self._render_win(canvas)
        elif state == GameState.GAME_OVER:
            canvas = self._render_game_over(canvas)

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

        # Render each tile at high intensity
        for tile in self.tile_grid.get_all_tiles():
            # Determine tile color based on state (bright, high intensity)
            if tile.state == TileState.GREEN:
                color = (0, 255, 0)  # Bright green - safe
            else:  # RED
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

            # Draw tile border
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
        canvas = self._render_upside_down_text(
            canvas, text, font, font_scale, color, thickness, y_offset=100
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

    def _render_red_light(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        frame_width: int,  # noqa: ARG002
        frame_height: int,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Render red light state (players must freeze).

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (Optional[Dict]): Player data.
            frame_width (int): Frame width.
            frame_height (int): Frame height.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Red tint overlay
        canvas[:, :] = (10, 0, 30)  # Dark red background

        # Render tile grid at high intensity (keep visible during red light)
        canvas = self._render_tile_grid(canvas, alpha=0.9)

        # Draw scanning laser effect
        laser_y = int((self.frame_counter * 5) % canvas.shape[0])
        cv2.line(canvas, (0, laser_y), (canvas.shape[1], laser_y), (0, 0, 255), 3)

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

        # Add "FREEZE!" text at bottom, upside down
        text = "FREEZE!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4

        canvas = self._render_upside_down_text(
            canvas, text, font, font_scale, (0, 0, 255), thickness, y_offset=80
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

        canvas = self._render_upside_down_text(
            canvas, message, font, font_scale, (0, 0, pulse), thickness, y_offset=80
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

    def _render_win(self, canvas: np.ndarray) -> np.ndarray:
        """
        Render win state.

        Args:
            canvas (np.ndarray): Canvas to render on.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Render tile grid at high intensity
        canvas = self._render_tile_grid(canvas, alpha=0.95)

        # Victory text - normal orientation since player is at bottom (close to camera)
        text = "YOU WIN!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 8

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = canvas.shape[0] - 200  # Near bottom

        # Pulsating gold color
        pulse = int(abs(np.sin(self.frame_counter * 0.15)) * 100 + 155)
        color = (0, pulse, 255)  # Gold

        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness)

        # Subtitle
        subtitle = "Press R to restart"
        sub_font_scale = 1.5
        sub_thickness = 3
        sub_size = cv2.getTextSize(subtitle, font, sub_font_scale, sub_thickness)[0]
        sub_x = (canvas.shape[1] - sub_size[0]) // 2
        sub_y = canvas.shape[0] - 100  # Below win text

        cv2.putText(
            canvas, subtitle, (sub_x, sub_y), font, sub_font_scale, (200, 200, 200), sub_thickness
        )

        return canvas

    def _render_game_over(self, canvas: np.ndarray) -> np.ndarray:
        """
        Render game over state.

        Args:
            canvas (np.ndarray): Canvas to render on.

        Returns:
            np.ndarray: Rendered canvas.
        """
        text = "GAME OVER"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 6

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = (canvas.shape[0] + text_size[1]) // 2

        cv2.putText(
            canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
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

