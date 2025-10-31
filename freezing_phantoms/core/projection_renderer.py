"""
Projection renderer for Freezing Phantoms game.

Renders game graphics using homography-based coordinate transformation.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.core.game_state import GameState

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
    ):
        """
        Initialize projection renderer.

        Args:
            homography_matrix (np.ndarray): Homography matrix (3x3).
            projector_resolution (Tuple[int, int]): Projector resolution (width, height).
            display_index (int): Display index for projector (0=primary, 1=secondary).
        """
        self.H = homography_matrix
        self.projector_resolution = projector_resolution
        self.display_index = display_index
        self.window_name = "Freezing Phantoms - Projector"

        # Animation frame counter
        self.frame_counter = 0

        hailo_logger.info(
            f"ProjectionRenderer initialized: {projector_resolution}, display={display_index}"
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

        hailo_logger.info(f"Created fullscreen window: {self.window_name}")

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

    def render_frame(
        self,
        state: GameState,
        player_data: Optional[Dict],
        caught_players: List[int] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> np.ndarray:
        """
        Render projection frame based on game state.

        Args:
            state (GameState): Current game state.
            player_data (Optional[Dict]): Current player data.
            caught_players (List[int], optional): List of caught player track IDs.
            frame_width (int): Camera frame width.
            frame_height (int): Camera frame height.

        Returns:
            np.ndarray: Rendered projection image.
        """
        self.frame_counter += 1

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
                canvas, player_data, caught_players, frame_width, frame_height
            )
        elif state == GameState.GAME_OVER:
            canvas = self._render_game_over(canvas)

        return canvas

    def _render_countdown(self, canvas: np.ndarray) -> np.ndarray:
        """
        Render countdown state.

        Args:
            canvas (np.ndarray): Canvas to render on.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Display "GET READY!" text
        text = "GET READY!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 6

        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = (canvas.shape[0] + text_size[1]) // 2

        # Pulsating effect
        pulse = int(abs(np.sin(self.frame_counter * 0.1)) * 100 + 155)
        color = (pulse, pulse, 0)  # Cyan

        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness)

        return canvas

    def _render_green_light(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        frame_width: int,
        frame_height: int,
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
        # Create "lava" or "phantom swirl" effect around players
        # For simplicity, draw animated circles around safe zones

        if player_data and "players" in player_data:
            for player in player_data["players"]:
                # Transform bbox to projector space
                proj_x, proj_y, proj_w, proj_h = self.transform_bbox(
                    player.bbox, frame_width, frame_height
                )

                # Skip if invalid bbox
                if proj_w <= 0 or proj_h <= 0:
                    continue

                # Calculate center and ensure it's within bounds
                center_x = int(proj_x + proj_w // 2)
                center_y = int(proj_y + proj_h // 2)

                # Skip if center is outside canvas
                if center_x < 0 or center_x >= canvas.shape[1] or center_y < 0 or center_y >= canvas.shape[0]:
                    continue

                # Calculate base radius with safety margins
                radius = max(proj_w, proj_h) // 2 + 20
                radius = max(10, min(radius, min(canvas.shape[:2]) // 4))  # Clamp between 10 and reasonable max

                # Animated pulsating circle with clamped variation
                pulse_variation = np.sin(self.frame_counter * 0.15) * 10
                pulse_radius = int(max(5, radius + pulse_variation))  # Ensure minimum radius

                # Draw translucent green circle only if radius is valid
                if pulse_radius > 0:
                    overlay = canvas.copy()
                    cv2.circle(overlay, (center_x, center_y), pulse_radius, (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Add swirling particles (optional enhancement)
        self._draw_particle_swirls(canvas, (150, 255, 150))

        return canvas

    def _render_red_light(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        frame_width: int,
        frame_height: int,
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

        # Draw scanning laser effect
        laser_y = int((self.frame_counter * 5) % canvas.shape[0])
        cv2.line(canvas, (0, laser_y), (canvas.shape[1], laser_y), (0, 0, 255), 3)

        # Draw "ice encasement" on players using segmentation masks
        if player_data and "players" in player_data:
            for player in player_data["players"]:
                if player.segmentation_mask is not None:
                    # Transform mask to projector space (simplified - use bbox for now)
                    proj_x, proj_y, proj_w, proj_h = self.transform_bbox(
                        player.bbox, frame_width, frame_height
                    )

                    # Draw "frozen" effect - blue/cyan overlay on player
                    if 0 <= proj_x < canvas.shape[1] and 0 <= proj_y < canvas.shape[0]:
                        end_x = min(proj_x + proj_w, canvas.shape[1])
                        end_y = min(proj_y + proj_h, canvas.shape[0])

                        if end_x > proj_x and end_y > proj_y:
                            overlay = canvas[proj_y:end_y, proj_x:end_x].copy()
                            ice_color = np.full(overlay.shape, (255, 200, 100), dtype=np.uint8)
                            canvas[proj_y:end_y, proj_x:end_x] = cv2.addWeighted(
                                overlay, 0.6, ice_color, 0.4, 0
                            )

        # Add "FREEZE!" text
        text = "FREEZE!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = 100
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

        return canvas

    def _render_penalty(
        self,
        canvas: np.ndarray,
        player_data: Optional[Dict],
        caught_players: List[int],
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Render penalty state (highlight caught players).

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (Optional[Dict]): Player data.
            caught_players (List[int]): Caught player track IDs.
            frame_width (int): Frame width.
            frame_height (int): Frame height.

        Returns:
            np.ndarray: Rendered canvas.
        """
        # Dim red background
        canvas[:, :] = (20, 0, 40)

        if player_data and "players" in player_data and caught_players:
            for player in player_data["players"]:
                if player.track_id in caught_players:
                    # Project red spotlight on caught player
                    proj_x, proj_y, proj_w, proj_h = self.transform_bbox(
                        player.bbox, frame_width, frame_height
                    )

                    # Skip if invalid bbox
                    if proj_w <= 0 or proj_h <= 0:
                        continue

                    # Calculate center and ensure it's within bounds
                    center_x = int(proj_x + proj_w // 2)
                    center_y = int(proj_y + proj_h // 2)

                    # Skip if center is outside canvas
                    if center_x < 0 or center_x >= canvas.shape[1] or center_y < 0 or center_y >= canvas.shape[0]:
                        continue

                    # Calculate radius with safety limits
                    radius = max(proj_w, proj_h)
                    radius = max(10, min(radius, min(canvas.shape[:2]) // 4))  # Clamp between 10 and reasonable max

                    # Pulsating red spotlight
                    pulse = int(abs(np.sin(self.frame_counter * 0.2)) * 100 + 155)

                    # Only draw if radius is valid
                    if radius > 0:
                        overlay = canvas.copy()
                        cv2.circle(overlay, (center_x, center_y), radius, (0, 0, pulse), -1)
                        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

                        # Draw "CAUGHT!" text
                        cv2.putText(
                            canvas,
                            "CAUGHT!",
                            (center_x - 80, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 255),
                            3,
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
        frame_width: int,
        frame_height: int,
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

