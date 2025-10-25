"""
Visual effects for RED_LIGHT game state.

Creates freeze detection effects and scanning animations.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class RedLightEffects:
    """
    Visual effects for the red light (freeze) phase.

    Attributes:
        frame_counter (int): Animation frame counter.
        laser_position (int): Current laser scan position.
    """

    def __init__(self):
        """Initialize red light effects."""
        self.frame_counter = 0
        self.laser_position = 0
        hailo_logger.debug("RedLightEffects initialized")

    def render(
        self,
        canvas: np.ndarray,
        player_data: List,
        homography_transform_func,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Render red light effects.

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_data (List): List of player data objects.
            homography_transform_func: Function to transform coordinates.
            frame_width (int): Camera frame width.
            frame_height (int): Camera frame height.

        Returns:
            np.ndarray: Rendered canvas with effects.
        """
        self.frame_counter += 1

        # Dark red background
        canvas[:, :] = (15, 0, 40)

        # Draw scanning laser
        canvas = self._draw_scanning_laser(canvas)

        # Draw ice encasement on players
        for player in player_data:
            canvas = self._draw_ice_effect(
                canvas, player, homography_transform_func, frame_width, frame_height
            )

        # Add "FREEZE!" warning text
        canvas = self._draw_freeze_warning(canvas)

        return canvas

    def _draw_scanning_laser(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draw animated scanning laser effect.

        Args:
            canvas (np.ndarray): Canvas to draw on.

        Returns:
            np.ndarray: Canvas with laser drawn.
        """
        height, width = canvas.shape[:2]

        # Vertical scanning laser
        laser_y = int((self.frame_counter * 8) % height)

        # Draw laser line with glow
        cv2.line(canvas, (0, laser_y), (width, laser_y), (0, 0, 255), 4)
        cv2.line(canvas, (0, laser_y - 2), (width, laser_y - 2), (0, 0, 150), 2)
        cv2.line(canvas, (0, laser_y + 2), (width, laser_y + 2), (0, 0, 150), 2)

        return canvas

    def _draw_ice_effect(
        self,
        canvas: np.ndarray,
        player,
        transform_func,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Draw ice/crystal effect on frozen player.

        Args:
            canvas (np.ndarray): Canvas to draw on.
            player: Player data object.
            transform_func: Coordinate transform function.
            frame_width (int): Frame width.
            frame_height (int): Frame height.

        Returns:
            np.ndarray: Canvas with ice effect drawn.
        """
        bbox = player.bbox

        # Transform bbox corners
        cam_x_min = bbox["xmin"] * frame_width
        cam_y_min = bbox["ymin"] * frame_height
        cam_x_max = (bbox["xmin"] + bbox["width"]) * frame_width
        cam_y_max = (bbox["ymin"] + bbox["height"]) * frame_height

        proj_x_min, proj_y_min = transform_func(cam_x_min, cam_y_min)
        proj_x_max, proj_y_max = transform_func(cam_x_max, cam_y_max)

        # Ensure valid bounds
        proj_x_min = max(0, min(proj_x_min, canvas.shape[1] - 1))
        proj_y_min = max(0, min(proj_y_min, canvas.shape[0] - 1))
        proj_x_max = max(0, min(proj_x_max, canvas.shape[1]))
        proj_y_max = max(0, min(proj_y_max, canvas.shape[0]))

        if proj_x_max <= proj_x_min or proj_y_max <= proj_y_min:
            return canvas

        # Create ice overlay
        overlay = canvas[proj_y_min:proj_y_max, proj_x_min:proj_x_max].copy()

        # Cyan/blue ice color
        ice_color = np.full(overlay.shape, (255, 200, 100), dtype=np.uint8)

        # Blend with pulsating alpha
        alpha = 0.4 + 0.2 * np.sin(self.frame_counter * 0.15)
        canvas[proj_y_min:proj_y_max, proj_x_min:proj_x_max] = cv2.addWeighted(
            overlay, 1 - alpha, ice_color, alpha, 0
        )

        # Draw ice crystal pattern
        center_x = (proj_x_min + proj_x_max) // 2
        center_y = (proj_y_min + proj_y_max) // 2

        # Draw crystalline lines
        crystal_size = min(proj_x_max - proj_x_min, proj_y_max - proj_y_min) // 2
        for angle in [0, 60, 120, 180, 240, 300]:
            end_x = int(center_x + crystal_size * np.cos(np.radians(angle)))
            end_y = int(center_y + crystal_size * np.sin(np.radians(angle)))
            cv2.line(canvas, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2)

        return canvas

    def _draw_freeze_warning(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draw "FREEZE!" warning text.

        Args:
            canvas (np.ndarray): Canvas to draw on.

        Returns:
            np.ndarray: Canvas with warning text.
        """
        text = "FREEZE!"
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 3
        thickness = 6

        # Calculate text position
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (canvas.shape[1] - text_size[0]) // 2
        text_y = 120

        # Pulsating red color
        pulse = int(abs(np.sin(self.frame_counter * 0.2)) * 100 + 155)

        cv2.putText(
            canvas, text, (text_x, text_y), font, font_scale, (0, 0, pulse), thickness
        )

        return canvas

    def reset(self) -> None:
        """Reset animation state."""
        self.frame_counter = 0
        self.laser_position = 0
        hailo_logger.debug("RedLightEffects reset")

