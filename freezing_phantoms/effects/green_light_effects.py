"""
Visual effects for GREEN_LIGHT game state.

Creates "safe zone" effects and ambient animations during move phase.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class GreenLightEffects:
    """
    Visual effects for the green light (move) phase.

    Attributes:
        frame_counter (int): Animation frame counter.
        particle_positions (List): Particle effect positions.
    """

    def __init__(self):
        """Initialize green light effects."""
        self.frame_counter = 0
        self.particle_positions = []
        hailo_logger.debug("GreenLightEffects initialized")

    def render(
        self,
        canvas: np.ndarray,
        player_bboxes: List[Dict],
        homography_transform_func,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Render green light effects.

        Args:
            canvas (np.ndarray): Canvas to render on.
            player_bboxes (List[Dict]): List of player bounding boxes.
            homography_transform_func: Function to transform camera to projector coords.
            frame_width (int): Camera frame width.
            frame_height (int): Camera frame height.

        Returns:
            np.ndarray: Rendered canvas with effects.
        """
        self.frame_counter += 1

        # Draw safe zones around players
        for bbox in player_bboxes:
            canvas = self._draw_safe_zone(
                canvas, bbox, homography_transform_func, frame_width, frame_height
            )

        # Draw ambient phantom swirls
        canvas = self._draw_phantom_swirls(canvas)

        return canvas

    def _draw_safe_zone(
        self,
        canvas: np.ndarray,
        bbox: Dict,
        transform_func,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Draw pulsating safe zone around player.

        Args:
            canvas (np.ndarray): Canvas to draw on.
            bbox (Dict): Player bounding box.
            transform_func: Homography transform function.
            frame_width (int): Frame width.
            frame_height (int): Frame height.

        Returns:
            np.ndarray: Canvas with safe zone drawn.
        """
        # Calculate bbox center in camera space
        cam_x = (bbox["xmin"] + bbox["width"] / 2) * frame_width
        cam_y = (bbox["ymin"] + bbox["height"] / 2) * frame_height

        # Transform to projector space
        proj_x, proj_y = transform_func(cam_x, cam_y)

        # Calculate radius based on bbox size
        bbox_width_px = bbox["width"] * frame_width
        bbox_height_px = bbox["height"] * frame_height
        base_radius = int(max(bbox_width_px, bbox_height_px) / 2) + 30

        # Pulsating animation
        pulse = np.sin(self.frame_counter * 0.15) * 15
        radius = int(base_radius + pulse)

        # Draw glowing green circle
        overlay = canvas.copy()
        cv2.circle(overlay, (proj_x, proj_y), radius, (0, 255, 100), -1)
        cv2.circle(overlay, (proj_x, proj_y), radius + 5, (100, 255, 150), 3)

        # Blend with canvas
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        return canvas

    def _draw_phantom_swirls(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draw ambient swirling phantom effects.

        Args:
            canvas (np.ndarray): Canvas to draw on.

        Returns:
            np.ndarray: Canvas with swirls drawn.
        """
        height, width = canvas.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw multiple swirling particles
        num_swirls = 5
        for i in range(num_swirls):
            angle = (self.frame_counter * 2 + i * (360 / num_swirls)) % 360
            radius = min(width, height) // 3

            x = int(center_x + radius * np.cos(np.radians(angle)))
            y = int(center_y + radius * np.sin(np.radians(angle)))

            # Draw ghostly particle
            color_intensity = int(abs(np.sin(self.frame_counter * 0.1 + i)) * 100 + 100)
            cv2.circle(canvas, (x, y), 15, (color_intensity, 255, color_intensity), -1)

            # Add glow
            cv2.circle(canvas, (x, y), 25, (0, 200, 0), 2)

        return canvas

    def reset(self) -> None:
        """Reset animation state."""
        self.frame_counter = 0
        self.particle_positions = []
        hailo_logger.debug("GreenLightEffects reset")

