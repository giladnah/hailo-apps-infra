"""
Visual effects for board tiles in Red Light Green Light game.

Provides lava animations, tile glows, and player indicators.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class BoardEffects:
    """
    Visual effects for game board tiles.

    Attributes:
        frame_counter (int): Animation frame counter.
        lava_texture (Optional[np.ndarray]): Loaded lava texture.
    """

    def __init__(self, lava_texture_path: Optional[str] = None):
        """
        Initialize board effects.

        Args:
            lava_texture_path (Optional[str]): Path to lava texture image.
        """
        self.frame_counter = 0
        self.lava_texture = None

        if lava_texture_path:
            self.load_lava_texture(lava_texture_path)

        hailo_logger.debug("BoardEffects initialized")

    def load_lava_texture(self, texture_path: str) -> None:
        """
        Load lava floor texture from file.

        Args:
            texture_path (str): Path to texture image file.
        """
        try:
            self.lava_texture = cv2.imread(texture_path)
            if self.lava_texture is not None:
                hailo_logger.info(
                    "Loaded lava texture: %dx%d",
                    self.lava_texture.shape[1],
                    self.lava_texture.shape[0],
                )
            else:
                hailo_logger.warning("Failed to load lava texture from %s", texture_path)
        except Exception as e:  # noqa: BLE001
            hailo_logger.error("Error loading lava texture: %s", e)

    def render_lava_background(
        self, canvas: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        """
        Render animated lava texture as background.

        Args:
            canvas (np.ndarray): Canvas to render on.
            alpha (float): Transparency of lava overlay (0-1).

        Returns:
            np.ndarray: Canvas with lava background.
        """
        if self.lava_texture is None:
            # Fallback: simple dark red/orange background
            canvas[:, :] = (20, 20, 80)  # Dark red
            return canvas

        # Resize texture to match canvas
        texture_resized = cv2.resize(
            self.lava_texture, (canvas.shape[1], canvas.shape[0])
        )

        # Blend with canvas
        cv2.addWeighted(texture_resized, alpha, canvas, 1 - alpha, 0, canvas)

        return canvas

    def render_tile_glow(
        self,
        canvas: np.ndarray,
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int,
        color: Tuple[int, int, int],
        intensity: float = 0.5,
    ) -> np.ndarray:
        """
        Render pulsating glow effect on a tile.

        Args:
            canvas (np.ndarray): Canvas to render on.
            tile_x (int): Tile top-left X coordinate.
            tile_y (int): Tile top-left Y coordinate.
            tile_width (int): Tile width.
            tile_height (int): Tile height.
            color (Tuple[int, int, int]): Glow color (BGR).
            intensity (float): Glow intensity (0-1).

        Returns:
            np.ndarray: Canvas with glow effect.
        """
        # Create pulsating effect
        pulse = np.sin(self.frame_counter * 0.1) * 0.3 + 0.7  # Range: 0.4 to 1.0
        alpha = intensity * pulse

        # Create overlay
        overlay = canvas.copy()

        # Draw filled rectangle
        cv2.rectangle(
            overlay,
            (tile_x, tile_y),
            (tile_x + tile_width, tile_y + tile_height),
            color,
            -1,
        )

        # Blend with canvas
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        return canvas

    def render_player_indicator(
        self,
        canvas: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int = 12,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Render player position indicator.

        Args:
            canvas (np.ndarray): Canvas to render on.
            center_x (int): Center X coordinate.
            center_y (int): Center Y coordinate.
            radius (int): Indicator radius (default: 12 for better accuracy).
            color (Tuple[int, int, int]): Indicator color (BGR).

        Returns:
            np.ndarray: Canvas with player indicator.
        """
        # Pulsating circle
        pulse = np.sin(self.frame_counter * 0.15) * 3
        current_radius = int(radius + pulse)

        # Draw outer ring
        cv2.circle(canvas, (center_x, center_y), current_radius, color, 2)

        # Draw inner dot
        cv2.circle(canvas, (center_x, center_y), 3, color, -1)

        return canvas

    def render_path_preview(
        self,
        canvas: np.ndarray,
        tiles: list,
        alpha: float = 0.2,
    ) -> np.ndarray:
        """
        Render subtle preview of upcoming path.

        Args:
            canvas (np.ndarray): Canvas to render on.
            tiles (list): List of tiles to highlight.
            alpha (float): Preview transparency (0-1).

        Returns:
            np.ndarray: Canvas with path preview.
        """
        if not tiles:
            return canvas

        overlay = canvas.copy()

        for tile in tiles:
            # Draw subtle highlight
            cv2.rectangle(
                overlay,
                (tile.proj_x, tile.proj_y),
                (tile.proj_x + tile.proj_width, tile.proj_y + tile.proj_height),
                (100, 255, 100),  # Light green
                -1,
            )

        # Blend
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        return canvas

    def update(self) -> None:
        """Update animation frame counter."""
        self.frame_counter += 1

    def reset(self) -> None:
        """Reset animation state."""
        self.frame_counter = 0
        hailo_logger.debug("BoardEffects reset")
