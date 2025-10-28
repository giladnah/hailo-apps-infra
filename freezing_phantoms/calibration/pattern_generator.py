"""
Pattern generator for camera-projector calibration.

Creates calibration patterns (grids of circles) for homography calculation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


def generate_calibration_pattern(
    resolution: Tuple[int, int],
    rows: int = 6,
    cols: int = 8,
    circle_radius: int = 40,
    spacing: int = 150,
    save_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a calibration pattern with a grid of circles.

    Args:
        resolution (Tuple[int, int]): Target resolution (width, height).
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        circle_radius (int): Radius of each circle in pixels.
        spacing (int): Spacing between circle centers in pixels.
        save_path (Path, optional): Path to save the pattern image.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Pattern image (BGR format)
            - Array of circle centers in projector space (cols*rows, 2)
    """
    width, height = resolution

    # Create black background
    pattern = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate grid dimensions
    grid_width = (cols - 1) * spacing + 2 * circle_radius
    grid_height = (rows - 1) * spacing + 2 * circle_radius

    # Center the grid in the image
    offset_x = (width - grid_width) // 2 + circle_radius
    offset_y = (height - grid_height) // 2 + circle_radius

    hailo_logger.debug(
        f"Generating {rows}x{cols} pattern, grid offset: ({offset_x}, {offset_y})"
    )

    # Store circle centers
    circle_centers = []

    # Draw circles
    for row in range(rows):
        for col in range(cols):
            center_x = offset_x + col * spacing
            center_y = offset_y + row * spacing

            # Draw white circle
            cv2.circle(pattern, (center_x, center_y), circle_radius, (255, 255, 255), -1)

            # Draw black center dot for easier detection
            cv2.circle(pattern, (center_x, center_y), 5, (0, 0, 0), -1)

            circle_centers.append([center_x, center_y])

    circle_centers = np.array(circle_centers, dtype=np.float32)

    hailo_logger.info(f"Generated calibration pattern with {len(circle_centers)} circles")

    # Save pattern if path provided
    if save_path:
        cv2.imwrite(str(save_path), pattern)
        hailo_logger.info(f"Saved calibration pattern to {save_path}")

    return pattern, circle_centers


def detect_circle_centers(
    image: np.ndarray, expected_count: int = 20, min_radius: int = 40, max_radius: int = 160
) -> np.ndarray:
    """
    Detect circle centers in the captured camera image.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        expected_count (int): Expected number of circles.
        min_radius (int): Minimum circle radius to detect.
        max_radius (int): Maximum circle radius to detect.

    Returns:
        np.ndarray: Array of detected circle centers (N, 2) or None if detection failed.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hailo_logger.debug(f"Detecting circles in image of shape {gray.shape}")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        hailo_logger.error("No circles detected in the image")
        return None

    circles = np.uint16(np.around(circles))
    centers = circles[0, :, :2].astype(np.float32)

    hailo_logger.info(f"Detected {len(centers)} circles (expected: {expected_count})")

    if len(centers) < expected_count * 0.8:  # Allow 20% tolerance
        hailo_logger.warning(
            f"Detected fewer circles than expected: {len(centers)} < {expected_count}"
        )

    return centers


def sort_circle_centers(
    centers: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    """
    Sort detected circle centers into a consistent grid order (left-to-right, top-to-bottom).

    Args:
        centers (np.ndarray): Unordered circle centers (N, 2).
        rows (int): Expected number of rows.
        cols (int): Expected number of columns.

    Returns:
        np.ndarray: Sorted circle centers (rows*cols, 2).
    """
    hailo_logger.debug(f"Sorting {len(centers)} centers into {rows}x{cols} grid")

    # Sort by y-coordinate to group into rows
    centers_sorted = centers[centers[:, 1].argsort()]

    # Split into rows (k-means clustering on y-coordinate would be more robust)
    sorted_grid = []
    points_per_row = len(centers) // rows

    for i in range(rows):
        # Get points in this row
        start_idx = i * points_per_row
        end_idx = start_idx + points_per_row if i < rows - 1 else len(centers)
        row_points = centers_sorted[start_idx:end_idx]

        # Sort this row by x-coordinate (left to right)
        row_points = row_points[row_points[:, 0].argsort()]
        sorted_grid.extend(row_points)

    sorted_centers = np.array(sorted_grid, dtype=np.float32)

    hailo_logger.info(f"Sorted centers into consistent grid order")

    return sorted_centers

