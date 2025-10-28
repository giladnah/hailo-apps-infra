"""
Pattern generator for camera-projector calibration using checkerboard patterns.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


def generate_checkerboard_pattern(
    resolution: Tuple[int, int],
    rows: int = 8,
    cols: int = 11,
    square_size: int = 100,
    save_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a checkerboard calibration pattern.

    Args:
        resolution (Tuple[int, int]): Target resolution (width, height).
        rows (int): Number of interior corners in rows.
        cols (int): Number of interior corners in columns.
        square_size (int): Size of each square in pixels.
        save_path (Path, optional): Path to save the pattern image.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Pattern image (BGR format)
            - Array of corner points in projector space (rows*cols, 2)
    """
    width, height = resolution

    # Create white background
    pattern = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Calculate board dimensions (rows x cols interior corners = (rows+1) x (cols+1) squares)
    board_width = (cols + 1) * square_size
    board_height = (rows + 1) * square_size

    # Center the board
    offset_x = (width - board_width) // 2
    offset_y = (height - board_height) // 2

    hailo_logger.debug(
        f"Generating {rows}x{cols} checkerboard, board offset: ({offset_x}, {offset_y})"
    )

    # Draw checkerboard pattern and generate corner points simultaneously
    corner_points = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            # Draw square
            if row < rows + 1 and col < cols + 1:
                is_white = (row + col) % 2 == 0
                color = (255, 255, 255) if is_white else (0, 0, 0)
                pt1 = (offset_x + col * square_size, offset_y + row * square_size)
                pt2 = (offset_x + (col + 1) * square_size, offset_y + (row + 1) * square_size)
                cv2.rectangle(pattern, pt1, pt2, color, -1)

            # Store interior corner points (matching OpenCV's findChessboardCorners)
            if 1 <= row <= rows and 1 <= col <= cols:
                corner_points.append([offset_x + col * square_size, offset_y + row * square_size])

    corner_points = np.array(corner_points, dtype=np.float32)

    hailo_logger.info(f"Generated checkerboard with {len(corner_points)} corner points")

    # Validate pattern generation
    expected_corners = rows * cols
    if len(corner_points) != expected_corners:
        hailo_logger.error(f"Pattern generation error: expected {expected_corners} corners, got {len(corner_points)}")
        return None, None

    # Check if pattern fits in resolution
    if board_width > width or board_height > height:
        hailo_logger.error(f"Pattern too large: {board_width}x{board_height} > {width}x{height}")
        return None, None

    hailo_logger.info(f"Pattern validation passed: {len(corner_points)} corners, size {board_width}x{board_height}")

    # Save pattern if path provided
    if save_path:
        cv2.imwrite(str(save_path), pattern)
        hailo_logger.info(f"Saved checkerboard pattern to {save_path}")

    return pattern, corner_points


def detect_checkerboard_corners(
    image: np.ndarray,
    rows: int = 8,
    cols: int = 11,
    refine: bool = True,
    debug_dir: str = None,
) -> np.ndarray:
    """
    Detect checkerboard corners in the captured camera image.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        rows (int): Number of interior corner rows.
        cols (int): Number of interior corner columns.
        refine (bool): Whether to refine corner positions.
        debug_dir (str, optional): Directory to save debug images.

    Returns:
        np.ndarray: Array of detected corner points (rows*cols, 2) or None if detection failed.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_color = image.copy()
    else:
        gray = image
        original_color = gray

    hailo_logger.debug(f"Detecting checkerboard corners in image of shape {gray.shape}")

    # Save original images for debugging
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/00_original_gray.png", gray)
        cv2.imwrite(f"{debug_dir}/00_original_color.png", original_color)

    # OpenCV expects (cols, rows) for pattern size
    pattern_size = (cols, rows)

    # Detection flags
    detection_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # Try multiple detection approaches in order
    detection_approaches = [
        ("standard", lambda img: (img, detection_flags)),
        ("histogram equalization", lambda img: (cv2.equalizeHist(img), detection_flags)),
        ("CLAHE", lambda img: (cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img), detection_flags)),
        ("CLAHE_strong", lambda img: (cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(img), detection_flags)),
        ("gaussian_blur", lambda img: (cv2.GaussianBlur(img, (3, 3), 0), detection_flags)),
        ("minimal flags", lambda img: (img, cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)),
        ("no flags", lambda img: (img, 0)),
    ]

    ret, corners = False, None
    for approach_name, preprocess_func in detection_approaches:
        gray_preprocessed, flags = preprocess_func(gray)

        # Save debug image for all approaches
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/01_{approach_name}.png", gray_preprocessed)

        ret, corners = cv2.findChessboardCorners(gray_preprocessed, pattern_size, flags)
        if ret:
            hailo_logger.info(f"Success with {approach_name} detection")
            break
        hailo_logger.debug(f"Failed with {approach_name}, trying next approach...")

    if not ret:
        hailo_logger.error("Failed to find checkerboard pattern with all approaches")
        if debug_dir:
            summary = np.hstack([gray, cv2.equalizeHist(gray)])
            cv2.imwrite(f"{debug_dir}/03_failed_summary.png", summary)
        return None

    hailo_logger.info(f"Found checkerboard with {len(corners)} corners")

    # Save detection visualization
    if debug_dir:
        img_with_corners = original_color.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret)
        cv2.imwrite(f"{debug_dir}/04_detected_corners.png", img_with_corners)

    # Refine corner positions
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if debug_dir:
            img_with_refined = original_color.copy()
            cv2.drawChessboardCorners(img_with_refined, pattern_size, corners, ret)
            cv2.imwrite(f"{debug_dir}/05_refined_corners.png", img_with_refined)

    # Convert to simple 2D array format
    corner_points = corners.reshape(-1, 2).astype(np.float32)

    hailo_logger.info(f"Detected {len(corner_points)} corners (expected: {rows * cols})")

    return corner_points


def generate_charuco_pattern(
    resolution: Tuple[int, int],
    rows: int = 5,
    cols: int = 7,
    square_size: int = 100,
    marker_size: int = 50,
    save_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a ChArUco calibration pattern (checkerboard with ArUco markers).

    ChArUco provides unique corner identification, solving ordering issues.

    Args:
        resolution (Tuple[int, int]): Target resolution (width, height).
        rows (int): Number of squares in rows.
        cols (int): Number of squares in columns.
        square_size (int): Size of each square in pixels.
        marker_size (int): Size of ArUco markers in pixels.
        save_path (Path, optional): Path to save the pattern image.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Pattern image (BGR format)
            - Array of corner points in projector space (rows*cols, 2)
    """
    width, height = resolution

    # Create white background
    pattern = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Calculate board dimensions
    board_width = cols * square_size
    board_height = rows * square_size

    # Center the board
    offset_x = (width - board_width) // 2
    offset_y = (height - board_height) // 2

    hailo_logger.info(f"Generating {rows}x{cols} ChArUco pattern")

    # Create ChArUco board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        square_size / 50.0,  # Square length in marker units
        marker_size / 50.0,   # Marker length in marker units
        dictionary
    )

    # Generate the pattern
    board_img = board.generateImage((board_width, board_height))

    # Paste onto background
    pattern[
        offset_y : offset_y + board_height,
        offset_x : offset_x + board_width
    ] = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    # Get expected corners from board
    expected_corners = rows * cols
    hailo_logger.info(f"Generated ChArUco with {expected_corners} expected corners")

    # Save pattern if path provided
    if save_path:
        cv2.imwrite(str(save_path), pattern)
        hailo_logger.info(f"Saved ChArUco pattern to {save_path}")

    # Return pattern and None for corners (detection will provide them)
    return pattern, None


def detect_charuco_corners(
    image: np.ndarray,
    rows: int = 5,
    cols: int = 7,
    square_size: int = 100,
    marker_size: int = 50,
    debug_dir: str = None,
) -> np.ndarray:
    """
    Detect ChArUco board corners with unique ID tracking.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        rows (int): Number of rows.
        cols (int): Number of columns.
        square_size (int): Square size in pixels.
        marker_size (int): Marker size in pixels.
        debug_dir (str, optional): Directory to save debug images.

    Returns:
        np.ndarray: Array of detected corner points (rows*cols, 2) or None.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_color = image.copy()
    else:
        gray = image
        original_color = gray

    hailo_logger.debug(f"Detecting ChArUco corners in image of shape {gray.shape}")

    # Save original images for debugging
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/00_original_gray.png", gray)
        cv2.imwrite(f"{debug_dir}/00_original_color.png", original_color)

    # Create Charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        square_size / 50.0,
        marker_size / 50.0,
        dictionary
    )

    # Create ChArUco detector
    charuco_detector = cv2.aruco.CharucoDetector(board)

    # Detect ChArUco corners and IDs
    # detectBoard returns: charucoCorners, charucoIds, markerCorners, markerIds
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) == 0:
        hailo_logger.error("No ChArUco corners detected")
        if debug_dir:
            cv2.imwrite(f"{debug_dir}/01_no_charuco.png", gray)
        return None

    # Save detection visualization
    if debug_dir:
        img_with_corners = original_color.copy()
        if marker_corners is not None and len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(img_with_corners, marker_corners, marker_ids)
        cv2.imwrite(f"{debug_dir}/02_detected_markers.png", img_with_corners)

        img_with_charuco = original_color.copy()
        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(img_with_charuco, charuco_corners, charuco_ids)
        cv2.imwrite(f"{debug_dir}/03_detected_charuco.png", img_with_charuco)

    # Convert to simple 2D array format
    corner_points = charuco_corners.reshape(-1, 2).astype(np.float32)

    hailo_logger.info(f"Detected {len(corner_points)} ChArUco corners")

    return corner_points
