"""
Camera-projector calibration tool.

Performs homography calibration by projecting a pattern and detecting it with the camera.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.calibration.pattern_generator import (
    detect_circle_centers,
    generate_calibration_pattern,
    sort_circle_centers,
)

hailo_logger = get_logger(__name__)


class CalibrationTool:
    """
    Tool for calibrating camera-projector homography mapping.

    Attributes:
        config (dict): Calibration configuration.
        camera_resolution (Tuple[int, int]): Camera resolution (width, height).
        projector_resolution (Tuple[int, int]): Projector resolution (width, height).
        pattern_rows (int): Number of rows in calibration pattern.
        pattern_cols (int): Number of columns in calibration pattern.
    """

    def __init__(self, config_path: Path):
        """
        Initialize the calibration tool.

        Args:
            config_path (Path): Path to game configuration YAML file.
        """
        hailo_logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.camera_resolution = tuple(self.config["camera"]["resolution"])
        self.projector_resolution = tuple(self.config["projector"]["resolution"])
        self.display_index = self.config["projector"]["display_index"]

        cal_config = self.config["calibration"]
        self.pattern_rows = cal_config["pattern_rows"]
        self.pattern_cols = cal_config["pattern_cols"]
        self.circle_radius = cal_config["circle_radius"]
        self.circle_spacing = cal_config["circle_spacing"]

        hailo_logger.info(
            f"Initialized calibration tool: "
            f"camera={self.camera_resolution}, "
            f"projector={self.projector_resolution}, "
            f"pattern={self.pattern_rows}x{self.pattern_cols}"
        )

    def _visualize_detected_points(
        self,
        image: np.ndarray,
        centers: np.ndarray,
        title: str = "Detected Calibration Points"
    ) -> None:
        """
        Visualize detected points on the image.

        Args:
            image (np.ndarray): Camera image.
            centers (np.ndarray): Detected circle centers.
            title (str): Window title.
        """
        vis_image = image.copy()

        # Draw detected circles in green
        for center in centers:
            cv2.circle(vis_image, tuple(center.astype(int)), 10, (0, 255, 0), 2)
            # Add semi-transparent fill
            overlay = vis_image.copy()
            cv2.circle(overlay, tuple(center.astype(int)), 15, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

        # Add information overlay
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)

        cv2.putText(vis_image, f"Detected: {len(centers)} points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_image, "Green circles = Detected points", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis_image, "Press any key to continue...", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow(title, vis_image)

    def _handle_insufficient_points(
        self,
        camera_image: np.ndarray,
        detected_centers: np.ndarray,
        camera_source: str,
        min_required_points: int,
        output_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Handle case when insufficient calibration points are detected.
        Shows visualization and allows parameter adjustment or retry.

        Args:
            camera_image (np.ndarray): Captured camera image.
            detected_centers (np.ndarray): Currently detected centers.
            camera_source (str): Camera device path.
            min_required_points (int): Minimum required points.
            output_path (Optional[Path]): Path to save calibration matrix.

        Returns:
            np.ndarray: Homography matrix if retry succeeds, None otherwise.
        """
        print("\n" + "="*70)
        print("CALIBRATION ADJUSTMENT MODE")
        print("="*70)

        # Show visualization of detected points
        self._visualize_detected_points(camera_image, detected_centers, "Current Detection - Press any key")
        print("\n✓ Showing detected points...")
        print(f"   Current: {len(detected_centers)} points detected")
        cv2.waitKey(0)

        print("\nOptions:")
        print("1. R/r - Retry with adjusted parameters")
        print("2. C/c - Capture new image")
        print("3. A/a - Accept current points (may be inaccurate)")
        print("4. Q/q or ESC - Cancel calibration")
        print("\n👉 Your choice: ", end="")

        # Interactive loop for adjustments
        while True:
            # Create adjustment window with trackbars
            adj_window = "Adjustment Mode - Press 'R' to retry, 'C' to recapture, 'A' to accept, ESC to cancel"
            cv2.namedWindow(adj_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(adj_window, 800, 600)

            # Current detection parameters
            param1_var = 50
            param2_var = 30
            min_dist_var = 150
            min_radius_var = max(40, self.circle_radius // 2)
            max_radius_var = min(250, self.circle_radius * 3)

            # Create trackbars for parameter adjustment
            cv2.createTrackbar('Param1', adj_window, param1_var, 200, lambda x: None)
            cv2.createTrackbar('Param2', adj_window, param2_var, 100, lambda x: None)
            cv2.createTrackbar('MinDist', adj_window, min_dist_var, 300, lambda x: None)
            cv2.createTrackbar('MinRadius', adj_window, min_radius_var, 150, lambda x: None)
            cv2.createTrackbar('MaxRadius', adj_window, max_radius_var, 300, lambda x: None)

            # Show instructions
            instruction_image = np.zeros((300, 600, 3), dtype=np.uint8)
            cv2.putText(instruction_image, "Adjustment Controls:", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(instruction_image, "R - Retry detection with current params", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(instruction_image, "C - Capture new image", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(instruction_image, "A - Accept current points", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(instruction_image, "ESC/Q - Cancel", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(instruction_image, f"Current: {len(detected_centers)} points", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(instruction_image, f"Required: {min_required_points} points", (10, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Instructions", instruction_image)

            # Show current detection result
            temp_vis = camera_image.copy()
            for center in detected_centers:
                cv2.circle(temp_vis, tuple(center.astype(int)), 10, (0, 255, 0), 2)
            cv2.imshow(adj_window, temp_vis)

            # Wait for user input
            key = cv2.waitKey(30) & 0xFF

            # Get current trackbar values
            if cv2.getWindowProperty(adj_window, cv2.WND_PROP_VISIBLE) >= 1:
                param1_var = cv2.getTrackbarPos('Param1', adj_window)
                param2_var = cv2.getTrackbarPos('Param2', adj_window)
                min_dist_var = cv2.getTrackbarPos('MinDist', adj_window)
                min_radius_var = cv2.getTrackbarPos('MinRadius', adj_window)
                max_radius_var = cv2.getTrackbarPos('MaxRadius', adj_window)

            # Handle key presses
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("Canceled")
                hailo_logger.info("Calibration adjustment cancelled by user")
                cv2.destroyAllWindows()
                return None

            elif key == ord('r') or key == ord('R'):
                print("Retrying detection with adjusted parameters...")
                hailo_logger.info(f"Retrying detection with params: param1={param1_var}, "
                                f"param2={param2_var}, minDist={min_dist_var}, "
                                f"minRadius={min_radius_var}, maxRadius={max_radius_var}")

                # Try detection with adjusted parameters
                new_centers = self._detect_with_params(
                    camera_image,
                    param1=param1_var if param1_var > 0 else 50,
                    param2=param2_var if param2_var > 0 else 30,
                    min_dist=min_dist_var if min_dist_var > 0 else 100,
                    min_radius=min_radius_var if min_radius_var > 0 else 40,
                    max_radius=max_radius_var if max_radius_var > 0 else 160
                )

                if new_centers is not None and len(new_centers) >= min_required_points:
                    print(f"✓ Success! Detected {len(new_centers)} points")
                    cv2.destroyAllWindows()
                    # Continue with new centers
                    detected_centers = new_centers
                    break
                else:
                    detected_count = len(new_centers) if new_centers is not None else 0
                    print(f"Still not enough points: {detected_count} < {min_required_points}")
                    if new_centers is not None:
                        detected_centers = new_centers
                        # Update visualization
                        temp_vis = camera_image.copy()
                        for center in detected_centers:
                            cv2.circle(temp_vis, tuple(center.astype(int)), 10, (0, 255, 0), 2)
                        cv2.imshow(adj_window, temp_vis)
                        cv2.waitKey(100)

            elif key == ord('c') or key == ord('C'):
                print("Capturing new image...")
                cv2.destroyAllWindows()
                # Recapture image and retry entire calibration
                return self.run_calibration(camera_source, output_path, min_required_points)

            elif key == ord('a') or key == ord('A'):
                print("Accepting current points (may be inaccurate)...")
                hailo_logger.warning("Accepting insufficient points for calibration")
                cv2.destroyAllWindows()
                # Continue with current detected centers
                break

        cv2.destroyAllWindows()

        # If we have the centers, continue with calibration using them
        if len(detected_centers) > 0:
            # Get expected pattern centers
            _, projector_centers = generate_calibration_pattern(
                resolution=self.projector_resolution,
                rows=self.pattern_rows,
                cols=self.pattern_cols,
                circle_radius=self.circle_radius,
                spacing=self.circle_spacing,
            )

            # Sort and match points
            camera_centers = sort_circle_centers(
                detected_centers, self.pattern_rows, self.pattern_cols
            )

            # Calculate homography
            min_points = min(len(camera_centers), len(projector_centers))
            camera_centers = camera_centers[:min_points]
            projector_centers = projector_centers[:min_points]

            print(f"\n🧮 Calculating homography with {min_points} points...")

            try:
                homography_matrix, mask = cv2.findHomography(
                    camera_centers, projector_centers, cv2.RANSAC, 5.0
                )

                if homography_matrix is not None:
                    inliers = np.sum(mask)
                    inlier_ratio = inliers / len(mask) * 100 if len(mask) > 0 else 0
                    print(f"✓ Homography calculated ({inliers}/{len(mask)} inliers, {inlier_ratio:.1f}%)")

                    # Save homography matrix if output_path provided
                    if output_path:
                        np.save(output_path, homography_matrix)
                        hailo_logger.info(f"Homography matrix saved to {output_path}")
                        print(f"\n✓ Calibration successful! Matrix saved to {output_path}\n")

                    return homography_matrix
                else:
                    print("❌ Homography calculation failed")
                    return None
            except Exception as e:
                hailo_logger.error(f"Failed to calculate homography: {e}")
                print(f"❌ Error: {e}")
                return None

        return None

    def _detect_with_params(
        self,
        image: np.ndarray,
        param1: int = 50,
        param2: int = 30,
        min_dist: int = 100,
        min_radius: int = 40,
        max_radius: int = 160
    ) -> Optional[np.ndarray]:
        """
        Detect circles with specific parameters.

        Args:
            image (np.ndarray): Input image.
            param1 (int): First HoughCircles parameter (edge detection).
            param2 (int): Second HoughCircles parameter (accumulator threshold).
            min_dist (int): Minimum distance between circles.
            min_radius (int): Minimum circle radius.
            max_radius (int): Maximum circle radius.

        Returns:
            np.ndarray: Detected centers or None.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None:
            return None

        circles = np.uint16(np.around(circles))
        centers = circles[0, :, :2].astype(np.float32)

        return centers

    def run_calibration(
        self,
        camera_source: str = "/dev/video0",
        output_path: Path = None,
        min_required_points: int = 12,
    ) -> Optional[np.ndarray]:
        """
        Run the calibration procedure.

        Args:
            camera_source (str): Camera device path or video file.
            output_path (Path, optional): Path to save calibration matrix.
            min_required_points (int): Minimum number of points needed for calibration.

        Returns:
            np.ndarray: Homography matrix (3x3) or None if calibration failed.
        """
        hailo_logger.info("Starting calibration procedure")

        # Generate calibration pattern
        pattern, projector_centers = generate_calibration_pattern(
            resolution=self.projector_resolution,
            rows=self.pattern_rows,
            cols=self.pattern_cols,
            circle_radius=self.circle_radius,
            spacing=self.circle_spacing,
        )

        # Open camera
        hailo_logger.info(f"Opening camera {camera_source}")
        # Try to parse as integer first (for compatibility with device indices)
        try:
            camera_source_parsed = int(camera_source)
        except (ValueError, TypeError):
            camera_source_parsed = camera_source

        cap = cv2.VideoCapture(camera_source_parsed)
        if not cap.isOpened():
            hailo_logger.error(f"Failed to open camera {camera_source}")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])

        # Create fullscreen window for projector
        window_name = "Calibration Pattern - Projector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Move window to projector display (if multiple displays)
        # Note: This is system-dependent; may need manual positioning
        cv2.moveWindow(window_name, self.projector_resolution[0], 0)

        # Display pattern
        cv2.imshow(window_name, pattern)
        cv2.waitKey(100)  # Give time for window to render

        hailo_logger.info("Calibration pattern displayed. Capturing camera image...")
        print("\n" + "="*70)
        print("CALIBRATION INSTRUCTIONS:")
        print("="*70)
        print("1. Calibration pattern is displayed on projector")
        print("2. Camera feed window will show below")
        print("3. Click on the camera feed window to ensure it has focus")
        print("4. Press SPACE to capture (or 'c' if SPACE doesn't work)")
        print("5. Press ESC to cancel (or 'q' to quit)")
        print("="*70)
        print("\n⏳ Waiting for camera to initialize...")

        # Wait for user input
        camera_image = None
        camera_feed_window = "Camera Feed - Press SPACE to capture, ESC to cancel"

        # Create window and make it visible
        cv2.namedWindow(camera_feed_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera_feed_window, 640, 480)

        # Give windows time to initialize
        time.sleep(0.5)

        frame_count = 0
        print("\n✓ Camera ready! Showing live feed...")
        print("👉 Click on the camera window and press SPACE to capture\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                hailo_logger.error("Failed to read from camera")
                cap.release()
                cv2.destroyAllWindows()
                return None

            frame_count += 1

            # Add instructions overlay on frame
            display_frame = frame.copy()

            # Add semi-transparent overlay at top
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            # Add text instructions
            cv2.putText(display_frame, "PRESS SPACE TO CAPTURE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press ESC to cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show frame number to confirm video is live
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show camera feed
            cv2.imshow(camera_feed_window, display_frame)

            # Try to keep window on top (may not work on all systems)
            try:
                cv2.setWindowProperty(camera_feed_window, cv2.WND_PROP_TOPMOST, 1)
            except:
                pass  # Some systems don't support this

            # Use longer wait and check multiple key codes
            key = cv2.waitKey(50) & 0xFF

            # Log key presses for debugging
            if key != 255:
                hailo_logger.info(f"Key detected: {key} (chr: '{chr(key) if 32 <= key < 127 else '?'}')")
                print(f"   Key detected: {key}")

            # Check for ESC or 'q'
            if key == 27 or key == ord('q') or key == ord('Q'):
                hailo_logger.info("Calibration cancelled by user")
                print("\n❌ Calibration cancelled\n")
                cap.release()
                cv2.destroyAllWindows()
                return None

            # Check for SPACE or 'c' (capture)
            elif key == 32 or key == ord('c') or key == ord('C'):
                camera_image = frame.copy()
                hailo_logger.info("Camera image captured")
                print("\n✓ Image captured!\n")
                break

        cap.release()

        # Detect circles in camera image
        expected_count = self.pattern_rows * self.pattern_cols
        hailo_logger.info(f"Detecting circles in camera image (expecting {expected_count})...")
        print(f"\n🔍 Detecting calibration points...")
        print(f"   Expected: {expected_count} circles")

        camera_centers = detect_circle_centers(
            camera_image,
            expected_count=expected_count,
            min_radius=self.circle_radius // 2,
            max_radius=self.circle_radius * 2,
        )

        if camera_centers is None:
            hailo_logger.error("Failed to detect any circles in camera image")
            print(f"❌ No circles detected!")
            print(f"   Check:")
            print(f"   • Camera focus and exposure")
            print(f"   • Pattern visibility and lighting")
            print(f"   • Projector brightness")
            cv2.destroyAllWindows()
            return None

        detected_count = len(camera_centers)
        print(f"   Detected: {detected_count} circles")

        # Check if we have enough points
        if detected_count < min_required_points:
            hailo_logger.error(
                f"Insufficient points detected: {detected_count} < {min_required_points}"
            )
            print(f"❌ Not enough points detected!")
            print(f"   Need at least {min_required_points} points")
            print(f"   Only found {detected_count} points")
            print(f"   Missing {min_required_points - detected_count} points")

            # Show visualization and allow adjustment
            return self._handle_insufficient_points(
                camera_image,
                camera_centers,
                camera_source,
                min_required_points,
                output_path
            )

        # Warn if significantly fewer than expected
        if detected_count < expected_count * 0.8:
            print(f"⚠️  Warning: Only {detected_count}/{expected_count} circles detected")
            print(f"   ({detected_count/expected_count*100:.1f}% of expected)")
        else:
            print(f"✅ Good detection: {detected_count}/{expected_count} circles")
            print(f"   ({detected_count/expected_count*100:.1f}% of expected)")

        # Sort camera centers to match projector pattern order
        hailo_logger.info("Sorting detected circle centers...")
        camera_centers = sort_circle_centers(
            camera_centers, self.pattern_rows, self.pattern_cols
        )

        # Ensure we have matching number of points
        if len(camera_centers) != len(projector_centers):
            hailo_logger.warning(
                f"Point count mismatch: camera={len(camera_centers)}, "
                f"projector={len(projector_centers)}"
            )
            # Use the minimum number of points
            min_points = min(len(camera_centers), len(projector_centers))
            print(f"⚠️  Using {min_points} points for calibration")
            camera_centers = camera_centers[:min_points]
            projector_centers = projector_centers[:min_points]
        else:
            print(f"✅ Using {len(camera_centers)} matched points for calibration")

        # Calculate homography matrix
        print(f"\n🧮 Calculating homography matrix...")
        hailo_logger.info("Calculating homography matrix...")

        try:
            homography_matrix, mask = cv2.findHomography(
                camera_centers, projector_centers, cv2.RANSAC, 5.0
            )
        except Exception as e:
            hailo_logger.error(f"Failed to calculate homography: {e}")
            print(f"❌ Homography calculation failed: {e}")
            cv2.destroyAllWindows()
            return None

        if homography_matrix is None:
            hailo_logger.error("Failed to calculate homography matrix")
            print(f"❌ Could not calculate homography matrix")
            print(f"   Points may be collinear or invalid")
            cv2.destroyAllWindows()
            return None

        # Calculate reprojection error
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(mask) * 100
        hailo_logger.info(f"Homography calculated with {inliers}/{len(mask)} inliers")

        print(f"✅ Homography matrix calculated")
        print(f"   Inliers: {inliers}/{len(mask)} ({inlier_ratio:.1f}%)")

        # Quality check
        if inlier_ratio < 70:
            print(f"⚠️  Warning: Low inlier ratio ({inlier_ratio:.1f}%)")
            print(f"   Calibration may be inaccurate")
        elif inlier_ratio < 90:
            print(f"✓ Acceptable quality ({inlier_ratio:.1f}%)")
        else:
            print(f"✓ Excellent quality ({inlier_ratio:.1f}%)")

        # Visualize detected points
        vis_image = camera_image.copy()
        for i, center in enumerate(camera_centers):
            cv2.circle(vis_image, tuple(center.astype(int)), 5, (0, 255, 0), -1)
            if i < 10:  # Label first few points
                cv2.putText(
                    vis_image,
                    str(i),
                    tuple(center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        cv2.imshow("Detected Centers", vis_image)
        hailo_logger.info("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save homography matrix
        if output_path:
            np.save(output_path, homography_matrix)
            hailo_logger.info(f"Homography matrix saved to {output_path}")
            print(f"\n✓ Calibration successful! Matrix saved to {output_path}\n")

        return homography_matrix

    def validate_calibration(
        self, homography_matrix: np.ndarray, camera_source: str = "/dev/video0"
    ) -> None:
        """
        Validate calibration by showing real-time projection alignment.

        Args:
            homography_matrix (np.ndarray): Homography matrix to validate.
            camera_source (str): Camera device path or video file.
        """
        hailo_logger.info("Starting calibration validation")

        # Open camera
        # Try to parse as integer first (for compatibility with device indices)
        try:
            camera_source_parsed = int(camera_source)
        except (ValueError, TypeError):
            camera_source_parsed = camera_source

        cap = cv2.VideoCapture(camera_source_parsed)
        if not cap.isOpened():
            hailo_logger.error(f"Failed to open camera {camera_source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])

        # Create fullscreen window for projector
        window_name = "Validation - Projector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window_name, self.projector_resolution[0], 0)

        print("\n" + "=" * 70)
        print("VALIDATION MODE:")
        print("="*70)
        print("Move your hand in the camera view.")
        print("A green circle should appear on the projector at the same location.")
        print("Press ESC to exit (or 'q' to quit).")
        print("="*70)
        print("\n⏳ Starting validation...")

        validation_camera_window = "Camera Feed - Validation (Press ESC to exit)"
        cv2.namedWindow(validation_camera_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(validation_camera_window, 640, 480)

        # Give windows time to initialize
        time.sleep(0.5)

        frame_count = 0
        print("✓ Validation active! Move your hand to test alignment.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Create projection frame
            projection = np.zeros(
                (self.projector_resolution[1], self.projector_resolution[0], 3),
                dtype=np.uint8,
            )

            # Get frame center as test point
            cam_h, cam_w = frame.shape[:2]
            test_point_cam = np.array([[[cam_w // 2, cam_h // 2]]], dtype=np.float32)

            # Transform to projector space
            test_point_proj = cv2.perspectiveTransform(test_point_cam, homography_matrix)
            proj_x, proj_y = test_point_proj[0][0].astype(int)

            # Draw circle on projection at transformed point
            if 0 <= proj_x < self.projector_resolution[0] and 0 <= proj_y < self.projector_resolution[1]:
                cv2.circle(projection, (proj_x, proj_y), 50, (0, 255, 0), -1)

            # Show camera feed with center point and instructions
            display_frame = frame.copy()

            # Draw center point
            cv2.circle(display_frame, (cam_w // 2, cam_h // 2), 10, (0, 0, 255), -1)
            cv2.circle(display_frame, (cam_w // 2, cam_h // 2), 15, (255, 255, 255), 2)

            # Add overlay with instructions
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            cv2.putText(display_frame, "VALIDATION MODE - Press ESC to exit", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, "Red dot should align with green circle on projector", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show frame count
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow(validation_camera_window, display_frame)
            cv2.imshow(window_name, projection)

            # Try to keep window on top for key input
            try:
                cv2.setWindowProperty(validation_camera_window, cv2.WND_PROP_TOPMOST, 1)
            except:
                pass

            key = cv2.waitKey(50) & 0xFF

            # Log key presses
            if key != 255:
                hailo_logger.info(f"Key detected in validation: {key}")
                print(f"   Key detected: {key}")

            # Check for ESC or 'q'
            if key == 27 or key == ord('q') or key == ord('Q'):
                hailo_logger.info("Validation ended by user")
                print("\n✓ Validation ended\n")
                break

        cap.release()
        cv2.destroyAllWindows()
        hailo_logger.info("Validation complete")


def main():
    """Main entry point for calibration tool."""
    parser = argparse.ArgumentParser(description="Freezing Phantoms Calibration Tool")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "game_config.yaml",
        help="Path to game configuration file",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default="/dev/video0",
        help="Video source (camera device path or video file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "calibration.npy",
        help="Output path for calibration matrix",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation mode (requires existing calibration)",
    )

    args = parser.parse_args()

    tool = CalibrationTool(args.config)

    if args.validate:
        # Load existing calibration
        if not args.output.exists():
            print(f"Error: Calibration file not found at {args.output}")
            sys.exit(1)

        homography = np.load(args.output)
        tool.validate_calibration(homography, args.video_source)
    else:
        # Run calibration
        homography = tool.run_calibration(args.video_source, args.output)
        if homography is None:
            print("Calibration failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()

