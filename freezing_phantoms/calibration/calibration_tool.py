"""
Camera-projector calibration tool using checkerboard patterns.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.calibration.pattern_generator import (
    detect_checkerboard_corners,
    generate_checkerboard_pattern,
)

hailo_logger = get_logger(__name__)


class CalibrationTool:
    """
    Tool for calibrating camera-projector homography mapping using checkerboard patterns.

    Attributes:
        config (dict): Calibration configuration.
        camera_resolution (Tuple[int, int]): Camera resolution (width, height).
        projector_resolution (Tuple[int, int]): Projector resolution (width, height).
        pattern_rows (int): Number of interior corner rows in checkerboard.
        pattern_cols (int): Number of interior corner columns in checkerboard.
        square_size (int): Size of each checkerboard square in pixels.
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
        self.square_size = cal_config.get("square_size", 100)

        hailo_logger.info(
            f"Initialized: camera={self.camera_resolution}, "
            f"projector={self.projector_resolution}, "
            f"pattern={self.pattern_rows}x{self.pattern_cols}, "
            f"square_size={self.square_size}"
        )

    def _visualize_detected_points(
        self, image: np.ndarray, corners: np.ndarray, title: str = "Detected Corners"
    ) -> None:
        """
        Visualize detected corners on the image.

        Args:
            image (np.ndarray): Camera image.
            corners (np.ndarray): Detected corner points.
            title (str): Window title.
        """
        vis_image = image.copy()

        # Draw detected corners
        for corner in corners:
            cv2.circle(vis_image, tuple(corner.astype(int)), 8, (0, 255, 0), 2)

        # Add text overlay
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0, vis_image)

        cv2.putText(
            vis_image,
            f"Detected: {len(corners)} points",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis_image,
            "Press any key to continue...",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow(title, vis_image)

    def run_calibration(
        self,
        camera_source: str = "/dev/video0",
        output_path: Path = None,
        min_required_points: int = None,
        auto_validate: bool = True,
        max_retries: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Run the calibration procedure with optional validation and retry logic.

        Args:
            camera_source (str): Camera device path or video file.
            output_path (Path, optional): Path to save calibration matrix.
            min_required_points (int, optional): Minimum number of points needed.
            auto_validate (bool): Whether to automatically validate calibration.
            max_retries (int): Maximum number of calibration attempts.

        Returns:
            np.ndarray: Homography matrix (3x3) or None if calibration failed.
        """
        # Set default minimum points (70% of expected)
        expected_points = self.pattern_rows * self.pattern_cols
        if min_required_points is None:
            min_required_points = int(expected_points * 0.7)

        hailo_logger.info(f"Starting calibration (pattern: {self.pattern_rows}x{self.pattern_cols})")

        # Create history directory for debugging
        history_dir = Path("calibration_history") / datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir.mkdir(parents=True, exist_ok=True)
        hailo_logger.info(f"Calibration history will be saved to: {history_dir}")

        for attempt in range(max_retries):
            print(f"\n{'='*70}")
            print(f"CALIBRATION ATTEMPT {attempt + 1}/{max_retries}")
            print(f"{'='*70}")

            if attempt > 0:
                print("🔄 Retrying calibration...")
                print("   Press SPACE to continue, ESC to exit")
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC
                    print("\n❌ Calibration cancelled by user\n")
                    return None

            # Generate checkerboard pattern
            pattern, projector_centers = generate_checkerboard_pattern(
                resolution=self.projector_resolution,
                rows=self.pattern_rows,
                cols=self.pattern_cols,
                square_size=self.square_size,
            )

            if pattern is None or projector_centers is None:
                hailo_logger.error("Failed to generate checkerboard pattern")
                print("❌ Failed to generate checkerboard pattern!")
                print("   Check pattern dimensions and resolution settings")
                continue  # Try next attempt

            hailo_logger.info(f"Generated checkerboard pattern with {len(projector_centers)} points")
            print(f"✓ Generated {self.pattern_rows}x{self.pattern_cols} pattern with {len(projector_centers)} corners")

            # Open camera (handle both device paths and numeric indices)
            hailo_logger.info(f"Opening camera {camera_source}")
            camera_source_parsed = (
                int(camera_source) if camera_source.isdigit() else camera_source
            )

            cap = cv2.VideoCapture(camera_source_parsed)
            if not cap.isOpened():
                hailo_logger.error(f"Failed to open camera {camera_source}")
                print(f"❌ Failed to open camera {camera_source}")
                continue  # Try next attempt

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])

            # Create fullscreen window for projector
            window_name = "Calibration Pattern"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # Move to projector display (HDMI-1 at position 0,0)
            cv2.moveWindow(window_name, 0, 0)

            # Display pattern
            # Note: Projectors may flip the image (mirror/reverse)
            # Common projector settings can cause:
            # 1. Horizontal flip (mirror mode) - reverses X axis
            # 2. Vertical flip (ceiling mount) - reverses Y axis
            # 3. Both flips - reverses both axes
            # This could explain axis inversions in calibration!
            cv2.imshow(window_name, pattern)
            cv2.waitKey(100)

            hailo_logger.info("Pattern displayed on projector")
            hailo_logger.warning(
                "If projector has any flip/mirror settings enabled, "
                "this will affect the calibration coordinate system"
            )

            hailo_logger.info("Pattern displayed. Capturing camera image...")
            print("\n" + "="*70)
            print("CALIBRATION")
            print("="*70)
            print("1. Checkerboard is displayed on projector")
            print("2. Press SPACE to capture image")
            print("3. Press ESC to cancel")
            print("="*70)
            print("\nWaiting for capture...\n")

            # Wait for user to capture
            camera_image = None
            camera_feed_window = "Camera Feed - Press SPACE to capture"
            cv2.namedWindow(camera_feed_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(camera_feed_window, 640, 480)

            time.sleep(0.5)

            frame_count = 0
            capture_successful = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    hailo_logger.error("Failed to read from camera")
                    print("❌ Failed to read from camera")
                    break

                frame_count += 1

                # Add instruction overlay
                display_frame = frame.copy()
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                cv2.putText(
                    display_frame,
                    "PRESS SPACE TO CAPTURE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Press ESC to cancel",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(camera_feed_window, display_frame)

                key = cv2.waitKey(50) & 0xFF

                if key == 27 or key == ord("q") or key == ord("Q"):  # ESC
                    hailo_logger.info("Calibration cancelled")
                    print("\n❌ Cancelled\n")
                    break

                elif key == 32 or key == ord("c") or key == ord("C"):  # SPACE or 'c'
                    camera_image = frame.copy()
                    hailo_logger.info("Image captured")
                    print("\n✓ Image captured\n")
                    capture_successful = True
                    break

            cap.release()
            cv2.destroyAllWindows()

            if not capture_successful:
                print("❌ Image capture failed or cancelled")
                continue  # Try next attempt

            # Detect checkerboard corners
            hailo_logger.info("Detecting corners in camera image...")
            print("🔍 Detecting checkerboard corners...")

            # Create debug directory for this attempt
            debug_dir = history_dir / f"attempt_{attempt + 1}"
            debug_dir.mkdir(parents=True, exist_ok=True)

            camera_centers = detect_checkerboard_corners(
                camera_image,
                rows=self.pattern_rows,
                cols=self.pattern_cols,
                refine=True,
                debug_dir=str(debug_dir),
            )

            if camera_centers is None:
                hailo_logger.error("No checkerboard detected")
                print("\n❌ No checkerboard detected!")
                print(f"   Debug images saved to: {debug_dir}")
                continue  # Try next attempt

            # Debug: Log corner positions
            hailo_logger.info(f"Camera corners shape: {camera_centers.shape}")
            hailo_logger.info(f"First 3 camera corners: {camera_centers[:3]}")
            hailo_logger.info(f"Projector corners shape: {projector_centers.shape}")
            hailo_logger.info(f"First 3 projector corners: {projector_centers[:3]}")

            detected_count = len(camera_centers)
            print(f"   Detected: {detected_count} corners")

            # Check if we have enough points
            if detected_count < min_required_points:
                hailo_logger.error(f"Insufficient points: {detected_count} < {min_required_points}")
                print(f"❌ Not enough points detected!")
                print(f"   Need at least {min_required_points} points")
                print(f"   Found {detected_count} points")
                print(f"   Debug images saved to: {debug_dir}")
                continue  # Try next attempt

            # Warn if significantly fewer than expected
            if detected_count < expected_points * 0.9:
                print(f"⚠️  Warning: Only {detected_count}/{expected_points} points detected")
            else:
                print(f"✅ Good detection: {detected_count}/{expected_points} points")

            # Match point counts
            min_points = min(len(camera_centers), len(projector_centers))
            if len(camera_centers) != len(projector_centers):
                print(f"⚠️  Using {min_points} points for calibration")
                camera_centers = camera_centers[:min_points]
                projector_centers = projector_centers[:min_points]
            else:
                print(f"✅ Using all {len(camera_centers)} points")

            # Calculate homography matrix
            print("\n🧮 Calculating homography matrix...")
            hailo_logger.info("Calculating homography...")

            try:
                # Calculate homography matrix - use OpenCV result directly
                # If axes are inverted, it's likely due to projector display flip settings
                # The homography matrix will naturally compensate for this
                homography_matrix, mask = cv2.findHomography(
                    camera_centers, projector_centers, cv2.RANSAC, 5.0
                )
            except (cv2.error, ValueError, TypeError) as e:
                hailo_logger.error("Homography calculation failed: %s", e)
                print(f"❌ Failed to calculate homography: {e}")
                continue  # Try next attempt

            if homography_matrix is None:
                hailo_logger.error("Homography is None")
                print("❌ Could not calculate homography")
                continue  # Try next attempt

            # Validate the transformation
            scale_x = homography_matrix[0, 0]
            scale_y = homography_matrix[1, 1]
            print(f"   Scale factors: X={scale_x:.3f}, Y={scale_y:.3f}")

            # Test the transformation quality
            test_points = [
                ("Center", (640, 360)),
                ("Right", (720, 360)),
                ("Left", (560, 360)),
                ("Up", (640, 300)),
                ("Down", (640, 420)),
            ]

            print("   Direction test:")
            for name, (cx, cy) in test_points:
                cam_pt = np.array([[[cx, cy]]], dtype=np.float32)
                proj_pt = cv2.perspectiveTransform(cam_pt, homography_matrix)
                px, py = proj_pt[0][0]
                within_bounds = 0 <= px < self.projector_resolution[0] and 0 <= py < self.projector_resolution[1]
                status = "✓" if within_bounds else "✗ OUT OF BOUNDS"
                print(f"     {name}: camera({cx},{cy}) -> projector({px:.1f},{py:.1f}) {status}")

            if abs(scale_x) > 3.0 or abs(scale_y) > 3.0:
                print("⚠️  Warning: Very large scale factors detected")
            elif scale_x < -1.5 or scale_y < -1.5:
                print("⚠️  Warning: Large negative scales (possible flip)")

            # Calculate quality metrics
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(mask) * 100
            hailo_logger.info(
                "Homography: %d/%d inliers (%.1f%%)", inliers, len(mask), inlier_ratio
            )

            print("✅ Homography calculated")
            print(f"   Inliers: {inliers}/{len(mask)} ({inlier_ratio:.1f}%)")

            # Quality check
            if inlier_ratio < 70:
                print(f"⚠️  Warning: Low quality ({inlier_ratio:.1f}%)")
                print("   This calibration may not be accurate enough")
                continue  # Try next attempt
            elif inlier_ratio < 90:
                print(f"✓ Acceptable quality ({inlier_ratio:.1f}%)")
            else:
                print(f"✓ Excellent quality ({inlier_ratio:.1f}%)")

            # Save homography matrix to history
            history_file = history_dir / f"homography_attempt_{attempt + 1}.npy"
            np.save(history_file, homography_matrix)
            hailo_logger.info(f"Saved homography to history: {history_file}")

            # Save transformation test results
            test_log_file = history_dir / f"transformation_test_{attempt + 1}.txt"
            with open(test_log_file, 'w') as f:
                f.write("Transformation Test Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Homography:\n{homography_matrix}\n\n")
                f.write("Test Points (camera -> projector):\n")
                for name, (cx, cy) in test_points:
                    cam_pt = np.array([[[cx, cy]]], dtype=np.float32)
                    proj_pt = cv2.perspectiveTransform(cam_pt, homography_matrix)
                    px, py = proj_pt[0][0]
                    within_bounds = 0 <= px < self.projector_resolution[0] and 0 <= py < self.projector_resolution[1]
                    f.write(f"{name}: ({cx},{cy}) -> ({px:.1f},{py:.1f}) "
                           f"{'✓' if within_bounds else '✗ OUT OF BOUNDS'}\n")
            hailo_logger.info(f"Saved transformation test to: {test_log_file}")

            # Visualize result
            self._visualize_detected_points(camera_image, camera_centers, "Calibration Result")
            print("\n✓ Calibration attempt successful!")
            print("   Press any key to continue to validation...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Run validation if requested
            if auto_validate:
                print(f"\n{'='*70}")
                print("VALIDATION TEST")
                print(f"{'='*70}")
                print("🔍 Testing calibration quality...")
                print()
                print("WHAT TO LOOK FOR:")
                print("• Red circle = camera center (on camera feed)")
                print("• Green circle = projector mapping (on projector screen)")
                print("• The green circle should align with the red circle")
                print("• Move around to test different positions")
                print()
                print("CONTROLS:")
                print("• SPACE = Accept this calibration (good alignment)")
                print("• ESC = Reject and try again (poor alignment)")
                print("• Q = Cancel calibration process")
                print("="*70)

                validation_result = self._run_validation_interactive(homography_matrix, camera_source)

                if validation_result == "accept":
                    print("\n✅ Calibration accepted!")
                    # Save final homography matrix
                    if output_path:
                        np.save(output_path, homography_matrix)
                        hailo_logger.info(f"Saved final calibration to {output_path}")
                        print(f"✓ Calibration saved to {output_path}")

                    # Save final result to history
                    final_file = history_dir / "final_homography.npy"
                    np.save(final_file, homography_matrix)
                    hailo_logger.info(f"Saved final homography to history: {final_file}")

                    return homography_matrix
                elif validation_result == "reject":
                    print("\n❌ Calibration rejected, trying again...")
                    continue  # Try next attempt
                else:  # cancelled
                    print("\n❌ Calibration cancelled by user")
                    return None
            else:
                # No validation, accept this calibration
                if output_path:
                    np.save(output_path, homography_matrix)
                    hailo_logger.info(f"Saved to {output_path}")
                    print(f"\n✓ Calibration successful! Saved to {output_path}\n")

                return homography_matrix

        # If we get here, all attempts failed
        print(f"\n❌ All {max_retries} calibration attempts failed!")
        print(f"   History saved to: {history_dir}")
        return None

    def _run_validation_interactive(
        self, homography_matrix: np.ndarray, camera_source: str = "/dev/video0"
    ) -> str:
        """
        Run interactive validation of calibration.

        Args:
            homography_matrix (np.ndarray): Homography matrix to validate.
            camera_source (str): Camera device path.

        Returns:
            str: "accept", "reject", or "cancel"
        """
        hailo_logger.info("Starting interactive validation")

        # Open camera (handle both device paths and numeric indices)
        camera_source_parsed = (
            int(camera_source) if camera_source.isdigit() else camera_source
        )

        cap = cv2.VideoCapture(camera_source_parsed)
        if not cap.isOpened():
            hailo_logger.error(f"Failed to open camera {camera_source}")
            print(f"❌ Failed to open camera {camera_source}")
            return "cancel"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])

        # Create fullscreen window
        window_name = "Validation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Move to projector display (HDMI-1 at position 0,0)
        cv2.moveWindow(window_name, 0, 0)

        validation_window = "Camera - Press SPACE to accept, ESC to reject"
        cv2.namedWindow(validation_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(validation_window, 640, 480)

        time.sleep(0.5)

        print("🔍 Enhanced Validation Mode")
        print("="*70)
        print("CONTROLS:")
        print("• WASD = Move test point (W=Up, S=Down, A=Left, D=Right)")
        print("• ARROW KEYS = Alternative movement (may have issues on some systems)")
        print("• + / - = Adjust movement speed")
        print("• SPACE = Accept this calibration")
        print("• ESC = Reject and retry")
        print("• Q = Cancel calibration")
        print("="*70)
        print("\nVISUALIZATION:")
        print("• Red circle on camera = interactive test point")
        print("• Cyan dots on camera = reference points")
        print("• Green circle on projector = transformed interactive point")
        print("• Yellow dots on projector = transformed reference points")
        print("• Check if green circle aligns with red circle on projector")
        print("="*70)
        print("\nNOTE:")
        print("• Use WASD for most reliable movement")
        print("• Move speed starts at 10px, adjust with +/-")
        print("• Watch console for movement feedback")
        print("="*70)

        # Test with a simple grid of points to verify transformation
        print("\n   Testing transformation with test points...")
        test_points = [(320, 180), (640, 360), (960, 540)]  # Left, center, right
        for tx, ty in test_points:
            test_pt = np.array([[[tx, ty]]], dtype=np.float32)
            result = cv2.perspectiveTransform(test_pt, homography_matrix)
            rx, ry = result[0][0]
            hailo_logger.info(f"Test point ({tx},{ty}) -> ({rx:.1f},{ry:.1f})")
            print(f"   Test: camera({tx},{ty}) -> projector({rx:.1f},{ry:.1f})")

        # Track current test point position
        test_point_x = self.camera_resolution[0] // 2  # Start at center
        test_point_y = self.camera_resolution[1] // 2
        move_speed = 10  # Pixels per keypress

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create projection frame
            projection = np.zeros(
                (self.projector_resolution[1], self.projector_resolution[0], 3),
                dtype=np.uint8,
            )

            # Show multiple test points for visual reference
            cam_h, cam_w = frame.shape[:2]

            # Fixed reference points (corners and center)
            reference_points = [
                (cam_w // 4, cam_h // 4),      # Top-left
                (3*cam_w // 4, cam_h // 4),     # Top-right
                (cam_w // 2, cam_h // 2),       # Center
                (cam_w // 4, 3*cam_h // 4),     # Bottom-left
                (3*cam_w // 4, 3*cam_h // 4),   # Bottom-right
            ]

            # Draw all reference points on camera frame
            display_frame = frame.copy()

            for i, (cx, cy) in enumerate(reference_points):
                # Transform to projector space
                cam_pt = np.array([[[cx, cy]]], dtype=np.float32)
                proj_pt = cv2.perspectiveTransform(cam_pt, homography_matrix)
                px, py = proj_pt[0][0]

                # Invert Y-axis if needed (check if moving up moves projection down)
                # This indicates Y-axis is flipped

                # Draw on camera frame (small reference dots)
                cv2.circle(display_frame, (cx, cy), 4, (0, 255, 255), -1)  # Cyan dots

                # Draw on projection if within bounds
                if 0 <= px < self.projector_resolution[0] and 0 <= py < self.projector_resolution[1]:
                    cv2.circle(projection, (int(px), int(py)), 4, (255, 255, 0), -1)  # Yellow dots

            # Draw the interactive test point (large and prominent)
            # Transform current test point
            cam_pt = np.array([[[test_point_x, test_point_y]]], dtype=np.float32)
            proj_pt = cv2.perspectiveTransform(cam_pt, homography_matrix)
            proj_x, proj_y = proj_pt[0][0]

            # Draw on camera frame (large, red)
            cv2.circle(display_frame, (test_point_x, test_point_y), 12, (0, 0, 255), -1)  # Red
            cv2.circle(display_frame, (test_point_x, test_point_y), 16, (255, 255, 255), 2)  # White border

            # Draw on projection if within bounds (large, green)
            if 0 <= proj_x < self.projector_resolution[0] and 0 <= proj_y < self.projector_resolution[1]:
                cv2.circle(projection, (int(proj_x), int(proj_y)), 12, (0, 255, 0), -1)  # Green
                cv2.circle(projection, (int(proj_x), int(proj_y)), 16, (255, 255, 255), 2)  # White border

                # Draw crosshair for alignment reference
                size = 40
                cv2.line(projection, (int(proj_x)-size, int(proj_y)), (int(proj_x)+size, int(proj_y)), (200, 200, 200), 2)
                cv2.line(projection, (int(proj_x), int(proj_y)-size), (int(proj_x), int(proj_y)+size), (200, 200, 200), 2)

            # Add status overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            # Show current position
            cv2.putText(
                display_frame,
                f"Test Point: ({test_point_x},{test_point_y})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Projection: ({int(proj_x) if 0<=proj_x<self.projector_resolution[0] else 'OUT'},{int(proj_y) if 0<=proj_y<self.projector_resolution[1] else 'OUT'})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                "ARROWS=Move | SPACE=Accept | ESC=Reject | Q=Cancel",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow(validation_window, display_frame)
            cv2.imshow(window_name, projection)

            # Enhanced key handling with arrow keys
            # Use longer timeout to ensure arrow keys are detected
            key = cv2.waitKey(100) & 0xFF

            # Debug: print key codes for troubleshooting
            if key != 255:  # 255 means no key pressed
                hailo_logger.debug(f"Key pressed: {key}")

            # Check for special keys (arrows, +, -)
            # Note: WASD works reliably, arrow keys have system-dependent issues
            # User reports right arrow sometimes detected as down (84)
            if key == ord('w') or key == ord('W') or key == 82:  # W or Up arrow
                test_point_y = max(0, test_point_y - move_speed)
                print(f"Moved UP to ({test_point_x}, {test_point_y})")
            elif key == ord('s') or key == ord('S') or key == 84:  # S or Down arrow
                test_point_y = min(cam_h - 1, test_point_y + move_speed)
                print(f"Moved DOWN to ({test_point_x}, {test_point_y})")
            elif key == ord('a') or key == ord('A') or key == 81:  # A or Left arrow
                test_point_x = max(0, test_point_x - move_speed)
                print(f"Moved LEFT to ({test_point_x}, {test_point_y})")
            elif key == ord('d') or key == ord('D') or key == 83:  # D or Right arrow
                test_point_x = min(cam_w - 1, test_point_x + move_speed)
                print(f"Moved RIGHT to ({test_point_x}, {test_point_y})")
            # Note: Some systems detect right arrow as multiple codes
            # Using WASD as primary method for reliability
            elif key == ord('+') or key == ord('='):  # Increase speed
                move_speed = min(50, move_speed + 5)
                print(f"Move speed: {move_speed}px")
            elif key == ord('-') or key == ord('_'):  # Decrease speed
                move_speed = max(1, move_speed - 5)
                print(f"Move speed: {move_speed}px")
            elif key == 32:  # SPACE
                print("\n✅ Calibration accepted!")
                cap.release()
                cv2.destroyAllWindows()
                return "accept"
            elif key == 27:  # ESC
                print("\n❌ Calibration rejected!")
                cap.release()
                cv2.destroyAllWindows()
                return "reject"
            elif key == ord("q") or key == ord("Q"):  # Q
                print("\n❌ Calibration cancelled!")
                cap.release()
                cv2.destroyAllWindows()
                return "cancel"

        cap.release()
        cv2.destroyAllWindows()
        return "cancel"

    def validate_calibration(
        self, homography_matrix: np.ndarray, camera_source: str = "/dev/video0"
    ) -> None:
        """
        Validate calibration by showing real-time projection alignment.

        Args:
            homography_matrix (np.ndarray): Homography matrix to validate.
            camera_source (str): Camera device path.
        """
        hailo_logger.info("Starting validation")

        # Open camera (handle both device paths and numeric indices)
        camera_source_parsed = (
            int(camera_source) if camera_source.isdigit() else camera_source
        )

        cap = cv2.VideoCapture(camera_source_parsed)
        if not cap.isOpened():
            hailo_logger.error(f"Failed to open camera {camera_source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])

        # Create fullscreen window
        window_name = "Validation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Move to projector display (HDMI-1 at position 0,0)
        cv2.moveWindow(window_name, 0, 0)

        print("\n" + "="*70)
        print("VALIDATION MODE")
        print("="*70)
        print("Testing homography matrix transformation.")
        print("Red circle shows camera center, green circle shows projector mapping.")
        print("Press ESC to exit.")
        print("="*70)
        print()

        validation_window = "Camera - Press ESC to exit"
        cv2.namedWindow(validation_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(validation_window, 640, 480)

        time.sleep(0.5)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create projection frame
            projection = np.zeros(
                (self.projector_resolution[1], self.projector_resolution[0], 3),
                dtype=np.uint8,
            )

            # Transform frame center to projector space
            cam_h, cam_w = frame.shape[:2]
            test_point_cam = np.array([[[cam_w // 2, cam_h // 2]]], dtype=np.float32)
            test_point_proj = cv2.perspectiveTransform(test_point_cam, homography_matrix)
            proj_x, proj_y = test_point_proj[0][0].astype(int)

            # Draw circle on projection
            if (
                0 <= proj_x < self.projector_resolution[0]
                and 0 <= proj_y < self.projector_resolution[1]
            ):
                cv2.circle(projection, (proj_x, proj_y), 50, (0, 255, 0), -1)

            # Show camera feed
            display_frame = frame.copy()
            cv2.circle(display_frame, (cam_w // 2, cam_h // 2), 10, (0, 0, 255), -1)
            cv2.circle(display_frame, (cam_w // 2, cam_h // 2), 15, (255, 255, 255), 2)

            # Add overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            cv2.putText(
                display_frame,
                "VALIDATION - Press ESC to exit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow(validation_window, display_frame)
            cv2.imshow(window_name, projection)

            key = cv2.waitKey(50) & 0xFF
            if key == 27 or key == ord("q") or key == ord("Q"):  # ESC
                print("\n✓ Validation ended\n")
                break

        cap.release()
        cv2.destroyAllWindows()
        hailo_logger.info("Validation complete")
