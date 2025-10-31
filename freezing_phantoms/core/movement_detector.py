"""
Movement detector for Freezing Phantoms game.

Simplified approach:
1. Smooth noisy keypoint positions using exponential moving average (EMA)
2. When freeze starts, store reference and bootstrap for 1 second
3. After bootstrap, compare current smoothed positions to reference positions
4. Detect movement when displacement exceeds threshold consistently

This approach is simpler and more robust to transition artifacts.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class MovementDetector:
    """
    Detects movement by comparing current positions to frozen reference positions.

    Simple and robust approach:
    1. Smooth noisy keypoint positions using exponential moving average (EMA)
    2. When freeze starts, store reference positions
    3. Compare current smoothed positions to reference
    4. Trigger when displacement exceeds threshold for consecutive frames

    Attributes:
        displacement_threshold (float): Displacement threshold in pixels.
        monitored_keypoints (List[str]): Keypoints to monitor for movement.
        smoothing_alpha (float): Smoothing for positions (0-1, lower = more smoothing).
        min_consecutive_frames (int): Consecutive frames above threshold to trigger detection.
        keypoint_confidence_threshold (float): Minimum confidence for keypoints.
    """

    def __init__(
        self,
        displacement_threshold: float = 50.0,
        monitored_keypoints: List[str] = None,
        smoothing_alpha: float = 0.3,
        min_consecutive_frames: int = 3,
        keypoint_confidence_threshold: float = 0.3,
        bootstrap_duration: float = 1.0,
    ):
        """
        Initialize displacement-based movement detector.

        Args:
            displacement_threshold (float): Displacement threshold in pixels (higher = less sensitive).
            monitored_keypoints (List[str], optional): List of keypoint names to monitor.
            smoothing_alpha (float): Smoothing factor for positions (0-1).
                Lower = more smoothing (less noise), Higher = more responsive.
            min_consecutive_frames (int): Consecutive frames above threshold to trigger.
            keypoint_confidence_threshold (float): Minimum confidence score for keypoints.
            bootstrap_duration (float): Time to smooth reference pose after freeze starts (seconds).
        """
        self.displacement_threshold = displacement_threshold
        self.smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))
        self.min_consecutive_frames = max(1, min_consecutive_frames)
        self.keypoint_confidence_threshold = max(0.0, min(1.0, keypoint_confidence_threshold))
        self.bootstrap_duration = bootstrap_duration

        if monitored_keypoints is None:
            self.monitored_keypoints = [
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
            ]
        else:
            self.monitored_keypoints = monitored_keypoints

        # Current smoothed pose: {track_id: {keypoint: (x, y)}}
        self.smoothed_pose: Dict[int, Dict[str, Tuple[float, float]]] = {}

        # Reference frozen pose: {track_id: {keypoint: (x, y)}}
        self.reference_pose: Dict[int, Dict[str, Tuple[float, float]]] = {}

        # Reference start time for bootstrap: {track_id: timestamp}
        self.reference_start_time: Dict[int, float] = {}

        # Consecutive violation counter: {track_id: int}
        self.consecutive_violations: Dict[int, int] = {}

        # Frame counter for warm-up: {track_id: int}
        self.frame_count: Dict[int, int] = {}

        # Max displacement tracking for freeze period: {track_id: (max_displacement, keypoint)}
        self.max_displacement_in_period: Dict[int, Tuple[float, Optional[str]]] = {}

        hailo_logger.info(
            "MovementDetector initialized: "
            "displacement_threshold=%.1fpx, "
            "monitoring %d keypoints, "
            "smoothing_alpha=%.2f, "
            "min_consecutive_frames=%d, "
            "keypoint_confidence_threshold=%.2f, "
            "bootstrap_duration=%.1fs",
            self.displacement_threshold,
            len(self.monitored_keypoints),
            self.smoothing_alpha,
            self.min_consecutive_frames,
            self.keypoint_confidence_threshold,
            self.bootstrap_duration
        )

    def _smooth_keypoint_position(
        self, track_id: int, keypoint_name: str, raw_position: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Smooth a single keypoint position using exponential moving average (EMA).

        Args:
            track_id (int): Track ID of the player.
            keypoint_name (str): Name of keypoint.
            raw_position (Tuple[float, float]): Raw (noisy) position (x, y).

        Returns:
            Tuple[float, float]: Smoothed position (x, y).
        """
        if track_id not in self.smoothed_pose:
            self.smoothed_pose[track_id] = {}

        if keypoint_name not in self.smoothed_pose[track_id]:
            # First measurement - no smoothing
            self.smoothed_pose[track_id][keypoint_name] = raw_position
            return raw_position

        # Apply exponential moving average: smoothed = prev + alpha * (raw - prev)
        x_prev, y_prev = self.smoothed_pose[track_id][keypoint_name]
        x_raw, y_raw = raw_position

        x_smooth = x_prev + self.smoothing_alpha * (x_raw - x_prev)
        y_smooth = y_prev + self.smoothing_alpha * (y_raw - y_prev)

        smoothed = (x_smooth, y_smooth)
        self.smoothed_pose[track_id][keypoint_name] = smoothed

        return smoothed

    def _filter_keypoints_by_confidence(
        self,
        pose_keypoints: Dict[str, Tuple[float, float]],
        pose_keypoint_confidences: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Filter keypoints based on confidence scores.

        Args:
            pose_keypoints (Dict[str, tuple]): Dictionary of keypoint names to (x, y) coordinates.
            pose_keypoint_confidences (Dict[str, float]): Dictionary of keypoint names to confidence scores.

        Returns:
            Dict[str, tuple]: Filtered keypoints with only high-confidence ones.
        """
        filtered_keypoints = {}

        for keypoint_name in self.monitored_keypoints:
            if (keypoint_name in pose_keypoints and
                keypoint_name in pose_keypoint_confidences and
                pose_keypoint_confidences[keypoint_name] >= self.keypoint_confidence_threshold):
                filtered_keypoints[keypoint_name] = pose_keypoints[keypoint_name]

        return filtered_keypoints

    def add_pose(self, track_id: int, raw_pose: Dict[str, Tuple[float, float]],
                 pose_confidences: Dict[str, float]) -> None:
        """
        Update smoothed pose with new raw measurements.

        Args:
            track_id (int): Track ID of player.
            raw_pose (Dict[str, Tuple[float, float]]): Raw pose keypoints.
            pose_confidences (Dict[str, float]): Keypoint confidence scores.
        """
        # Filter by confidence
        filtered_pose = self._filter_keypoints_by_confidence(raw_pose, pose_confidences)

        # Smooth each keypoint position
        for keypoint_name, raw_position in filtered_pose.items():
            self._smooth_keypoint_position(track_id, keypoint_name, raw_position)

        # Update frame count for warm-up
        if track_id not in self.frame_count:
            self.frame_count[track_id] = 0
        self.frame_count[track_id] += 1

    def set_reference_pose(self, track_id: int) -> None:
        """
        Store current smoothed pose as the reference (frozen) pose.
        Call this when red light starts. Starts bootstrap period.

        Args:
            track_id (int): Track ID of player.
        """
        if track_id not in self.smoothed_pose:
            hailo_logger.warning(
                "Cannot set reference pose for track %d: no smoothed pose available", track_id
            )
            return

        # Deep copy the current smoothed pose as reference
        self.reference_pose[track_id] = self.smoothed_pose[track_id].copy()

        # Store the time when reference was set (for bootstrap)
        self.reference_start_time[track_id] = time.time()

        # Reset consecutive violations counter
        self.consecutive_violations[track_id] = 0

        # Reset max displacement tracking for new freeze period
        self.max_displacement_in_period[track_id] = (0.0, None)

        hailo_logger.debug(
            "[Track %d] Set reference pose with %d keypoints, starting %.1fs bootstrap",
            track_id, len(self.reference_pose[track_id]), self.bootstrap_duration
        )

    def calculate_displacement(self, track_id: int) -> Tuple[float, Optional[str]]:
        """
        Calculate displacement from reference (frozen) pose.

        Args:
            track_id (int): Track ID of player.

        Returns:
            Tuple[float, Optional[str]]: (max_displacement in pixels, keypoint_name with max displacement)
        """
        if track_id not in self.reference_pose:
            return 0.0, None

        if track_id not in self.smoothed_pose:
            return 0.0, None

        reference = self.reference_pose[track_id]
        current = self.smoothed_pose[track_id]

        max_displacement = 0.0
        max_keypoint = None

        for keypoint_name in self.monitored_keypoints:
            if keypoint_name not in reference or keypoint_name not in current:
                continue

            x_ref, y_ref = reference[keypoint_name]
            x_cur, y_cur = current[keypoint_name]

            # Calculate Euclidean distance
            displacement = math.sqrt((x_cur - x_ref)**2 + (y_cur - y_ref)**2)

            if displacement > max_displacement:
                max_displacement = displacement
                max_keypoint = keypoint_name

        return max_displacement, max_keypoint

    def detect_movement(self, track_id: int, debug_log: bool = False) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if player moved from reference (frozen) pose.
        Requires displacement above threshold for multiple consecutive frames.

        Args:
            track_id (int): Track ID of player.
            debug_log (bool): Ignored, kept for compatibility.

        Returns:
            Tuple[bool, float, Optional[str]]: (movement_detected, displacement, keypoint_name)
        """
        # Need at least 10 frames of smoothing before detecting (warm-up period)
        min_frames_for_warmup = 10
        if track_id not in self.frame_count or self.frame_count[track_id] < min_frames_for_warmup:
            return False, 0.0, None

        # Check if we're still in bootstrap period (smoothing reference pose)
        if track_id in self.reference_start_time:
            elapsed = time.time() - self.reference_start_time[track_id]
            if elapsed < self.bootstrap_duration:
                hailo_logger.debug(
                    "[Track %d] Bootstrap: %.2fs / %.1fs",
                    track_id, elapsed, self.bootstrap_duration
                )
                return False, 0.0, None

        # Calculate displacement from reference pose
        displacement, keypoint = self.calculate_displacement(track_id)

        # Track max displacement during this freeze period
        if track_id in self.max_displacement_in_period:
            current_max, _ = self.max_displacement_in_period[track_id]
            if displacement > current_max:
                self.max_displacement_in_period[track_id] = (displacement, keypoint)

        # Check if displacement exceeds threshold
        if displacement > self.displacement_threshold:
            # Increment consecutive violations
            if track_id not in self.consecutive_violations:
                self.consecutive_violations[track_id] = 0
            self.consecutive_violations[track_id] += 1

            # Require multiple consecutive frames above threshold
            if self.consecutive_violations[track_id] >= self.min_consecutive_frames:
                hailo_logger.warning(
                    "Movement detected for track %d: %s moved %.1f px (threshold: %.1f px)",
                    track_id, keypoint, displacement, self.displacement_threshold
                )
                return True, displacement, keypoint
        else:
            # Reset counter if below threshold
            self.consecutive_violations[track_id] = 0

        return False, displacement, keypoint

    def get_freeze_period_summary(self, track_id: int) -> Tuple[float, Optional[str]]:
        """
        Get max displacement during the current freeze period and reset it.

        Args:
            track_id (int): Track ID of player.

        Returns:
            Tuple[float, Optional[str]]: (max_displacement, keypoint_name)
        """
        if track_id not in self.max_displacement_in_period:
            return 0.0, None

        max_disp, keypoint = self.max_displacement_in_period[track_id]
        # Reset for next period
        self.max_displacement_in_period[track_id] = (0.0, None)
        return max_disp, keypoint

    def get_smoothed_pose(self, track_id: int) -> Dict[str, Tuple[float, float]]:
        """
        Get the current smoothed pose for a player.

        Args:
            track_id (int): Track ID of player.

        Returns:
            Dict[str, Tuple[float, float]]: Smoothed keypoint positions.
        """
        if track_id not in self.smoothed_pose:
            return {}
        return self.smoothed_pose[track_id].copy()

    def detect_player_movement(
        self,
        players_current: List,
        debug_log: bool = False,
    ) -> Dict[int, Tuple[bool, float, Optional[str]]]:
        """
        Detect movement for multiple players.
        First updates smoothed poses, then checks for movement.

        Args:
            players_current (List[PlayerData]): Current player states.
            debug_log (bool): If True, log displacement details to terminal.

        Returns:
            Dict[int, Tuple[bool, float, Optional[str]]]:
                Dictionary mapping track_id to (moved, displacement, keypoint).
        """
        movement_results = {}

        # Update smoothed poses
        for player in players_current:
            track_id = player.track_id
            self.add_pose(
                track_id,
                player.pose_keypoints,
                getattr(player, 'pose_keypoint_confidences', {})
            )

        # Detect movement for each player
        for player in players_current:
            track_id = player.track_id
            moved, displacement, keypoint = self.detect_movement(track_id, debug_log=debug_log)
            movement_results[track_id] = (moved, displacement, keypoint)

        return movement_results

    def draw_displacement_arrows(
        self,
        canvas: np.ndarray,
        track_id: int,
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """
        Draw displacement arrows from reference to current position.

        Args:
            canvas (np.ndarray): Image to draw on.
            track_id (int): Track ID of player.
            color (Tuple[int, int, int]): Arrow color (BGR).

        Returns:
            np.ndarray: Canvas with arrows drawn.
        """
        if track_id not in self.reference_pose or track_id not in self.smoothed_pose:
            return canvas

        reference = self.reference_pose[track_id]
        current = self.smoothed_pose[track_id]

        for keypoint_name in self.monitored_keypoints:
            if keypoint_name not in reference or keypoint_name not in current:
                continue

            x_ref, y_ref = reference[keypoint_name]
            x_cur, y_cur = current[keypoint_name]

            # Calculate displacement
            displacement = math.sqrt((x_cur - x_ref)**2 + (y_cur - y_ref)**2)

            # Color based on threshold
            if displacement > self.displacement_threshold:
                arrow_color = (0, 0, 255)  # Red - above threshold
            else:
                arrow_color = color  # Yellow - below threshold

            # Draw arrow from reference to current
            cv2.arrowedLine(canvas, (int(x_ref), int(y_ref)), (int(x_cur), int(y_cur)),
                          arrow_color, 2, tipLength=0.3)

            # Draw displacement text
            cv2.putText(canvas, f"{displacement:.0f}px", (int(x_cur) + 5, int(y_cur) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1)

        return canvas

    def reset_player(self, track_id: int) -> None:
        """
        Reset all data for a specific player.

        Args:
            track_id (int): Track ID of player to reset.
        """
        if track_id in self.smoothed_pose:
            del self.smoothed_pose[track_id]
        if track_id in self.reference_pose:
            del self.reference_pose[track_id]
        if track_id in self.reference_start_time:
            del self.reference_start_time[track_id]
        if track_id in self.consecutive_violations:
            del self.consecutive_violations[track_id]
        if track_id in self.frame_count:
            del self.frame_count[track_id]
        if track_id in self.max_displacement_in_period:
            del self.max_displacement_in_period[track_id]

        hailo_logger.debug("[Track %d] Reset all movement detection data", track_id)

    def clear_all(self) -> None:
        """Clear all tracking data for all players."""
        self.smoothed_pose.clear()
        self.reference_pose.clear()
        self.reference_start_time.clear()
        self.consecutive_violations.clear()
        self.frame_count.clear()
        self.max_displacement_in_period.clear()
        hailo_logger.info("All movement detection data cleared")

    def set_threshold(self, displacement_threshold: float) -> None:
        """
        Update displacement threshold.

        Args:
            displacement_threshold (float): New threshold in pixels.
        """
        self.displacement_threshold = displacement_threshold
        hailo_logger.info("Displacement threshold updated to %.1f px", displacement_threshold)

    def set_smoothing_alpha(self, alpha: float) -> None:
        """
        Update position smoothing factor.

        Args:
            alpha (float): Smoothing factor (0-1). Lower = more smoothing.
        """
        self.smoothing_alpha = max(0.0, min(1.0, alpha))
        hailo_logger.info("Smoothing alpha updated to %.2f", self.smoothing_alpha)

    def get_stats(self) -> Dict:
        """
        Get current detector statistics.

        Returns:
            Dict: Statistics including thresholds and active tracks.
        """
        return {
            "displacement_threshold": self.displacement_threshold,
            "smoothing_alpha": self.smoothing_alpha,
            "min_consecutive_frames": self.min_consecutive_frames,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold,
            "active_tracks": len(self.smoothed_pose),
            "monitored_keypoints": len(self.monitored_keypoints),
            "tracks_with_reference": len(self.reference_pose)
        }
