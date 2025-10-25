"""
Movement detector for Freezing Phantoms game.

Detects player movement by comparing pose keypoints between frames.
Includes smoothing and sensitivity controls for better stability.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class MovementDetector:
    """
    Detects movement by comparing pose keypoints over time.
    Includes smoothing and sensitivity controls for better stability.

    Attributes:
        threshold_pixels (float): Movement threshold in pixels.
        monitored_keypoints (List[str]): Keypoints to monitor for movement.
        smoothing_alpha (float): Smoothing factor for exponential moving average (0-1).
        pose_history_size (int): Number of poses to keep in history for smoothing.
        sensitivity_multiplier (float): Multiplier for sensitivity adjustment.
        pose_histories (Dict[int, deque]): Pose history for each track ID.
    """

    def __init__(
        self,
        threshold_pixels: float = 20.0,
        monitored_keypoints: List[str] = None,
        smoothing_alpha: float = 0.3,
        pose_history_size: int = 5,
        sensitivity_multiplier: float = 1.0,
        keypoint_confidence_threshold: float = 0.5
    ):
        """
        Initialize movement detector.

        Args:
            threshold_pixels (float): Movement threshold in pixels.
            monitored_keypoints (List[str], optional): List of keypoint names to monitor.
                Defaults to wrists, ankles, and shoulders.
            smoothing_alpha (float): Smoothing factor for exponential moving average (0-1).
                Higher values = more responsive, lower values = more stable.
            pose_history_size (int): Number of poses to keep in history for smoothing.
            sensitivity_multiplier (float): Multiplier for sensitivity adjustment.
            keypoint_confidence_threshold (float): Minimum confidence score for keypoints to use.
        """
        self.threshold_pixels = threshold_pixels
        self.smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))  # Clamp to [0,1]
        self.pose_history_size = max(1, pose_history_size)
        self.sensitivity_multiplier = max(0.1, sensitivity_multiplier)  # Minimum 0.1
        self.keypoint_confidence_threshold = max(0.0, min(1.0, keypoint_confidence_threshold))  # Clamp to [0,1]

        if monitored_keypoints is None:
            self.monitored_keypoints = [
                "left_wrist",
                "right_wrist",
                "left_ankle",
                "right_ankle",
                "left_shoulder",
                "right_shoulder",
            ]
        else:
            self.monitored_keypoints = monitored_keypoints

        # Pose history for smoothing - track_id -> deque of poses
        self.pose_histories: Dict[int, deque] = {}

        hailo_logger.info(
            "MovementDetector initialized: threshold=%spx, "
            "monitoring %d keypoints, "
            "smoothing_alpha=%s, "
            "sensitivity=%sx, "
            "keypoint_confidence_threshold=%s",
            threshold_pixels,
            len(self.monitored_keypoints),
            self.smoothing_alpha,
            self.sensitivity_multiplier,
            self.keypoint_confidence_threshold
        )

    def _smooth_pose(self, track_id: int, pose: Dict[str, tuple]) -> Dict[str, tuple]:
        """
        Apply smoothing to a pose using exponential moving average.

        Args:
            track_id (int): Track ID of the player.
            pose (Dict[str, tuple]): Current pose keypoints.

        Returns:
            Dict[str, tuple]: Smoothed pose keypoints.
        """
        if track_id not in self.pose_histories:
            self.pose_histories[track_id] = deque(maxlen=self.pose_history_size)

        history = self.pose_histories[track_id]

        if not history:
            # First pose - no smoothing needed
            history.append(pose)
            return pose

        # Apply exponential moving average
        smoothed_pose = {}
        last_pose = history[-1]

        for keypoint_name in self.monitored_keypoints:
            if keypoint_name in pose and keypoint_name in last_pose:
                x0, y0 = last_pose[keypoint_name]
                x1, y1 = pose[keypoint_name]

                # Exponential moving average
                smoothed_x = x0 + self.smoothing_alpha * (x1 - x0)
                smoothed_y = y0 + self.smoothing_alpha * (y1 - y0)

                smoothed_pose[keypoint_name] = (smoothed_x, smoothed_y)
            elif keypoint_name in pose:
                # New keypoint detected
                smoothed_pose[keypoint_name] = pose[keypoint_name]

        # Add to history
        history.append(smoothed_pose)
        return smoothed_pose

    def _get_effective_threshold(self) -> float:
        """
        Get the effective threshold adjusted by sensitivity multiplier.

        Returns:
            float: Adjusted threshold in pixels.
        """
        return self.threshold_pixels * self.sensitivity_multiplier

    def _filter_keypoints_by_confidence(
        self,
        pose_keypoints: Dict[str, tuple],
        pose_keypoint_confidences: Dict[str, float]
    ) -> Dict[str, tuple]:
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

        hailo_logger.debug(
            "Filtered keypoints: %d/%d passed confidence threshold (%.2f)",
            len(filtered_keypoints),
            len(pose_keypoints),
            self.keypoint_confidence_threshold
        )

        return filtered_keypoints

    def detect_movement(
        self, pose_t0: Dict[str, tuple], pose_t1: Dict[str, tuple]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if movement occurred between two poses.

        Args:
            pose_t0 (Dict[str, tuple]): Initial pose keypoints {keypoint_name: (x, y)}.
            pose_t1 (Dict[str, tuple]): Current pose keypoints {keypoint_name: (x, y)}.

        Returns:
            Tuple[bool, float, Optional[str]]:
                - Whether movement detected (True/False)
                - Maximum displacement in pixels
                - Name of keypoint with maximum movement (or None)
        """
        if not pose_t0 or not pose_t1:
            hailo_logger.debug("Empty pose data provided")
            return False, 0.0, None

        max_displacement = 0.0
        max_keypoint = None
        moved = False

        # Compare each monitored keypoint
        for keypoint_name in self.monitored_keypoints:
            if keypoint_name not in pose_t0 or keypoint_name not in pose_t1:
                continue

            # Calculate Euclidean distance
            x0, y0 = pose_t0[keypoint_name]
            x1, y1 = pose_t1[keypoint_name]

            displacement = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            if displacement > max_displacement:
                max_displacement = displacement
                max_keypoint = keypoint_name

            effective_threshold = self._get_effective_threshold()
            if displacement > effective_threshold:
                moved = True

        if moved:
            effective_threshold = self._get_effective_threshold()
            hailo_logger.debug(
                "Movement detected: %s moved %.1fpx (threshold: %.1fpx)",
                max_keypoint,
                max_displacement,
                effective_threshold
            )

        return moved, max_displacement, max_keypoint

    def detect_player_movement(
        self,
        players_t0: List,
        players_t1: List,
    ) -> Dict[int, Tuple[bool, float, Optional[str]]]:
        """
        Detect movement for multiple players by matching track IDs.
        Applies confidence filtering and smoothing to current poses before comparison.

        Args:
            players_t0 (List[PlayerData]): Initial player states.
            players_t1 (List[PlayerData]): Current player states.

        Returns:
            Dict[int, Tuple[bool, float, Optional[str]]]:
                Dictionary mapping track_id to (moved, displacement, keypoint).
        """
        movement_results = {}

        # Create lookup dictionary for T0 players
        players_t0_dict = {player.track_id: player for player in players_t0}

        # Check each current player against their initial state
        for player_t1 in players_t1:
            track_id = player_t1.track_id

            if track_id not in players_t0_dict:
                hailo_logger.debug(
                    "Track ID %d not found in initial state (new player)",
                    track_id
                )
                continue

            player_t0 = players_t0_dict[track_id]

            # Filter keypoints by confidence for both poses
            filtered_pose_t0 = self._filter_keypoints_by_confidence(
                player_t0.pose_keypoints,
                getattr(player_t0, 'pose_keypoint_confidences', {})
            )
            filtered_pose_t1 = self._filter_keypoints_by_confidence(
                player_t1.pose_keypoints,
                getattr(player_t1, 'pose_keypoint_confidences', {})
            )

            # Apply smoothing to current pose
            smoothed_pose_t1 = self._smooth_pose(track_id, filtered_pose_t1)

            # Compare poses (using filtered and smoothed poses)
            moved, displacement, keypoint = self.detect_movement(
                filtered_pose_t0, smoothed_pose_t1
            )

            movement_results[track_id] = (moved, displacement, keypoint)

        return movement_results

    def calculate_pose_displacement(
        self, pose_t0: Dict[str, tuple], pose_t1: Dict[str, tuple]
    ) -> float:
        """
        Calculate average displacement across all monitored keypoints.

        Args:
            pose_t0 (Dict[str, tuple]): Initial pose.
            pose_t1 (Dict[str, tuple]): Current pose.

        Returns:
            float: Average displacement in pixels.
        """
        displacements = []

        for keypoint_name in self.monitored_keypoints:
            if keypoint_name not in pose_t0 or keypoint_name not in pose_t1:
                continue

            x0, y0 = pose_t0[keypoint_name]
            x1, y1 = pose_t1[keypoint_name]

            displacement = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            displacements.append(displacement)

        if not displacements:
            return 0.0

        return sum(displacements) / len(displacements)

    def get_keypoint_velocity(
        self,
        pose_history: List[Dict[str, tuple]],
        keypoint_name: str,
        time_delta: float = 1.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate velocity of a specific keypoint over time.

        Args:
            pose_history (List[Dict[str, tuple]]): List of poses over time.
            keypoint_name (str): Name of keypoint to track.
            time_delta (float): Time between poses in seconds.

        Returns:
            Optional[Tuple[float, float]]: (vx, vy) velocity in pixels/second or None.
        """
        if len(pose_history) < 2:
            return None

        # Get first and last poses
        pose_start = pose_history[0]
        pose_end = pose_history[-1]

        if keypoint_name not in pose_start or keypoint_name not in pose_end:
            return None

        x0, y0 = pose_start[keypoint_name]
        x1, y1 = pose_end[keypoint_name]

        # Calculate velocity
        total_time = time_delta * (len(pose_history) - 1)
        vx = (x1 - x0) / total_time
        vy = (y1 - y0) / total_time

        return vx, vy

    def set_threshold(self, threshold_pixels: float) -> None:
        """
        Update movement threshold.

        Args:
            threshold_pixels (float): New threshold in pixels.
        """
        self.threshold_pixels = threshold_pixels
        hailo_logger.info("Movement threshold updated to %spx", threshold_pixels)

    def set_monitored_keypoints(self, keypoints: List[str]) -> None:
        """
        Update list of monitored keypoints.

        Args:
            keypoints (List[str]): New list of keypoint names.
        """
        self.monitored_keypoints = keypoints
        hailo_logger.info("Monitoring %d keypoints: %s", len(keypoints), keypoints)

    def set_smoothing_alpha(self, alpha: float) -> None:
        """
        Update smoothing factor.

        Args:
            alpha (float): Smoothing factor (0-1). Higher = more responsive, lower = more stable.
        """
        self.smoothing_alpha = max(0.0, min(1.0, alpha))
        hailo_logger.info("Smoothing alpha updated to %s", self.smoothing_alpha)

    def set_sensitivity_multiplier(self, multiplier: float) -> None:
        """
        Update sensitivity multiplier.

        Args:
            multiplier (float): Sensitivity multiplier (>0). Higher = more sensitive.
        """
        self.sensitivity_multiplier = max(0.1, multiplier)
        hailo_logger.info("Sensitivity multiplier updated to %sx", self.sensitivity_multiplier)

    def set_pose_history_size(self, size: int) -> None:
        """
        Update pose history size.

        Args:
            size (int): Number of poses to keep in history.
        """
        self.pose_history_size = max(1, size)
        # Update existing histories
        for history in self.pose_histories.values():
            history.maxlen = self.pose_history_size
        hailo_logger.info("Pose history size updated to %d", self.pose_history_size)

    def clear_pose_histories(self) -> None:
        """Clear all pose histories (useful for resetting smoothing)."""
        self.pose_histories.clear()
        hailo_logger.info("Pose histories cleared")

    def set_keypoint_confidence_threshold(self, threshold: float) -> None:
        """
        Update keypoint confidence threshold.

        Args:
            threshold (float): New confidence threshold (0-1).
        """
        self.keypoint_confidence_threshold = max(0.0, min(1.0, threshold))
        hailo_logger.info("Keypoint confidence threshold updated to %s", self.keypoint_confidence_threshold)

    def get_smoothing_stats(self) -> Dict[str, float]:
        """
        Get current smoothing statistics.

        Returns:
            Dict[str, float]: Statistics including effective threshold and smoothing parameters.
        """
        return {
            "threshold_pixels": self.threshold_pixels,
            "effective_threshold": self._get_effective_threshold(),
            "smoothing_alpha": self.smoothing_alpha,
            "sensitivity_multiplier": self.sensitivity_multiplier,
            "pose_history_size": self.pose_history_size,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold,
            "active_tracks": len(self.pose_histories)
        }

