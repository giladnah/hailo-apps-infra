"""
Movement detector for Freezing Phantoms game.

Uses a two-stage approach:
1. Smooth noisy raw keypoint positions (exponential moving average)
2. Calculate velocity from smoothed positions over rolling time window
3. Detect movement when velocity exceeds threshold consistently

This approach handles noisy measurements and distinguishes between
intentional movement and slow drift/sway.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class MovementDetector:
    """
    Detects movement using velocity-based rolling window approach.
    
    Two-stage process:
    1. Position Smoothing: Apply EMA to noisy raw keypoint positions
    2. Velocity Calculation: Calculate velocity from smoothed positions over time window
    
    Movement is detected when velocity exceeds threshold for multiple consecutive frames.

    Attributes:
        velocity_threshold (float): Velocity threshold in pixels/second.
        monitored_keypoints (List[str]): Keypoints to monitor for movement.
        position_smoothing_alpha (float): Smoothing for raw positions (0-1, lower = more smoothing).
        velocity_window_duration (float): Time window for velocity calculation (seconds).
        min_consecutive_frames (int): Consecutive frames above threshold to trigger detection.
        keypoint_confidence_threshold (float): Minimum confidence for keypoints.
        fps (float): Expected frame rate for time calculations.
    """

    def __init__(
        self,
        velocity_threshold: float = 100.0,  # pixels/second (not just pixels!)
        monitored_keypoints: List[str] = None,
        position_smoothing_alpha: float = 0.3,  # For smoothing raw positions
        velocity_window_duration: float = 0.3,  # seconds
        min_consecutive_frames: int = 2,
        keypoint_confidence_threshold: float = 0.5,
        fps: float = 30.0
    ):
        """
        Initialize velocity-based movement detector.

        Args:
            velocity_threshold (float): Velocity threshold in pixels/second.
            monitored_keypoints (List[str], optional): List of keypoint names to monitor.
            position_smoothing_alpha (float): Smoothing factor for raw positions (0-1).
                Lower = more smoothing (less noise), Higher = more responsive.
            velocity_window_duration (float): Time window for velocity calculation (seconds).
            min_consecutive_frames (int): Consecutive frames above threshold to trigger.
            keypoint_confidence_threshold (float): Minimum confidence score for keypoints.
            fps (float): Expected frame rate for time calculations.
        """
        self.velocity_threshold = velocity_threshold
        self.position_smoothing_alpha = max(0.0, min(1.0, position_smoothing_alpha))
        self.velocity_window_duration = velocity_window_duration
        self.min_consecutive_frames = max(1, min_consecutive_frames)
        self.keypoint_confidence_threshold = max(0.0, min(1.0, keypoint_confidence_threshold))
        self.fps = fps

        if monitored_keypoints is None:
            self.monitored_keypoints = [
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
            ]
        else:
            self.monitored_keypoints = monitored_keypoints

        # Smoothed pose history with timestamps: {track_id: deque of (timestamp, smoothed_pose)}
        max_history_frames = int(fps * velocity_window_duration * 2)  # Keep 2x window for safety
        self.pose_history_timestamped: Dict[int, deque] = {}
        
        # Last smoothed pose for EMA: {track_id: {keypoint: (x, y)}}
        self.last_smoothed_pose: Dict[int, Dict[str, Tuple[float, float]]] = {}
        
        # Consecutive violation counter: {track_id: int}
        self.consecutive_violations: Dict[int, int] = {}
        
        # Velocity cache for visualization: {track_id: {keypoint: (vx, vy)}}
        self.velocity_vectors: Dict[int, Dict[str, Tuple[float, float]]] = {}
        
        self.max_history_frames = max_history_frames

        hailo_logger.info(
            "MovementDetector initialized (velocity-based): "
            "threshold=%.1fpx/s, "
            "monitoring %d keypoints, "
            "position_smoothing_alpha=%.2f, "
            "velocity_window=%.2fs, "
            "min_consecutive_frames=%d, "
            "keypoint_confidence_threshold=%.2f, "
            "fps=%.1f",
            velocity_threshold,
            len(self.monitored_keypoints),
            self.position_smoothing_alpha,
            velocity_window_duration,
            min_consecutive_frames,
            keypoint_confidence_threshold,
            fps
        )

    def _smooth_keypoint_position(
        self, track_id: int, keypoint_name: str, raw_position: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Smooth a single keypoint position using exponential moving average.
        Stage 1: Position smoothing to reduce measurement noise.

        Args:
            track_id (int): Track ID of the player.
            keypoint_name (str): Name of keypoint.
            raw_position (Tuple[float, float]): Raw (noisy) position (x, y).

        Returns:
            Tuple[float, float]: Smoothed position (x, y).
        """
        if track_id not in self.last_smoothed_pose:
            self.last_smoothed_pose[track_id] = {}
        
        if keypoint_name not in self.last_smoothed_pose[track_id]:
            # First measurement - no smoothing
            self.last_smoothed_pose[track_id][keypoint_name] = raw_position
            return raw_position
        
        # Apply exponential moving average
        x_prev, y_prev = self.last_smoothed_pose[track_id][keypoint_name]
        x_raw, y_raw = raw_position
        
        alpha = self.position_smoothing_alpha
        x_smooth = x_prev + alpha * (x_raw - x_prev)
        y_smooth = y_prev + alpha * (y_raw - y_prev)
        
        smoothed = (x_smooth, y_smooth)
        self.last_smoothed_pose[track_id][keypoint_name] = smoothed
        
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
                 pose_confidences: Dict[str, float], current_time: Optional[float] = None) -> None:
        """
        Add a new pose measurement.
        Stage 1: Smooth raw positions, then add to timestamped history.

        Args:
            track_id (int): Track ID of player.
            raw_pose (Dict[str, Tuple[float, float]]): Raw pose keypoints.
            pose_confidences (Dict[str, float]): Keypoint confidence scores.
            current_time (Optional[float]): Timestamp (seconds). If None, uses time.time().
        """
        if current_time is None:
            current_time = time.time()
        
        # Filter by confidence
        filtered_pose = self._filter_keypoints_by_confidence(raw_pose, pose_confidences)
        
        if not filtered_pose:
            return
        
        # Smooth each keypoint position
        smoothed_pose = {}
        for keypoint_name, raw_position in filtered_pose.items():
            smoothed_position = self._smooth_keypoint_position(track_id, keypoint_name, raw_position)
            smoothed_pose[keypoint_name] = smoothed_position
        
        # Add to timestamped history
        if track_id not in self.pose_history_timestamped:
            self.pose_history_timestamped[track_id] = deque(maxlen=self.max_history_frames)
        
        self.pose_history_timestamped[track_id].append((current_time, smoothed_pose))

    def calculate_velocity(self, track_id: int) -> Tuple[float, Optional[str]]:
        """
        Calculate current velocity over rolling time window.
        Stage 2: Calculate velocity from smoothed positions.

        Args:
            track_id (int): Track ID of player.

        Returns:
            Tuple[float, Optional[str]]: (max_velocity in px/s, keypoint_name with max velocity)
        """
        if track_id not in self.pose_history_timestamped:
            return 0.0, None
        
        history = self.pose_history_timestamped[track_id]
        
        if len(history) < 2:
            return 0.0, None
        
        # Get poses within velocity window
        current_time = history[-1][0]
        window_start_time = current_time - self.velocity_window_duration
        
        window_poses = [(t, p) for t, p in history if t >= window_start_time]
        
        if len(window_poses) < 2:
            return 0.0, None
        
        # Calculate velocity for each keypoint
        max_velocity = 0.0
        max_keypoint = None
        velocity_dict = {}
        
        t_start, pose_start = window_poses[0]
        t_end, pose_end = window_poses[-1]
        time_delta = t_end - t_start
        
        if time_delta < 0.01:  # Avoid division by zero
            return 0.0, None
        
        for keypoint_name in self.monitored_keypoints:
            if keypoint_name in pose_start and keypoint_name in pose_end:
                x0, y0 = pose_start[keypoint_name]
                x1, y1 = pose_end[keypoint_name]
                
                # Calculate displacement and velocity
                dx = x1 - x0
                dy = y1 - y0
                displacement = math.sqrt(dx**2 + dy**2)
                velocity = displacement / time_delta  # pixels/second
                
                # Store velocity vector for visualization
                vx = dx / time_delta
                vy = dy / time_delta
                velocity_dict[keypoint_name] = (vx, vy)
                
                if velocity > max_velocity:
                    max_velocity = velocity
                    max_keypoint = keypoint_name
        
        # Cache velocity vectors for visualization
        if track_id not in self.velocity_vectors:
            self.velocity_vectors[track_id] = {}
        self.velocity_vectors[track_id] = velocity_dict
        
        return max_velocity, max_keypoint

    def detect_movement(self, track_id: int) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if player is currently moving.
        Requires sustained velocity over multiple consecutive frames.

        Args:
            track_id (int): Track ID of player.

        Returns:
            Tuple[bool, float, Optional[str]]: (movement_detected, velocity, keypoint_name)
        """
        velocity, keypoint = self.calculate_velocity(track_id)
        
        # Check if velocity exceeds threshold
        if velocity > self.velocity_threshold:
            # Increment consecutive violations
            if track_id not in self.consecutive_violations:
                self.consecutive_violations[track_id] = 0
            self.consecutive_violations[track_id] += 1
            
            # Require multiple consecutive frames above threshold
            if self.consecutive_violations[track_id] >= self.min_consecutive_frames:
                hailo_logger.debug(
                    "Movement detected for track %d: %s moving at %.1f px/s (threshold: %.1f px/s, %d consecutive frames)",
                    track_id, keypoint, velocity, self.velocity_threshold, self.consecutive_violations[track_id]
                )
                return True, velocity, keypoint
        else:
            # Reset counter if below threshold
            self.consecutive_violations[track_id] = 0
        
        return False, velocity, keypoint

    def detect_player_movement(
        self,
        players_current: List,
    ) -> Dict[int, Tuple[bool, float, Optional[str]]]:
        """
        Detect movement for multiple players.
        First adds current poses to history, then checks for movement.

        Args:
            players_current (List[PlayerData]): Current player states.

        Returns:
            Dict[int, Tuple[bool, float, Optional[str]]]:
                Dictionary mapping track_id to (moved, velocity, keypoint).
        """
        current_time = time.time()
        movement_results = {}
        
        # Add current poses to history (with smoothing)
        for player in players_current:
            track_id = player.track_id
            self.add_pose(
                track_id,
                player.pose_keypoints,
                getattr(player, 'pose_keypoint_confidences', {}),
                current_time
            )
        
        # Detect movement for each player
        for player in players_current:
            track_id = player.track_id
            moved, velocity, keypoint = self.detect_movement(track_id)
            movement_results[track_id] = (moved, velocity, keypoint)
        
        return movement_results

    def get_velocity_vectors_for_visualization(self, track_id: int) -> Dict[str, Tuple[float, float]]:
        """
        Get velocity vectors for visualization (arrows).

        Args:
            track_id (int): Track ID of player.

        Returns:
            Dict[str, Tuple[float, float]]: Keypoint name to (vx, vy) velocity in px/s.
        """
        return self.velocity_vectors.get(track_id, {})

    def get_smoothed_pose(self, track_id: int) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get the most recent smoothed pose for a player.

        Args:
            track_id (int): Track ID of player.

        Returns:
            Optional[Dict[str, Tuple[float, float]]]: Smoothed pose or None.
        """
        if track_id not in self.pose_history_timestamped:
            return None
        
        history = self.pose_history_timestamped[track_id]
        if not history:
            return None
        
        _, smoothed_pose = history[-1]
        return smoothed_pose

    def draw_velocity_arrows(
        self,
        canvas: np.ndarray,
        track_id: int,
        pose_keypoints: Dict[str, Tuple[float, float]],
        scale_factor: float = 0.05,
        color: Tuple[int, int, int] = (0, 255, 255)
    ) -> np.ndarray:
        """
        Draw velocity arrows on debug visualization.

        Args:
            canvas (np.ndarray): Image to draw on.
            track_id (int): Track ID of player.
            pose_keypoints (Dict[str, Tuple[float, float]]): Current keypoint positions.
            scale_factor (float): Scale factor for arrow length (velocity in px/s * scale).
            color (Tuple[int, int, int]): Arrow color (BGR).

        Returns:
            np.ndarray: Canvas with arrows drawn.
        """
        velocity_vectors = self.get_velocity_vectors_for_visualization(track_id)
        
        if not velocity_vectors:
            return canvas
        
        for keypoint_name, (vx, vy) in velocity_vectors.items():
            if keypoint_name not in pose_keypoints:
                continue
            
            x, y = pose_keypoints[keypoint_name]
            x, y = int(x), int(y)
            
            # Calculate arrow end point (scale velocity for visualization)
            end_x = int(x + vx * scale_factor)
            end_y = int(y + vy * scale_factor)
            
            # Calculate velocity magnitude
            velocity_mag = math.sqrt(vx**2 + vy**2)
            
            # Color intensity based on velocity
            if velocity_mag > self.velocity_threshold:
                # Above threshold - red
                arrow_color = (0, 0, 255)
            else:
                # Below threshold - yellow
                arrow_color = color
            
            # Draw arrow
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y), arrow_color, 2, tipLength=0.3)
            
            # Draw velocity text
            velocity_text = f"{velocity_mag:.0f}"
            cv2.putText(canvas, velocity_text, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1)
        
        return canvas

    def reset_player(self, track_id: int) -> None:
        """
        Reset all data for a specific player.

        Args:
            track_id (int): Track ID of player to reset.
        """
        if track_id in self.pose_history_timestamped:
            del self.pose_history_timestamped[track_id]
        if track_id in self.last_smoothed_pose:
            del self.last_smoothed_pose[track_id]
        if track_id in self.consecutive_violations:
            del self.consecutive_violations[track_id]
        if track_id in self.velocity_vectors:
            del self.velocity_vectors[track_id]

    def clear_all(self) -> None:
        """Clear all tracking data for all players."""
        self.pose_history_timestamped.clear()
        self.last_smoothed_pose.clear()
        self.consecutive_violations.clear()
        self.velocity_vectors.clear()
        hailo_logger.info("All movement detection data cleared")

    def set_threshold(self, velocity_threshold: float) -> None:
        """
        Update velocity threshold.

        Args:
            velocity_threshold (float): New threshold in pixels/second.
        """
        self.velocity_threshold = velocity_threshold
        hailo_logger.info("Velocity threshold updated to %.1f px/s", velocity_threshold)

    def set_position_smoothing_alpha(self, alpha: float) -> None:
        """
        Update position smoothing factor.

        Args:
            alpha (float): Smoothing factor (0-1). Lower = more smoothing.
        """
        self.position_smoothing_alpha = max(0.0, min(1.0, alpha))
        hailo_logger.info("Position smoothing alpha updated to %.2f", self.position_smoothing_alpha)

    def get_smoothing_stats(self) -> Dict:
        """
        Get current detector statistics.

        Returns:
            Dict: Statistics including thresholds and active tracks.
        """
        return {
            "velocity_threshold": self.velocity_threshold,
            "position_smoothing_alpha": self.position_smoothing_alpha,
            "velocity_window_duration": self.velocity_window_duration,
            "min_consecutive_frames": self.min_consecutive_frames,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold,
            "fps": self.fps,
            "active_tracks": len(self.pose_history_timestamped),
            "monitored_keypoints": len(self.monitored_keypoints)
        }
