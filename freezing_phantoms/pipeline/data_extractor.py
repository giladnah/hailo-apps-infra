"""
Data extractor for Hailo AI pipeline.

Extracts structured player information from Hailo buffer data.
"""

import time
from typing import Dict, List, Optional, Tuple

import hailo

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


# COCO keypoint mapping
COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


class PlayerData:
    """
    Structured data for a single player.

    Attributes:
        track_id (int): Unique tracking ID.
        bbox (Dict[str, float]): Bounding box (xmin, ymin, width, height).
        confidence (float): Detection confidence.
        pose_keypoints (Dict[str, tuple]): Dictionary of keypoint names to (x, y) coordinates.
        pose_keypoint_confidences (Dict[str, float]): Dictionary of keypoint names to confidence scores.
        leg_position (Optional[Tuple[float, float]]): Detected leg position (x, y) in camera coordinates.
        leg_confidence (float): Confidence score for leg position detection.
    """

    def __init__(
        self,
        track_id: int,
        bbox: Dict[str, float],
        confidence: float,
        pose_keypoints: Dict[str, tuple],
        pose_keypoint_confidences: Dict[str, float] = None,
        leg_position: Optional[Tuple[float, float]] = None,
        leg_confidence: float = 0.0,
    ):
        """
        Initialize player data.

        Args:
            track_id (int): Player tracking ID.
            bbox (Dict[str, float]): Bounding box dictionary.
            confidence (float): Detection confidence score.
            pose_keypoints (Dict[str, tuple]): Pose keypoint coordinates.
            pose_keypoint_confidences (Dict[str, float], optional): Keypoint confidence scores.
            leg_position (Optional[Tuple[float, float]]): Detected leg position in camera coordinates.
            leg_confidence (float): Confidence score for leg position detection.
        """
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.pose_keypoints = pose_keypoints
        self.pose_keypoint_confidences = pose_keypoint_confidences or {}
        self.leg_position = leg_position
        self.leg_confidence = leg_confidence

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format.

        Returns:
            Dict: Player data as dictionary.
        """
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "pose_keypoints": self.pose_keypoints,
            "pose_keypoint_confidences": self.pose_keypoint_confidences,
            "leg_position": self.leg_position,
            "leg_confidence": self.leg_confidence,
        }


class DataExtractor:
    """
    Extracts player data from Hailo buffer.

    Combines detection, pose estimation, and segmentation data into structured format.
    """

    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize data extractor.

        Args:
            frame_width (int): Video frame width in pixels.
            frame_height (int): Video frame height in pixels.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        hailo_logger.info(
            "DataExtractor initialized for %dx%d frames",
            frame_width, frame_height
        )

    def extract_players(self, buffer) -> Dict:
        """
        Extract all player data from Hailo buffer.

        Args:
            buffer: GStreamer buffer containing Hailo metadata.

        Returns:
            Dict: Structured player data with timestamp.
                {
                    "players": [PlayerData, ...],
                    "timestamp": float
                }
        """
        players = []
        timestamp = time.time()

        # Get ROI from buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        hailo_logger.debug("Processing %d detections", len(detections))

        for detection in detections:
            label = detection.get_label()

            # Only process person detections
            if label != "person":
                continue

            # Extract basic detection info
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()

            # Get tracking ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()

            # Extract pose keypoints and confidence scores
            pose_keypoints, pose_keypoint_confidences = self._extract_pose_keypoints(detection, bbox)

            # Calculate leg position from pose keypoints (average of ankle positions and bbox bottom)
            leg_position, leg_confidence = self._calculate_leg_position_from_pose(
                pose_keypoints, pose_keypoint_confidences, bbox
            )

            # Create player data object
            player = PlayerData(
                track_id=track_id,
                bbox={
                    "xmin": bbox.xmin(),
                    "ymin": bbox.ymin(),
                    "width": bbox.width(),
                    "height": bbox.height(),
                },
                confidence=confidence,
                pose_keypoints=pose_keypoints,
                pose_keypoint_confidences=pose_keypoint_confidences,
                leg_position=leg_position,
                leg_confidence=leg_confidence,
            )

            players.append(player)
            hailo_logger.debug(
                "Extracted player: track_id=%d, keypoints=%d, leg_pos=%s",
                track_id,
                len(pose_keypoints),
                leg_position
            )

        return {"players": players, "timestamp": timestamp}

    def _extract_pose_keypoints(
        self, detection, bbox
    ) -> tuple[Dict[str, tuple], Dict[str, float]]:
        """
        Extract pose keypoints and confidence scores from detection.

        Args:
            detection: Hailo detection object.
            bbox: Bounding box object.

        Returns:
            tuple: (keypoints_dict, confidences_dict) where:
                - keypoints_dict: Dictionary mapping keypoint names to (x, y) pixel coordinates.
                - confidences_dict: Dictionary mapping keypoint names to confidence scores.
        """
        keypoints = {}
        confidences = {}

        # Get landmarks from detection
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)

        if not landmarks:
            hailo_logger.debug("No landmarks found in detection")
            return keypoints, confidences

        points = landmarks[0].get_points()

        # Convert normalized coordinates to pixel coordinates
        for keypoint_name, keypoint_index in COCO_KEYPOINTS.items():
            if keypoint_index < len(points):
                point = points[keypoint_index]

                # Transform from bbox-relative to frame-absolute coordinates
                x = (point.x() * bbox.width() + bbox.xmin()) * self.frame_width
                y = (point.y() * bbox.height() + bbox.ymin()) * self.frame_height

                keypoints[keypoint_name] = (x, y)
                confidences[keypoint_name] = point.confidence()

        hailo_logger.debug("Extracted %d keypoints with confidence scores", len(keypoints))
        return keypoints, confidences

    def _calculate_leg_position_from_pose(
        self, pose_keypoints: Dict[str, tuple], pose_keypoint_confidences: Dict[str, float], bbox
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Calculate leg position from pose keypoints.

        Uses the average between ankle positions and the bottom of the detection box
        to better approximate foot location on the ground.

        Args:
            pose_keypoints: Dictionary of keypoint names to (x, y) coordinates.
            pose_keypoint_confidences: Dictionary of keypoint names to confidence scores.
            bbox: Bounding box object containing detection boundaries.

        Returns:
            Tuple[Optional[Tuple[float, float]], float]:
                (leg_position, confidence) where leg_position is (x, y) in camera coordinates
        """
        left_ankle = pose_keypoints.get("left_ankle")
        right_ankle = pose_keypoints.get("right_ankle")

        left_conf = pose_keypoint_confidences.get("left_ankle", 0.0)
        right_conf = pose_keypoint_confidences.get("right_ankle", 0.0)

        # Calculate bottom of detection box in pixel coordinates
        bbox_bottom_y = (bbox.ymin() + bbox.height()) * self.frame_height

        # If we have both ankles, average them with bbox bottom
        if left_ankle and right_ankle and left_conf > 0.3 and right_conf > 0.3:
            ankle_x = (left_ankle[0] + right_ankle[0]) / 2
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2

            # Average ankle position with bbox bottom to approximate foot position
            leg_x = ankle_x
            leg_y = (ankle_y + bbox_bottom_y) / 2
            confidence = (left_conf + right_conf) / 2

            hailo_logger.debug(
                "Leg position from ankles + bbox bottom: (%.1f, %.1f) confidence=%.2f",
                leg_x, leg_y, confidence
            )
            return (leg_x, leg_y), confidence

        # If we only have one ankle, use that with bbox bottom
        if left_ankle and left_conf > 0.3:
            leg_x = left_ankle[0]
            leg_y = (left_ankle[1] + bbox_bottom_y) / 2
            hailo_logger.debug(
                "Leg position from left ankle + bbox bottom: (%.1f, %.1f) confidence=%.2f",
                leg_x, leg_y, left_conf
            )
            return (leg_x, leg_y), left_conf

        if right_ankle and right_conf > 0.3:
            leg_x = right_ankle[0]
            leg_y = (right_ankle[1] + bbox_bottom_y) / 2
            hailo_logger.debug(
                "Leg position from right ankle + bbox bottom: (%.1f, %.1f) confidence=%.2f",
                leg_x, leg_y, right_conf
            )
            return (leg_x, leg_y), right_conf

        # Fallback: use hip position if ankles are not available
        left_hip = pose_keypoints.get("left_hip")
        right_hip = pose_keypoints.get("right_hip")
        left_hip_conf = pose_keypoint_confidences.get("left_hip", 0.0)
        right_hip_conf = pose_keypoint_confidences.get("right_hip", 0.0)

        if left_hip and right_hip and left_hip_conf > 0.3 and right_hip_conf > 0.3:
            leg_x = (left_hip[0] + right_hip[0]) / 2
            leg_y = (left_hip[1] + right_hip[1]) / 2
            confidence = (left_hip_conf + right_hip_conf) / 2 * 0.7  # Lower confidence for hip

            hailo_logger.debug(
                "Leg position from hips (fallback): (%.1f, %.1f) confidence=%.2f",
                leg_x, leg_y, confidence
            )
            return (leg_x, leg_y), confidence

        hailo_logger.debug("No valid leg position found from pose keypoints")
        return None, 0.0


def get_keypoint_names() -> List[str]:
    """
    Get list of all COCO keypoint names.

    Returns:
        List[str]: List of keypoint names.
    """
    return list(COCO_KEYPOINTS.keys())

