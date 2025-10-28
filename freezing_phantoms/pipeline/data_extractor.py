"""
Data extractor for Hailo AI pipeline.

Extracts structured player information from Hailo buffer data.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import hailo
import numpy as np

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
        segmentation_mask (Optional[np.ndarray]): Segmentation mask (H, W).
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
        segmentation_mask: Optional[np.ndarray] = None,
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
            segmentation_mask (Optional[np.ndarray]): Segmentation mask.
            leg_position (Optional[Tuple[float, float]]): Detected leg position in camera coordinates.
            leg_confidence (float): Confidence score for leg position detection.
        """
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.pose_keypoints = pose_keypoints
        self.pose_keypoint_confidences = pose_keypoint_confidences or {}
        self.segmentation_mask = segmentation_mask
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
            "segmentation_mask": self.segmentation_mask,
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

            # Extract segmentation mask
            segmentation_mask = self._extract_segmentation_mask(detection, bbox)

            # Create temporary player data for leg detection
            temp_player = PlayerData(
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
                segmentation_mask=segmentation_mask,
            )

            # Detect leg position using segmentation mask
            leg_position, leg_confidence = self.detect_leg_position(temp_player)

            # Create final player data object with leg position
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
                segmentation_mask=segmentation_mask,
                leg_position=leg_position,
                leg_confidence=leg_confidence,
            )

            players.append(player)
            hailo_logger.debug(
                "Extracted player: track_id=%d, keypoints=%d, has_mask=%s, leg_pos=%s",
                track_id,
                len(pose_keypoints),
                segmentation_mask is not None,
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

    def _extract_segmentation_mask(
        self, detection, bbox
    ) -> Optional[np.ndarray]:
        """
        Extract segmentation mask from detection.

        Args:
            detection: Hailo detection object.
            bbox: Bounding box object.

        Returns:
            Optional[np.ndarray]: Binary segmentation mask or None if not available.
        """
        # Get segmentation masks
        masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)

        if not masks or len(masks) == 0:
            hailo_logger.debug("No segmentation mask found in detection")
            return None

        mask = masks[0]
        mask_height = mask.get_height()
        mask_width = mask.get_width()

        # Get mask data and reshape
        data = np.array(mask.get_data())
        data = data.reshape((mask_height, mask_width))

        # Resize mask to bbox size in frame coordinates
        bbox_width = int(bbox.width() * self.frame_width)
        bbox_height = int(bbox.height() * self.frame_height)

        if bbox_width > 0 and bbox_height > 0:
            resized_mask = cv2.resize(
                data, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR
            )

            # Create full-frame mask
            full_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)

            # Calculate bbox position in frame
            x_min = int(bbox.xmin() * self.frame_width)
            y_min = int(bbox.ymin() * self.frame_height)
            x_max = min(x_min + bbox_width, self.frame_width)
            y_max = min(y_min + bbox_height, self.frame_height)

            # Ensure valid bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            # Place mask in full frame
            if x_max > x_min and y_max > y_min:
                full_mask[y_min:y_max, x_min:x_max] = resized_mask[
                    : y_max - y_min, : x_max - x_min
                ]

            hailo_logger.debug(
                "Extracted segmentation mask: original=%dx%d, resized=%dx%d",
                mask_width, mask_height, bbox_width, bbox_height
            )

            return full_mask

        return None

    def detect_leg_position(self, player_data: PlayerData) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Detect leg position using segmentation mask.

        Uses the lowest pixel in the segmentation mask to find foot position.
        This is more efficient than pose keypoints and works even when pose is occluded.

        Args:
            player_data (PlayerData): Player data with segmentation mask.

        Returns:
            Tuple[Optional[Tuple[float, float]], float]:
                (leg_position, confidence) where leg_position is (x, y) in camera coordinates
        """
        if player_data.segmentation_mask is None:
            hailo_logger.debug("No segmentation mask available for leg detection")
            return None, 0.0

        mask = player_data.segmentation_mask

        # Find the lowest (highest y-coordinate) non-zero pixel in the mask
        # This represents the bottom of the feet
        leg_y = -1
        leg_x = -1

        # Search from bottom to top for efficiency
        for y in range(mask.shape[0] - 1, -1, -1):
            row = mask[y, :]
            non_zero_indices = np.where(row > 0)[0]

            if len(non_zero_indices) > 0:
                # Found the bottom row with non-zero pixels
                leg_y = y
                # Use the center of the non-zero pixels as x position
                leg_x = int(np.mean(non_zero_indices))
                break

        if leg_y == -1 or leg_x == -1:
            hailo_logger.debug("No valid leg position found in segmentation mask")
            return None, 0.0

        # Calculate confidence based on mask quality
        # More non-zero pixels in the bottom row = higher confidence
        bottom_row = mask[leg_y, :]
        non_zero_count = np.count_nonzero(bottom_row)
        confidence = min(1.0, non_zero_count / 50.0)  # Normalize to 0-1

        hailo_logger.debug(
            "Leg position detected: (%d, %d) with confidence %.2f",
            leg_x, leg_y, confidence
        )

        return (float(leg_x), float(leg_y)), confidence


def get_keypoint_names() -> List[str]:
    """
    Get list of all COCO keypoint names.

    Returns:
        List[str]: List of keypoint names.
    """
    return list(COCO_KEYPOINTS.keys())

