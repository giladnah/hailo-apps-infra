"""
Unit tests for confidence-based keypoint filtering functionality.
"""

from unittest.mock import Mock

from freezing_phantoms.core.movement_detector import MovementDetector
from freezing_phantoms.pipeline.data_extractor import PlayerData


class TestConfidenceFiltering:
    """Test cases for confidence-based keypoint filtering."""

    def test_confidence_threshold_initialization(self):
        """Test that confidence threshold is properly initialized."""
        detector = MovementDetector(keypoint_confidence_threshold=0.7)
        assert detector.keypoint_confidence_threshold == 0.7

    def test_confidence_threshold_clamping(self):
        """Test that confidence threshold is properly clamped to [0,1]."""
        # Test values outside valid range
        detector1 = MovementDetector(keypoint_confidence_threshold=-0.5)
        assert detector1.keypoint_confidence_threshold == 0.0

        detector2 = MovementDetector(keypoint_confidence_threshold=1.5)
        assert detector2.keypoint_confidence_threshold == 1.0

        # Test valid values
        detector3 = MovementDetector(keypoint_confidence_threshold=0.8)
        assert detector3.keypoint_confidence_threshold == 0.8

    def test_filter_keypoints_by_confidence(self):
        """Test filtering keypoints based on confidence scores."""
        detector = MovementDetector(keypoint_confidence_threshold=0.6)

        pose_keypoints = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200),
            "left_ankle": (150, 400),
            "right_ankle": (250, 400)
        }

        pose_confidences = {
            "left_wrist": 0.8,    # Above threshold
            "right_wrist": 0.4,   # Below threshold
            "left_ankle": 0.7,    # Above threshold
            "right_ankle": 0.3    # Below threshold
        }

        # pylint: disable=protected-access
        filtered = detector._filter_keypoints_by_confidence(pose_keypoints, pose_confidences)

        # Only high-confidence keypoints should be included
        expected = {
            "left_wrist": (100, 200),
            "left_ankle": (150, 400)
        }
        assert filtered == expected

    def test_filter_keypoints_missing_confidence(self):
        """Test filtering when confidence scores are missing."""
        detector = MovementDetector(keypoint_confidence_threshold=0.5)

        pose_keypoints = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200)
        }

        pose_confidences = {
            "left_wrist": 0.8
            # right_wrist confidence missing
        }

        # pylint: disable=protected-access
        filtered = detector._filter_keypoints_by_confidence(pose_keypoints, pose_confidences)

        # Only keypoints with confidence scores should be included
        expected = {"left_wrist": (100, 200)}
        assert filtered == expected

    def test_filter_keypoints_empty_confidences(self):
        """Test filtering when no confidence scores are provided."""
        detector = MovementDetector(keypoint_confidence_threshold=0.5)

        pose_keypoints = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200)
        }

        # pylint: disable=protected-access
        filtered = detector._filter_keypoints_by_confidence(pose_keypoints, {})

        # No keypoints should pass the filter
        assert filtered == {}

    def test_detect_player_movement_with_confidence_filtering(self):
        """Test that player movement detection uses confidence filtering."""
        detector = MovementDetector(keypoint_confidence_threshold=0.6)

        # Create mock players with confidence scores
        player_t0 = Mock()
        player_t0.track_id = 1
        player_t0.pose_keypoints = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200)
        }
        player_t0.pose_keypoint_confidences = {
            "left_wrist": 0.8,    # High confidence
            "right_wrist": 0.4    # Low confidence
        }

        player_t1 = Mock()
        player_t1.track_id = 1
        player_t1.pose_keypoints = {
            "left_wrist": (120, 220),  # Moved
            "right_wrist": (280, 180)  # Moved but low confidence
        }
        player_t1.pose_keypoint_confidences = {
            "left_wrist": 0.9,    # High confidence
            "right_wrist": 0.3    # Low confidence
        }

        # Test movement detection
        results = detector.detect_player_movement([player_t0], [player_t1])

        assert 1 in results
        moved, _displacement, keypoint = results[1]
        # Should detect movement based on high-confidence keypoints only
        assert moved is True
        assert keypoint == "left_wrist"  # Only high-confidence keypoint

    def test_detect_player_movement_no_high_confidence_keypoints(self):
        """Test movement detection when no keypoints have high confidence."""
        detector = MovementDetector(keypoint_confidence_threshold=0.9)  # Very high threshold

        # Create mock players with low confidence scores
        player_t0 = Mock()
        player_t0.track_id = 1
        player_t0.pose_keypoints = {"left_wrist": (100, 200)}
        player_t0.pose_keypoint_confidences = {"left_wrist": 0.5}  # Below threshold

        player_t1 = Mock()
        player_t1.track_id = 1
        player_t1.pose_keypoints = {"left_wrist": (150, 250)}  # Moved
        player_t1.pose_keypoint_confidences = {"left_wrist": 0.6}  # Still below threshold

        # Test movement detection
        results = detector.detect_player_movement([player_t0], [player_t1])

        assert 1 in results
        moved, displacement, keypoint = results[1]
        # Should not detect movement since no keypoints pass confidence filter
        assert moved is False
        assert displacement == 0.0
        assert keypoint is None

    def test_set_confidence_threshold(self):
        """Test setting confidence threshold at runtime."""
        detector = MovementDetector()

        detector.set_keypoint_confidence_threshold(0.8)
        assert detector.keypoint_confidence_threshold == 0.8

        # Test clamping
        detector.set_keypoint_confidence_threshold(1.5)
        assert detector.keypoint_confidence_threshold == 1.0

        detector.set_keypoint_confidence_threshold(-0.5)
        assert detector.keypoint_confidence_threshold == 0.0

    def test_smoothing_stats_includes_confidence_threshold(self):
        """Test that smoothing stats include confidence threshold."""
        detector = MovementDetector(
            keypoint_confidence_threshold=0.7,
            smoothing_alpha=0.4,
            sensitivity_multiplier=1.2
        )

        stats = detector.get_smoothing_stats()

        assert "keypoint_confidence_threshold" in stats
        assert stats["keypoint_confidence_threshold"] == 0.7
        assert stats["smoothing_alpha"] == 0.4
        assert stats["sensitivity_multiplier"] == 1.2

    def test_player_data_with_confidence_scores(self):
        """Test PlayerData structure with confidence scores."""
        player = PlayerData(
            track_id=1,
            bbox={"xmin": 0, "ymin": 0, "width": 100, "height": 200},
            confidence=0.9,
            pose_keypoints={"left_wrist": (50, 100)},
            pose_keypoint_confidences={"left_wrist": 0.8}
        )

        assert player.track_id == 1
        assert player.pose_keypoints == {"left_wrist": (50, 100)}
        assert player.pose_keypoint_confidences == {"left_wrist": 0.8}

        # Test to_dict includes confidence scores
        player_dict = player.to_dict()
        assert "pose_keypoint_confidences" in player_dict
        assert player_dict["pose_keypoint_confidences"] == {"left_wrist": 0.8}

    def test_player_data_without_confidence_scores(self):
        """Test PlayerData structure without confidence scores (backward compatibility)."""
        player = PlayerData(
            track_id=1,
            bbox={"xmin": 0, "ymin": 0, "width": 100, "height": 200},
            confidence=0.9,
            pose_keypoints={"left_wrist": (50, 100)}
            # No confidence scores provided
        )

        assert player.track_id == 1
        assert player.pose_keypoints == {"left_wrist": (50, 100)}
        assert player.pose_keypoint_confidences == {}  # Should default to empty dict

        # Test to_dict includes empty confidence scores
        player_dict = player.to_dict()
        assert "pose_keypoint_confidences" in player_dict
        assert player_dict["pose_keypoint_confidences"] == {}

    def test_confidence_filtering_with_different_thresholds(self):
        """Test confidence filtering with different threshold values."""
        pose_keypoints = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200),
            "left_ankle": (150, 400)
        }

        pose_confidences = {
            "left_wrist": 0.9,
            "right_wrist": 0.6,
            "left_ankle": 0.3
        }

        # Test with low threshold (0.2) - should include all
        detector_low = MovementDetector(keypoint_confidence_threshold=0.2)
        # pylint: disable=protected-access
        filtered_low = detector_low._filter_keypoints_by_confidence(pose_keypoints, pose_confidences)
        assert len(filtered_low) == 3

        # Test with medium threshold (0.5) - should include 2
        detector_medium = MovementDetector(keypoint_confidence_threshold=0.5)
        # pylint: disable=protected-access
        filtered_medium = detector_medium._filter_keypoints_by_confidence(pose_keypoints, pose_confidences)
        assert len(filtered_medium) == 2
        assert "left_wrist" in filtered_medium
        assert "right_wrist" in filtered_medium

        # Test with high threshold (0.8) - should include 1
        detector_high = MovementDetector(keypoint_confidence_threshold=0.8)
        # pylint: disable=protected-access
        filtered_high = detector_high._filter_keypoints_by_confidence(pose_keypoints, pose_confidences)
        assert len(filtered_high) == 1
        assert "left_wrist" in filtered_high
