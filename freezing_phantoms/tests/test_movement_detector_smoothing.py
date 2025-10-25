"""
Unit tests for MovementDetector smoothing functionality.
"""

from unittest.mock import Mock

from freezing_phantoms.core.movement_detector import MovementDetector


class TestMovementDetectorSmoothing:
    """Test cases for movement detector smoothing features."""

    def test_smoothing_initialization(self):
        """Test that smoothing parameters are properly initialized."""
        detector = MovementDetector(
            threshold_pixels=20.0,
            smoothing_alpha=0.5,
            pose_history_size=3,
            sensitivity_multiplier=1.5
        )

        assert detector.smoothing_alpha == 0.5
        assert detector.pose_history_size == 3
        assert detector.sensitivity_multiplier == 1.5
        assert len(detector.pose_histories) == 0

    def test_smoothing_alpha_clamping(self):
        """Test that smoothing alpha is properly clamped to [0,1]."""
        # Test values outside valid range
        detector1 = MovementDetector(smoothing_alpha=-0.5)
        assert detector1.smoothing_alpha == 0.0

        detector2 = MovementDetector(smoothing_alpha=1.5)
        assert detector2.smoothing_alpha == 1.0

        # Test valid values
        detector3 = MovementDetector(smoothing_alpha=0.7)
        assert detector3.smoothing_alpha == 0.7

    def test_sensitivity_multiplier_clamping(self):
        """Test that sensitivity multiplier is properly clamped to minimum 0.1."""
        detector1 = MovementDetector(sensitivity_multiplier=0.05)
        assert detector1.sensitivity_multiplier == 0.1

        detector2 = MovementDetector(sensitivity_multiplier=2.0)
        assert detector2.sensitivity_multiplier == 2.0

    def test_pose_history_size_clamping(self):
        """Test that pose history size is properly clamped to minimum 1."""
        detector1 = MovementDetector(pose_history_size=0)
        assert detector1.pose_history_size == 1

        detector2 = MovementDetector(pose_history_size=5)
        assert detector2.pose_history_size == 5

    def test_smooth_pose_first_pose(self):
        """Test smoothing behavior for the first pose (no smoothing applied)."""
        detector = MovementDetector(smoothing_alpha=0.5)
        track_id = 1

        pose = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200)
        }

        # pylint: disable=protected-access
        smoothed = detector._smooth_pose(track_id, pose)

        # First pose should be returned unchanged
        assert smoothed == pose
        assert track_id in detector.pose_histories
        assert len(detector.pose_histories[track_id]) == 1

    def test_smooth_pose_exponential_moving_average(self):
        """Test exponential moving average smoothing."""
        detector = MovementDetector(smoothing_alpha=0.5)
        track_id = 1

        # First pose
        pose1 = {
            "left_wrist": (100, 200),
            "right_wrist": (300, 200)
        }
        # pylint: disable=protected-access
        smoothed1 = detector._smooth_pose(track_id, pose1)
        assert smoothed1 == pose1

        # Second pose with movement
        pose2 = {
            "left_wrist": (120, 220),  # +20, +20
            "right_wrist": (280, 180)  # -20, -20
        }
        # pylint: disable=protected-access
        smoothed2 = detector._smooth_pose(track_id, pose2)

        # With alpha=0.5, smoothed should be halfway between poses
        expected_left = (100 + 0.5 * (120 - 100), 200 + 0.5 * (220 - 200))
        expected_right = (300 + 0.5 * (280 - 300), 200 + 0.5 * (180 - 200))

        assert smoothed2["left_wrist"] == expected_left
        assert smoothed2["right_wrist"] == expected_right

    def test_smooth_pose_history_size_limit(self):
        """Test that pose history is limited to specified size."""
        detector = MovementDetector(pose_history_size=2)
        track_id = 1

        # Add more poses than history size
        for i in range(5):
            pose = {"left_wrist": (100 + i, 200 + i)}
            # pylint: disable=protected-access
            detector._smooth_pose(track_id, pose)

        # History should be limited to 2 poses
        assert len(detector.pose_histories[track_id]) == 2

    def test_effective_threshold_calculation(self):
        """Test effective threshold calculation with sensitivity multiplier."""
        detector = MovementDetector(
            threshold_pixels=20.0,
            sensitivity_multiplier=1.5
        )

        # pylint: disable=protected-access
        effective_threshold = detector._get_effective_threshold()
        assert effective_threshold == 30.0  # 20.0 * 1.5

    def test_movement_detection_with_sensitivity(self):
        """Test movement detection uses effective threshold."""
        detector = MovementDetector(
            threshold_pixels=20.0,
            sensitivity_multiplier=0.5  # Less sensitive
        )

        pose_t0 = {"left_wrist": (100, 200)}
        pose_t1 = {"left_wrist": (115, 200)}  # 15px movement

        # With 0.5x sensitivity, effective threshold is 10px
        # 15px > 10px, so movement should be detected
        moved, displacement, keypoint = detector.detect_movement(pose_t0, pose_t1)

        assert moved is True
        assert displacement == 15.0
        assert keypoint == "left_wrist"

    def test_detect_player_movement_with_smoothing(self):
        """Test that player movement detection applies smoothing."""
        detector = MovementDetector(smoothing_alpha=0.5)

        # Create mock players
        player_t0 = Mock()
        player_t0.track_id = 1
        player_t0.pose_keypoints = {"left_wrist": (100, 200)}

        player_t1 = Mock()
        player_t1.track_id = 1
        player_t1.pose_keypoints = {"left_wrist": (120, 200)}

        # Test movement detection
        results = detector.detect_player_movement([player_t0], [player_t1])

        assert 1 in results
        moved, displacement, _keypoint = results[1]
        assert moved is True
        assert displacement > 0

    def test_set_smoothing_parameters(self):
        """Test setting smoothing parameters at runtime."""
        detector = MovementDetector()

        detector.set_smoothing_alpha(0.8)
        assert detector.smoothing_alpha == 0.8

        detector.set_sensitivity_multiplier(2.0)
        assert detector.sensitivity_multiplier == 2.0

        detector.set_pose_history_size(10)
        assert detector.pose_history_size == 10

    def test_clear_pose_histories(self):
        """Test clearing pose histories."""
        detector = MovementDetector()
        track_id = 1

        # Add some pose history
        # pylint: disable=protected-access
        detector._smooth_pose(track_id, {"left_wrist": (100, 200)})
        assert track_id in detector.pose_histories

        # Clear histories
        detector.clear_pose_histories()
        assert len(detector.pose_histories) == 0

    def test_get_smoothing_stats(self):
        """Test getting smoothing statistics."""
        detector = MovementDetector(
            threshold_pixels=25.0,
            smoothing_alpha=0.4,
            pose_history_size=3,
            sensitivity_multiplier=1.2
        )

        # Add some pose history
        # pylint: disable=protected-access
        detector._smooth_pose(1, {"left_wrist": (100, 200)})
        # pylint: disable=protected-access
        detector._smooth_pose(2, {"left_wrist": (200, 300)})

        stats = detector.get_smoothing_stats()

        assert stats["threshold_pixels"] == 25.0
        assert stats["effective_threshold"] == 30.0  # 25.0 * 1.2
        assert stats["smoothing_alpha"] == 0.4
        assert stats["sensitivity_multiplier"] == 1.2
        assert stats["pose_history_size"] == 3
        assert stats["active_tracks"] == 2

    def test_smoothing_with_missing_keypoints(self):
        """Test smoothing behavior when keypoints are missing."""
        detector = MovementDetector(smoothing_alpha=0.5)
        track_id = 1

        # First pose
        pose1 = {"left_wrist": (100, 200)}
        # pylint: disable=protected-access
        detector._smooth_pose(track_id, pose1)

        # Second pose with new keypoint
        pose2 = {
            "left_wrist": (120, 220),
            "right_wrist": (300, 200)  # New keypoint
        }
        # pylint: disable=protected-access
        smoothed = detector._smooth_pose(track_id, pose2)

        # Existing keypoint should be smoothed
        expected_left = (100 + 0.5 * (120 - 100), 200 + 0.5 * (220 - 200))
        assert smoothed["left_wrist"] == expected_left

        # New keypoint should be added as-is
        assert smoothed["right_wrist"] == (300, 200)
