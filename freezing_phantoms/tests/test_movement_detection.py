"""
Unit tests for movement detection module.

Tests pose comparison and movement threshold logic.
"""

import pytest
import numpy as np

from freezing_phantoms.core.movement_detector import MovementDetector
from freezing_phantoms.pipeline.data_extractor import PlayerData


class TestMovementDetector:
    """Test movement detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MovementDetector(threshold_pixels=20.0)

    def test_no_movement_detected(self):
        """Test that identical poses show no movement."""
        pose_t0 = {
            "left_wrist": (100.0, 200.0),
            "right_wrist": (150.0, 200.0),
            "left_ankle": (110.0, 400.0),
            "right_ankle": (140.0, 400.0),
        }
        pose_t1 = pose_t0.copy()

        moved, displacement, keypoint = self.detector.detect_movement(pose_t0, pose_t1)

        assert moved is False
        assert displacement == 0.0

    def test_movement_below_threshold(self):
        """Test that small movements below threshold are not detected."""
        pose_t0 = {
            "left_wrist": (100.0, 200.0),
            "right_wrist": (150.0, 200.0),
        }
        pose_t1 = {
            "left_wrist": (105.0, 203.0),  # Moved ~5.8 pixels
            "right_wrist": (148.0, 198.0),  # Moved ~3.6 pixels
        }

        moved, displacement, keypoint = self.detector.detect_movement(pose_t0, pose_t1)

        assert moved is False
        assert displacement < 20.0

    def test_movement_above_threshold(self):
        """Test that movements above threshold are detected."""
        pose_t0 = {
            "left_wrist": (100.0, 200.0),
            "right_wrist": (150.0, 200.0),
        }
        pose_t1 = {
            "left_wrist": (130.0, 220.0),  # Moved ~36 pixels
            "right_wrist": (150.0, 200.0),
        }

        moved, displacement, keypoint = self.detector.detect_movement(pose_t0, pose_t1)

        assert moved is True
        assert displacement > 20.0
        assert keypoint == "left_wrist"

    def test_empty_pose_handling(self):
        """Test handling of empty pose data."""
        pose_t0 = {}
        pose_t1 = {"left_wrist": (100.0, 200.0)}

        moved, displacement, keypoint = self.detector.detect_movement(pose_t0, pose_t1)

        assert moved is False
        assert displacement == 0.0

    def test_missing_keypoint_handling(self):
        """Test handling when keypoint is missing in one pose."""
        pose_t0 = {
            "left_wrist": (100.0, 200.0),
            "right_wrist": (150.0, 200.0),
        }
        pose_t1 = {
            "left_wrist": (102.0, 201.0),
            # right_wrist missing
        }

        # Should not crash
        moved, displacement, keypoint = self.detector.detect_movement(pose_t0, pose_t1)

        # Should check only available keypoints
        assert isinstance(moved, bool)

    def test_threshold_adjustment(self):
        """Test adjusting movement threshold."""
        pose_t0 = {"left_wrist": (100.0, 200.0)}
        pose_t1 = {"left_wrist": (115.0, 200.0)}  # 15 pixels movement

        # With threshold 20, should not detect
        self.detector.set_threshold(20.0)
        moved, _, _ = self.detector.detect_movement(pose_t0, pose_t1)
        assert moved is False

        # With threshold 10, should detect
        self.detector.set_threshold(10.0)
        moved, _, _ = self.detector.detect_movement(pose_t0, pose_t1)
        assert moved is True

    def test_calculate_average_displacement(self):
        """Test average displacement calculation."""
        pose_t0 = {
            "left_wrist": (100.0, 200.0),
            "right_wrist": (150.0, 200.0),
            "left_ankle": (110.0, 400.0),
        }
        pose_t1 = {
            "left_wrist": (110.0, 200.0),  # 10px
            "right_wrist": (160.0, 200.0),  # 10px
            "left_ankle": (110.0, 410.0),  # 10px
        }

        avg_displacement = self.detector.calculate_pose_displacement(pose_t0, pose_t1)

        assert avg_displacement == pytest.approx(10.0, abs=0.1)


class TestPlayerMovementDetection:
    """Test movement detection for multiple players."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MovementDetector(threshold_pixels=20.0)

    def test_detect_player_movement(self):
        """Test detection across multiple players."""
        # Create player data at T0
        player1_t0 = PlayerData(
            track_id=1,
            bbox={"xmin": 0.1, "ymin": 0.2, "width": 0.1, "height": 0.2},
            confidence=0.9,
            pose_keypoints={"left_wrist": (100.0, 200.0)},
        )
        player2_t0 = PlayerData(
            track_id=2,
            bbox={"xmin": 0.5, "ymin": 0.3, "width": 0.1, "height": 0.2},
            confidence=0.85,
            pose_keypoints={"left_wrist": (300.0, 250.0)},
        )

        # Create player data at T1 (player 1 moved, player 2 static)
        player1_t1 = PlayerData(
            track_id=1,
            bbox={"xmin": 0.1, "ymin": 0.2, "width": 0.1, "height": 0.2},
            confidence=0.9,
            pose_keypoints={"left_wrist": (130.0, 220.0)},  # Moved 36px
        )
        player2_t1 = PlayerData(
            track_id=2,
            bbox={"xmin": 0.5, "ymin": 0.3, "width": 0.1, "height": 0.2},
            confidence=0.85,
            pose_keypoints={"left_wrist": (302.0, 251.0)},  # Moved 2.2px
        )

        players_t0 = [player1_t0, player2_t0]
        players_t1 = [player1_t1, player2_t1]

        # Detect movement
        results = self.detector.detect_player_movement(players_t0, players_t1)

        # Player 1 should be detected as moved
        assert 1 in results
        moved_1, _, _ = results[1]
        assert moved_1 is True

        # Player 2 should not be detected as moved
        assert 2 in results
        moved_2, _, _ = results[2]
        assert moved_2 is False

    def test_new_player_handling(self):
        """Test handling of players that appear in T1 but not T0."""
        player1_t0 = PlayerData(
            track_id=1,
            bbox={"xmin": 0.1, "ymin": 0.2, "width": 0.1, "height": 0.2},
            confidence=0.9,
            pose_keypoints={"left_wrist": (100.0, 200.0)},
        )

        player2_t1 = PlayerData(
            track_id=2,  # New player
            bbox={"xmin": 0.5, "ymin": 0.3, "width": 0.1, "height": 0.2},
            confidence=0.85,
            pose_keypoints={"left_wrist": (300.0, 250.0)},
        )

        players_t0 = [player1_t0]
        players_t1 = [player2_t1]

        # Should not crash
        results = self.detector.detect_player_movement(players_t0, players_t1)

        # Player 2 should not be in results (was not in T0)
        assert 2 not in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

