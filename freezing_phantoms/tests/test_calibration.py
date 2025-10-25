"""
Unit tests for calibration module.

Tests pattern generation, circle detection, and homography calculation.
"""

import numpy as np
import pytest
import cv2

from freezing_phantoms.calibration.pattern_generator import (
    generate_calibration_pattern,
    detect_circle_centers,
    sort_circle_centers,
)


class TestPatternGeneration:
    """Test calibration pattern generation."""

    def test_generate_pattern_basic(self):
        """Test basic pattern generation."""
        resolution = (1920, 1080)
        rows, cols = 6, 8

        pattern, centers = generate_calibration_pattern(
            resolution=resolution,
            rows=rows,
            cols=cols,
        )

        # Check pattern shape
        assert pattern.shape == (resolution[1], resolution[0], 3)

        # Check number of centers
        assert len(centers) == rows * cols

        # Check centers are within bounds
        assert np.all(centers[:, 0] >= 0)
        assert np.all(centers[:, 0] < resolution[0])
        assert np.all(centers[:, 1] >= 0)
        assert np.all(centers[:, 1] < resolution[1])

    def test_pattern_is_bgr(self):
        """Test that pattern is in BGR format."""
        pattern, _ = generate_calibration_pattern(
            resolution=(800, 600),
            rows=4,
            cols=5,
        )

        # Pattern should have 3 channels (BGR)
        assert pattern.ndim == 3
        assert pattern.shape[2] == 3

    def test_circles_are_white(self):
        """Test that circles are drawn in white."""
        pattern, centers = generate_calibration_pattern(
            resolution=(800, 600),
            rows=2,
            cols=2,
            circle_radius=50,
        )

        # Check that pattern has white pixels (circles)
        white_pixels = np.sum(np.all(pattern == [255, 255, 255], axis=2))
        assert white_pixels > 0


class TestCircleDetection:
    """Test circle detection in images."""

    def test_detect_circles_in_synthetic_image(self):
        """Test circle detection in a synthetic pattern."""
        # Create synthetic pattern
        pattern, expected_centers = generate_calibration_pattern(
            resolution=(1280, 720),
            rows=4,
            cols=5,
            circle_radius=30,
            spacing=120,
        )

        # Detect circles
        detected_centers = detect_circle_centers(
            pattern,
            expected_count=20,
            min_radius=20,
            max_radius=50,
        )

        # Should detect most circles
        assert detected_centers is not None
        assert len(detected_centers) >= 15  # Allow some tolerance

    def test_detect_no_circles_in_blank_image(self):
        """Test that no circles are detected in blank image."""
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)

        detected = detect_circle_centers(blank, expected_count=10)

        # Should detect no circles
        assert detected is None


class TestCircleSorting:
    """Test circle center sorting."""

    def test_sort_grid_centers(self):
        """Test sorting of grid-arranged centers."""
        # Create a regular grid of points
        rows, cols = 4, 5
        centers = []

        for r in range(rows):
            for c in range(cols):
                centers.append([c * 100 + 50, r * 100 + 50])

        centers = np.array(centers, dtype=np.float32)

        # Shuffle centers
        np.random.shuffle(centers)

        # Sort them
        sorted_centers = sort_circle_centers(centers, rows, cols)

        # Check that sorted centers are in correct order
        assert len(sorted_centers) == rows * cols

        # First point should be top-left
        assert sorted_centers[0][1] < sorted_centers[-1][1]  # Top before bottom

    def test_sort_maintains_count(self):
        """Test that sorting doesn't lose points."""
        centers = np.random.rand(24, 2).astype(np.float32) * 1000

        sorted_centers = sort_circle_centers(centers, rows=4, cols=6)

        assert len(sorted_centers) == len(centers)


class TestHomographyCalculation:
    """Test homography matrix calculation."""

    def test_identity_homography(self):
        """Test that identical point sets produce identity-like homography."""
        # Create matching point sets
        points = np.array(
            [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32
        )

        H, mask = cv2.findHomography(points, points, cv2.RANSAC)

        # Should be close to identity
        identity = np.eye(3)

        # Normalize H
        H_normalized = H / H[2, 2]

        # Check similarity to identity (allowing some numerical error)
        assert np.allclose(H_normalized, identity, atol=0.1)

    def test_translation_homography(self):
        """Test homography with translation."""
        # Source points
        src = np.array(
            [[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32
        )

        # Translated destination points
        dst = src + np.array([50, 50], dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC)

        # Transform source points
        src_homogeneous = np.hstack([src, np.ones((4, 1))]).T
        transformed = H @ src_homogeneous
        transformed = (transformed[:2] / transformed[2]).T

        # Should match destination
        assert np.allclose(transformed, dst, atol=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

