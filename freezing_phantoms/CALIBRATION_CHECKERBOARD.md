# Checkerboard Calibration for Floor Projection

## Overview

The calibration system now supports **checkerboard patterns** which are much more robust to perspective distortion when projecting onto angled surfaces (like floors).

## Configuration

In `config/game_config.yaml`:

```yaml
calibration:
  pattern_type: "checkerboard"  # or "circles"
  pattern_rows: 8               # Interior corner rows
  pattern_cols: 11             # Interior corner columns
  square_size: 100             # Size of each square (pixels)
```

## Why Checkerboard?

1. **Robust to perspective distortion**: When projecting onto a floor, circles appear as ellipses due to perspective. Checkerboard corner detection (using OpenCV's `findChessboardCorners`) handles this naturally.

2. **Better corner detection**: OpenCV's checkerboard detector is highly optimized and uses adaptive thresholds, making it work well in varying lighting conditions.

3. **More points**: The 8x11 pattern provides 88 corner points vs 20 circle centers, giving better calibration accuracy.

4. **Industry standard**: Checkerboards are the gold standard for camera calibration.

## Pattern Dimensions

- **Projector pattern**: 8 rows × 11 columns of interior corners
- **Each square**: 100×100 pixels
- **Total size**: Approximately 1100×900 pixels (centered in 1920×1080 resolution)

## Usage

Run calibration as before:

```bash
python main.py --run-calibration --video-source /dev/video2
```

The system will:
1. Display the checkerboard pattern on the projector
2. Detect corners automatically (robust to floor projection angle)
3. Calculate homography between camera and projector spaces
4. Save calibration to `config/calibration.npy`

## Detection Flow

1. **Pattern Generation**: Creates a checkerboard pattern with alternating white/black squares
2. **Corner Detection**: Uses `cv2.findChessboardCorners()` with adaptive thresholding
3. **Sub-pixel Refinement**: Refines corner positions for accuracy
4. **Homography Calculation**: Matches detected corners to projector coordinates using RANSAC

## Advantages over Circle Patterns

- ✅ Handles perspective distortion naturally
- ✅ Works well with floor projection at angles
- ✅ More robust to lighting variations
- ✅ Provides more calibration points
- ✅ Better accuracy for homography estimation

## Pattern Visualization

The checkerboard shows alternating black and white squares. In the camera view (especially when projecting on a floor), the squares will appear trapezoidal due to perspective, but the corner detector handles this automatically.

## Tips for Best Results

1. **Full visibility**: Ensure the entire checkerboard is visible to the camera
2. **Flat surface**: Keep the projection surface as flat as possible
3. **Good lighting**: Adequate lighting helps corner detection
4. **Camera angle**: Position camera to see most or all of the pattern
5. **Pattern size**: Adjust `square_size` if needed for better visibility

## Switching Back to Circles

To use circle patterns instead:

```yaml
calibration:
  pattern_type: "circles"
  pattern_rows: 4
  pattern_cols: 5
  circle_radius: 80
  circle_spacing: 300
```

Circle patterns may require more parameter tuning and are less robust to perspective distortion on angled surfaces.

