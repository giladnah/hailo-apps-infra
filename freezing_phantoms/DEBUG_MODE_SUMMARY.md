# Debug Mode Feature

## Overview

Added a `--show-video` flag to enable visual debugging of the AI pipeline. This allows developers to see what the camera sees and verify that AI detections (pose estimation and segmentation) are working correctly.

## What Was Added

### 1. Command Line Flag

```bash
python main.py --video-source /dev/video0 --show-video
```

### 2. Pipeline Modifications

**File:** `pipeline/combined_pipeline.py`

- Added `debug_mode` parameter to `CombinedPoseSegmentationPipeline`
- When `debug_mode=True`:
  - Uses `hailooverlay + xvimagesink` instead of `fakesink`
  - Shows video window with AI overlays (detection boxes, pose keypoints, segmentation masks)
- When `debug_mode=False` (default):
  - Uses `fakesink` (no video output, lower latency)

**Pipeline Strings:**
```python
# Debug mode (with display)
sink = "hailooverlay name=hailo_overlay ! videoconvert ! xvimagesink name=hailo_display sync=false"

# Game mode (no display)
sink = "fakesink name=game_sink sync=false"
```

### 3. Game Application Changes

**File:** `main.py`

- Added `debug_mode` parameter to `FreezingPhantomsGame.__init__()`
- Added `--show-video` command line argument
- Passes `debug_mode` flag to pipeline initialization

### 4. Documentation Updates

**File:** `README.md`

- Added debug mode section with usage instructions
- Listed what's visible in debug mode
- Explained use cases for debug mode

## Use Cases

Debug mode is useful for:
- **Verifying AI Pipeline**: Ensure the Hailo models are loading and running correctly
- **Camera Positioning**: Check if the camera has a good view of the play area
- **Detection Validation**: Confirm that people are being detected and tracked
- **Pose Estimation Check**: Verify keypoints are correctly identified
- **Segmentation Quality**: See if instance segmentation masks are accurate
- **Lighting Conditions**: Identify if lighting is adequate for detection
- **Troubleshooting**: Debug issues before running the full game

## Technical Details

### Why No FPS Counter?

The `hailooverlay` element in the debug pipeline doesn't support the `fps-measurements` signal that's used in other Hailo apps. This is because we're using a custom pipeline structure with two inference stages (pose + segmentation).

The FPS counter isn't critical for debugging - the visual overlays showing detections are more important for verifying the pipeline is working correctly.

### Performance Impact

- **Normal mode** (`fakesink`): Lower latency, no video rendering overhead
- **Debug mode** (`xvimagesink`): Slightly higher CPU usage due to video rendering, but negligible impact on AI inference

### Display Requirements

Debug mode requires:
- X11 display server (automatically available on most Linux desktops)
- OpenGL/X video output support
- May not work over SSH without X forwarding

## Testing

Confirmed working:
```bash
cd /home/giladn/tappas_apps/repos/cursor_agent/hailo-apps-infra
source setup_env.sh
cd freezing_phantoms
python main.py --video-source /dev/video2 --show-video
```

Result:
- Pipeline starts successfully
- "Freezing Phantoms - AI Pipeline" process running
- Video window displays with AI overlays
- No errors in initialization or pipeline creation

## Future Enhancements

Potential improvements:
1. Add FPS display using custom overlay text
2. Add keyboard shortcuts to toggle overlays (detections, poses, masks)
3. Support for recording debug video to file
4. Optional secondary debug window (separate from projector output)
5. Real-time performance metrics display

## Related Files

- `freezing_phantoms/pipeline/combined_pipeline.py` - Pipeline implementation
- `freezing_phantoms/main.py` - Main application entry point
- `freezing_phantoms/README.md` - User documentation
- `freezing_phantoms/DEBUG_MODE_SUMMARY.md` - This file

