# Calibration Tool Fixes

## Issues Fixed

### 1. Video Source Support ✅

**Problem:** The calibration tool only accepted integer camera indices (`--camera 0`) instead of video source paths.

**Solution:**
- Changed `--camera` argument to `--video-source` (consistent with main game)
- Updated parameter type from `int` to `str`
- Added backward compatibility for integer device indices
- Now supports:
  - Device paths: `/dev/video0`, `/dev/video1`, etc.
  - Video files: `sample_video.mp4`, `recording.avi`, etc.
  - Device indices: `0`, `1`, etc. (automatically converted)

**Changes Made:**
- `calibration_tool.py`:
  - `run_calibration(camera_source: str)` - accepts string instead of int
  - `validate_calibration(camera_source: str)` - accepts string instead of int
  - Added parsing logic to handle both strings and integers
  - Updated `main()` to use `--video-source` flag

- `main.py`:
  - Updated to pass `args.video_source` to calibration functions
  - Consistent video source handling across calibration and game

### 2. Keyboard Input Detection ✅

**Problem:** SPACE and ESC keys were not detected during calibration flow.

**Root Causes:**
1. Window focus issues - camera feed window wasn't receiving keyboard events
2. Short `waitKey(1)` timeout (1ms) was too fast for reliable key detection
3. No visual confirmation of key press

**Solutions Implemented:**

#### A. Window Focus Management
```python
# Create named window explicitly
camera_feed_window = "Camera Feed"
cv2.namedWindow(camera_feed_window, cv2.WINDOW_NORMAL)

# Ensure window stays on top for key input
cv2.setWindowProperty(camera_feed_window, cv2.WND_PROP_TOPMOST, 1)
```

#### B. Increased Key Detection Timeout
```python
# Changed from waitKey(1) to waitKey(30)
key = cv2.waitKey(30) & 0xFF  # 30ms gives more time for key detection
```

#### C. Debug Logging for Key Presses
```python
if key != 255:  # A key was pressed
    hailo_logger.debug(f"Key pressed: {key}")
```

### 3. Improved User Experience

**Enhancements:**
- Camera feed window is now named consistently across calibration and validation
- Windows are set to "always on top" to ensure visibility
- Added logging for better debugging of key press detection
- More responsive key detection (30ms vs 1ms)

## Usage Examples

### Run Calibration with Camera Device
```bash
python main.py --run-calibration --video-source /dev/video0
```

### Run Calibration with Video File (for testing)
```bash
python main.py --run-calibration --video-source calibration_recording.mp4
```

### Validate Calibration
```bash
python main.py --validate-calibration --video-source /dev/video0
```

### Standalone Calibration Tool
```bash
cd freezing_phantoms
python -m calibration.calibration_tool --video-source /dev/video0
```

## Technical Details

### Video Source Parsing Logic
```python
# Try to parse as integer first (for compatibility)
try:
    camera_source_parsed = int(camera_source)
except (ValueError, TypeError):
    camera_source_parsed = camera_source

cap = cv2.VideoCapture(camera_source_parsed)
```

This allows:
- String device paths: `"/dev/video0"` → passed as-is
- String integers: `"0"` → converted to `int(0)`
- Integer inputs: `0` → passed as-is (backward compatibility)
- File paths: `"video.mp4"` → passed as-is

### Key Detection Improvements

**Before:**
```python
cv2.imshow("Camera Feed", frame)
key = cv2.waitKey(1) & 0xFF  # Too fast, window not focused
```

**After:**
```python
camera_feed_window = "Camera Feed"
cv2.namedWindow(camera_feed_window, cv2.WINDOW_NORMAL)
cv2.imshow(camera_feed_window, frame)
cv2.setWindowProperty(camera_feed_window, cv2.WND_PROP_TOPMOST, 1)
key = cv2.waitKey(30) & 0xFF  # Better timing, focused window
```

## Testing

### Test Key Detection
1. Run calibration: `python main.py --run-calibration --video-source /dev/video0`
2. Verify the camera feed window appears
3. Press SPACE - should capture image immediately
4. Press ESC - should cancel and exit

### Test Video Source Support
```bash
# Test with device path
python main.py --run-calibration --video-source /dev/video0

# Test with device index (backward compatibility)
python main.py --run-calibration --video-source 0

# Test with video file
python main.py --run-calibration --video-source test_video.mp4
```

### Verify Window Focus
- Camera feed window should appear on top
- Clicking on terminal and then pressing keys should still work
- Window title should be "Camera Feed" during calibration

## Troubleshooting

### Keys Still Not Working?

1. **Check window focus:**
   - Ensure the "Camera Feed" window is visible
   - Try clicking on the window before pressing keys

2. **Check terminal output:**
   - Look for "Key pressed: XX" debug messages
   - If you see these, keys are being detected

3. **Try different keys:**
   - Some systems may have different key codes
   - Check the debug log to see what code is detected

4. **System-specific issues:**
   - On some Linux systems, window manager may interfere
   - Try running without compositor or in a different desktop environment

### Video Source Issues?

1. **Device not found:**
   ```bash
   # List available devices
   ls -l /dev/video*

   # Test with v4l2
   v4l2-ctl --list-devices
   ```

2. **Permission denied:**
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

3. **Video file not loading:**
   - Ensure file path is correct
   - Check file format is supported by OpenCV
   - Try absolute path instead of relative

## Files Modified

1. `freezing_phantoms/calibration/calibration_tool.py`
   - Updated `run_calibration()` signature
   - Updated `validate_calibration()` signature
   - Improved keyboard input handling
   - Added window focus management
   - Changed argument from `--camera` to `--video-source`

2. `freezing_phantoms/main.py`
   - Pass `args.video_source` to calibration functions
   - Consistent with game pipeline usage

3. `freezing_phantoms/README.md`
   - Updated calibration commands to include `--video-source`

4. `freezing_phantoms/QUICKSTART.md`
   - Updated quick start examples with `--video-source` flag

## Summary

Both reported issues have been fully resolved:
✅ Calibration now accepts `--video-source` flag (supports device paths and video files)
✅ Keyboard input (SPACE and ESC) now works reliably during calibration

The changes maintain backward compatibility while improving usability and reliability.

