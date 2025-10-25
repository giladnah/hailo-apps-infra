# Freezing Phantoms - Quick Start Guide

Get the game running in 5 steps!

## Prerequisites

- Hailo Apps Infrastructure installed
- Camera connected (USB webcam or built-in)
- Projector connected (or use monitor for testing)
- Python 3.7+

## Step 1: Setup

```bash
cd freezing_phantoms
./setup.sh
```

This installs dependencies and checks your system.

## Step 2: Test Camera

Verify your camera works:

```bash
# List available cameras
ls -l /dev/video*

# Test with GStreamer (optional)
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

Press Ctrl+C to stop the test.

## Step 3: Run Calibration

**Important**: Set up your hardware first:
- Mount camera overhead looking down
- Position projector to cover the play area
- Ensure camera can see projected image

**Run the standalone calibration tool (Recommended):**

```bash
python calibrate.py --video-source /dev/video0
```

The tool will:
1. Display pattern on projector
2. Wait for SPACE key press
3. Detect circles (shows count and quality)
4. Calculate homography with metrics
5. Retry automatically if detection fails
6. Save to `config/calibration.npy`

**Expected Output:**
```
🔍 Detecting calibration points...
   Expected: 48 circles
   Detected: 46 circles
✅ Good detection: 46/48 circles (95.8%)
✅ Homography matrix calculated
   Inliers: 44/46 (95.7%)
✅ CALIBRATION SUCCESSFUL!
```

## Step 4: Test Calibration (Recommended!)

Validate the calibration:

```bash
python calibrate.py --validate --video-source /dev/video0
```

Move your hand in the camera view. A green circle should appear on the projector at the same location. Press ESC when satisfied.

**Why validate?**
- Confirms accuracy before starting game
- Catches alignment issues early
- Verifies tracking across entire play area

## Step 5: Start the Game!

```bash
python main.py --video-source /dev/video0
```

The game will start automatically. Press ESC to quit, R to reset.

## Troubleshooting Quick Fixes

### "Calibration file not found"
Run standalone calibration: `python calibrate.py --video-source /dev/video0`

### "Not enough points detected"
- Improve lighting
- Check projector brightness
- Move camera closer
- See [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) for detailed help

### "Keys don't work during calibration"
- **Click on the camera feed window** to give it focus
- Try alternative keys: 'c' to capture, 'q' to quit
- Check terminal for "Key detected: XX" messages
- See [KEYBOARD_TROUBLESHOOTING.md](KEYBOARD_TROUBLESHOOTING.md) for complete solutions

### "Could not detect Hailo architecture"
Ensure Hailo device is connected and drivers installed.
Check: `lsusb | grep Hailo`

### Pipeline errors
Try with a test video file first:
```bash
python main.py --video-source /path/to/video.mp4
```

### No audio
Audio is optional. The game will work without sound.
To add audio, place files in `assets/audio/` (see audio README).

### Effects not aligned with players
Re-run calibration. Ensure camera and projector didn't move.

## Game Controls

- **ESC**: Quit game
- **R**: Reset to countdown

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Customize game settings in `config/game_config.yaml`
- Add custom audio in `assets/audio/`
- Run tests: `pytest tests/`

## Testing Without Hardware

For development without projector/camera:

```bash
# Use a video file as camera source
python main.py --video-source sample_video.mp4

# Skip calibration in code for testing
# (requires manual code modification in main.py)
```

## Getting Help

- Check logs in terminal for error messages
- Review `README.md` for detailed documentation
- Ensure Hailo pipeline works: `hailo-pose --input /dev/video0`

## Project Structure

```
freezing_phantoms/
├── main.py              # Start here
├── config/              # Game configuration
├── calibration/         # Camera-projector calibration
├── core/                # Game logic
├── pipeline/            # AI inference
├── effects/             # Visual effects
├── assets/              # Audio and textures
└── tests/               # Unit tests
```

Happy haunting! 👻

