# Calibration Guide - Freezing Phantoms

Complete guide for calibrating camera-projector alignment.

## Overview

The calibration process establishes a mathematical mapping (homography matrix) between camera coordinates and projector coordinates. This allows the game to project visual effects precisely onto players.

## Prerequisites

- Overhead camera mounted and focused on play area
- Projector mounted overhead covering the play area
- Both devices connected and working
- Play area should be flat (floor or table)

## Quick Start

### 1. Run Standalone Calibration

```bash
cd freezing_phantoms
python calibrate.py --video-source /dev/video0
```

### 2. Follow On-Screen Instructions

1. Calibration pattern displays on projector
2. Camera feed shows in separate window
3. **Press SPACE** when camera can see the pattern clearly
4. System detects circles and calculates calibration
5. Results displayed with quality metrics

### 3. Validate Calibration (Optional)

```bash
python calibrate.py --validate --video-source /dev/video0
```

Move your hand - green circle should track on projector.

## Standalone Calibration Tool

The `calibrate.py` script is completely independent from the game.

### Command Line Options

```bash
python calibrate.py [OPTIONS]

Options:
  --video-source PATH    Camera device or video file (default: /dev/video0)
  --config PATH          Configuration file (default: config/game_config.yaml)
  --output PATH          Output calibration file (default: config/calibration.npy)
  --validate            Validate existing calibration
  --min-points N        Minimum points required (default: 20)
  -h, --help            Show help message
```

### Examples

```bash
# Basic calibration
python calibrate.py

# With specific camera
python calibrate.py --video-source /dev/video1

# Require more points for accuracy
python calibrate.py --min-points 35

# Save to custom location
python calibrate.py --output /tmp/my_calibration.npy

# Validate existing calibration
python calibrate.py --validate
```

## Understanding the Process

### Pattern Generation

The tool generates a grid of white circles on black background:
- Default: 6 rows × 8 columns = 48 circles
- Configurable in `config/game_config.yaml`
- Pattern displayed fullscreen on projector

### Circle Detection

The camera captures the projected pattern:
- **Minimum Required:** 20 points (configurable)
- **Expected:** 48 points (6×8 grid)
- **Good Quality:** ≥80% of expected points
- **Acceptable:** ≥40% of expected points

### Quality Metrics

The tool reports:
1. **Points Detected:** How many circles found
2. **Inlier Ratio:** Percentage of points fitting the model
   - **Excellent:** ≥90%
   - **Acceptable:** 70-90%
   - **Poor:** <70%

### Example Output

```
🔍 Detecting calibration points...
   Expected: 48 circles
   Detected: 46 circles
✅ Good detection: 46/48 circles
   (95.8% of expected)

✅ Using 46 matched points for calibration

🧮 Calculating homography matrix...
✅ Homography matrix calculated
   Inliers: 44/46 (95.7%)
✓ Excellent quality (95.7%)

✅ CALIBRATION SUCCESSFUL!
📊 Calibration matrix saved to: config/calibration.npy
```

## Troubleshooting

### Too Few Points Detected

**Problem:** Less than 20 circles detected

**Solutions:**
1. **Improve Lighting**
   ```bash
   # Check if camera sees the pattern
   v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1
   v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=100
   ```

2. **Adjust Camera Position**
   - Move closer to pattern
   - Ensure entire pattern is visible
   - Check focus

3. **Increase Projector Brightness**
   - Check projector settings
   - Reduce ambient light
   - Clean projector lens

4. **Check Pattern Configuration**
   Edit `config/game_config.yaml`:
   ```yaml
   calibration:
     pattern_rows: 5        # Reduce from 6
     pattern_cols: 6        # Reduce from 8
     circle_radius: 50      # Increase from 40
     circle_spacing: 180    # Increase from 150
   ```

### Low Inlier Ratio

**Problem:** Inlier ratio < 70%

**Causes:**
- Camera distortion
- Pattern not flat
- Movement during capture
- Bad focus

**Solutions:**
- Use wide-angle camera with less distortion
- Ensure play surface is completely flat
- Keep camera/projector stable
- Capture with better focus

### Pattern Not Visible on Projector

**Problem:** No pattern displayed

**Solutions:**
1. **Check Display Configuration**
   ```bash
   # List displays
   xrandr

   # Extend desktop to projector
   xrandr --output HDMI-1 --auto --right-of eDP-1
   ```

2. **Adjust display_index in config:**
   ```yaml
   projector:
     display_index: 0  # Try 0, 1, or 2
   ```

### Keyboard Input Not Working

**Solutions:**
- Click on "Camera Feed" window before pressing keys
- Use longer key press (not quick tap)
- Check terminal for "Key pressed: XX" debug messages
- Try ESC if SPACE doesn't work, restart and try again

## Advanced Configuration

### Custom Pattern

Edit `config/game_config.yaml`:

```yaml
calibration:
  # Grid size
  pattern_rows: 6        # Number of rows
  pattern_cols: 8        # Number of columns

  # Circle properties
  circle_radius: 40      # Radius in pixels
  circle_spacing: 150    # Distance between centers
```

**Trade-offs:**
- **More circles:** Better accuracy, harder to detect all
- **Fewer circles:** Easier detection, less accuracy
- **Larger circles:** Easier detection, need more space
- **Smaller circles:** Fit more, harder to detect

### Minimum Points Threshold

```bash
# Stricter (better quality, may fail more often)
python calibrate.py --min-points 35

# More lenient (accepts lower quality)
python calibrate.py --min-points 15
```

**Recommendations:**
- **Simple setup:** 15-20 points
- **Standard setup:** 20-30 points
- **High accuracy:** 30-40 points

## Integration with Game

The game automatically uses calibration from `config/calibration.npy`.

### Option 1: Standalone Calibration (Recommended)

```bash
# Step 1: Run calibration separately
python calibrate.py --video-source /dev/video0

# Step 2: Run game
python main.py --video-source /dev/video0
```

### Option 2: Integrated Calibration

```bash
# Calibration + game in one command
python main.py --run-calibration --video-source /dev/video0
```

## Validation

Always validate after calibration:

```bash
python calibrate.py --validate --video-source /dev/video0
```

**What to check:**
- Green circle tracks your hand position accurately
- No significant offset between hand and circle
- Tracking works across entire play area
- Circle stays stable (not jittering)

## Best Practices

1. **Hardware Setup**
   - Mount camera and projector securely
   - Calibrate after any movement
   - Use tripods or permanent mounting

2. **Environment**
   - Reduce ambient light
   - Avoid direct sunlight
   - Keep consistent lighting

3. **Calibration Timing**
   - Calibrate before each game session
   - Re-calibrate if hardware moves
   - Save multiple calibration files for different setups

4. **Quality Checks**
   - Aim for >40 detected points
   - Verify >85% inlier ratio
   - Always run validation
   - Test at edges and center of play area

## Retry Logic

The standalone tool automatically retries failed calibrations:

```
📸 Calibration attempt 1/3
❌ Not enough points detected!
   Need at least 20 points
   Only found 12 points
   Missing 8 points

⚠️  Calibration failed. Retrying (1/3)...
   Tips:
   - Ensure better lighting
   - Move camera closer or adjust angle
   - Ensure pattern is fully visible and in focus

Press ENTER to retry or Ctrl+C to cancel...
```

You get 3 attempts to improve conditions between tries.

## Saving Calibrations

```bash
# Save different calibrations for different setups
python calibrate.py --output calibration_outdoor.npy
python calibrate.py --output calibration_indoor.npy

# Use specific calibration with game
python main.py --calibration calibration_outdoor.npy
```

## Technical Details

### Homography Matrix

The calibration produces a 3×3 transformation matrix:

```
H = | h11  h12  h13 |
    | h21  h22  h23 |
    | h31  h32  h33 |
```

This transforms any camera coordinate (xc, yc) to projector coordinate (xp, yp):

```python
[xp']   [h11  h12  h13]   [xc]
[yp'] = [h21  h22  h23] × [yc]
[w' ]   [h31  h32  h33]   [1 ]

xp = xp' / w'
yp = yp' / w'
```

### File Format

Calibration saved as NumPy binary (`.npy`):
- File size: ~128 bytes
- Contains: 3×3 float64 matrix
- Platform independent
- Fast to load

## Troubleshooting by Error Message

### "Failed to open camera"
```bash
# Check camera exists
ls -l /dev/video*

# Check permissions
sudo usermod -a -G video $USER
# Log out and back in

# Test camera
ffplay /dev/video0
```

### "Configuration file not found"
```bash
# Ensure you're in the right directory
cd freezing_phantoms

# Check config exists
ls -l config/game_config.yaml
```

### "No circles detected"
- Increase exposure: Brighter projector, longer exposure
- Check focus: Camera must focus at distance
- Clean lenses: Both camera and projector
- Reduce ambient light

### "Homography calculation failed"
- Points may be collinear (in a line)
- Ensure camera sees full pattern
- Pattern should cover large area
- Check camera isn't too tilted

## Support

If calibration continues to fail:
1. Save the camera image: Look for `vis_image` display
2. Check the debug logs in terminal
3. Try with a recorded video for testing
4. Verify hardware is working independently

For additional help, see main `README.md` or raise an issue.

