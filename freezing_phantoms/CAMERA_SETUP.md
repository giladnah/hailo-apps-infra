# Camera Setup Guide

## Overview

The game automatically manages camera settings to ensure consistent image quality for pose detection and calibration accuracy. Camera settings are configured during calibration and automatically applied during gameplay.

## Automatic Setup (Recommended)

**Camera settings are fully automated:**

1. **During Calibration**:
   - Auto-focus, auto-exposure, and auto-brightness are disabled
   - Interactive keyboard controls let you adjust settings in real-time
   - Settings are **saved automatically** when calibration succeeds
   - Saved as `config/calibration_camera_settings.json`

2. **During Gameplay**:
   - Saved camera settings are loaded automatically
   - Applied before starting the AI pipeline
   - Ensures consistent camera behavior every session

**No manual setup required!** Just run calibration, adjust if needed, and the game uses those settings automatically.

## Camera Settings During Calibration

The calibration tool provides **interactive manual controls** while viewing the camera feed:

- **1 / 2** - Decrease / Increase brightness
- **3 / 4** - Decrease / Increase exposure
- **5 / 6** - Decrease / Increase focus
- **A** - Toggle auto-exposure on/off
- **R** - Reset to default values

Adjust these settings until the checkerboard pattern is clearly visible, then capture. Your settings will be saved automatically!

## Saved Settings File

After calibration, you'll have:
- `config/calibration.npy` - The homography matrix
- `config/calibration_camera_settings.json` - Your camera settings (brightness, exposure, focus, etc.)

The game automatically loads both files when starting.

## Manual Setup Script (Optional)

If you need to apply settings outside of calibration, you can use:

```bash
./setup_camera.sh /dev/video0
```

Replace `/dev/video0` with your camera device (check with `ls /dev/video*`).

Note: This applies **default** settings, not your calibrated settings.

## What This Does

The script disables:
- **Auto-focus** → Sets focus to infinity (best for overhead camera)
- **Auto-exposure** → Sets fixed exposure value
- **Auto-brightness** → Sets fixed brightness value
- **Auto white balance** → Sets fixed white balance

## Manual Camera Settings

If you prefer manual control, you can use `v4l2-ctl` directly:

### Install v4l-utils (if not installed)
```bash
sudo apt-get install v4l-utils
```

### View all available camera controls
```bash
v4l2-ctl -d /dev/video0 -L
```

### Common adjustments

**Disable auto-focus:**
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=0
```

**Disable auto-exposure:**
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=200
```

**Set fixed brightness:**
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=128
```

**Disable auto white balance:**
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=4000
```

## Fine-Tuning

After running the setup script, you may need to adjust based on your lighting:

### Too Dark?
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=150
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=250
```

### Too Bright?
```bash
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=100
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=150
```

### Blurry?
```bash
# Try different focus values (0 = infinity, higher = closer focus)
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=0
```

## Important Notes

⚠️ **Camera settings reset when unplugged!**
- Run the setup script again after reconnecting the camera
- Or add it to your startup script

💡 **Optimal Settings**
- For overhead installation: Focus should be at infinity (0)
- Lighting: Consistent, bright lighting works best
- Exposure: Adjust based on your specific lighting conditions

## Testing Your Settings

After applying settings, test with:
```bash
# View camera feed
ffplay /dev/video0

# Or check with the game in debug mode
python main.py --input /dev/video0 --show-video
```

Look for:
- ✅ Stable image (no sudden brightness changes)
- ✅ Sharp focus (not constantly adjusting)
- ✅ Consistent colors (no white balance shifts)

## Troubleshooting

### Settings don't persist
Some cameras don't support certain controls. Check available controls:
```bash
v4l2-ctl -d /dev/video0 -L
```

### Permission denied
Add your user to the video group:
```bash
sudo usermod -aG video $USER
# Then log out and back in
```

### Different camera models
Each camera model supports different controls. Use:
```bash
v4l2-ctl -d /dev/video0 --list-ctrls
```
to see what your specific camera supports.

## Automation

To automatically apply settings on boot, create a systemd service or udev rule. Example udev rule:

Create `/etc/udev/rules.d/99-camera-settings.rules`:
```
ACTION=="add", SUBSYSTEM=="video4linux", KERNEL=="video0", RUN+="/path/to/setup_camera.sh /dev/video0"
```

Then reload udev:
```bash
sudo udevadm control --reload-rules
```

