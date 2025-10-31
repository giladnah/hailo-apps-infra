# Camera Setup Guide

## Problem: Camera Auto-Focus and Auto-Brightness

If your camera keeps adjusting focus, brightness, or exposure during gameplay, it can cause detection issues and visual instability.

## Automatic Setup During Calibration

**Good news!** Camera settings are now automatically applied during the calibration process. When you run calibration, the tool will:

1. Check for `v4l2-ctl` availability
2. Automatically disable auto-focus, auto-exposure, and auto-brightness
3. Set fixed values for optimal detection

No manual setup needed before calibration!

## Manual Setup (Optional)

If you want to apply settings before starting the game (or re-apply after reconnecting the camera), run:

```bash
./setup_camera.sh /dev/video0
```

Replace `/dev/video0` with your camera device (check with `ls /dev/video*`).

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

