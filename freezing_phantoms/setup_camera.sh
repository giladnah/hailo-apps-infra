#!/bin/bash
# Camera setup script for Red Light Green Light game
# Disables auto-focus and auto-brightness for stable detection

CAMERA_DEVICE="${1:-/dev/video0}"

echo "====================================="
echo "Camera Setup for Red Light Green Light"
echo "====================================="
echo "Camera device: $CAMERA_DEVICE"
echo ""

# Check if v4l2-ctl is installed
if ! command -v v4l2-ctl &> /dev/null; then
    echo "Error: v4l2-ctl not found!"
    echo "Please install it with: sudo apt-get install v4l-utils"
    exit 1
fi

echo "Current camera settings:"
v4l2-ctl -d "$CAMERA_DEVICE" -L | grep -E "(focus|exposure|brightness|white_balance)"
echo ""

echo "Applying fixed camera settings..."

# Disable auto-focus and set to infinity (for overhead camera)
if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "focus_auto"; then
    echo "- Disabling auto-focus..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=focus_auto=0 2>/dev/null || echo "  (focus_auto not available)"
fi

if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "focus_absolute"; then
    echo "- Setting focus to infinity..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=focus_absolute=0 2>/dev/null || echo "  (focus_absolute not available)"
fi

# Disable auto-exposure
if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "exposure_auto"; then
    echo "- Disabling auto-exposure..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=exposure_auto=1 2>/dev/null || echo "  (exposure_auto not available)"
fi

if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "exposure_absolute"; then
    echo "- Setting fixed exposure..."
    # Set to a reasonable value (adjust based on your lighting)
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=exposure_absolute=200 2>/dev/null || echo "  (exposure_absolute not available)"
fi

# Disable auto-brightness
if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "brightness"; then
    echo "- Setting fixed brightness..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=brightness=128 2>/dev/null || echo "  (brightness not available)"
fi

# Disable auto white balance
if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "white_balance_temperature_auto"; then
    echo "- Disabling auto white balance..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=white_balance_temperature_auto=0 2>/dev/null || echo "  (white_balance_temperature_auto not available)"
fi

if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "white_balance_temperature"; then
    echo "- Setting white balance..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=white_balance_temperature=4000 2>/dev/null || echo "  (white_balance_temperature not available)"
fi

# Disable power line frequency (reduces flicker)
if v4l2-ctl -d "$CAMERA_DEVICE" -l | grep -q "power_line_frequency"; then
    echo "- Setting power line frequency..."
    v4l2-ctl -d "$CAMERA_DEVICE" --set-ctrl=power_line_frequency=2 2>/dev/null || echo "  (power_line_frequency not available)"
fi

echo ""
echo "New camera settings:"
v4l2-ctl -d "$CAMERA_DEVICE" -L | grep -E "(focus|exposure|brightness|white_balance)"
echo ""
echo "====================================="
echo "Camera setup complete!"
echo "====================================="
echo ""
echo "NOTE: These settings will be reset if you unplug the camera."
echo "Run this script again after reconnecting the camera."
echo ""
echo "If you need to adjust settings:"
echo "  - Brightness is too low: v4l2-ctl -d $CAMERA_DEVICE --set-ctrl=brightness=150"
echo "  - Brightness is too high: v4l2-ctl -d $CAMERA_DEVICE --set-ctrl=brightness=100"
echo "  - Exposure is too low: v4l2-ctl -d $CAMERA_DEVICE --set-ctrl=exposure_absolute=250"
echo "  - Exposure is too high: v4l2-ctl -d $CAMERA_DEVICE --set-ctrl=exposure_absolute=150"
echo ""

