# Keyboard Input Troubleshooting

## Latest Fixes Applied

The calibration tool has been updated with multiple improvements to keyboard detection:

### What Was Fixed

1. **Visual Feedback**
   - On-screen instructions overlaid on camera feed
   - Live frame counter to confirm video is working
   - Clear text: "PRESS SPACE TO CAPTURE"

2. **Multiple Key Options**
   - SPACE or 'c' to capture
   - ESC or 'q' to quit
   - Increased chances of working on different systems

3. **Better Window Management**
   - Resizable windows (640x480)
   - Initialization delay (0.5s) to ensure windows render
   - Clear window titles with instructions

4. **Improved Key Detection**
   - Increased `waitKey` to 50ms (from 30ms)
   - Real-time key logging to terminal
   - Shows key code when detected

5. **User Guidance**
   - Explicit instruction to click on camera window
   - Frame counter confirms video is live
   - Terminal feedback when keys are pressed

## How to Use

### Calibration Mode

When you run calibration:

```bash
python calibrate.py --video-source /dev/video0
```

You'll see:

```
======================================================================
CALIBRATION INSTRUCTIONS:
======================================================================
1. Calibration pattern is displayed on projector
2. Camera feed window will show below
3. Click on the camera feed window to ensure it has focus
4. Press SPACE to capture (or 'c' if SPACE doesn't work)
5. Press ESC to cancel (or 'q' to quit)
======================================================================

⏳ Waiting for camera to initialize...

✓ Camera ready! Showing live feed...
👉 Click on the camera window and press SPACE to capture
```

The camera window will show:
- **"PRESS SPACE TO CAPTURE"** at the top
- **"Press ESC to cancel"** below
- **"Frame: XXX"** at the bottom (confirms video is running)

### What to Do

1. **Wait for camera window to appear**
2. **Click on the camera feed window** (very important!)
3. **Press SPACE** (or 'c' if SPACE doesn't work)
4. Watch terminal for **"Key detected: XX"** message

### Validation Mode

```bash
python calibrate.py --validate --video-source /dev/video0
```

Window shows:
- **"VALIDATION MODE - Press ESC to exit"**
- Red dot in center
- **"Red dot should align with green circle on projector"**
- Frame counter

## If Keys Still Don't Work

### Step 1: Verify Window Has Focus

**Problem:** OpenCV window might not have focus

**Solution:**
```bash
# After the camera window appears:
1. Click directly on the camera feed window
2. Wait a moment
3. Try pressing SPACE
4. Check terminal for "Key detected: XX" message
```

### Step 2: Try Alternative Keys

If SPACE doesn't work:
- Try **'c'** (capture)
- Try **'q'** (quit)
- Check if ANY key is detected in terminal

### Step 3: Check Window Manager

**Problem:** Some Linux window managers intercept keys

**Solution:**
```bash
# Check if running Wayland
echo $XDG_SESSION_TYPE

# If Wayland, try X11 session instead
# Or disable window manager compositing temporarily
```

### Step 4: Test OpenCV Key Detection

Run this test script:

```python
#!/usr/bin/env python3
import cv2
import numpy as np

print("OpenCV Key Detection Test")
print("A window will appear. Click it and press keys.")
print("Press ESC to exit")
print()

cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(frame, "Press any key", (200, 240),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

frame_count = 0
while True:
    display = frame.copy()
    cv2.putText(display, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Test Window", display)

    key = cv2.waitKey(50) & 0xFF
    frame_count += 1

    if key != 255:
        print(f"Key detected: {key} ('{chr(key) if 32 <= key < 127 else '?'}')")

        if key == 27:  # ESC
            print("ESC pressed - exiting")
            break

cv2.destroyAllWindows()
print("Test complete")
```

Save as `test_keys.py` and run:
```bash
python test_keys.py
```

### Step 5: System-Specific Workarounds

#### For Ubuntu/Debian
```bash
# Install missing OpenCV dependencies
sudo apt-get install libgtk-3-0 libgtk-3-dev

# Ensure video group permissions
sudo usermod -a -G video $USER
# Log out and back in
```

#### For Wayland Systems
```bash
# OpenCV works better with X11
# Start X11 session or run with:
export GDK_BACKEND=x11
python calibrate.py --video-source /dev/video0
```

#### For Remote SSH/X11 Forwarding
```bash
# X11 forwarding might not work well with OpenCV
# Try running locally or use VNC instead
```

### Step 6: Use Mouse Callback (Fallback)

If keyboard still doesn't work, we can add mouse click support:

Edit `calibration_tool.py` and add this before the main loop:

```python
# Mouse callback for capture
captured = [False]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        captured[0] = True
        print("✓ Click detected - capturing!")

cv2.setMouseCallback(camera_feed_window, mouse_callback)
```

Then in the loop:
```python
# Check mouse click
if captured[0]:
    camera_image = frame.copy()
    print("\n✓ Image captured!\n")
    break
```

## Debugging Information

### What the Terminal Shows

**Successful key press:**
```
✓ Camera ready! Showing live feed...
👉 Click on the camera window and press SPACE to capture

   Key detected: 32

✓ Image captured!
```

**No key detected:**
```
✓ Camera ready! Showing live feed...
👉 Click on the camera window and press SPACE to capture

[No "Key detected" messages appear]
```

### Key Code Reference

- **32** = SPACE
- **27** = ESC
- **99** / **67** = 'c' / 'C'
- **113** / **81** = 'q' / 'Q'
- **255** = No key pressed

## Known Issues

### Issue 1: Window Behind Fullscreen Pattern

**Symptom:** Can't see camera window because pattern is fullscreen

**Solution:** The pattern window is on the projector (secondary display). Camera window should be on your primary display.

### Issue 2: No Window Appears

**Symptom:** No camera feed window shows up

**Check:**
```bash
# Test camera works
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --all

# Test OpenCV can open camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Success' if cap.isOpened() else 'Failed')"
```

### Issue 3: Window Appears But Frozen

**Symptom:** Window shows but doesn't update

**Check:**
- Frame counter should increment
- If frozen, camera might not be streaming
- Try different video source

### Issue 4: Keys Work in Terminal But Not OpenCV

**Symptom:** Terminal receives input, OpenCV doesn't

**Reason:** OpenCV window needs focus

**Solution:**
1. Click the OpenCV window
2. Don't click back to terminal
3. Press keys while OpenCV window is focused

## Alternative: Use Terminal Input

If OpenCV keys absolutely don't work, you can modify the code to use terminal input:

```python
import sys
import select

# In the loop, add timeout check for stdin
while True:
    ret, frame = cap.read()
    cv2.imshow(camera_feed_window, frame)
    cv2.waitKey(50)

    # Check for terminal input (non-blocking)
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if line.strip() == 'c' or line.strip() == '':
            camera_image = frame.copy()
            print("✓ Captured!")
            break
```

Then press ENTER in terminal to capture.

## Getting Help

If keyboard input still doesn't work after trying all the above:

1. **Run the key detection test** (`test_keys.py`)
2. **Note your system:**
   - OS and version
   - Desktop environment (GNOME, KDE, etc.)
   - Display server (X11 or Wayland)
   - OpenCV version

3. **Check terminal output** for "Key detected" messages

4. **Try the fallback:** Use mouse click capture (code provided above)

5. **Report the issue** with:
   - System information
   - Terminal output
   - Whether test_keys.py works

## Success Indicators

You'll know it's working when you see:

✅ Camera window appears with live video
✅ Frame counter increments
✅ On-screen text is visible
✅ Terminal shows "Key detected: XX" when you press keys
✅ Capture succeeds after pressing SPACE

If you see all of the above, the keyboard input is working correctly!

