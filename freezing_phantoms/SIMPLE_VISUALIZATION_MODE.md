# Simple Pose Visualization Mode

## Overview

The Freezing Phantoms game has been simplified to **visualization mode only** - displaying just the pose landmarks projected through the calibration homography. This is perfect for testing that the calibration is working correctly.

## What Was Removed

All game logic and visual effects have been disabled:

- ❌ Game state machine (countdown, red/green light, penalty)
- ❌ Movement detection
- ❌ Audio system
- ❌ Board obstacles
- ❌ Ghosts
- ❌ Board effects
- ❌ Score tracking
- ❌ All visual effects

## What Remains

Only the essentials for pose visualization:

- ✅ **Pose estimation pipeline** (720p)
- ✅ **Calibration homography** transformation
- ✅ **Projection renderer** (simple mode)
- ✅ **Real-time pose keypoint display**

## What You'll See

When you run the game, the projector will display:

### Pose Keypoints
- Each player's pose keypoints as **colored circles**
- Circle size varies by confidence (bigger = more confident)
- Different colors for different players (green, yellow, magenta, orange, blue)
- Short labels showing keypoint names ("nose", "left", "righ", etc.)

### Leg Position
- **White circle** showing calculated leg position (from ankle keypoints)
- Player ID label
- This is the position used for game collision detection

### Info Display
- "Pose Landmarks Projection Test" title
- Player count
- "Press ESC to exit" instruction

## Running the Visualization

```bash
cd freezing_phantoms
python main.py --input /dev/video0
```

You should see:
1. Terminal message: "POSE LANDMARKS VISUALIZATION"
2. Blank projector screen (black background)
3. When you step into view, colored circles appear at your joints
4. Circles follow your movements in real-time
5. All positions are transformed through calibration homography

## Verifying Calibration

This mode is perfect for checking calibration accuracy:

### Good Calibration Signs ✅
- Keypoints appear exactly where your body parts are
- Circles don't slide or drift
- Movement is smooth and accurate
- Leg position circle aligns with your feet

### Bad Calibration Signs ❌
- Keypoints are offset from your body
- Circles appear stretched or distorted
- Different body parts have different offsets
- Leg position doesn't match your feet location

## Code Changes

### Simplified `main.py`

**Before (Full Game):**
```python
# Initialize all components
self.game_state = GameStateMachine(...)
self.movement_detector = MovementDetector(...)
self.audio_manager = AudioManager(...)
self.board_system = BoardSystem(...)
self.ghost_manager = GhostManager(...)
self.board_effects = BoardEffects(...)

# Game loop with full logic
self.game_state.update(dt, player_data)
self.ghost_manager.update(positions, dt)
self._check_board_collisions(player_data)
self._check_for_movement(player_data)
projection_frame = self.renderer.render_frame(...)
projection_frame = self.board_effects.render_board(...)
```

**After (Simple Visualization):**
```python
# Initialize only renderer
self.renderer = ProjectionRenderer(...)

# Simple loop - just render landmarks
projection_frame = self.renderer.render_simple_pose_landmarks(
    player_data,
    frame_width,
    frame_height,
)
```

### New Render Method

Added `render_simple_pose_landmarks()` in `projection_renderer.py`:

```python
def render_simple_pose_landmarks(self, player_data, frame_width, frame_height):
    # Create blank canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for player in players:
        for keypoint_name, (cam_x, cam_y) in pose_keypoints.items():
            # Transform to projector coordinates
            proj_x, proj_y = self.transform_point(cam_x, cam_y)

            # Draw circle at keypoint
            cv2.circle(canvas, (proj_x, proj_y), radius, color, -1)
            cv2.putText(canvas, keypoint_name[:4], ...)

    return canvas
```

## Technical Details

### Resolution
- Camera: **1280x720** (720p)
- Pipeline: **1280x720** (720p)
- Projector: **1280x720** (720p)
- All consistent with calibration

### Pose Keypoints (COCO Format)
17 keypoints detected per person:
- Head: nose, eyes, ears
- Torso: shoulders, hips
- Arms: elbows, wrists
- Legs: knees, ankles

### Coordinate Transformation
```
Camera Coordinates (x, y)
         ↓
  Homography Matrix H
         ↓
Projector Coordinates (px, py)
```

Each keypoint goes through this transformation before being displayed.

### Confidence Filtering
- Only keypoints with confidence > 0.3 are displayed
- Circle radius = `5 + confidence * 10` pixels
- Low confidence = small circles
- High confidence = large circles

## Debugging Tips

### No Circles Appearing?
1. Check if camera is working: `ls -la /dev/video*`
2. Check if pose detection is running: add `--show-video` flag
3. Verify calibration file exists: `ls -la config/calibration.npy`
4. Check logs for errors: add `--log-level DEBUG`

### Circles in Wrong Places?
1. **Re-run calibration**: `python main.py --run-calibration`
2. Ensure camera and projector haven't moved
3. Check calibration matrix: `python -c "import numpy as np; print(np.load('config/calibration.npy'))"`

### Circles Jittering?
- This is normal with pose estimation
- Jitter is reduced by confidence-based sizing
- Full game has smoothing enabled (currently disabled)

### Performance Issues?
- Check FPS: add `--show-fps` flag
- Reduce camera resolution if needed (edit config)
- Ensure Hailo device is properly connected

## Next Steps

Once calibration is verified:

1. **Re-enable game logic**: Uncomment game components in `main.py`
2. **Test movement detection**: Enable MovementDetector
3. **Add game effects**: Uncomment BoardEffects
4. **Enable audio**: Uncomment AudioManager
5. **Add obstacles**: Uncomment BoardSystem

## Restoring Full Game

To restore the full game functionality:

1. **Uncomment imports** in `main.py`:
```python
from freezing_phantoms.core.audio_manager import AudioManager
from freezing_phantoms.core.board_system import BoardSystem
from freezing_phantoms.core.game_state import GameState, GameStateMachine
# ... etc
```

2. **Uncomment initialization** code:
```python
self.game_state = GameStateMachine(self.config)
self.movement_detector = MovementDetector(...)
# ... etc
```

3. **Replace render call** with full game logic:
```python
# Replace
projection_frame = self.renderer.render_simple_pose_landmarks(...)

# With
self.game_state.update(dt, player_data)
projection_frame = self.renderer.render_frame(...)
projection_frame = self.board_effects.render_board(...)
```

## Summary

✅ **Minimal visualization mode** for calibration testing
✅ **Clean projection** - just pose keypoints
✅ **Real-time display** through homography
✅ **Easy to verify** calibration accuracy
✅ **Simple to restore** full game when ready

The game is now a pure pose visualization tool - perfect for testing your camera-to-projector calibration!

