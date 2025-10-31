# Development Guide

## Project Status

This project has been transformed from a "Freezing Phantoms Halloween Game" into a "Red Light, Green Light - Floor is Lava" tile-based game with full camera settings management.

## Architecture Overview

### Core Components

1. **Tile Grid System** (`core/tile_grid.py`)
   - Manages 5x10 grid of tiles
   - Automatic path generation from start to goal
   - Player position tracking via leg detection
   - Progressive path revelation during gameplay

2. **Game State Machine** (`core/game_state.py`)
   - States: COUNTDOWN, GREEN_LIGHT, RED_LIGHT, PENALTY, WIN
   - Automatic state transitions based on timers
   - Handles movement violations and tile violations
   - Win condition detection

3. **Projection Renderer** (`core/projection_renderer.py`)
   - Renders game graphics using homography transformation
   - Lava floor texture with tile grid overlay
   - Player position indicators
   - State-specific visual effects

4. **Camera Settings Management** (`core/camera_utils.py`)
   - Save/load camera settings to JSON
   - Apply settings via `v4l2-ctl`
   - Automatic application on game start

5. **AI Pipeline** (`pipeline/combined_pipeline.py`, `pipeline/data_extractor.py`)
   - Hailo pose estimation at 720p
   - Player data extraction from Hailo buffer
   - Leg position calculation (ankle + bbox bottom average)
   - Coordinate transformation to tile grid

6. **Calibration System** (`calibration/calibration_tool.py`)
   - Interactive checkerboard-based calibration
   - Real-time camera settings adjustment (keys 1-6)
   - Automatic camera settings save
   - Validation mode for testing accuracy

### Data Flow

```
Camera (720p)
    ↓
GStreamer Pipeline
    ↓
Hailo Pose Estimation
    ↓
Data Extractor
    ↓
Player Tracking (leg position)
    ↓
Tile Grid (detect current tile)
    ↓
Game State Machine (check violations, win condition)
    ↓
Projection Renderer (homography transform)
    ↓
Projector Display
```

### Coordinate Spaces

1. **Camera Space**: Raw camera pixels (1280x720)
2. **Normalized Space**: Pose keypoints relative to detection box (0.0-1.0)
3. **Camera Absolute**: Keypoints in camera pixel coordinates
4. **Projector Space**: Transformed via homography (1280x720)
5. **Tile Space**: Grid row/column coordinates (0-9 rows, 0-4 cols)

## Key Features

### Leg Position Detection

Player position is determined by approximating foot location:

```python
# 1. Get ankle keypoints from pose estimation
left_ankle = pose_keypoints.get("left_ankle")
right_ankle = pose_keypoints.get("right_ankle")

# 2. Average ankle positions
ankle_x = (left_ankle[0] + right_ankle[0]) / 2
ankle_y = (left_ankle[1] + right_ankle[1]) / 2

# 3. Get bbox bottom
bbox_bottom_y = (bbox.ymin() + bbox.height()) * frame_height

# 4. Average for ground contact approximation
leg_y = (ankle_y + bbox_bottom_y) / 2
```

### Path Generation

The tile grid automatically generates a winding path:

```python
# Start at middle of starting row
current_col = cols // 2

# Move towards goal row
for row in range(starting_row, goal_row + 1):
    path.append((row, current_col))
    
    # Randomly move left/right
    direction = random.choice([-1, 0, 1])
    current_col = max(0, min(cols - 1, current_col + direction))
```

### Camera Settings Workflow

1. **During Calibration**:
   - Automatic camera setup (disable auto-focus, etc.)
   - Interactive manual controls (keys 1-6)
   - Settings saved as `config/calibration_camera_settings.json`

2. **During Game**:
   - Settings automatically loaded and applied
   - Consistent camera behavior every session

## Code Organization

### Core Modules
- `tile_grid.py`: Grid system, path generation, player position
- `game_state.py`: State machine, transitions, timers
- `movement_detector.py`: Pose keypoint comparison for movement
- `projection_renderer.py`: Visual rendering with homography
- `camera_utils.py`: Camera settings save/load/apply

### Pipeline Modules
- `combined_pipeline.py`: Hailo pose estimation pipeline
- `data_extractor.py`: Player data extraction from Hailo

### Calibration Modules
- `calibration_tool.py`: Interactive calibration with camera controls
- `pattern_generator.py`: Checkerboard pattern generation/detection

### Effects Modules
- `board_effects.py`: Lava texture, player indicator, tile effects
- `green_light_effects.py`: Green light visual effects
- `red_light_effects.py`: Red light visual effects

## Configuration

Edit `config/game_config.yaml` to customize:

```yaml
game:
  countdown_duration: 3.0
  green_light_duration: [5.0, 8.0]
  red_light_duration: [3.0, 5.0]
  penalty_duration: 2.0

tile_grid:
  columns: 5
  rows: 10
  starting_row: 0
  goal_row: 9
  penalty_rows_back: 2
  path_width: 1
  obstacle_density: 0.3

movement:
  threshold_pixels: 20.0
  smoothing_alpha: 0.3
  keypoint_confidence_threshold: 0.5

camera:
  resolution: [1280, 720]
  fps: 30

projector:
  resolution: [1280, 720]
  display_index: 1
```

## Testing

### Unit Tests

```bash
pytest tests/
```

Test files:
- `test_game_state.py`: State machine transitions
- `test_movement_detection.py`: Movement detector
- `test_calibration.py`: Calibration utilities

### Manual Testing

1. **Calibration**:
   ```bash
   python main.py --run-calibration --input /dev/video0
   ```

2. **Validation**:
   ```bash
   python main.py --validate-calibration --input /dev/video0
   ```

3. **Debug Mode**:
   ```bash
   python main.py --input /dev/video0 --show-video
   ```

## Common Issues and Solutions

### Calibration

**Pattern not visible:**
- Use keyboard controls (1-6) to adjust brightness/exposure/focus
- Ensure good, even lighting
- Check projector brightness

**Detection fails:**
- Pattern must be fully in camera view
- Reduce glare and reflections
- Try different exposure settings

### Camera Settings

**Auto-focus/auto-brightness issues:**
- Camera settings saved during calibration
- Automatically applied on game start
- Manual script: `./setup_camera.sh /dev/video0`

**Settings don't persist:**
- Install `v4l2-ctl`: `sudo apt-get install v4l-utils`
- Camera settings reset when unplugged

### Game Play

**Position inaccurate:**
- Re-run calibration with better accuracy
- Check lighting for pose detection
- Ensure camera sees player's full body

**Movement detection issues:**
- Adjust `threshold_pixels` in config
- Modify `monitored_keypoints` list
- Adjust `smoothing_alpha` for stability

## Adding Features

### New Tile Effects

Edit `effects/board_effects.py`:

```python
def render_tile_sparkle(self, canvas, tile, color):
    """Add sparkle effect to tile."""
    # Your effect code here
    pass
```

Call from `projection_renderer.py`:

```python
for tile in green_tiles:
    self.board_effects.render_tile_sparkle(canvas, tile, (255, 255, 0))
```

### New Game States

1. Add to `GameState` enum in `game_state.py`:
```python
class GameState(Enum):
    # ... existing states ...
    BONUS_ROUND = "bonus_round"
```

2. Add transition logic in `_transition_to_next_state()`:
```python
elif current == GameState.WIN:
    self._change_state(GameState.BONUS_ROUND)
```

3. Add rendering in `projection_renderer.py`:
```python
def _render_bonus_round(self, canvas, player_data):
    """Render bonus round state."""
    # Your rendering code
    pass
```

### Custom Path Generation

Edit `_generate_path()` in `tile_grid.py`:

```python
def _generate_path(self):
    """Custom path generation algorithm."""
    # Your path generation code
    pass
```

## Performance Optimization

### Profiling

Enable performance monitoring in `main.py`:

```python
from freezing_phantoms.core.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_frame()
# ... game loop ...
print(monitor.get_report())
```

### Typical Timings

- AI Inference: ~33ms (30 FPS)
- Movement Detection: <1ms
- Rendering: ~16ms
- Total Loop: ~50ms

### Optimization Tips

1. **Reduce Camera Resolution**: Lower resolution = faster processing
2. **Simplify Visual Effects**: Complex rendering slows down
3. **Reduce Monitored Keypoints**: Fewer keypoints = faster movement detection
4. **Cooling**: Ensure Hailo device has adequate cooling

## Documentation

- **README.md**: Main user documentation
- **CAMERA_SETUP.md**: Camera configuration guide
- **CALIBRATION_CHECKERBOARD.md**: Calibration details
- **CHANGELOG.md**: Version history
- **DEVELOPMENT.md**: This file

## Historical Context

This project evolved from a "Freezing Phantoms Halloween Game" with ghosts and ice effects to the current "Red Light, Green Light - Floor is Lava" tile-based game. Intermediate documentation has been archived in `docs/archive/` for reference.

## Contributing

1. Make changes on a feature branch
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass
5. Create pull request with clear description

## License

See LICENSE file in repository root.

