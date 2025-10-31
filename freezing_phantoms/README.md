# Freezing Phantoms - Red Light, Green Light Game

An AI-powered "Red Light, Green Light" game with floor-is-lava theme using Hailo AI for real-time player tracking and projection mapping.

## Overview

"Freezing Phantoms" is an interactive projection-mapped game based on the classic "Red Light, Green Light". Players navigate a 5x10 tile grid projected on the floor, avoiding red lava tiles while racing to the goal. The game uses Hailo AI for real-time pose estimation to track player position and detect movement during freeze phases.

## Game Features

- **5x10 Tile Grid**: Play area divided into 50 tiles with automatic path generation
- **Lava Floor Theme**: Red tiles represent lava (forbidden), green tiles are safe
- **Real-time AI Tracking**: Uses Hailo AI for pose estimation and player position tracking
- **Movement Detection**: Detects illegal movement during "Red Light" freeze phases
- **Projection Mapping**: Precise camera-projector calibration with saved camera settings
- **Automatic Gameplay**: Self-running game with timer-based state transitions
- **Win Condition**: Reach the goal row at the bottom of the screen

## Hardware Requirements

1. **Compute Device**: PC or SBC capable of running Hailo AI (e.g., Raspberry Pi 5 with Hailo-8L)
2. **AI Accelerator**: Hailo-8 or Hailo-8L
3. **Camera**: Overhead RGB camera (USB webcam, 720p minimum)
4. **Projector**: Bright projector, mounted overhead facing down
5. **v4l2-ctl** (optional): For camera settings management

## Software Requirements

- Python 3.7+
- Hailo Apps Infrastructure
- GStreamer with Hailo plugins
- OpenCV (cv2)
- PyYAML
- NumPy

## Quick Start

### 1. Installation
```bash
cd freezing_phantoms
./setup.sh
```

### 2. Run Calibration (First Time Only)
```bash
python main.py --run-calibration --input /dev/video0
```

**During calibration:**
- The checkerboard pattern displays on the projector
- Adjust camera settings using keyboard controls (1-6 keys)
- Press SPACE when pattern is clearly visible
- Camera settings are automatically saved

### 3. Start the Game
```bash
python main.py --input /dev/video0
```

The saved camera settings are automatically applied when starting the game.

## Game Rules

### Objective
Navigate from the starting row (top of screen) to the goal row (bottom of screen) by stepping only on GREEN tiles.

### Gameplay

1. **Countdown** (3 seconds): 
   - Players prepare at starting position
   - Game displays countdown text

2. **Green Light** (5-8 seconds random):
   - Players can move on GREEN tiles
   - RED tiles represent lava - stepping on them causes penalty
   - New green path tiles are revealed each cycle

3. **Red Light** (3-5 seconds):
   - Players must **FREEZE immediately**
   - Any movement detected results in penalty
   - Movement detector monitors pose keypoints

4. **Penalty** (2 seconds):
   - Triggered by movement during Red Light OR stepping on red lava tile
   - Player moves back 2 rows toward start
   - Displays violation message ("MOVED!" or "STEPPED ON LAVA!")

5. **Win**:
   - Player reaches the goal row
   - Victory message displays
   - Press R to restart

### Controls
- **R** - Restart game
- **ESC** - Quit

## Installation & Setup

### 1. Install Dependencies

```bash
cd freezing_phantoms
pip install -r requirements.txt
```

### 2. Camera and Projector Setup

1. Mount camera overhead, looking down at play area
2. Mount projector overhead, covering same area
3. Ensure camera can see entire projected area
4. Connect both devices to your system

### 3. Calibration

Run the calibration tool:

```bash
python main.py --run-calibration --input /dev/video0
```

**Calibration Process:**

1. **Checkerboard Pattern**: Displayed on projector
2. **Camera Settings Adjustment** (if pattern not visible):
   - **1/2** - Decrease/Increase brightness
   - **3/4** - Decrease/Increase exposure
   - **5/6** - Decrease/Increase focus
   - **A** - Toggle auto-exposure
   - **R** - Reset to defaults
3. **Capture**: Press SPACE when pattern is clearly visible
4. **Validation**: Interactive validation to test accuracy
5. **Save**: Calibration matrix and camera settings saved automatically

**Files Created:**
- `config/calibration.npy` - Homography transformation matrix
- `config/calibration_camera_settings.json` - Your camera settings

### 4. Validate Calibration (Optional)

```bash
python main.py --validate-calibration --input /dev/video0
```

Move in the camera view - a green circle should follow your position on the projector.

## Configuration

Edit `config/game_config.yaml` to customize:

```yaml
# Game timing
game:
  countdown_duration: 3.0
  green_light_duration: [5.0, 8.0]
  red_light_duration: [3.0, 5.0]
  penalty_duration: 2.0

# Tile grid configuration
tile_grid:
  columns: 5
  rows: 10
  starting_row: 0          # Top of screen (far from camera)
  goal_row: 9              # Bottom of screen (near camera)
  penalty_rows_back: 2     # How many rows to move back on penalty
  path_width: 1            # Safe tiles per row
  obstacle_density: 0.3    # Percentage of non-path tiles that are red

# Movement detection
movement:
  threshold_pixels: 20.0
  monitored_keypoints:
    - left_wrist
    - right_wrist
    - left_ankle
    - right_ankle

# Camera and projector
camera:
  resolution: [1280, 720]
  fps: 30

projector:
  resolution: [1920, 1080]
  display_index: 1         # 0=primary, 1=secondary
```

## Command Line Options

### Game Options
- `--input`, `-i`: Input source (camera device or file)
- `--config`: Path to configuration file (default: `config/game_config.yaml`)
- `--calibration`: Path to calibration matrix (default: `config/calibration.npy`)
- `--run-calibration`: Run calibration before starting game
- `--validate-calibration`: Validate existing calibration
- `--show-video`: Show video output with AI overlays (debug mode)

### Hailo Options
- `--arch`: Hailo architecture (`hailo8`, `hailo8l`, `hailo10h`)
- `--hef-path`: Custom HEF model file path
- `--frame-rate`, `-r`: Frame rate (default: 30)
- `--show-fps`, `-f`: Show FPS measurements

### Examples

```bash
# Basic game
python main.py --input /dev/video0

# With debug visualization
python main.py --input /dev/video0 --show-video

# Specify architecture
python main.py --input /dev/video0 --arch hailo8l

# Run calibration first
python main.py --run-calibration --input /dev/video0
```

## Architecture

### AI Pipeline

```
Camera → GStreamer → Hailo Pose Estimation → Data Extraction
                                                    ↓
                                            Player Tracking
                                                    ↓
                                            Game State Machine
                                                    ↓
                                            Projection Renderer → Projector
```

### Key Components

- **`main.py`**: Application entry point and game loop
- **`core/tile_grid.py`**: 5x10 grid management, path generation, player position
- **`core/game_state.py`**: State machine (COUNTDOWN, GREEN_LIGHT, RED_LIGHT, PENALTY, WIN)
- **`core/movement_detector.py`**: Detects player movement during RED_LIGHT
- **`core/projection_renderer.py`**: Renders game graphics with homography transform
- **`core/camera_utils.py`**: Camera settings management (save/load/apply)
- **`pipeline/combined_pipeline.py`**: Hailo AI pose estimation pipeline
- **`pipeline/data_extractor.py`**: Extracts player data from Hailo buffer
- **`calibration/calibration_tool.py`**: Camera-projector calibration with interactive controls
- **`effects/board_effects.py`**: Visual effects (lava texture, player indicator, glows)

### Coordinate System

1. **Camera Space**: Raw pixel coordinates from camera (1280x720)
2. **Normalized Space**: Pose keypoints relative to detection box (0.0-1.0)
3. **Camera Absolute**: Keypoints converted to camera pixel coordinates
4. **Projector Space**: Transformed via homography matrix (1920x1080)
5. **Tile Space**: Mapped to grid row/column (0-9 rows, 0-4 cols)

### Player Position Tracking

Player position is determined by leg location:
1. Extract ankle keypoints from pose estimation
2. Average left and right ankle positions
3. Take midpoint between ankles and detection box bottom (for ground contact)
4. Transform to projector coordinates via homography
5. Map to tile grid row/column
6. Check tile state for violations

## Troubleshooting

### Calibration Issues

**Pattern not visible:**
- Adjust camera settings using keyboard controls (1-6 keys)
- Improve lighting conditions
- Check projector brightness and focus

**Detection fails:**
- Ensure checkerboard pattern is fully in camera view
- Reduce glare and reflections
- Try different camera exposure settings

**Poor accuracy:**
- Re-run calibration with better conditions
- Ensure camera and projector are stable and not moving
- Use validation mode to verify accuracy

### Camera Issues

**Auto-focus/auto-brightness problems:**
- Camera settings are automatically saved during calibration
- Manually apply settings: `./setup_camera.sh /dev/video0`
- See `CAMERA_SETUP.md` for details

**Settings don't persist:**
- Ensure `v4l2-ctl` is installed: `sudo apt-get install v4l-utils`
- Camera settings reset when unplugged - re-run calibration

### Game Issues

**Player position inaccurate:**
- Check calibration accuracy with validation mode
- Ensure good lighting for pose detection
- Verify camera can see player's full body

**Movement detection too sensitive/insensitive:**
- Adjust `threshold_pixels` in config (increase = less sensitive)
- Modify `monitored_keypoints` list

**Low frame rate:**
- Reduce camera resolution in config
- Ensure adequate cooling for Hailo device
- Close other applications

## Project Structure

```
freezing_phantoms/
├── main.py                          # Main application entry point
├── config/
│   ├── game_config.yaml             # Game configuration
│   ├── calibration.npy              # Calibration matrix (generated)
│   └── calibration_camera_settings.json  # Camera settings (generated)
├── core/                            # Core game logic
│   ├── tile_grid.py                 # Tile grid system and path generation
│   ├── game_state.py                # Game state machine
│   ├── movement_detector.py         # Movement detection
│   ├── projection_renderer.py       # Visual rendering
│   ├── camera_utils.py              # Camera settings management
│   └── ...                          # Other utilities
├── pipeline/                        # AI inference
│   ├── combined_pipeline.py         # Hailo pose estimation pipeline
│   └── data_extractor.py            # Player data extraction
├── calibration/                     # Calibration system
│   ├── calibration_tool.py          # Interactive calibration
│   └── pattern_generator.py         # Pattern generation
├── effects/                         # Visual effects
│   ├── board_effects.py             # Lava, tiles, player indicator
│   ├── green_light_effects.py       # Green light effects
│   └── red_light_effects.py         # Red light effects
├── tests/                           # Unit tests
├── setup_camera.sh                  # Camera setup script
├── setup.sh                         # Installation script
└── README.md                        # This file
```

## Technical Details

### Leg Position Calculation

The system calculates player "foot position" for tile detection:

1. Extract left and right ankle keypoints from pose estimation
2. Calculate average ankle position: `(left_ankle + right_ankle) / 2`
3. Get detection box bottom: `bbox.ymin + bbox.height`
4. Take midpoint: `leg_y = (ankle_y + bbox_bottom) / 2`
5. This approximates where the feet touch the ground

### Tile Path Generation

The tile grid uses automatic path generation:

1. Start at `starting_row` (default: row 0, top of screen)
2. Generate winding path to `goal_row` (default: row 9, bottom)
3. Path width is configurable (default: 1 tile wide)
4. Non-path tiles have chance to be RED based on `obstacle_density`
5. Path tiles are revealed progressively during gameplay

### Camera Settings Management

Camera settings are saved during calibration and automatically applied:

1. **During Calibration**: Manual adjustment with keyboard (keys 1-6)
2. **Saved**: Settings written to JSON file alongside calibration matrix
3. **Game Startup**: Settings automatically loaded and applied via `v4l2-ctl`
4. **Benefits**: Consistent camera behavior, no manual setup needed

## Development

### Running Tests

```bash
pytest tests/
```

### Debug Mode

```bash
# Show video with AI overlays
python main.py --input /dev/video0 --show-video

# Enable debug logging
LOG_LEVEL=DEBUG python main.py --input /dev/video0
```

### Adding Features

See `CONTRIBUTING.md` for development guidelines.

## Documentation

- **README.md** - This file, main documentation
- **CAMERA_SETUP.md** - Camera configuration guide
- **CALIBRATION_CHECKERBOARD.md** - Calibration details
- **CONTRIBUTING.md** - Development guidelines (if exists)
- **docs/** - Additional documentation

## Performance

- **AI Inference**: ~33ms per frame (30 FPS)
- **Movement Detection**: <1ms
- **Rendering**: ~16ms (60 FPS capable)
- **Total Loop Time**: ~50ms
- **Memory Usage**: ~500MB with Hailo pipeline

## Credits

Built using:
- [Hailo AI](https://hailo.ai/) - Edge AI acceleration
- [GStreamer](https://gstreamer.freedesktop.org/) - Video pipeline
- [OpenCV](https://opencv.org/) - Computer vision
- COCO pose estimation model

## License

See LICENSE file in repository root.
