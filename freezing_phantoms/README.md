# Freezing Phantoms

An AI-augmented Halloween game using Hailo AI for real-time player tracking and projection mapping.

## Overview

"Freezing Phantoms" is an interactive projection-mapped game based on the classic "Red Light, Green Light" (known in Hebrew as "Dag Maluach"). The game uses a Hailo-powered AI pipeline to track children in a play area, projecting dynamic graphics and game elements onto the floor and the players themselves.

## Features

- **Real-time AI Tracking**: Uses Hailo AI for person detection, pose estimation, and instance segmentation
- **Projection Mapping**: Precise camera-projector calibration for interactive visual effects
- **Automated Gameplay**: Self-running game with timer-based state transitions
- **Movement Detection**: Detects player movement during "freeze" phases using pose keypoint comparison
- **Dynamic Visual Effects**:
  - Green Light: Safe zones with swirling phantoms
  - Red Light: Scanning lasers and ice encasement effects
  - Penalty: Spotlight on caught players
- **Audio Cues**: Spooky sound effects and background music
- **Debug Mode**: Visual debugging with AI overlays (`--show-video`)

## Hardware Requirements

1. **Compute Device**: PC or SBC capable of running Hailo AI (e.g., Raspberry Pi 5 with Hailo-8L)
2. **AI Accelerator**: Hailo-8 or Hailo-8L
3. **Camera**: Overhead RGB camera (e.g., USB webcam, minimum 720p recommended)
4. **Projector**: Bright projector with low input lag, mounted overhead
5. **Speakers**: For audio playback

## Software Requirements

- Python 3.7+
- Hailo Apps Infrastructure (this repository)
- GStreamer with Hailo plugins
- OpenCV
- Pygame
- PyYAML
- NumPy

## Quick Start

### 1. Setup
```bash
cd freezing_phantoms
./setup.sh
```

### 2. Run Calibration
```bash
python calibrate.py --video-source /dev/video0
```

### 3. Start the Game
```bash
python main.py --video-source /dev/video0
```

## Installation

1. **Install the main Hailo Apps infrastructure** (if not already done):
   ```bash
   cd /path/to/hailo-apps-infra
   ./install.sh
   ```

2. **Install Freezing Phantoms dependencies**:
   ```bash
   cd freezing_phantoms
   pip install -r requirements.txt
   ```

3. **Prepare audio assets** (optional but recommended):
   - Place audio files in `assets/audio/`:
     - `green_light_music.mp3`: Background music for move phase
     - `freeze_sound.wav`: "Freeze!" command sound
     - `caught_buzzer.wav`: Penalty sound effect
     - `game_start.wav`: Countdown sound

## Setup and Calibration

### 1. Camera and Projector Setup

1. Mount the camera overhead, looking down at the play area
2. Mount the projector overhead, covering the same play area
3. Ensure the camera can see the entire projected area
4. Connect both devices to your compute system

### 2. Run Calibration

**Recommended: Use the standalone calibration tool**

```bash
python calibrate.py --video-source /dev/video0
```

The standalone tool provides:
- Better point detection validation
- Automatic retry on failure
- Detailed quality metrics
- Independent from game

Alternatively, calibrate through the game:
```bash
python main.py --run-calibration --video-source /dev/video0
```

**Calibration Steps:**
1. A calibration pattern (grid of circles) will be displayed on the projector
2. Ensure the pattern is fully visible in the camera view
3. Press **SPACE** to capture the camera image
4. The system will detect the circles and calculate the homography matrix
5. A visualization shows detected points for verification
6. The calibration matrix is saved to `config/calibration.npy`

**Tips:**
- Ensure good lighting for circle detection
- The pattern should fill most of the play area
- If detection fails, adjust camera position or lighting and try again

### 3. Validate Calibration (Optional but Recommended)

Test the calibration accuracy:

```bash
python calibrate.py --validate --video-source /dev/video0
```

Move your hand in the camera view - a green circle should appear on the projector at the same location. Press ESC to exit validation.

The validation shows:
- Real-time tracking accuracy
- Alignment across play area
- Any offset or distortion issues

## Running the Game

Once calibration is complete, start the game:

```bash
python main.py --video-source /dev/video0
```

### Command Line Options

**Game-Specific Options:**
- `--config`: Path to configuration file (default: `config/game_config.yaml`)
- `--calibration`: Path to calibration matrix (default: `config/calibration.npy`)
- `--run-calibration`: Run calibration tool before starting game
- `--validate-calibration`: Validate existing calibration
- `--show-video`: Show video output with AI overlays (debug mode)
- `--video-source`: Video source path (alias for `--input`)

**Standard Hailo Options** (inherited from `hailo-apps-infra`):
- `--input`, `-i`: Input source (camera device or file path)
- `--arch`: Hailo architecture (`hailo8`, `hailo8l`, `hailo10h`)
- `--hef-path`: Custom path to HEF model file
- `--frame-rate`, `-r`: Frame rate of video source (default: 30)
- `--disable-sync`: Disable display sink synchronization
- `--show-fps`, `-f`: Show FPS measurements
- `--dump-dot`: Dump pipeline graph to DOT file
- `--disable-callback`: Disable user callback function
- `--use-frame`, `-u`: Use frame from callback

**Examples:**
```bash
# Basic usage
python main.py --input /dev/video0

# Specify architecture
python main.py --input /dev/video0 --arch hailo8l

# Custom frame rate
python main.py --input /dev/video2 --frame-rate 60

# Debug mode with pipeline graph
python main.py --input /dev/video0 --show-video --dump-dot
```

### Debug Mode

To see what the AI pipeline is detecting, use the `--show-video` flag:
```bash
python main.py --video-source /dev/video0 --show-video
```

This will open a window showing:
- Camera feed
- Detection boxes around people
- Pose keypoints (skeleton overlay)
- Segmentation masks

**Note:** FPS counter is not available in this debug mode, but all AI detections will be visible.

Debug mode is useful for:
- Verifying the AI pipeline is working
- Checking camera positioning
- Troubleshooting detection issues
- Adjusting lighting conditions

**Controls During Game:**
- **ESC**: Quit game
- **R**: Reset game to countdown state

## Game Rules

### Objective
Be the first player to cross the finish line!

### Gameplay

1. **Countdown** (3 seconds): Players prepare

2. **Green Light** (5-8 seconds random):
   - Players can move freely
   - Safe zones project around each player
   - Swirling phantom effects create ambiance

3. **Red Light** (3-5 seconds):
   - Players must **freeze immediately**
   - AI captures initial pose at freeze moment
   - Movement detection monitors for any motion
   - Ice effects project on frozen players
   - Scanning laser sweeps the area

4. **Penalty** (2 seconds):
   - If movement detected during Red Light
   - Red spotlight highlights caught player
   - Caught player must return to start
   - Game returns to Green Light

5. **Game Over**:
   - A player reaches the finish line
   - Press R to restart

## Configuration

Edit `config/game_config.yaml` to customize game parameters:

```yaml
game:
  countdown_duration: 3.0              # Countdown time
  green_light_duration: [5.0, 8.0]     # Move phase duration range
  red_light_duration: [3.0, 5.0]       # Freeze phase duration range
  penalty_duration: 2.0                 # Penalty display time

movement:
  threshold_pixels: 20.0                # Movement detection threshold
  monitored_keypoints:                  # Body parts to monitor
    - left_wrist
    - right_wrist
    - left_ankle
    - right_ankle

camera:
  resolution: [1280, 720]               # Camera resolution
  fps: 30                                # Frame rate

projector:
  resolution: [1920, 1080]              # Projector resolution
  display_index: 1                       # Display number (0=primary, 1=secondary)
```

## Architecture

### AI Pipeline

The game uses a combined Hailo pipeline that runs two models:

1. **Pose Estimation**: Detects people and extracts 17 COCO keypoints per person
2. **Instance Segmentation**: Provides pixel-accurate masks for each person

Both models process the same video stream with shared tracking IDs.

### Data Flow

```
Camera → Hailo Pipeline → Game State Machine → Projection Renderer → Projector
                              ↓
                        Movement Detector
                              ↓
                        Audio Manager → Speakers
```

### Key Components

- **`combined_pipeline.py`**: Manages dual-model Hailo inference
- **`data_extractor.py`**: Parses Hailo buffer into structured player data
- **`game_state.py`**: Automated state machine with timer-based transitions
- **`movement_detector.py`**: Compares pose keypoints to detect motion
- **`projection_renderer.py`**: Renders effects using homography transformation
- **`audio_manager.py`**: Handles sound playback via pygame
- **`calibration_tool.py`**: Camera-projector calibration utility

## Troubleshooting

### Calibration Issues

**Problem**: Circle detection fails
- **Solution**: Improve lighting, ensure pattern is fully visible, try adjusting camera exposure

**Problem**: Calibration seems inaccurate
- **Solution**: Re-run calibration with better alignment, ensure camera and projector are stable

**Problem**: Keyboard input not working during calibration
- **Solution**: Click on the camera feed window before pressing keys, try 'c' instead of SPACE, check terminal for "Key detected" messages

### Pipeline Issues

**Problem**: "Could not detect Hailo architecture"
- **Solution**: Ensure Hailo device is connected and drivers are installed

**Problem**: Low frame rate
- **Solution**: Reduce camera resolution in config, ensure adequate cooling for Hailo device

### Projection Issues

**Problem**: Effects not aligning with players
- **Solution**: Re-run calibration, ensure camera and projector haven't moved

**Problem**: Projector on wrong display
- **Solution**: Adjust `display_index` in config, or manually move the window

### Debug Logging

Enable debug logging to see detailed information from all components:

```bash
# Method 1: Command line
python main.py --input /dev/video0 --log-level DEBUG

# Method 2: Environment variable
LOG_LEVEL=DEBUG python main.py --input /dev/video0
```

Debug logging shows:
- Movement detector operations
- Game state transitions
- Pipeline initialization
- Data extraction details
- Projection rendering

## Development and Testing

### Running Unit Tests

```bash
cd tests
pytest test_calibration.py
pytest test_movement_detection.py
pytest test_game_state.py
```

### Adding Custom Effects

1. Edit `effects/green_light_effects.py` or `effects/red_light_effects.py`
2. Add new rendering methods
3. Call from projection renderer

### Adjusting Movement Sensitivity

Edit `config/game_config.yaml`:
- Increase `threshold_pixels` to make detection less sensitive
- Add/remove keypoints from `monitored_keypoints`

## Project Structure

```
freezing_phantoms/
├── main.py                    # Main application entry point
├── calibrate.py              # Standalone calibration tool
├── setup.sh                  # Setup script
├── requirements.txt          # Python dependencies
├── config/
│   ├── game_config.yaml      # Game configuration
│   └── calibration.npy       # Calibration matrix (generated)
├── core/                     # Core game logic
│   ├── audio_manager.py      # Audio playback
│   ├── game_state.py         # Game state machine
│   ├── movement_detector.py  # Movement detection
│   └── projection_renderer.py # Visual effects rendering
├── pipeline/                 # AI inference
│   ├── combined_pipeline.py  # Dual-model pipeline
│   └── data_extractor.py     # Data parsing
├── calibration/              # Calibration system
│   ├── calibration_tool.py   # Calibration logic
│   └── pattern_generator.py  # Pattern generation
├── effects/                  # Visual effects
│   ├── green_light_effects.py
│   └── red_light_effects.py
├── assets/                   # Game assets
│   └── audio/               # Audio files
└── tests/                   # Unit tests
    ├── test_calibration.py
    ├── test_movement_detection.py
    └── test_game_state.py
```

## Performance Characteristics

### Latency
- AI inference: ~33ms (30 FPS)
- Movement detection: <1ms
- Rendering: ~16ms (60 FPS capable)
- Total loop time: ~50ms

### Resource Usage
- Memory: ~500MB (with Hailo pipeline)
- CPU: Moderate (mostly GStreamer)
- GPU/Hailo: High during inference

## Future Enhancements

See [docs/TODO_LEADERBOARD.md](docs/TODO_LEADERBOARD.md) for planned leaderboard system with:
- Web-based leaderboard display
- Player image capture and storage
- Score tracking and statistics
- Local web server for viewing results

## Credits

Built using:
- [Hailo AI](https://hailo.ai/) - Edge AI acceleration
- [GStreamer](https://gstreamer.freedesktop.org/) - Video pipeline
- [OpenCV](https://opencv.org/) - Computer vision
- [Pygame](https://www.pygame.org/) - Audio playback

## License

See LICENSE file in repository root.

## Contact

For issues or questions, please open an issue on the repository.