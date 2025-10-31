# Changelog

All notable changes to the Freezing Phantoms project are documented here.

## [Unreleased] - Current Development

### Added
- **Red Light, Green Light - Floor is Lava Game**: Complete transformation to tile-based game
  - 5x10 tile grid system with automatic path generation
  - Lava floor theme (RED tiles = lava, GREEN tiles = safe)
  - Player tracking via leg position (ankle + bbox bottom averaging)
  - Win condition when reaching goal row
  - Penalty system (move back 2 rows on violation)
  - Tile violation detection (stepping on red lava tiles)

- **Camera Settings Management**:
  - Automatic camera setup during calibration using `v4l2-ctl`
  - Interactive manual controls during calibration (keys 1-6)
  - Camera settings saved alongside calibration matrix (JSON format)
  - Automatic settings application when starting game
  - Manual setup script (`setup_camera.sh`)

- **Enhanced Calibration Tool**:
  - Interactive camera controls during calibration
  - Real-time setting adjustments (brightness, exposure, focus)
  - Auto-exposure toggle
  - Reset to defaults option
  - Settings displayed on screen during calibration
  - Better keyboard mapping (number keys 1-6 for increment/decrement)

- **Improved Player Position Tracking**:
  - More accurate leg position using average of ankles and bbox bottom
  - Smaller player marker (12px radius) for better accuracy
  - Better ground contact approximation

- **Core Components**:
  - `core/tile_grid.py`: Complete tile grid system
  - `core/camera_utils.py`: Camera settings management utilities
  - `effects/board_effects.py`: Lava texture, tile effects, player indicators

### Changed
- **Game Theme**: From "Halloween Freezing Phantoms" to "Red Light, Green Light - Floor is Lava"
- **Gameplay**: From finish-line racing to tile-based navigation
- **Player Position**: Now uses leg position (feet approximation) instead of center of mass
- **Camera Settings**: Now managed automatically instead of manual setup

### Documentation
- Complete README rewrite reflecting current game mechanics
- Enhanced CAMERA_SETUP.md with auto-save workflow
- Consolidated CALIBRATION_CHECKERBOARD.md
- Archived outdated intermediate state documentation

## [Previous] - Earlier Development

### Pipeline Simplification
- Removed instance segmentation, keeping only pose estimation
- Fixed resolution to 720p (1280x720) for consistency with calibration
- Simplified pipeline architecture
- Leg position calculation using ankle keypoints
- Renamed pipeline class to `PoseEstimationPipeline`

### Calibration Improvements
- Checkerboard pattern instead of circle grid
- Better point detection with validation
- Interactive validation mode
- Retry logic with history tracking
- Detailed quality metrics

### Core Infrastructure
- Hailo AI integration for pose estimation
- GStreamer pipeline for video processing
- Homography-based projection mapping
- Movement detection using pose keypoints
- Real-time rendering with OpenCV

### Game State Machine
- COUNTDOWN state for game start
- GREEN_LIGHT state for player movement
- RED_LIGHT state with freeze detection
- PENALTY state for violations
- WIN state for game completion
- Automatic state transitions with timers

### Visual Effects
- Projection rendering with homography transform
- Green light effects (safe zones, movement allowed)
- Red light effects (freeze, scanning lasers)
- Penalty effects (spotlight on violator)
- Win effects (victory message)

## Version History Notes

This project has evolved significantly from its initial "Freezing Phantoms Halloween Game" concept to the current "Red Light, Green Light - Floor is Lava" tile-based game. Many intermediate improvements and refactorings are archived in `docs/archive/` for historical reference.

## Contributing

See CONTRIBUTING.md for development guidelines and how to contribute to the project.
