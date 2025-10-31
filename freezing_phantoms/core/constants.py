"""
Constants for Freezing Phantoms game.

Centralized configuration constants to avoid hardcoded values throughout the codebase.
"""

from enum import Enum
from typing import Final

# Version
__version__: Final[str] = "1.0.0"

# Default paths
DEFAULT_CONFIG_DIR: Final[str] = "config"
DEFAULT_ASSETS_DIR: Final[str] = "assets"
DEFAULT_AUDIO_DIR: Final[str] = "assets/audio"
DEFAULT_CALIBRATION_FILE: Final[str] = "config/calibration.npy"
DEFAULT_GAME_CONFIG_FILE: Final[str] = "config/game_config.yaml"

# Video defaults
DEFAULT_FRAME_WIDTH: Final[int] = 1280
DEFAULT_FRAME_HEIGHT: Final[int] = 720
DEFAULT_FPS: Final[int] = 30
DEFAULT_VIDEO_DEVICE: Final[str] = "/dev/video0"

# Projector defaults
DEFAULT_PROJECTOR_WIDTH: Final[int] = 1920
DEFAULT_PROJECTOR_HEIGHT: Final[int] = 1080
DEFAULT_DISPLAY_INDEX: Final[int] = 1

# Movement detection defaults
DEFAULT_MOVEMENT_THRESHOLD_PX: Final[float] = 20.0
DEFAULT_SMOOTHING_ALPHA: Final[float] = 0.3
DEFAULT_POSE_HISTORY_SIZE: Final[int] = 5
DEFAULT_SENSITIVITY_MULTIPLIER: Final[float] = 1.0
DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD: Final[float] = 0.5

# Game timing defaults (in seconds)
DEFAULT_COUNTDOWN_DURATION: Final[float] = 3.0
DEFAULT_GREEN_LIGHT_MIN: Final[float] = 5.0
DEFAULT_GREEN_LIGHT_MAX: Final[float] = 8.0
DEFAULT_RED_LIGHT_MIN: Final[float] = 3.0
DEFAULT_RED_LIGHT_MAX: Final[float] = 5.0
DEFAULT_PENALTY_DURATION: Final[float] = 2.0

# Queue configuration
PIPELINE_DATA_QUEUE_SIZE: Final[int] = 10
PIPELINE_TIMEOUT_SECONDS: Final[float] = 0.1

# Calibration defaults
DEFAULT_CALIBRATION_PATTERN_ROWS: Final[int] = 6
DEFAULT_CALIBRATION_PATTERN_COLS: Final[int] = 9
DEFAULT_CALIBRATION_SQUARE_SIZE: Final[int] = 80
MAX_CALIBRATION_RETRIES: Final[int] = 3

# Performance settings
TARGET_GAME_LOOP_FPS: Final[int] = 30
PIPELINE_WARMUP_SECONDS: Final[float] = 2.0

# Collision detection
DEFAULT_COLLISION_RADIUS_PX: Final[float] = 15.0

# Logging
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File size limits (for validation)
MAX_CONFIG_FILE_SIZE_MB: Final[int] = 10
MAX_CALIBRATION_FILE_SIZE_MB: Final[int] = 10

# Exit codes
EXIT_SUCCESS: Final[int] = 0
EXIT_CALIBRATION_FAILED: Final[int] = 1
EXIT_CONFIG_ERROR: Final[int] = 2
EXIT_PIPELINE_ERROR: Final[int] = 3
EXIT_RUNTIME_ERROR: Final[int] = 4


class KeyboardKeys(Enum):
    """Keyboard key codes for game controls."""
    ESC = 27
    R = ord('r')
    R_UPPER = ord('R')
    SPACE = ord(' ')
    C = ord('c')
    C_UPPER = ord('C')


class COCOKeypoints(Enum):
    """COCO pose estimation keypoint names and indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Monitored keypoints for movement detection
DEFAULT_MONITORED_KEYPOINTS: Final[list[str]] = [
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
    "left_shoulder",
    "right_shoulder",
]

# Audio file names
AUDIO_FILES: Final[dict[str, str]] = {
    "green_light_music": "green_light_music.mp3",
    "freeze_sound": "freeze_sound.wav",
    "caught_buzzer": "caught_buzzer.wav",
    "game_start": "game_start.wav",
}

# Color constants (BGR format for OpenCV)
class Colors:
    """Common colors used in rendering (BGR format)."""
    RED: Final[tuple[int, int, int]] = (0, 0, 255)
    GREEN: Final[tuple[int, int, int]] = (0, 255, 0)
    BLUE: Final[tuple[int, int, int]] = (255, 0, 0)
    YELLOW: Final[tuple[int, int, int]] = (0, 255, 255)
    CYAN: Final[tuple[int, int, int]] = (255, 255, 0)
    MAGENTA: Final[tuple[int, int, int]] = (255, 0, 255)
    WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
    BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
    GRAY: Final[tuple[int, int, int]] = (128, 128, 128)

