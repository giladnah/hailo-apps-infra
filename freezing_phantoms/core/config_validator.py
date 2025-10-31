"""
Configuration validation utilities for Freezing Phantoms.

Provides comprehensive validation for game configuration files.
"""

from pathlib import Path
from typing import Optional

import yaml
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

from freezing_phantoms.core.constants import (
    MAX_CONFIG_FILE_SIZE_MB,
    DEFAULT_COUNTDOWN_DURATION,
    DEFAULT_GREEN_LIGHT_MIN,
    DEFAULT_GREEN_LIGHT_MAX,
    DEFAULT_RED_LIGHT_MIN,
    DEFAULT_RED_LIGHT_MAX,
    DEFAULT_PENALTY_DURATION,
    DEFAULT_MOVEMENT_THRESHOLD_PX,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FPS,
    DEFAULT_PROJECTOR_WIDTH,
    DEFAULT_PROJECTOR_HEIGHT,
    DEFAULT_DISPLAY_INDEX,
)

hailo_logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""


class ConfigValidator:
    """
    Validates game configuration files and ensures all required fields are present.
    """

    # Define required schema
    REQUIRED_SCHEMA = {
        "game": {
            "countdown_duration": (float, int),
            "green_light_duration": list,
            "red_light_duration": list,
            "penalty_duration": (float, int),
        },
        "movement": {
            "threshold_pixels": (float, int),
            "monitored_keypoints": list,
            "smoothing_alpha": (float, int),
            "pose_history_size": int,
            "sensitivity_multiplier": (float, int),
            "keypoint_confidence_threshold": (float, int),
        },
        "camera": {
            "resolution": list,
            "fps": int,
        },
        "projector": {
            "resolution": list,
            "display_index": int,
        },
    }

    def __init__(self, config_path: Path):
        """
        Initialize configuration validator.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.config: Optional[dict] = None

    def validate(self) -> dict:
        """
        Validate configuration file.

        Returns:
            dict: Validated configuration.

        Raises:
            ConfigValidationError: If validation fails.
        """
        # Check file exists
        if not self.config_path.exists():
            raise ConfigValidationError(
                f"Configuration file not found: {self.config_path}"
            )

        # Check file size
        file_size_mb = self.config_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_CONFIG_FILE_SIZE_MB:
            raise ConfigValidationError(
                f"Configuration file too large: {file_size_mb:.1f}MB "
                f"(max: {MAX_CONFIG_FILE_SIZE_MB}MB)"
            )

        # Load YAML
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML syntax: {e}") from e
        except Exception as e:
            raise ConfigValidationError(f"Failed to read config file: {e}") from e

        if not isinstance(self.config, dict):
            raise ConfigValidationError("Configuration must be a dictionary")

        # Validate schema
        self._validate_schema()

        # Validate specific fields
        self._validate_game_section()
        self._validate_movement_section()
        self._validate_camera_section()
        self._validate_projector_section()

        hailo_logger.info("Configuration validation passed")
        return self.config

    def _validate_schema(self):
        """Validate that all required sections and fields are present."""
        for section, fields in self.REQUIRED_SCHEMA.items():
            if section not in self.config:
                raise ConfigValidationError(f"Missing required section: '{section}'")

            section_data = self.config[section]
            if not isinstance(section_data, dict):
                raise ConfigValidationError(
                    f"Section '{section}' must be a dictionary"
                )

            for field, expected_type in fields.items():
                if field not in section_data:
                    raise ConfigValidationError(
                        f"Missing required field: '{section}.{field}'"
                    )

                value = section_data[field]
                if not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Field '{section}.{field}' has incorrect type. "
                        f"Expected {expected_type}, got {type(value)}"
                    )

    def _validate_game_section(self):
        """Validate game configuration section."""
        game = self.config["game"]

        # Validate countdown_duration
        if game["countdown_duration"] <= 0:
            raise ConfigValidationError(
                "game.countdown_duration must be positive"
            )

        # Validate green_light_duration
        green_light = game["green_light_duration"]
        if len(green_light) != 2:
            raise ConfigValidationError(
                "game.green_light_duration must be a list of 2 values [min, max]"
            )
        if green_light[0] <= 0 or green_light[1] <= 0:
            raise ConfigValidationError(
                "game.green_light_duration values must be positive"
            )
        if green_light[0] > green_light[1]:
            raise ConfigValidationError(
                "game.green_light_duration: min must be <= max"
            )

        # Validate red_light_duration
        red_light = game["red_light_duration"]
        if len(red_light) != 2:
            raise ConfigValidationError(
                "game.red_light_duration must be a list of 2 values [min, max]"
            )
        if red_light[0] <= 0 or red_light[1] <= 0:
            raise ConfigValidationError(
                "game.red_light_duration values must be positive"
            )
        if red_light[0] > red_light[1]:
            raise ConfigValidationError(
                "game.red_light_duration: min must be <= max"
            )

        # Validate penalty_duration
        if game["penalty_duration"] <= 0:
            raise ConfigValidationError(
                "game.penalty_duration must be positive"
            )

    def _validate_movement_section(self):
        """Validate movement configuration section."""
        movement = self.config["movement"]

        # Validate threshold_pixels
        if movement["threshold_pixels"] <= 0:
            raise ConfigValidationError(
                "movement.threshold_pixels must be positive"
            )

        # Validate monitored_keypoints
        keypoints = movement["monitored_keypoints"]
        if not keypoints:
            raise ConfigValidationError(
                "movement.monitored_keypoints must not be empty"
            )
        if not all(isinstance(k, str) for k in keypoints):
            raise ConfigValidationError(
                "movement.monitored_keypoints must be a list of strings"
            )

        # Validate smoothing_alpha
        alpha = movement["smoothing_alpha"]
        if not 0 <= alpha <= 1:
            raise ConfigValidationError(
                "movement.smoothing_alpha must be between 0 and 1"
            )

        # Validate pose_history_size
        if movement["pose_history_size"] < 1:
            raise ConfigValidationError(
                "movement.pose_history_size must be at least 1"
            )

        # Validate sensitivity_multiplier
        if movement["sensitivity_multiplier"] <= 0:
            raise ConfigValidationError(
                "movement.sensitivity_multiplier must be positive"
            )

        # Validate keypoint_confidence_threshold
        threshold = movement["keypoint_confidence_threshold"]
        if not 0 <= threshold <= 1:
            raise ConfigValidationError(
                "movement.keypoint_confidence_threshold must be between 0 and 1"
            )

    def _validate_camera_section(self):
        """Validate camera configuration section."""
        camera = self.config["camera"]

        # Validate resolution
        resolution = camera["resolution"]
        if len(resolution) != 2:
            raise ConfigValidationError(
                "camera.resolution must be a list of 2 values [width, height]"
            )
        if not all(isinstance(v, int) and v > 0 for v in resolution):
            raise ConfigValidationError(
                "camera.resolution values must be positive integers"
            )

        # Validate fps
        if camera["fps"] <= 0:
            raise ConfigValidationError(
                "camera.fps must be positive"
            )

    def _validate_projector_section(self):
        """Validate projector configuration section."""
        projector = self.config["projector"]

        # Validate resolution
        resolution = projector["resolution"]
        if len(resolution) != 2:
            raise ConfigValidationError(
                "projector.resolution must be a list of 2 values [width, height]"
            )
        if not all(isinstance(v, int) and v > 0 for v in resolution):
            raise ConfigValidationError(
                "projector.resolution values must be positive integers"
            )

        # Validate display_index
        if projector["display_index"] < 0:
            raise ConfigValidationError(
                "projector.display_index must be non-negative"
            )


def load_and_validate_config(config_path: Path) -> dict:
    """
    Load and validate configuration file.

    Args:
        config_path: Path to configuration file.

    Returns:
        dict: Validated configuration.

    Raises:
        ConfigValidationError: If validation fails.
    """
    validator = ConfigValidator(config_path)
    return validator.validate()


def create_default_config(output_path: Path) -> dict:
    """
    Create a default configuration file.

    Args:
        output_path: Path where to save the configuration.

    Returns:
        dict: Default configuration dictionary.
    """
    default_config = {
        "game": {
            "countdown_duration": DEFAULT_COUNTDOWN_DURATION,
            "green_light_duration": [DEFAULT_GREEN_LIGHT_MIN, DEFAULT_GREEN_LIGHT_MAX],
            "red_light_duration": [DEFAULT_RED_LIGHT_MIN, DEFAULT_RED_LIGHT_MAX],
            "penalty_duration": DEFAULT_PENALTY_DURATION,
        },
        "movement": {
            "threshold_pixels": DEFAULT_MOVEMENT_THRESHOLD_PX,
            "monitored_keypoints": [
                "left_wrist",
                "right_wrist",
                "left_ankle",
                "right_ankle",
                "left_shoulder",
                "right_shoulder",
            ],
            "smoothing_alpha": 0.3,
            "pose_history_size": 5,
            "sensitivity_multiplier": 1.0,
            "keypoint_confidence_threshold": 0.5,
        },
        "camera": {
            "resolution": [DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT],
            "fps": DEFAULT_FPS,
        },
        "projector": {
            "resolution": [DEFAULT_PROJECTOR_WIDTH, DEFAULT_PROJECTOR_HEIGHT],
            "display_index": DEFAULT_DISPLAY_INDEX,
        },
    }

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write configuration
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    hailo_logger.info("Created default configuration at %s", output_path)
    return default_config

