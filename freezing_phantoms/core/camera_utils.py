"""
Utility functions for camera configuration.

Provides functions to save, load, and apply camera settings using v4l2-ctl.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


def apply_camera_settings(
    camera_device: str = "/dev/video0",
    brightness: Optional[int] = None,
    exposure: Optional[int] = None,
    focus: Optional[int] = None,
    auto_exposure: Optional[int] = None,
    auto_focus: Optional[int] = None,
    auto_white_balance: Optional[int] = None,
    power_line_frequency: Optional[int] = None,
) -> bool:
    """
    Apply camera settings using v4l2-ctl.

    Args:
        camera_device (str): Camera device path (e.g., "/dev/video0").
        brightness (int, optional): Brightness value.
        exposure (int, optional): Exposure value.
        focus (int, optional): Focus value.
        auto_exposure (int, optional): Auto-exposure mode (1=manual, 3=auto).
        auto_focus (int, optional): Auto-focus mode (0=off, 1=on).
        auto_white_balance (int, optional): Auto white balance (0=off, 1=on).
        power_line_frequency (int, optional): Power line frequency (0=disabled, 1=50Hz, 2=60Hz).

    Returns:
        bool: True if settings were applied successfully, False otherwise.
    """
    try:
        # Check if v4l2-ctl is available
        result = subprocess.run(
            ["which", "v4l2-ctl"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            hailo_logger.warning("v4l2-ctl not found. Camera settings cannot be applied.")
            return False

        # Build command to apply settings
        cmd = ["v4l2-ctl", "-d", camera_device]
        
        settings_applied = []
        
        if auto_focus is not None:
            cmd.extend(["--set-ctrl", f"focus_automatic_continuous={auto_focus}"])
            settings_applied.append(f"auto_focus={auto_focus}")
        
        if focus is not None:
            cmd.extend(["--set-ctrl", f"focus_absolute={focus}"])
            settings_applied.append(f"focus={focus}")
        
        if auto_exposure is not None:
            cmd.extend(["--set-ctrl", f"auto_exposure={auto_exposure}"])
            settings_applied.append(f"auto_exposure={auto_exposure}")
        
        if exposure is not None:
            cmd.extend(["--set-ctrl", f"exposure_absolute={exposure}"])
            settings_applied.append(f"exposure={exposure}")
        
        if brightness is not None:
            cmd.extend(["--set-ctrl", f"brightness={brightness}"])
            settings_applied.append(f"brightness={brightness}")
        
        if auto_white_balance is not None:
            cmd.extend(["--set-ctrl", f"white_balance_automatic={auto_white_balance}"])
            settings_applied.append(f"auto_white_balance={auto_white_balance}")
        
        if power_line_frequency is not None:
            cmd.extend(["--set-ctrl", f"power_line_frequency={power_line_frequency}"])
            settings_applied.append(f"power_line_frequency={power_line_frequency}")

        if len(settings_applied) > 0:
            subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            hailo_logger.info("Applied camera settings: %s", ", ".join(settings_applied))
            return True
        else:
            hailo_logger.debug("No camera settings to apply")
            return True

    except Exception as e:  # noqa: BLE001
        hailo_logger.warning("Failed to apply camera settings: %s", e)
        return False


def load_camera_settings(settings_path: Path) -> Optional[Dict]:
    """
    Load camera settings from a JSON file.

    Args:
        settings_path (Path): Path to camera settings JSON file.

    Returns:
        Optional[Dict]: Dictionary of camera settings, or None if file doesn't exist.
    """
    if not settings_path.exists():
        hailo_logger.debug("Camera settings file not found: %s", settings_path)
        return None

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        hailo_logger.info("Loaded camera settings from %s", settings_path)
        return settings
    except Exception as e:  # noqa: BLE001
        hailo_logger.warning("Failed to load camera settings: %s", e)
        return None


def apply_camera_settings_from_file(
    camera_device: str = "/dev/video0",
    settings_path: Optional[Path] = None,
) -> bool:
    """
    Load and apply camera settings from a JSON file.

    Args:
        camera_device (str): Camera device path (e.g., "/dev/video0").
        settings_path (Path, optional): Path to camera settings JSON file.

    Returns:
        bool: True if settings were applied successfully, False otherwise.
    """
    if settings_path is None:
        hailo_logger.debug("No camera settings path provided")
        return False

    settings = load_camera_settings(settings_path)
    if settings is None:
        return False

    # Apply settings
    return apply_camera_settings(
        camera_device=camera_device,
        brightness=settings.get("brightness"),
        exposure=settings.get("exposure"),
        focus=settings.get("focus"),
        auto_exposure=1 if settings.get("auto_exposure") == "MANUAL" else 3,
        auto_focus=0,  # Always disable auto-focus
        auto_white_balance=0,  # Always disable auto white balance
        power_line_frequency=0,  # Always disable power line frequency
    )

