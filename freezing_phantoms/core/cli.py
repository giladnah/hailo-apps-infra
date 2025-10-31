"""
Command-line interface utilities for Freezing Phantoms.

Handles argument parsing and CLI-related functionality.
"""

import argparse
import sys
from pathlib import Path

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

from freezing_phantoms.core.constants import (
    DEFAULT_CALIBRATION_FILE,
    DEFAULT_GAME_CONFIG_FILE,
    DEFAULT_VIDEO_DEVICE,
    EXIT_CALIBRATION_FAILED,
)

hailo_logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for the game.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Freezing Phantoms - AI-Augmented Halloween Game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --input /dev/video0

  # Specify architecture
  python main.py --input /dev/video0 --arch hailo8l

  # Debug mode with video display
  python main.py --input /dev/video0 --show-video

  # Run calibration first
  python main.py --run-calibration --input /dev/video0

For more information, see README.md
        """
    )

    # Standard Hailo options
    hailo_group = parser.add_argument_group('Hailo Options')
    hailo_group.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input source (camera device or file path)"
    )
    hailo_group.add_argument(
        "--use-frame", "-u",
        action="store_true",
        help="Use frame from the callback function"
    )
    hailo_group.add_argument(
        "--show-fps", "-f",
        action="store_true",
        help="Print FPS on sink"
    )
    hailo_group.add_argument(
        "--arch",
        default=None,
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Specify the Hailo architecture"
    )
    hailo_group.add_argument(
        "--hef-path",
        default=None,
        help="Path to HEF file"
    )
    hailo_group.add_argument(
        "--disable-sync",
        action="store_true",
        help="Disables display sink synchronization"
    )
    hailo_group.add_argument(
        "--disable-callback",
        action="store_true",
        help="Disables the user's custom callback"
    )
    hailo_group.add_argument(
        "--dump-dot",
        action="store_true",
        help="Dump the pipeline graph to DOT file"
    )
    hailo_group.add_argument(
        "--frame-rate", "-r",
        type=int,
        default=30,
        help="Frame rate of the video source (default: 30)"
    )

    # Game-specific options
    game_group = parser.add_argument_group('Game Options')
    game_group.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / DEFAULT_GAME_CONFIG_FILE,
        help=f"Path to game configuration file (default: {DEFAULT_GAME_CONFIG_FILE})"
    )
    game_group.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).parent.parent / DEFAULT_CALIBRATION_FILE,
        help=f"Path to calibration matrix file (default: {DEFAULT_CALIBRATION_FILE})"
    )
    game_group.add_argument(
        "--video-source",
        type=str,
        default=None,
        help="Video source (camera device or file path). Alias for --input."
    )
    game_group.add_argument(
        "--show-video",
        action="store_true",
        help="Show video output with AI overlays (debug mode)"
    )

    # Calibration options
    calib_group = parser.add_argument_group('Calibration Options')
    calib_group.add_argument(
        "--run-calibration",
        action="store_true",
        help="Run calibration tool before starting game"
    )
    calib_group.add_argument(
        "--validate-calibration",
        action="store_true",
        help="Validate existing calibration"
    )

    # Logging options
    log_group = parser.add_argument_group('Logging Options')
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable performance metrics collection and reporting"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments.

    Args:
        args: Parsed arguments.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Validate config file exists
    if not args.config.exists():
        raise ValueError(f"Configuration file not found: {args.config}")

    # If not running calibration, check that calibration file exists
    if not args.run_calibration and not args.validate_calibration:
        if not args.calibration.exists():
            hailo_logger.error("Calibration file not found: %s", args.calibration)
            print(f"\nError: Calibration file not found at {args.calibration}")
            print("Please run calibration first:")
            print(f"  python main.py --run-calibration --input {DEFAULT_VIDEO_DEVICE}")
            sys.exit(EXIT_CALIBRATION_FAILED)


def resolve_video_source(args: argparse.Namespace) -> str:
    """
    Resolve video source from arguments.

    Args:
        args: Parsed arguments.

    Returns:
        str: Video source path.
    """
    # Use --input if provided, otherwise --video-source, otherwise default
    video_source = args.input or args.video_source or DEFAULT_VIDEO_DEVICE
    hailo_logger.debug("Resolved video source: %s", video_source)
    return video_source


def print_startup_banner() -> None:
    """Print startup banner with game information."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                   FREEZING PHANTOMS                        ║
║              AI-Augmented Halloween Game                   ║
╚════════════════════════════════════════════════════════════╝

Controls:
  ESC - Quit game
  R   - Reset game

Press Ctrl+C to exit at any time.
    """
    print(banner)


def print_calibration_banner() -> None:
    """Print calibration banner with instructions."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║              CALIBRATION WITH VALIDATION                   ║
╚════════════════════════════════════════════════════════════╝

The calibration process will:
1. Generate and display checkerboard pattern
2. Capture camera image and detect corners
3. Calculate homography matrix
4. Run validation test
5. Allow you to accept/reject the calibration
6. Retry if rejected (up to 3 attempts)
    """
    print(banner)

