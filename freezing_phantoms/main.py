"""
Main application for Freezing Phantoms game.

Orchestrates all components: AI pipeline, game logic, projection, and audio.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.calibration.calibration_tool import CalibrationTool
# Disabled for simple visualization mode:
# from freezing_phantoms.core.audio_manager import AudioManager
# from freezing_phantoms.core.board_system import BoardSystem
# from freezing_phantoms.core.game_state import GameState, GameStateMachine
# from freezing_phantoms.core.ghost_system import GhostManager
# from freezing_phantoms.core.movement_detector import MovementDetector
# from freezing_phantoms.effects.board_effects import BoardEffects
from freezing_phantoms.core.projection_renderer import ProjectionRenderer
from freezing_phantoms.pipeline.combined_pipeline import PoseEstimationPipeline

hailo_logger = get_logger(__name__)

class FreezingPhantomsGame:
    """
    Main game application.

    Orchestrates the AI pipeline, game state, projection, and audio systems.
    """

    def __init__(self, config_path: Path, calibration_path: Path, debug_mode: bool = False, hailo_options=None):
        """
        Initialize game application.

        Args:
            config_path (Path): Path to game configuration file.
            calibration_path (Path): Path to calibration matrix file.
            debug_mode (bool): If True, show video output with AI overlays.
            hailo_options: Parsed Hailo standard options (arch, hef-path, etc.)
        """
        hailo_logger.info("Initializing Freezing Phantoms game...")

        # Store debug mode flag and Hailo options
        self.debug_mode = debug_mode
        self.hailo_options = hailo_options or argparse.Namespace()

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        hailo_logger.info("Loaded configuration from %s", config_path)
        if debug_mode:
            hailo_logger.info("DEBUG MODE: Video output enabled")

        # Load calibration matrix
        if not calibration_path.exists():
            hailo_logger.error("Calibration file not found: %s", calibration_path)
            raise FileNotFoundError(
                f"Calibration matrix not found at {calibration_path}. "
                f"Please run calibration first."
            )

        self.homography_matrix = np.load(calibration_path)
        hailo_logger.info("Loaded calibration matrix from %s", calibration_path)

        # Extract config values
        self.camera_resolution = tuple(self.config["camera"]["resolution"])
        self.projector_resolution = tuple(self.config["projector"]["resolution"])
        self.fps = self.config["camera"]["fps"]

        # Initialize components
        hailo_logger.info("Initializing pose visualization (simplified mode)...")

        # Projection renderer (only component needed for simple visualization)
        self.renderer = ProjectionRenderer(
            homography_matrix=self.homography_matrix,
            projector_resolution=self.projector_resolution,
            display_index=self.config["projector"]["display_index"],
        )

        # Disabled game components for simple visualization mode
        # self.game_state = GameStateMachine(self.config)
        # self.movement_detector = MovementDetector(...)
        # self.audio_manager = AudioManager(assets_path)
        # self.board_system = BoardSystem(board_config)
        # self.ghost_manager = GhostManager(board_config, self.board_system)
        # self.board_effects = BoardEffects(...)

        # AI pipeline (will be started later)
        self.pipeline = None

        # Game loop control
        self.running = False
        self.last_frame_time = 0

        hailo_logger.info("Game initialization complete")

    def start_pipeline(self, video_source: str = "/dev/video0") -> None:
        """
        Start the AI pipeline.

        Args:
            video_source (str): Video source path or device.
        """
        hailo_logger.info("Starting AI pipeline with source: %s", video_source)

        # Use frame rate from Hailo options if provided, otherwise use config
        frame_rate = getattr(self.hailo_options, 'frame_rate', None) or self.fps

        # Force 720p resolution (1280x720) to match calibration
        hailo_logger.info("Using 720p resolution: 1280x720")
        self.pipeline = PoseEstimationPipeline(
            video_source=video_source,
            frame_width=1280,
            frame_height=720,
            frame_rate=frame_rate,
            debug_mode=self.debug_mode,
            hailo_options=self.hailo_options,
        )

        self.pipeline.start()
        hailo_logger.info("AI pipeline started")

        # Wait for pipeline to warm up
        time.sleep(2.0)

    def run(self) -> None:
        """Run the main game loop."""
        hailo_logger.info("Starting pose visualization loop...")
        print("\n" + "=" * 60)
        print("POSE LANDMARKS VISUALIZATION")
        print("=" * 60)
        print("Displaying pose keypoints projected through calibration.")
        print("")
        print("Controls:")
        print("  ESC - Quit")
        print("=" * 60 + "\n")

        self.running = True
        self.last_frame_time = time.time()

        try:
            while self.running:
                current_time = time.time()
                # dt = current_time - self.last_frame_time  # Not used in simple mode
                self.last_frame_time = current_time

                # Get latest player data from pipeline
                player_data = None
                if self.pipeline:
                    player_data = self.pipeline.get_latest_data(timeout=0.01)
                    if player_data:
                        hailo_logger.debug("Got player data: %d players", len(player_data.get('players', [])))
                    else:
                        hailo_logger.debug("No player data from pipeline")

                # SIMPLE MODE: Just render pose landmarks (no game logic)
                projection_frame = self.renderer.render_simple_pose_landmarks(
                    player_data,
                    self.camera_resolution[0],
                    self.camera_resolution[1],
                )

                # Display projection
                self.renderer.display(projection_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF  # noqa: PGH003,E1101
                if key == 27:  # ESC
                    hailo_logger.info("ESC pressed, exiting visualization")
                    self.running = False

                # Maintain frame rate
                target_frame_time = 1.0 / 30  # 30 FPS
                elapsed = time.time() - current_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        except KeyboardInterrupt:
            hailo_logger.info("Keyboard interrupt received")

        except Exception as e:
            hailo_logger.error("Error in game loop: %s", e, exc_info=True)

        finally:
            self.cleanup()

    # Disabled game methods (not needed in simple visualization mode)
    # def _check_for_movement(self, player_data: dict) -> None:
    #     ...
    # def _check_board_collisions(self, player_data: dict) -> None:
    #     ...

    def cleanup(self) -> None:
        """Clean up all resources."""
        hailo_logger.info("Cleaning up visualization resources...")

        # Stop pipeline
        if self.pipeline:
            self.pipeline.stop()

        # Clean up renderer
        self.renderer.cleanup()

        # Close all OpenCV windows
        cv2.destroyAllWindows()  # noqa: PGH003,E1101

        hailo_logger.info("Cleanup complete")


def main():
    """Main entry point."""
    # Create parser with standard Hailo options
    parser = argparse.ArgumentParser(
        description="Freezing Phantoms - AI-Augmented Halloween Game"
    )

    # Add standard Hailo options (from get_default_parser)
    parser.add_argument("--input", "-i", type=str, default=None, help="Input source (camera device or file path)")
    parser.add_argument("--use-frame", "-u", action="store_true", help="Use frame from the callback function")
    parser.add_argument("--show-fps", "-f", action="store_true", help="Print FPS on sink")
    parser.add_argument("--arch", default=None, choices=["hailo8", "hailo8l", "hailo10h"],
                        help="Specify the Hailo architecture")
    parser.add_argument("--hef-path", default=None, help="Path to HEF file")
    parser.add_argument("--disable-sync", action="store_true", help="Disables display sink sync")
    parser.add_argument("--disable-callback", action="store_true", help="Disables the user's custom callback")
    parser.add_argument("--dump-dot", action="store_true", help="Dump the pipeline graph to DOT file")
    parser.add_argument("--frame-rate", "-r", type=int, default=30, help="Frame rate of the video source")

    # Add game-specific arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "game_config.yaml",
        help="Path to game configuration file",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).parent / "config" / "calibration.npy",
        help="Path to calibration matrix file",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default=None,
        help="Video source (camera device or file path). Alias for --input.",
    )
    parser.add_argument(
        "--run-calibration",
        action="store_true",
        help="Run calibration tool before starting game",
    )
    parser.add_argument(
        "--validate-calibration",
        action="store_true",
        help="Validate existing calibration",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show video output with AI overlays (debug mode)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Update log level if specified
    if args.log_level != "INFO":
        os.environ["LOG_LEVEL"] = args.log_level
        # Update existing loggers
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        hailo_logger.info("Log level set to %s", args.log_level)

    # Handle video source: use --input if provided, otherwise --video-source
    video_source = args.input or args.video_source or "/dev/video0"

    # Run calibration if requested
    if args.run_calibration or args.validate_calibration:
        tool = CalibrationTool(args.config)

        if args.validate_calibration:
            if not args.calibration.exists():
                print(f"Error: Calibration file not found at {args.calibration}")
                sys.exit(1)

            homography = np.load(args.calibration)
            tool.validate_calibration(homography, camera_source=video_source)

            if not args.run_calibration:
                return

        if args.run_calibration:
            print("\n" + "="*70)
            print("CALIBRATION WITH VALIDATION")
            print("="*70)
            print("The calibration process will:")
            print("1. Generate and display checkerboard pattern")
            print("2. Capture camera image and detect corners")
            print("3. Calculate homography matrix")
            print("4. Run validation test")
            print("5. Allow you to accept/reject the calibration")
            print("6. Retry if rejected (up to 3 attempts)")
            print("="*70)
            print()

            homography = tool.run_calibration(
                camera_source=video_source,
                output_path=args.calibration,
                auto_validate=True,
                max_retries=3
            )
            if homography is None:
                print("\n❌ Calibration failed or was cancelled!")
                print("   Check calibration_history/ directory for debug information")
                sys.exit(1)

            print("\n✅ Calibration completed successfully!")
            print("   Calibration matrix saved and ready for use.")
            print("   Run the game separately with: python main.py")
            return  # Exit without starting the game

    # Check if calibration exists
    if not args.calibration.exists():
        print(f"\nError: Calibration file not found at {args.calibration}")
        print("Please run calibration first:")
        print(f"  python {__file__} --run-calibration --input {video_source}")
        sys.exit(1)

    # Log standard Hailo options if provided
    if args.arch:
        hailo_logger.info("Using specified architecture: %s", args.arch)
    if args.hef_path:
        hailo_logger.info("Using custom HEF path: %s", args.hef_path)
    if args.dump_dot:
        hailo_logger.info("Pipeline graph dumping enabled")

    # Create and run game with Hailo options
    game = FreezingPhantomsGame(
        args.config,
        args.calibration,
        debug_mode=args.show_video,
        hailo_options=args
    )
    game.start_pipeline(video_source)
    game.run()


if __name__ == "__main__":
    main()

