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
from freezing_phantoms.core.game_state import GameState, GameStateMachine
from freezing_phantoms.core.movement_detector import MovementDetector
from freezing_phantoms.core.tile_grid import TileGrid
from freezing_phantoms.effects.board_effects import BoardEffects
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
        hailo_logger.info("Initializing Red Light Green Light game...")

        # Initialize tile grid
        tile_config = self.config.get("tile_grid", {})
        self.tile_grid = TileGrid(
            projector_width=self.projector_resolution[0],
            projector_height=self.projector_resolution[1],
            rows=tile_config.get("rows", 10),
            cols=tile_config.get("columns", 5),
            starting_row=tile_config.get("starting_row", 9),
            goal_row=tile_config.get("goal_row", 0),
            penalty_rows_back=tile_config.get("penalty_rows_back", 2),
            path_width=tile_config.get("path_width", 1),
            obstacle_density=tile_config.get("obstacle_density", 0.3),
        )

        # Initialize board effects
        lava_config = self.config.get("lava_floor", {})
        lava_texture_path = Path(__file__).parent / lava_config.get("texture_path", "lava-floor-2.jpg")
        self.board_effects = BoardEffects(lava_texture_path=str(lava_texture_path))

        # Initialize projection renderer with tile grid
        self.renderer = ProjectionRenderer(
            homography_matrix=self.homography_matrix,
            projector_resolution=self.projector_resolution,
            display_index=self.config["projector"]["display_index"],
            tile_grid=self.tile_grid,
            board_effects=self.board_effects,
        )

        # Initialize game state machine
        self.game_state = GameStateMachine(self.config, tile_grid=self.tile_grid)

        # Initialize movement detector
        movement_config = self.config.get("movement", {})
        self.movement_detector = MovementDetector(
            threshold_pixels=movement_config.get("threshold_pixels", 20.0),
            monitored_keypoints=movement_config.get("monitored_keypoints"),
            smoothing_alpha=movement_config.get("smoothing_alpha", 0.3),
            pose_history_size=movement_config.get("pose_history_size", 5),
            sensitivity_multiplier=movement_config.get("sensitivity_multiplier", 1.0),
            keypoint_confidence_threshold=movement_config.get("keypoint_confidence_threshold", 0.5),
        )

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
        hailo_logger.info("Starting Red Light Green Light game loop...")
        print("\n" + "=" * 60)
        print("RED LIGHT GREEN LIGHT - Floor is Lava Edition")
        print("=" * 60)
        print("Game Rules:")
        print("  - Move on GREEN tiles during green light")
        print("  - FREEZE during red light")
        print("  - Avoid RED lava tiles")
        print("  - Reach the GOAL at the bottom")
        print("")
        print("Controls:")
        print("  R - Restart game")
        print("  ESC - Quit")
        print("=" * 60)
        print("")
        print("⚠️  CAMERA TIP: If the camera keeps adjusting focus/brightness,")
        print("    run: ./setup_camera.sh [camera_device]")
        print("    Example: ./setup_camera.sh /dev/video0")
        print("=" * 60 + "\n")

        self.running = True
        self.last_frame_time = time.time()

        try:
            while self.running:
                current_time = time.time()
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time

                # Get latest player data from pipeline
                player_data = None
                if self.pipeline:
                    player_data = self.pipeline.get_latest_data(timeout=0.01)
                    if player_data:
                        hailo_logger.debug("Got player data: %d players", len(player_data.get('players', [])))

                # Update game state
                self.game_state.update(dt, player_data)

                # Check for tile violations during GREEN_LIGHT
                if self.game_state.current == GameState.GREEN_LIGHT and player_data:
                    self._check_tile_violations(player_data)

                # Check for movement during RED_LIGHT
                if self.game_state.current == GameState.RED_LIGHT and player_data:
                    self._check_for_movement(player_data)

                # Render projection frame
                projection_frame = self.renderer.render_frame(
                    state=self.game_state.current,
                    player_data=player_data,
                    caught_players=self.game_state.get_caught_players(),
                    frame_width=self.camera_resolution[0],
                    frame_height=self.camera_resolution[1],
                    violation_type=self.game_state.violation_type,
                )

                # Display projection
                self.renderer.display(projection_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF  # noqa: PGH003,E1101
                if key == 27:  # ESC
                    hailo_logger.info("ESC pressed, exiting game")
                    self.running = False
                elif key == ord('r') or key == ord('R'):
                    hailo_logger.info("R pressed, restarting game")
                    self.game_state.reset()

                # Maintain frame rate
                target_frame_time = 1.0 / 30  # 30 FPS
                elapsed = time.time() - current_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        except KeyboardInterrupt:
            hailo_logger.info("Keyboard interrupt received")

        except Exception as e:  # noqa: BLE001
            hailo_logger.error("Error in game loop: %s", e, exc_info=True)

        finally:
            self.cleanup()

    def _check_tile_violations(self, player_data: dict) -> None:
        """
        Check if player stepped on a red tile during green light.

        Args:
            player_data (dict): Current player data from pipeline.
        """
        if not player_data or "players" not in player_data:
            return

        for player in player_data["players"]:
            # Get leg position
            if hasattr(player, 'leg_position') and player.leg_position:
                leg_x, leg_y = player.leg_position
                # Transform to projector space
                proj_x, proj_y = self.renderer.transform_point(leg_x, leg_y)

                # Get tile at position
                current_tile = self.tile_grid.get_player_current_tile((proj_x, proj_y))

                if current_tile and self.tile_grid.is_tile_dangerous(current_tile):
                    hailo_logger.warning("Player %d stepped on lava tile!", player.track_id)
                    self.game_state.register_tile_violation(player.track_id)

    def _check_for_movement(self, player_data: dict) -> None:
        """
        Check for player movement during red light phase.

        Args:
            player_data (dict): Current player data from pipeline.
        """
        if not player_data or "players" not in player_data:
            return

        frozen_poses = self.game_state.get_frozen_poses()
        if not frozen_poses:
            return

        current_players = player_data["players"]
        movement_results = self.movement_detector.detect_player_movement(
            frozen_poses, current_players
        )

        for track_id, (moved, displacement, keypoint) in movement_results.items():
            if moved:
                hailo_logger.warning(
                    "Player %d moved during red light! %s displaced %.1fpx",
                    track_id,
                    keypoint,
                    displacement,
                )
                self.game_state.register_caught_player(track_id)

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

