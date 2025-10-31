"""
Main application for Freezing Phantoms game.

Orchestrates all components: AI pipeline, game logic, projection, and audio.
"""

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.calibration.calibration_tool import CalibrationTool
from freezing_phantoms.core.camera_utils import apply_camera_settings_from_file
from freezing_phantoms.core.game_state import GameState, GameStateMachine
from freezing_phantoms.core.movement_detector import MovementDetector
from freezing_phantoms.core.tile_grid import TileGrid, TileState
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

        # Store calibration path for camera settings
        self.calibration_path = calibration_path

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
            velocity_threshold=movement_config.get("velocity_threshold", 120.0),
            monitored_keypoints=movement_config.get("monitored_keypoints"),
            position_smoothing_alpha=movement_config.get("position_smoothing_alpha", 0.2),
            velocity_window_duration=movement_config.get("velocity_window_duration", 0.3),
            min_consecutive_frames=movement_config.get("min_consecutive_frames", 2),
            keypoint_confidence_threshold=movement_config.get("keypoint_confidence_threshold", 0.5),
            fps=movement_config.get("fps", 30.0),
        )

        # AI pipeline (will be started later)
        self.pipeline = None

        # Game loop control
        self.running = False
        self.last_frame_time = 0

        # Terminal settings for non-blocking input
        self.old_terminal_settings = None

        # Active player tracking
        self.active_player_id = None  # Track ID of player currently on board
        self.last_debug_time = 0  # For periodic debug output
        
        # Lava step tracking (delay before counting as violation)
        self.lava_step_start_time = {}  # track_id -> time when first stepped on lava
        self.lava_step_delay = 1.5  # seconds to wait before counting as violation
        
        # Game start tracking
        self.player_on_start_square = False  # Wait for player to reach start before monitoring

        hailo_logger.info("Game initialization complete")

    def _get_terminal_key(self) -> Optional[str]:
        """
        Get keyboard input from terminal (non-blocking).

        Returns:
            Optional[str]: Key pressed, or None if no key.
        """
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def _setup_terminal(self) -> None:
        """Set terminal to non-blocking raw mode."""
        self.old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal(self) -> None:
        """Restore terminal to original mode."""
        if self.old_terminal_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)

    def start_pipeline(self, video_source: str = "/dev/video0") -> None:
        """
        Start the AI pipeline.

        Args:
            video_source (str): Video source path or device.
        """
        hailo_logger.info("Starting AI pipeline with source: %s", video_source)

        # Apply saved camera settings if available
        camera_settings_path = self.calibration_path.with_suffix('.json').parent / (
            self.calibration_path.stem + '_camera_settings.json'
        )
        if camera_settings_path.exists():
            hailo_logger.info("Applying saved camera settings from calibration...")
            print(f"📷 Applying camera settings from {camera_settings_path.name}")
            if apply_camera_settings_from_file(video_source, camera_settings_path):
                print("✓ Camera settings applied successfully")
            else:
                print("⚠ Camera settings could not be applied (continuing anyway)")
        else:
            hailo_logger.info("No saved camera settings found, using camera defaults")
            print("ℹ No saved camera settings found (run calibration to save settings)")

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

        # Setup terminal for non-blocking input
        self._setup_terminal()

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
        print("  + / - - Increase / Decrease freeze detection sensitivity")
        print("  S - Show current sensitivity settings")
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
                    all_player_data = self.pipeline.get_latest_data(timeout=0.01)
                    if all_player_data:
                        # Filter to only include players on the game board
                        player_data = self._filter_players_on_board(all_player_data)
                        if player_data:
                            hailo_logger.debug("Got player data: %d players (filtered from %d total)",
                                             len(player_data.get('players', [])),
                                             len(all_player_data.get('players', [])))

                # Update game state
                self.game_state.update(dt, player_data)
                
                # If in countdown, wait for player to be on start square
                if self.game_state.current_state == GameState.COUNTDOWN:
                    if player_data and self._check_player_on_start_square(player_data):
                        if not self.player_on_start_square:
                            self.player_on_start_square = True
                            hailo_logger.info("Player on start square, starting game!")
                            print("✅ Player ready! Starting game...")
                            # Start game timer
                            if self.game_state.game_start_time is None:
                                self.game_state.game_start_time = time.time()
                            # Manually transition to GREEN_LIGHT
                            self.game_state._change_state(GameState.GREEN_LIGHT)
                            self.game_state.state_duration = float("inf")
                            if self.game_state.tile_grid:
                                self.game_state.green_light_starting_row = self.game_state.tile_grid.player_row
                
                # Print periodic debug info about tracked player (every 2 seconds)
                if current_time - self.last_debug_time >= 2.0:
                    self._print_player_debug_info(player_data)
                    self.last_debug_time = current_time

                # Check for tile violations during GREEN_LIGHT (only after player started)
                if self.game_state.current_state == GameState.GREEN_LIGHT and player_data:
                    self._check_tile_violations(player_data)

                    # Check if player is on front tile for 2 seconds
                    if self.game_state.check_player_on_front_tile():
                        print("✓ Player on front tile for 2s! Starting transition to RED LIGHT...")
                        # Remove tiles behind player when transitioning (not every frame)
                        self.tile_grid.remove_tiles_behind_player()
                        self.game_state.trigger_transition_to_red_light(player_data)

                # Check for movement during RED_LIGHT
                if self.game_state.current_state == GameState.RED_LIGHT and player_data:
                    self._check_for_movement(player_data)

                # Get game stats if in WIN state
                game_stats = None
                if self.game_state.current_state == GameState.WIN:
                    game_stats = self.game_state.get_game_stats()
                
                # Calculate current score for display
                current_score = 0
                if self.game_state.game_start_time:
                    elapsed_time = time.time() - self.game_state.game_start_time
                    base_score = 1000
                    penalty_deduction = self.game_state.total_penalties * 100
                    time_deduction = int(elapsed_time)
                    current_score = max(0, base_score - penalty_deduction - time_deduction)

                # Render projection frame
                projection_frame = self.renderer.render_frame(
                    state=self.game_state.current_state,
                    player_data=player_data,
                    caught_players=self.game_state.get_caught_players(),
                    frame_width=self.camera_resolution[0],
                    frame_height=self.camera_resolution[1],
                    violation_type=self.game_state.violation_type,
                    game_stats=game_stats,
                    current_hearts=self.game_state.current_hearts,
                    current_score=current_score,
                )

                # Display projection
                self.renderer.display(projection_frame)

                # Handle keyboard input from terminal (non-blocking)
                key = self._get_terminal_key()
                if key:
                    if key == '\x1b':  # ESC
                        hailo_logger.info("ESC pressed, exiting game")
                        self.running = False
                    elif key in ('r', 'R'):
                        hailo_logger.info("R pressed, restarting game")
                        self.game_state.reset()
                        self.player_on_start_square = False  # Reset start square flag
                        self.lava_step_start_time.clear()  # Clear lava timers
                        print("\n♻️  Game restarted! Stand on green tile to begin.\n")

                    # Sensitivity controls for freeze detection (velocity-based)
                    elif key in ('+', '='):
                        # Increase sensitivity (lower velocity threshold)
                        new_threshold = max(50.0, self.movement_detector.velocity_threshold - 20.0)
                        self.movement_detector.set_threshold(new_threshold)
                        print(f"🔼 Freeze detection MORE sensitive: {new_threshold:.1f} px/s threshold")
                    elif key in ('-', '_'):
                        # Decrease sensitivity (higher velocity threshold)
                        new_threshold = min(300.0, self.movement_detector.velocity_threshold + 20.0)
                        self.movement_detector.set_threshold(new_threshold)
                        print(f"🔽 Freeze detection LESS sensitive: {new_threshold:.1f} px/s threshold")
                    elif key in ('s', 'S'):
                        # Show current sensitivity settings
                        stats = self.movement_detector.get_smoothing_stats()
                        print("\n=== Freeze Detection Settings (Velocity-Based) ===")
                        print(f"Velocity threshold: {stats['velocity_threshold']:.1f} px/s")
                        print(f"Position smoothing: {stats['position_smoothing_alpha']:.2f}")
                        print(f"Velocity window: {stats['velocity_window_duration']:.2f}s")
                        print(f"Min consecutive frames: {stats['min_consecutive_frames']}")
                        print(f"Keypoint confidence: {stats['keypoint_confidence_threshold']:.2f}")
                        print(f"Active tracks: {stats['active_tracks']}")
                        print("==================================================\n")

                # Still need cv2.waitKey for window event handling
                cv2.waitKey(1)  # noqa: PGH003,E1101

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

    def _filter_players_on_board(self, all_player_data: dict) -> Optional[dict]:
        """
        Filter player data to only include players on the game board.

        Args:
            all_player_data (dict): All player data from pipeline.

        Returns:
            Optional[dict]: Filtered player data with only players on board, or None.
        """
        if "players" not in all_player_data or not all_player_data["players"]:
            return None

        players_on_board = []
        for player in all_player_data["players"]:
            if hasattr(player, 'leg_position') and player.leg_position:
                leg_x, leg_y = player.leg_position
                proj_x, proj_y = self.renderer.transform_point(leg_x, leg_y)

                # Check if player is on the game board
                if self.tile_grid.is_position_on_board(proj_x, proj_y):
                    players_on_board.append(player)
                    # Update active player ID (use the first player on board)
                    if not hasattr(self, 'active_player_id') or self.active_player_id is None or self.active_player_id != player.track_id:
                        self.active_player_id = player.track_id
                        hailo_logger.info("🎯 Now tracking player ID: %d (on board)", player.track_id)

        if not players_on_board:
            return None

        # Return filtered data with same structure
        return {
            "players": players_on_board,
            "frame_id": all_player_data.get("frame_id", 0),
        }

    def _print_player_debug_info(self, player_data: Optional[dict]) -> None:
        """
        Print debug information about currently tracked player.

        Args:
            player_data (Optional[dict]): Current player data.
        """
        if player_data and "players" in player_data and player_data["players"]:
            player = player_data["players"][0]  # Primary tracked player
            if hasattr(player, 'leg_position') and player.leg_position:
                leg_x, leg_y = player.leg_position
                proj_x, proj_y = self.renderer.transform_point(leg_x, leg_y)
                current_tile = self.tile_grid.get_player_current_tile((proj_x, proj_y))

                tile_info = f"Row {current_tile.row}, Col {current_tile.col}" if current_tile else "No tile"
                tile_state = current_tile.state.value if current_tile else "N/A"

                # Count green tiles for debugging
                green_count = len(self.tile_grid.get_tiles_by_state(TileState.GREEN))

                print(f"👤 Player ID: {player.track_id} | Position: ({proj_x}, {proj_y}) | "
                      f"Tile: {tile_info} | State: {tile_state} | "
                      f"Game Row: {self.tile_grid.player_row}/{self.tile_grid.current_revealed_row} | "
                      f"Green tiles: {green_count}")
        elif self.active_player_id is not None:
            print(f"⚠️  No player currently on board (last tracked: {self.active_player_id})")
    
    def _check_player_on_start_square(self, player_data: dict) -> bool:
        """
        Check if player is standing on a green tile in the starting row.

        Args:
            player_data (dict): Player data from pipeline.

        Returns:
            bool: True if player is on a green tile in starting row.
        """
        if not player_data or "players" not in player_data:
            return False

        for player in player_data["players"]:
            if hasattr(player, 'leg_position') and player.leg_position:
                leg_x, leg_y = player.leg_position
                proj_x, proj_y = self.renderer.transform_point(leg_x, leg_y)
                current_tile = self.tile_grid.get_player_current_tile((proj_x, proj_y))
                
                # Check if on starting row and on a green tile
                if current_tile and current_tile.row == self.tile_grid.starting_row:
                    if current_tile.state == TileState.GREEN:
                        return True
        
        return False

    def _check_tile_violations(self, player_data: dict) -> None:
        """
        Check if player stepped on a red tile during green light.
        Only triggers after being on lava for 1.5 seconds.

        Args:
            player_data (dict): Player data from pipeline (already filtered to board players).
        """
        if not player_data or "players" not in player_data:
            return

        current_time = time.time()

        for player in player_data["players"]:
            # Get leg position
            if hasattr(player, 'leg_position') and player.leg_position:
                leg_x, leg_y = player.leg_position
                # Transform to projector space
                proj_x, proj_y = self.renderer.transform_point(leg_x, leg_y)

                # Get tile at position
                current_tile = self.tile_grid.get_player_current_tile((proj_x, proj_y))
                track_id = player.track_id

                if current_tile and self.tile_grid.is_tile_dangerous(current_tile):
                    # Player is on lava
                    if track_id not in self.lava_step_start_time:
                        # First time stepping on lava, start timer
                        self.lava_step_start_time[track_id] = current_time
                        hailo_logger.debug("Player %d stepped on lava, starting %ss timer", 
                                          track_id, self.lava_step_delay)
                    else:
                        # Check if enough time has passed
                        elapsed = current_time - self.lava_step_start_time[track_id]
                        if elapsed >= self.lava_step_delay:
                            # Trigger violation
                            hailo_logger.warning("Player %d on lava for %.1fs - violation!", 
                                                track_id, elapsed)
                            self.game_state.register_tile_violation(track_id)
                            # Trigger visual effect
                            self.renderer.trigger_lava_step_effect()
                            print(f"🔥 LAVA! Player {track_id} on red tile too long!")
                            # Reset timer
                            del self.lava_step_start_time[track_id]
                else:
                    # Player is on safe tile, reset lava timer if exists
                    if track_id in self.lava_step_start_time:
                        hailo_logger.debug("Player %d moved off lava, resetting timer", track_id)
                        del self.lava_step_start_time[track_id]

    def _check_for_movement(self, player_data: dict) -> None:
        """
        Check for player movement during red light phase.
        Uses velocity-based detection (tracks current velocity, not displacement from frozen pose).

        Args:
            player_data (dict): Current player data from pipeline.
        """
        if not player_data or "players" not in player_data:
            return

        current_players = player_data["players"]
        
        # Enable debug logging to see all velocities during RED_LIGHT
        debug_log = True
        movement_results = self.movement_detector.detect_player_movement(current_players, debug_log=debug_log)

        for track_id, (moved, velocity, keypoint) in movement_results.items():
            if moved:
                hailo_logger.warning(
                    "Player %d moved during red light! %s velocity: %.1f px/s",
                    track_id,
                    keypoint,
                    velocity,
                )
                self.game_state.register_caught_player(track_id)

    def cleanup(self) -> None:
        """Clean up all resources."""
        hailo_logger.info("Cleaning up visualization resources...")

        # Restore terminal settings
        self._restore_terminal()

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

