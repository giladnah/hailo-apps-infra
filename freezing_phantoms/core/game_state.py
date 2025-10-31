"""
Game state machine for Freezing Phantoms.

Manages automated game state transitions and timing.
"""

import random
import time
from enum import Enum
from typing import Dict, List, Optional

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class GameState(Enum):
    """
    Game state enumeration.

    States:
        COUNTDOWN: Initial countdown before game starts.
        GREEN_LIGHT: Players can move.
        TRANSITION: Flashing transition to red light.
        RED_LIGHT: Players must freeze.
        PENALTY: Display caught players.
        WIN: Player reached the goal.
        GAME_OVER: Game ended.
    """

    COUNTDOWN = "countdown"
    GREEN_LIGHT = "green_light"
    TRANSITION = "transition"
    RED_LIGHT = "red_light"
    PENALTY = "penalty"
    WIN = "win"
    GAME_OVER = "game_over"


class GameStateMachine:
    """
    Automated game state machine with timer-based transitions.

    Attributes:
        current_state (GameState): Current game state.
        state_start_time (float): Timestamp when current state started.
        config (dict): Game configuration.
    """

    def __init__(self, config: dict, tile_grid=None):
        """
        Initialize game state machine.

        Args:
            config (dict): Game configuration dictionary.
            tile_grid (TileGrid, optional): Reference to tile grid system.
        """
        self.config = config
        self.current_state = GameState.COUNTDOWN
        self.state_start_time = time.time()
        self.state_duration = self.config["game"]["countdown_duration"]

        # Storage for frozen poses
        self.frozen_poses = []  # List of PlayerData at freeze moment

        # Caught players
        self.caught_players = []  # List of track IDs

        # Tile violation tracking
        self.tile_violators = []  # List of track IDs who stepped on red tiles

        # Tile grid reference
        self.tile_grid = tile_grid

        # Player position tracking
        self.player_current_row = config.get("tile_grid", {}).get("starting_row", 9)
        self.penalty_rows_back = config.get("tile_grid", {}).get("penalty_rows_back", 2)

        # Violation type (for penalty display)
        self.violation_type = None  # "movement" or "tile"

        # Audio callback (will be set by main app)
        self.audio_callback = None

        # State change listeners
        self.state_change_listeners = []

        # Track player's starting row at beginning of green light
        self.green_light_starting_row = None

        # Track when player reaches front tile
        self.front_tile_reached_time = None
        self.front_tile_wait_duration = config.get("game", {}).get("front_tile_wait_duration", 2.0)

        # Flashing transition parameters
        self.flash_count = 0
        self.last_flash_time = 0
        self.flash_interval = config.get("game", {}).get("flash_interval", 0.25)
        self.flash_total = config.get("game", {}).get("flash_total", 3)

        # Score tracking
        self.game_start_time = None
        self.game_end_time = None
        self.total_penalties = 0
        self.movement_penalties = 0
        self.lava_penalties = 0

        hailo_logger.info("Game state machine initialized in %s", self.current_state.value)

    def update(self, dt: float, player_data: Optional[Dict]) -> None:  # noqa: ARG002
        """
        Update state machine (call every frame).

        Args:
            dt (float): Time delta since last update.
            player_data (Optional[Dict]): Current player data from pipeline.
        """
        # Handle flashing transition state
        if self.current_state == GameState.TRANSITION:
            current_time = time.time()
            if current_time - self.last_flash_time >= self.flash_interval:
                self.flash_count += 1
                self.last_flash_time = current_time
                hailo_logger.debug("Flash %d/%d", self.flash_count, self.flash_total)

                # After all flashes, transition to red light
                if self.flash_count >= self.flash_total:
                    self._transition_to_next_state(player_data)
        else:
            # Normal time-based transitions for other states
            elapsed_time = time.time() - self.state_start_time

            # Check if state duration expired
            if elapsed_time >= self.state_duration:
                self._transition_to_next_state(player_data)

    def _transition_to_next_state(self, player_data: Optional[Dict]) -> None:
        """
        Transition to the next appropriate game state.

        Args:
            player_data (Optional[Dict]): Current player data.
        """
        current = self.current_state

        if current == GameState.COUNTDOWN:
            # Countdown -> Green Light
            self._change_state(GameState.GREEN_LIGHT)
            self.state_duration = float("inf")  # No time limit, ends when player advances

            # Store player's starting row
            if self.tile_grid:
                self.green_light_starting_row = self.tile_grid.player_row
                hailo_logger.info("Green light started at row %d", self.green_light_starting_row)

        elif current == GameState.GREEN_LIGHT:
            # This transition is now triggered manually, not by timer
            # See check_player_advancement()
            pass

        elif current == GameState.TRANSITION:
            # Transition -> Red Light after flashing completes
            self._change_state(GameState.RED_LIGHT)
            self._set_random_duration("red_light_duration")

            # Reset flash counter
            self.flash_count = 0

            # Store frozen poses
            if player_data and "players" in player_data:
                self.frozen_poses = player_data["players"]
                hailo_logger.info(
                    "Stored %d frozen poses for movement detection", len(self.frozen_poses)
                )

        elif current == GameState.RED_LIGHT:
            # Red Light -> Penalty (if players caught) or Green Light
            if self.caught_players:
                self._change_state(GameState.PENALTY)
                self.state_duration = self.config["game"]["penalty_duration"]
                self.violation_type = "movement"

                # Apply penalty to tile grid
                if self.tile_grid:
                    self.tile_grid.apply_penalty()
            else:
                # Advance path before green light
                if self.tile_grid:
                    self.tile_grid.advance_path(rows_to_reveal=1)

                self._change_state(GameState.GREEN_LIGHT)
                self.state_duration = float("inf")  # No time limit, ends when player advances

                # Store player's starting row
                if self.tile_grid:
                    self.green_light_starting_row = self.tile_grid.player_row
                    hailo_logger.info("Green light started at row %d", self.green_light_starting_row)

        elif current == GameState.PENALTY:
            # Penalty -> Green Light
            self.caught_players = []  # Clear caught players
            self.tile_violators = []  # Clear tile violators
            self.violation_type = None
            self._change_state(GameState.GREEN_LIGHT)
            self.state_duration = float("inf")  # No time limit, ends when player advances

            # Store player's starting row
            if self.tile_grid:
                self.green_light_starting_row = self.tile_grid.player_row
                hailo_logger.info("Green light started at row %d", self.green_light_starting_row)

        elif current == GameState.WIN:
            # Stay in win state
            pass

        elif current == GameState.GAME_OVER:
            # Stay in game over
            pass

    def _change_state(self, new_state: GameState) -> None:
        """
        Change to a new state.

        Args:
            new_state (GameState): The state to transition to.
        """
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()

        hailo_logger.info("State transition: %s -> %s", old_state.value, new_state.value)

        # Trigger audio callback
        if self.audio_callback:
            self.audio_callback(new_state)

        # Notify listeners
        for listener in self.state_change_listeners:
            listener(old_state, new_state)

    def _set_random_duration(self, config_key: str) -> None:
        """
        Set random duration from config range.

        Args:
            config_key (str): Configuration key for duration range.
        """
        duration_range = self.config["game"][config_key]
        self.state_duration = random.uniform(duration_range[0], duration_range[1])
        hailo_logger.debug(
            "Set %s duration to %.1fs (range: %s)", config_key, self.state_duration, duration_range
        )

    def register_caught_player(self, track_id: int) -> None:
        """
        Register a player as caught.

        Args:
            track_id (int): Track ID of caught player.
        """
        if track_id not in self.caught_players:
            self.caught_players.append(track_id)
            self.movement_penalties += 1
            self.total_penalties += 1
            hailo_logger.info("Player %d caught! Total penalties: %d", track_id, self.total_penalties)

    def register_tile_violation(self, track_id: int) -> None:
        """
        Register a player who stepped on a red tile.

        Args:
            track_id (int): Track ID of violating player.
        """
        if track_id not in self.tile_violators:
            self.tile_violators.append(track_id)
            self.lava_penalties += 1
            self.total_penalties += 1
            hailo_logger.info("Player %d stepped on lava! Total penalties: %d", track_id, self.total_penalties)

            # Trigger audio callback for lava step
            if self.audio_callback:
                self.audio_callback("lava_step")

            # Transition to penalty immediately
            if self.current_state == GameState.GREEN_LIGHT:
                self._change_state(GameState.PENALTY)
                self.state_duration = self.config["game"]["penalty_duration"]
                self.violation_type = "tile"

                # Apply penalty to tile grid
                if self.tile_grid:
                    self.tile_grid.apply_penalty()

    def get_frozen_poses(self) -> List:
        """
        Get the stored frozen poses from the start of RED_LIGHT.

        Returns:
            List: List of PlayerData objects.
        """
        return self.frozen_poses

    def get_caught_players(self) -> List[int]:
        """
        Get list of caught player track IDs.

        Returns:
            List[int]: List of track IDs.
        """
        return self.caught_players

    def get_time_remaining(self) -> float:
        """
        Get time remaining in current state.

        Returns:
            float: Seconds remaining.
        """
        elapsed = time.time() - self.state_start_time
        return max(0.0, self.state_duration - elapsed)

    def get_time_in_state(self) -> float:
        """
        Get time elapsed in current state.

        Returns:
            float: Seconds elapsed.
        """
        return time.time() - self.state_start_time

    def set_audio_callback(self, callback) -> None:
        """
        Set callback function for audio cues.

        Args:
            callback: Function that takes GameState as argument.
        """
        self.audio_callback = callback
        hailo_logger.debug("Audio callback registered")

    def add_state_change_listener(self, listener) -> None:
        """
        Add a listener for state changes.

        Args:
            listener: Function that takes (old_state, new_state) as arguments.
        """
        self.state_change_listeners.append(listener)
        hailo_logger.debug("State change listener added")

    def force_state(self, state: GameState, duration: float = None) -> None:
        """
        Force transition to a specific state (for testing/debugging).

        Args:
            state (GameState): State to transition to.
            duration (float, optional): Override state duration.
        """
        self._change_state(state)
        if duration is not None:
            self.state_duration = duration
        hailo_logger.warning("Forced state change to %s", state.value)

    def check_player_on_front_tile(self) -> bool:
        """
        Check if player is on the furthest revealed row during green light.
        Starts a 2-second timer when reached.

        Returns:
            bool: True if 2 seconds have passed since reaching front tile.
        """
        if self.current_state != GameState.GREEN_LIGHT:
            return False

        if not self.tile_grid:
            return False

        current_row = self.tile_grid.player_row
        revealed_row = self.tile_grid.current_revealed_row

        # Check if player is on the furthest revealed row
        on_front_tile = (current_row == revealed_row)

        if on_front_tile:
            # Start timer if not already started
            if self.front_tile_reached_time is None:
                self.front_tile_reached_time = time.time()
                hailo_logger.info("Player reached front tile (row %d), starting 2s timer", current_row)
                return False

            # Check if 2 seconds have passed
            elapsed = time.time() - self.front_tile_reached_time
            if elapsed >= self.front_tile_wait_duration:
                hailo_logger.info("Front tile timer expired (%.1fs), triggering transition", elapsed)
                return True
        else:
            # Reset timer if player moves away
            if self.front_tile_reached_time is not None:
                hailo_logger.debug("Player moved away from front tile, resetting timer")
                self.front_tile_reached_time = None

        return False

    def trigger_transition_to_red_light(self, player_data: Optional[Dict] = None) -> None:
        """
        Trigger flashing transition to red light.

        Args:
            player_data (Optional[Dict]): Current player data.
        """
        if self.current_state == GameState.GREEN_LIGHT:
            # Check for win condition first
            if self.tile_grid and self.tile_grid.has_reached_goal():
                # Record end time
                self.game_end_time = time.time()
                self._change_state(GameState.WIN)
                self.state_duration = float("inf")
                hailo_logger.info("Player reached goal! Transitioning to WIN state")
            else:
                # Start flashing transition
                self._change_state(GameState.TRANSITION)
                self.flash_count = 0
                self.last_flash_time = time.time()
                self.state_duration = self.flash_interval * self.flash_total
                self.front_tile_reached_time = None  # Reset timer
                hailo_logger.info("Starting transition with %d flashes", self.flash_total)

    def reset(self) -> None:
        """Reset the game state machine to initial state."""
        self.current_state = GameState.COUNTDOWN
        self.state_start_time = time.time()
        self.state_duration = self.config["game"]["countdown_duration"]
        self.frozen_poses = []
        self.caught_players = []
        self.tile_violators = []
        self.violation_type = None
        self.green_light_starting_row = None
        self.flash_count = 0
        self.last_flash_time = 0
        self.front_tile_reached_time = None

        # Reset score tracking
        self.game_start_time = None
        self.game_end_time = None
        self.total_penalties = 0
        self.movement_penalties = 0
        self.lava_penalties = 0

        # Reset tile grid
        if self.tile_grid:
            self.tile_grid.reset()
            self.player_current_row = self.tile_grid.player_row

        hailo_logger.info("Game state machine reset")

    def end_game(self) -> None:
        """Transition to game over state."""
        self._change_state(GameState.GAME_OVER)
        self.state_duration = float("inf")
        hailo_logger.info("Game ended")

    def get_game_stats(self) -> dict:
        """
        Get current game statistics for scoring.

        Returns:
            dict: Game statistics including time, penalties, and score.
        """
        if self.game_start_time and self.game_end_time:
            elapsed_time = self.game_end_time - self.game_start_time
        elif self.game_start_time:
            elapsed_time = time.time() - self.game_start_time
        else:
            elapsed_time = 0

        # Calculate score (higher is better)
        # Base score: 1000 points
        # Subtract penalties: -100 per penalty
        # Time bonus: faster = better (subtract seconds)
        base_score = 1000
        penalty_deduction = self.total_penalties * 100
        time_deduction = int(elapsed_time)

        final_score = max(0, base_score - penalty_deduction - time_deduction)

        # Determine performance rating
        if self.total_penalties == 0:
            rating = "PERFECT!"
        elif self.total_penalties <= 2:
            rating = "EXCELLENT"
        elif self.total_penalties <= 5:
            rating = "GOOD"
        else:
            rating = "COMPLETED"

        return {
            "time": elapsed_time,
            "total_penalties": self.total_penalties,
            "movement_penalties": self.movement_penalties,
            "lava_penalties": self.lava_penalties,
            "score": final_score,
            "rating": rating
        }

    @property
    def current(self) -> GameState:
        """
        Get current game state.

        Returns:
            GameState: Current state.
        """
        return self.current_state

