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
        RED_LIGHT: Players must freeze.
        PENALTY: Display caught players.
        WIN: Player reached the goal.
        GAME_OVER: Game ended.
    """

    COUNTDOWN = "countdown"
    GREEN_LIGHT = "green_light"
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

        hailo_logger.info("Game state machine initialized in %s", self.current_state.value)

    def update(self, dt: float, player_data: Optional[Dict]) -> None:  # noqa: ARG002
        """
        Update state machine (call every frame).

        Args:
            dt (float): Time delta since last update.
            player_data (Optional[Dict]): Current player data from pipeline.
        """
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
            self._set_random_duration("green_light_duration")

        elif current == GameState.GREEN_LIGHT:
            # Check if player reached goal
            if self.tile_grid and self.tile_grid.has_reached_goal():
                self._change_state(GameState.WIN)
                self.state_duration = float("inf")
                hailo_logger.info("Player reached goal! Transitioning to WIN state")
            else:
                # Green Light -> Red Light
                self._change_state(GameState.RED_LIGHT)
                self._set_random_duration("red_light_duration")

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
                self._set_random_duration("green_light_duration")

        elif current == GameState.PENALTY:
            # Penalty -> Green Light
            self.caught_players = []  # Clear caught players
            self.tile_violators = []  # Clear tile violators
            self.violation_type = None
            self._change_state(GameState.GREEN_LIGHT)
            self._set_random_duration("green_light_duration")

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
            hailo_logger.info("Player %d caught!", track_id)

    def register_tile_violation(self, track_id: int) -> None:
        """
        Register a player who stepped on a red tile.

        Args:
            track_id (int): Track ID of violating player.
        """
        if track_id not in self.tile_violators:
            self.tile_violators.append(track_id)
            hailo_logger.info("Player %d stepped on lava!", track_id)

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

    def reset(self) -> None:
        """Reset the game state machine to initial state."""
        self.current_state = GameState.COUNTDOWN
        self.state_start_time = time.time()
        self.state_duration = self.config["game"]["countdown_duration"]
        self.frozen_poses = []
        self.caught_players = []
        self.tile_violators = []
        self.violation_type = None

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

    @property
    def current(self) -> GameState:
        """
        Get current game state.

        Returns:
            GameState: Current state.
        """
        return self.current_state

