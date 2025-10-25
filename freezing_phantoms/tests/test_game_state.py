"""
Unit tests for game state machine.

Tests state transitions, timing, and game logic.
"""

import time
import pytest

from freezing_phantoms.core.game_state import GameState, GameStateMachine


class TestGameStateMachine:
    """Test game state machine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "game": {
                "countdown_duration": 1.0,
                "green_light_duration": [2.0, 3.0],
                "red_light_duration": [1.5, 2.5],
                "penalty_duration": 1.0,
            }
        }
        self.state_machine = GameStateMachine(self.config)

    def test_initial_state(self):
        """Test that initial state is COUNTDOWN."""
        assert self.state_machine.current == GameState.COUNTDOWN

    def test_countdown_to_green_light_transition(self):
        """Test transition from COUNTDOWN to GREEN_LIGHT."""
        # Fast-forward time
        self.state_machine.state_duration = 0.1
        time.sleep(0.15)

        self.state_machine.update(0.15, None)

        assert self.state_machine.current == GameState.GREEN_LIGHT

    def test_green_light_to_red_light_transition(self):
        """Test transition from GREEN_LIGHT to RED_LIGHT."""
        # Force to GREEN_LIGHT
        self.state_machine.force_state(GameState.GREEN_LIGHT, duration=0.1)

        time.sleep(0.15)
        self.state_machine.update(0.15, None)

        assert self.state_machine.current == GameState.RED_LIGHT

    def test_red_light_to_green_light_no_caught(self):
        """Test transition from RED_LIGHT to GREEN_LIGHT when no players caught."""
        # Force to RED_LIGHT
        self.state_machine.force_state(GameState.RED_LIGHT, duration=0.1)

        time.sleep(0.15)
        self.state_machine.update(0.15, None)

        # Should go to GREEN_LIGHT (no caught players)
        assert self.state_machine.current == GameState.GREEN_LIGHT

    def test_red_light_to_penalty_with_caught(self):
        """Test transition from RED_LIGHT to PENALTY when players caught."""
        # Force to RED_LIGHT
        self.state_machine.force_state(GameState.RED_LIGHT, duration=0.1)

        # Register a caught player
        self.state_machine.register_caught_player(track_id=1)

        time.sleep(0.15)
        self.state_machine.update(0.15, None)

        # Should go to PENALTY
        assert self.state_machine.current == GameState.PENALTY

    def test_penalty_to_green_light_transition(self):
        """Test transition from PENALTY to GREEN_LIGHT."""
        # Force to PENALTY
        self.state_machine.force_state(GameState.PENALTY, duration=0.1)
        self.state_machine.caught_players = [1, 2]

        time.sleep(0.15)
        self.state_machine.update(0.15, None)

        # Should go to GREEN_LIGHT and clear caught players
        assert self.state_machine.current == GameState.GREEN_LIGHT
        assert len(self.state_machine.caught_players) == 0

    def test_register_caught_player(self):
        """Test registering caught players."""
        self.state_machine.register_caught_player(1)
        self.state_machine.register_caught_player(2)

        caught = self.state_machine.get_caught_players()

        assert 1 in caught
        assert 2 in caught
        assert len(caught) == 2

    def test_register_same_player_twice(self):
        """Test that registering same player twice doesn't duplicate."""
        self.state_machine.register_caught_player(1)
        self.state_machine.register_caught_player(1)

        caught = self.state_machine.get_caught_players()

        assert len(caught) == 1

    def test_frozen_poses_storage(self):
        """Test storage of frozen poses during RED_LIGHT."""
        from freezing_phantoms.pipeline.data_extractor import PlayerData

        # Create mock player data
        player_data = {
            "players": [
                PlayerData(
                    track_id=1,
                    bbox={"xmin": 0.1, "ymin": 0.2, "width": 0.1, "height": 0.2},
                    confidence=0.9,
                    pose_keypoints={"left_wrist": (100.0, 200.0)},
                )
            ]
        }

        # Force to GREEN_LIGHT
        self.state_machine.force_state(GameState.GREEN_LIGHT, duration=0.1)

        time.sleep(0.15)
        self.state_machine.update(0.15, player_data)

        # Should transition to RED_LIGHT and store poses
        assert self.state_machine.current == GameState.RED_LIGHT
        frozen = self.state_machine.get_frozen_poses()
        assert len(frozen) == 1
        assert frozen[0].track_id == 1

    def test_time_remaining(self):
        """Test time remaining calculation."""
        self.state_machine.force_state(GameState.GREEN_LIGHT, duration=5.0)

        time.sleep(0.1)

        remaining = self.state_machine.get_time_remaining()

        assert remaining < 5.0
        assert remaining > 4.5

    def test_time_in_state(self):
        """Test time elapsed in state."""
        self.state_machine.force_state(GameState.GREEN_LIGHT, duration=5.0)

        time.sleep(0.2)

        elapsed = self.state_machine.get_time_in_state()

        assert elapsed >= 0.2
        assert elapsed < 0.5

    def test_reset(self):
        """Test state machine reset."""
        # Advance to RED_LIGHT with caught players
        self.state_machine.force_state(GameState.RED_LIGHT)
        self.state_machine.register_caught_player(1)

        # Reset
        self.state_machine.reset()

        assert self.state_machine.current == GameState.COUNTDOWN
        assert len(self.state_machine.caught_players) == 0
        assert len(self.state_machine.frozen_poses) == 0

    def test_end_game(self):
        """Test transitioning to game over."""
        self.state_machine.end_game()

        assert self.state_machine.current == GameState.GAME_OVER

    def test_audio_callback_invoked(self):
        """Test that audio callback is invoked on state change."""
        called_states = []

        def mock_audio_callback(state):
            called_states.append(state)

        self.state_machine.set_audio_callback(mock_audio_callback)

        # Force state change
        self.state_machine.force_state(GameState.GREEN_LIGHT)

        assert GameState.GREEN_LIGHT in called_states

    def test_state_change_listener(self):
        """Test state change listener notification."""
        transitions = []

        def listener(old_state, new_state):
            transitions.append((old_state, new_state))

        self.state_machine.add_state_change_listener(listener)

        # Force state change
        old = self.state_machine.current
        self.state_machine.force_state(GameState.GREEN_LIGHT)

        assert len(transitions) == 1
        assert transitions[0][0] == old
        assert transitions[0][1] == GameState.GREEN_LIGHT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

