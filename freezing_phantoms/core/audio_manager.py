"""
Audio manager for Freezing Phantoms game.

Handles all game audio using pygame mixer.
"""

from pathlib import Path
from typing import Dict, Optional

import pygame.mixer

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from freezing_phantoms.core.game_state import GameState

hailo_logger = get_logger(__name__)


class AudioManager:
    """
    Manages game audio playback.

    Attributes:
        assets_path (Path): Path to audio assets directory.
        sounds (Dict[str, pygame.mixer.Sound]): Loaded sound effects.
        music_channel (pygame.mixer.Channel): Channel for background music.
    """

    def __init__(self, assets_path: Path):
        """
        Initialize audio manager.

        Args:
            assets_path (Path): Path to assets directory containing audio files.
        """
        self.assets_path = assets_path / "audio"
        self.sounds = {}
        self.music_channel = None
        self.freeze_sound_channel = None
        self.enabled = True

        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            hailo_logger.info("Pygame mixer initialized")
        except Exception as e:
            hailo_logger.error(f"Failed to initialize pygame mixer: {e}")
            self.enabled = False
            return

        # Load audio files
        self._load_audio_files()

    def _load_audio_files(self) -> None:
        """Load all audio files from assets directory."""
        audio_files = {
            "green_light_music": "green_light_music.mp3",
            "freeze_sound": "freeze_sound.wav",
            "caught_buzzer": "caught_buzzer.wav",
            "game_start": "game_start.wav",
            "win_sound": "Epic Stock Media - Undead Character - You Win.wav",
            "lose_sound": "Epic Stock Media - Undead Character - You Lose.wav",
        }

        for sound_name, filename in audio_files.items():
            filepath = self.assets_path / filename

            if filepath.exists():
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(str(filepath))
                    hailo_logger.info(f"Loaded audio: {sound_name} from {filename}")
                except Exception as e:
                    hailo_logger.warning(f"Failed to load {filename}: {e}")
            else:
                hailo_logger.warning(
                    f"Audio file not found: {filepath}. "
                    f"Game will run without this sound."
                )

    def play_state_audio(self, state: GameState) -> None:
        """
        Play appropriate audio for game state.

        Args:
            state (GameState): Game state to play audio for.
        """
        if not self.enabled:
            hailo_logger.warning(f"Audio disabled, skipping audio for state: {state.value}")
            return

        hailo_logger.info(f"🔊 Triggering audio for state: {state.value}")

        if state == GameState.COUNTDOWN:
            self.play_game_start()
        elif state == GameState.GREEN_LIGHT:
            self.play_green_light_music()
        elif state == GameState.RED_LIGHT:
            self.play_freeze_sound()
        elif state == GameState.PENALTY:
            self.play_caught_buzzer()
        elif state == GameState.WIN:
            self.play_win_sound()
        elif state == GameState.GAME_OVER:
            self.play_lose_sound()

    def play_green_light_music(self) -> None:
        """Play green light background music (looping)."""
        if not self.enabled or "green_light_music" not in self.sounds:
            return

        self.stop_music()

        try:
            # Play on a dedicated channel with loop
            self.music_channel = self.sounds["green_light_music"].play(loops=-1)
            hailo_logger.debug("Playing green light music")
        except Exception as e:
            hailo_logger.error(f"Failed to play green light music: {e}")

    def play_freeze_sound(self) -> None:
        """Play freeze sound effect (stops music)."""
        if not self.enabled or "freeze_sound" not in self.sounds:
            return

        self.stop_music()

        try:
            self.freeze_sound_channel = self.sounds["freeze_sound"].play()
            hailo_logger.debug("Playing freeze sound")
        except Exception as e:
            hailo_logger.error(f"Failed to play freeze sound: {e}")

    def play_caught_buzzer(self) -> None:
        """Play caught/penalty sound effect. Stops all other audio first."""
        if not self.enabled or "caught_buzzer" not in self.sounds:
            return

        # Stop all audio before playing buzzer
        self.stop_all()

        try:
            self.sounds["caught_buzzer"].play()
            hailo_logger.info("🚨 Playing caught buzzer (stopped all other audio)")
        except Exception as e:
            hailo_logger.error(f"Failed to play caught buzzer: {e}")

    def play_game_start(self) -> None:
        """Play game start/countdown sound."""
        if not self.enabled or "game_start" not in self.sounds:
            return

        try:
            self.sounds["game_start"].play()
            hailo_logger.debug("Playing game start sound")
        except Exception as e:
            hailo_logger.error(f"Failed to play game start sound: {e}")

    def play_win_sound(self) -> None:
        """Play win sound effect."""
        if not self.enabled:
            hailo_logger.warning("Audio not enabled, cannot play win sound")
            return
            
        if "win_sound" not in self.sounds:
            hailo_logger.warning("Win sound not loaded")
            return

        self.stop_all()

        try:
            self.sounds["win_sound"].play()
            hailo_logger.info("🎉 Playing WIN sound!")
        except Exception as e:
            hailo_logger.error(f"Failed to play win sound: {e}")

    def play_lose_sound(self) -> None:
        """Play lose/game over sound effect."""
        if not self.enabled:
            hailo_logger.warning("Audio not enabled, cannot play lose sound")
            return
            
        if "lose_sound" not in self.sounds:
            hailo_logger.warning("Lose sound not loaded")
            return

        self.stop_all()

        try:
            self.sounds["lose_sound"].play()
            hailo_logger.info("💀 Playing GAME OVER sound!")
        except Exception as e:
            hailo_logger.error(f"Failed to play lose sound: {e}")

    def stop_music(self) -> None:
        """Stop background music."""
        if self.music_channel and self.music_channel.get_busy():
            self.music_channel.stop()
            hailo_logger.debug("Stopped background music")

    def stop_freeze_sound(self) -> None:
        """Stop freeze sound effect."""
        if self.freeze_sound_channel and self.freeze_sound_channel.get_busy():
            self.freeze_sound_channel.stop()
            hailo_logger.debug("Stopped freeze sound")

    def stop_all(self) -> None:
        """Stop all audio playback."""
        if not self.enabled:
            return

        self.stop_music()
        self.stop_freeze_sound()
        pygame.mixer.stop()
        hailo_logger.debug("Stopped all audio")

    def set_volume(self, volume: float) -> None:
        """
        Set master volume.

        Args:
            volume (float): Volume level (0.0 to 1.0).
        """
        if not self.enabled:
            return

        volume = max(0.0, min(1.0, volume))

        for sound in self.sounds.values():
            sound.set_volume(volume)

        hailo_logger.info(f"Volume set to {volume:.2f}")

    def set_sound_volume(self, sound_name: str, volume: float) -> None:
        """
        Set volume for a specific sound.

        Args:
            sound_name (str): Name of the sound.
            volume (float): Volume level (0.0 to 1.0).
        """
        if not self.enabled or sound_name not in self.sounds:
            return

        volume = max(0.0, min(1.0, volume))
        self.sounds[sound_name].set_volume(volume)
        hailo_logger.debug(f"Set {sound_name} volume to {volume:.2f}")

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if not self.enabled:
            return

        self.stop_all()
        pygame.mixer.quit()
        hailo_logger.info("Audio manager cleaned up")

    def create_placeholder_sounds(self) -> None:
        """
        Create placeholder audio files if they don't exist.

        Note: This creates empty files. Actual audio should be provided by the user.
        """
        self.assets_path.mkdir(parents=True, exist_ok=True)

        placeholder_files = [
            "green_light_music.mp3",
            "freeze_sound.wav",
            "caught_buzzer.wav",
            "game_start.wav",
        ]

        readme_content = """# Audio Assets

Place your audio files in this directory:

- green_light_music.mp3: Spooky, upbeat background music for the "move" phase
- freeze_sound.wav: Sharp "Freeze!" command sound (e.g., witch cackle)
- caught_buzzer.wav: Penalty/caught sound effect (e.g., buzzer or alarm)
- game_start.wav: Countdown/game start sound

You can use royalty-free sound effects from sources like:
- https://freesound.org/
- https://www.zapsplat.com/
- https://mixkit.co/free-sound-effects/

Format recommendations:
- MP3 for music (longer files)
- WAV for sound effects (better for short, precise sounds)
"""

        readme_path = self.assets_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w") as f:
                f.write(readme_content)
            hailo_logger.info(f"Created audio assets README at {readme_path}")

