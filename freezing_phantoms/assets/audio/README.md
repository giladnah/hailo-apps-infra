# Audio Assets

Place your audio files in this directory for the Freezing Phantoms game.

## Required Audio Files

### 1. green_light_music.mp3
**Purpose**: Background music during the "move" phase
**Description**: Spooky, upbeat music that creates atmosphere while players can move
**Duration**: 10-30 seconds (will loop)
**Format**: MP3
**Mood**: Halloween-themed, playful, slightly eerie

### 2. freeze_sound.wav
**Purpose**: "Freeze!" command sound (RED LIGHT stage)
**Description**: Sharp, attention-grabbing sound when players must freeze
**Duration**: 1-3 seconds
**Format**: WAV
**Examples**: Witch cackle, ghost sound, sudden music stop with alarm
**Current**: Submarine Sonar Sound Effect - Underwater Ping (creates eerie freeze atmosphere)

### 3. caught_buzzer.wav
**Purpose**: Penalty/caught sound effect
**Description**: Sound when a player is caught moving
**Duration**: 1-2 seconds
**Format**: WAV
**Examples**: Buzzer, error sound, "gotcha" sound, ghost laugh

### 4. game_start.wav
**Purpose**: Countdown/game start sound
**Description**: Sound played during the initial countdown
**Duration**: 2-3 seconds
**Format**: WAV
**Examples**: "Get Ready!", countdown beeps, spooky announcement

## Sources for Royalty-Free Sound Effects

### Free Resources
- **Freesound.org**: https://freesound.org/ (requires attribution)
- **Zapsplat**: https://www.zapsplat.com/ (free with account)
- **Mixkit**: https://mixkit.co/free-sound-effects/ (royalty-free)
- **BBC Sound Effects**: https://sound-effects.bbcrewind.co.uk/ (free for personal use)

### Search Terms
- Halloween sounds
- Spooky music
- Game sound effects
- Freeze sound
- Alert buzzer
- Witch cackle
- Ghost sounds

## Format Recommendations

### For Music (green_light_music.mp3)
- **Format**: MP3 (smaller file size)
- **Bitrate**: 128-192 kbps
- **Sample Rate**: 44.1 kHz
- **Channels**: Stereo

### For Sound Effects (*.wav)
- **Format**: WAV (better for short, precise sounds)
- **Bitrate**: 16-bit
- **Sample Rate**: 44.1 kHz
- **Channels**: Mono or Stereo

## Testing Your Audio

After placing audio files in this directory, test them:

```bash
cd /path/to/freezing_phantoms
python -c "from core.audio_manager import AudioManager; from pathlib import Path; am = AudioManager(Path('assets')); am.play_game_start()"
```

## Current Audio Files

### freeze_sound.wav
- **Source**: YouTube - Submarine Sonar Sound Effect - Underwater Ping
- **URL**: https://www.youtube.com/watch?v=JCLg_jSs6Tg
- **Added**: October 31, 2025
- **Usage**: RED LIGHT stage - creates an eerie, underwater detection atmosphere

## Notes

- If audio files are missing, the game will still run but without sound
- Check the terminal for warnings about missing audio files
- Adjust volume in the game configuration or using system volume controls
- Consider the age of players when selecting sound intensity (not too scary for young children)

