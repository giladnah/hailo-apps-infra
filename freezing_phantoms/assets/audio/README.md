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
**Duration**: 12 seconds
**Format**: WAV (24-bit, 96kHz, Stereo)
**Examples**: Witch cackle, ghost sound, sudden music stop with alarm
**Current**: Space Launch - Scanner Sonar Radar Beeping (creates sci-fi scanning/detection atmosphere)

### 3. caught_buzzer.wav
**Purpose**: Penalty/caught sound effect (movement detected during RED LIGHT)
**Description**: Sound when a player is caught moving
**Duration**: 2 seconds
**Format**: WAV (24-bit, 96kHz, Stereo)
**Examples**: Buzzer, error sound, "gotcha" sound, ghost laugh
**Current**: Witch Spooky Laugher Cartoonish - playful Halloween penalty sound

### 4. game_start.wav
**Purpose**: Countdown/game start sound
**Description**: Sound played during the initial countdown
**Duration**: 2-3 seconds
**Format**: WAV
**Examples**: "Get Ready!", countdown beeps, spooky announcement
**Status**: ⚠️ Not yet added

### 5. Epic Stock Media - Undead Character - You Win.wav
**Purpose**: Victory sound (WIN state)
**Description**: Sound played when player successfully reaches the goal
**Duration**: ~2 seconds
**Format**: WAV (24-bit, 96kHz, Stereo)
**Current**: Epic Stock Media - Undead Character saying "You Win" ✅

### 6. Epic Stock Media - Undead Character - You Lose.wav
**Purpose**: Game over sound (GAME_OVER state)
**Description**: Sound played when player runs out of lives
**Duration**: ~2 seconds
**Format**: WAV (24-bit, 96kHz, Stereo)
**Current**: Epic Stock Media - Undead Character saying "You Lose" ✅

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
- **Artist**: Cristian Lucchetta
- **Track**: Space Launch - Scanner Sonar Radar Beeping
- **Added**: October 31, 2025
- **Duration**: 12 seconds
- **Usage**: RED LIGHT stage - creates a sci-fi scanning/detection atmosphere with radar beeping

### caught_buzzer.wav
- **Artist**: Vadi Sound
- **Track**: Spooktacular - Witch Spooky Laugher Cartoonish
- **Added**: October 31, 2025
- **Duration**: 2 seconds
- **Usage**: Penalty sound when movement is detected during RED LIGHT - playful Halloween witch laugh

### Epic Stock Media - Undead Character - You Win.wav
- **Artist**: Epic Stock Media
- **Track**: Undead Character - You Win
- **Added**: October 31, 2025
- **Duration**: ~2 seconds
- **Usage**: WIN state - played when player successfully reaches the goal

### Epic Stock Media - Undead Character - You Lose.wav
- **Artist**: Epic Stock Media
- **Track**: Undead Character - You Lose
- **Added**: October 31, 2025
- **Duration**: ~2 seconds
- **Usage**: GAME_OVER state - played when player runs out of lives

## Audio System Integration

The audio system is fully integrated into the game and will automatically play sounds based on game state:

- **COUNTDOWN** → game_start.wav (if available)
- **GREEN_LIGHT** → green_light_music.mp3 (loops, if available)
- **RED_LIGHT** → freeze_sound.wav ✅
- **PENALTY** → caught_buzzer.wav ✅
- **WIN** → You Win.wav ✅
- **GAME_OVER** → You Lose.wav ✅

## Notes

- If audio files are missing, the game will still run but without sound
- Check the terminal for warnings about missing audio files
- Adjust volume in the game configuration or using system volume controls
- Consider the age of players when selecting sound intensity (not too scary for young children)
- The audio manager uses pygame.mixer for playback
- Audio is cleaned up automatically when the game exits

