# Logging Guide for Freezing Phantoms

## Overview

The Freezing Phantoms game uses Hailo's logging system, which supports different log levels to control output verbosity. This guide explains how to enable debug logging to see detailed information from all game components.

## Log Levels

Available log levels (in order of verbosity):
- **DEBUG**: Most verbose - shows all internal operations
- **INFO**: Normal verbosity - shows important events
- **WARNING**: Only warnings and errors
- **ERROR**: Only errors

## How to Enable Debug Logging

### Method 1: Command-Line Argument (Recommended)

Use the `--log-level` flag when running the game:

```bash
python main.py --input /dev/video0 --log-level DEBUG
```

This will show debug messages from all modules including:
- Movement detector
- Game state transitions
- Pipeline initialization
- Data extraction
- Projection rendering

### Method 2: Environment Variable

Set the `LOG_LEVEL` environment variable before running:

```bash
export LOG_LEVEL=DEBUG
python main.py --input /dev/video0
```

Or in one line:

```bash
LOG_LEVEL=DEBUG python main.py --input /dev/video0
```

## What Debug Logging Shows

### Movement Detector Debug Messages

When enabled, you'll see messages like:

```
23:45:12 | DEBUG | freezing_phantoms.core.movement_detector | Empty pose data provided
23:45:13 | DEBUG | freezing_phantoms.core.movement_detector | Movement detected: left_wrist moved 25.3px (threshold: 20.0px)
23:45:14 | DEBUG | freezing_phantoms.core.movement_detector | Track ID 2 not found in initial state (new player)
```

### Game State Debug Messages

```
23:45:10 | DEBUG | freezing_phantoms.core.game_state | Timer countdown: 2.5s remaining
23:45:12 | DEBUG | freezing_phantoms.core.game_state | Capturing freeze snapshot for 3 players
```

### Pipeline Debug Messages

```
23:45:00 | DEBUG | freezing_phantoms.pipeline.combined_pipeline | Building pipeline string...
23:45:01 | DEBUG | freezing_phantoms.pipeline.data_extractor | Extracted 2 players from buffer
```

## Debug Logging in Practice

### Example: Debugging Movement Detection

If players are being caught unexpectedly, enable debug mode to see exactly which keypoints are moving:

```bash
python main.py --input /dev/video0 --log-level DEBUG --show-video
```

Output will show:
```
23:45:20 | INFO | freezing_phantoms.core.game_state | State transition: green_light -> red_light
23:45:20 | DEBUG | freezing_phantoms.core.game_state | Capturing freeze snapshot for 2 players
23:45:21 | DEBUG | freezing_phantoms.core.movement_detector | Movement detected: left_wrist moved 25.3px (threshold: 20.0px)
23:45:21 | INFO | __main__ | Player 1 moved! left_wrist displaced 25.3px
```

### Example: Debugging Pipeline Issues

If the AI pipeline isn't detecting players:

```bash
LOG_LEVEL=DEBUG python main.py --input /dev/video0 --show-video
```

Look for messages about:
- Pipeline initialization
- HEF loading
- Data extraction
- Player detection

## Filtering Specific Modules

To see debug messages from only specific modules, you can use `grep`:

```bash
# Only movement detector messages
python main.py --log-level DEBUG 2>&1 | grep movement_detector

# Only game state messages
python main.py --log-level DEBUG 2>&1 | grep game_state

# Only pipeline messages
python main.py --log-level DEBUG 2>&1 | grep pipeline
```

## Performance Impact

Debug logging has minimal performance impact:
- **DEBUG level**: ~1-2% CPU overhead from logging
- **INFO level** (default): < 0.5% overhead
- **WARNING/ERROR**: Negligible

For production gameplay, use INFO level. Use DEBUG only when troubleshooting.

## Log File Output

To save logs to a file while also seeing them on screen:

```bash
python main.py --log-level DEBUG 2>&1 | tee game_debug.log
```

This creates `game_debug.log` with all debug messages for later analysis.

## Common Debug Scenarios

### 1. Players Not Being Detected

```bash
LOG_LEVEL=DEBUG python main.py --input /dev/video0 --show-video
```

Look for: "Extracted X players from buffer"

### 2. False Movement Detection

```bash
python main.py --log-level DEBUG --input /dev/video0
```

Watch for: "Movement detected" messages during red light

### 3. Calibration Issues

```bash
LOG_LEVEL=DEBUG python calibrate.py --video-source /dev/video0
```

Shows circle detection and homography calculation details

### 4. Game State Not Transitioning

```bash
python main.py --log-level DEBUG --input /dev/video0
```

Look for: "State transition" and timer messages

## Environment Variables

The game respects these environment variables:

- `LOG_LEVEL`: Sets default logging level (DEBUG, INFO, WARNING, ERROR)
- `HAILO_LOG_LEVEL`: Alternative Hailo-specific log level
- `GST_DEBUG`: GStreamer pipeline debug level (0-5)

Example with all debug options:

```bash
LOG_LEVEL=DEBUG GST_DEBUG=3 python main.py --input /dev/video0 --show-video --dump-dot
```

## Troubleshooting

### Debug Messages Not Showing

**Problem**: Debug logging enabled but no debug messages appear

**Solution**: Make sure you're using `--log-level DEBUG` not `--show-fps` (different flag)

```bash
# Wrong
python main.py --show-fps  # This is for FPS display, not logging

# Correct
python main.py --log-level DEBUG
```

### Too Much Output

**Problem**: Debug logging is overwhelming

**Solutions**:
1. Filter specific modules with `grep`
2. Use INFO level with specific debug calls:
   ```bash
   python main.py --log-level INFO 2>&1 | grep -E "(WARNING|ERROR|movement)"
   ```
3. Save to file and analyze later:
   ```bash
   python main.py --log-level DEBUG 2>&1 > debug.log
   ```

### Module-Specific Debug Not Working

**Problem**: Movement detector debug not showing even with `--log-level DEBUG`

**Root Cause**: The log level flag is processed after modules are imported, so their loggers are already created with the default level.

**Solution**: Use environment variable instead:
```bash
export LOG_LEVEL=DEBUG
python main.py --input /dev/video0
```

Or in one line:
```bash
LOG_LEVEL=DEBUG python main.py --input /dev/video0
```

## Related Files

- `main.py` - Main application with log level argument
- `core/movement_detector.py` - Movement detection debug messages
- `core/game_state.py` - Game state debug messages
- `pipeline/combined_pipeline.py` - Pipeline debug messages
- `hailo_apps/hailo_app_python/core/common/hailo_logger.py` - Logging system implementation

