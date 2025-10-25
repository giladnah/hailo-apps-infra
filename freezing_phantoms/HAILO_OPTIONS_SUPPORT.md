# Hailo Standard Options Support

## Overview

The Freezing Phantoms game now supports all standard Hailo GStreamer application options from `hailo-apps-infra`. This ensures consistency with other Hailo applications and provides familiar command-line options for users.

## Implementation

### What Was Done

1. **Added Standard Options to main.py**
   - Manually added all options from `get_default_parser()`
   - Avoided parent parser conflicts with help messages
   - All standard options are now available in the game

2. **Updated Pipeline to Accept Options**
   - Modified `CombinedPoseSegmentationPipeline` to accept `hailo_options` parameter
   - Pipeline now respects `--arch`, `--hef-path`, `--disable-sync`, `--show-fps`, `--dump-dot`, etc.
   - Options are properly passed through to GStreamer pipeline

3. **Maintained Backward Compatibility**
   - `--video-source` is kept as an alias for `--input` for convenience
   - Default values are preserved from the original game configuration
   - Game-specific options (calibration, etc.) work alongside standard options

## Supported Standard Options

All options from `get_default_parser()` are supported:

| Option               | Short | Type   | Description                                          |
| -------------------- | ----- | ------ | ---------------------------------------------------- |
| `--input`            | `-i`  | string | Input source (camera device or file path)            |
| `--use-frame`        | `-u`  | flag   | Use frame from callback function                     |
| `--show-fps`         | `-f`  | flag   | Print FPS on sink                                    |
| `--arch`             |       | choice | Hailo architecture (`hailo8`, `hailo8l`, `hailo10h`) |
| `--hef-path`         |       | string | Path to HEF file                                     |
| `--disable-sync`     |       | flag   | Disable display sink synchronization                 |
| `--disable-callback` |       | flag   | Disable user's custom callback                       |
| `--dump-dot`         |       | flag   | Dump pipeline graph to DOT file                      |
| `--frame-rate`       | `-r`  | int    | Frame rate of video source (default: 30)             |

## Game-Specific Options

These options are unique to Freezing Phantoms:

| Option                   | Type   | Description                                |
| ------------------------ | ------ | ------------------------------------------ |
| `--config`               | path   | Path to game configuration file            |
| `--calibration`          | path   | Path to calibration matrix file            |
| `--video-source`         | string | Alias for `--input` (for convenience)      |
| `--run-calibration`      | flag   | Run calibration tool before starting       |
| `--validate-calibration` | flag   | Validate existing calibration              |
| `--show-video`           | flag   | Show video output with AI overlays (debug) |

## Usage Examples

### Basic Usage with Standard Options

```bash
# Use --input (standard Hailo option)
python main.py --input /dev/video0

# Specify architecture
python main.py --input /dev/video2 --arch hailo8

# Custom frame rate
python main.py --input /dev/video0 --frame-rate 60 --show-fps

# Disable sync for lower latency
python main.py --input /dev/video0 --disable-sync

# Dump pipeline graph for debugging
python main.py --input /dev/video0 --dump-dot
```

### Combining Standard and Game Options

```bash
# Architecture + debug mode
python main.py --input /dev/video2 --arch hailo8 --show-video

# Custom frame rate + calibration
python main.py --input /dev/video0 --frame-rate 30 --run-calibration

# All debug options enabled
python main.py --input /dev/video0 --show-fps --show-video --dump-dot
```

### Backward Compatibility

```bash
# Still works with --video-source
python main.py --video-source /dev/video0

# Mixing old and new options
python main.py --video-source /dev/video0 --arch hailo8
```

## How Options Are Passed Through

1. **Command Line → main.py**
   - All options are parsed by `argparse` in `main()` function
   - Video source resolved: `args.input` or `args.video_source` or default

2. **main.py → FreezingPhantomsGame**
   - Game instance receives `hailo_options=args` parameter
   - Options stored in `self.hailo_options`

3. **FreezingPhantomsGame → CombinedPoseSegmentationPipeline**
   - Options passed during pipeline initialization
   - Pipeline uses options to build GStreamer command

4. **CombinedPoseSegmentationPipeline → GStreamerApp**
   - Options converted to `sys.argv` format
   - Passed to parent class `__init__`
   - GStreamer pipeline respects all flags

## Option Priority

When multiple sources provide values:

1. **Command-line arguments** (highest priority)
2. **Game configuration file** (`game_config.yaml`)
3. **Default values** (lowest priority)

Example:
- Frame rate from `--frame-rate` overrides `game_config.yaml`
- Architecture from `--arch` overrides auto-detection

## Testing

Verified with:

```bash
# Test help message shows all options
python main.py --help

# Test architecture option
python main.py --input /dev/video2 --arch hailo8 --show-fps

# Output confirms:
# - "Using specified architecture: hailo8"
# - "Using Hailo architecture: hailo8"
# - Pipeline runs successfully
```

## Benefits

1. **Consistency**: Same options as other Hailo apps
2. **Familiarity**: Users don't need to learn new flags
3. **Flexibility**: All GStreamer features accessible
4. **Debugging**: Standard debugging options (`--dump-dot`, `--show-fps`)
5. **Compatibility**: Works with existing Hailo workflows and scripts

## Future Enhancements

Potential improvements:

1. Support for custom post-processing SO files
2. Multiple HEF paths (separate pose and segmentation models)
3. Advanced GStreamer pipeline customization options
4. Configuration file overrides for all standard options

## Related Files

- `freezing_phantoms/main.py` - Argument parsing and option handling
- `freezing_phantoms/pipeline/combined_pipeline.py` - Pipeline option forwarding
- `hailo_apps/hailo_app_python/core/common/core.py` - Standard options definition
- `freezing_phantoms/README.md` - User documentation

