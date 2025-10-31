# Movement Detection System

## Overview

The movement detection system is responsible for catching players who move during the "RED LIGHT" (freeze) phase of the game. It uses pose estimation keypoint tracking to detect movement.

## How It Works

### 1. **Baseline Capture (Start of RED LIGHT)**
When RED LIGHT begins, the system captures the current pose of all players as a "frozen" reference state. This includes the positions of all monitored keypoints (wrists, ankles, shoulders).

### 2. **Continuous Comparison (During RED LIGHT)**
Every frame during RED LIGHT, the system:
1. Gets the current pose of each player
2. Compares it to their frozen reference pose
3. Calculates how much each keypoint has moved
4. Determines if any movement exceeds the threshold

### 3. **Movement Calculation**
For each monitored keypoint:
```
displacement = sqrt((x1 - x0)² + (y1 - y0)²)
```
Where:
- `(x0, y0)` = keypoint position at start of RED LIGHT
- `(x1, y1)` = current keypoint position
- `displacement` = Euclidean distance in pixels

### 4. **Threshold Comparison**
Movement is detected if:
```
displacement > (threshold_pixels * sensitivity_multiplier)
```

**Example** (default settings):
- `threshold_pixels = 40`
- `sensitivity_multiplier = 0.8`
- `effective_threshold = 40 * 0.8 = 32 pixels`
- If any keypoint moves more than 32 pixels → **CAUGHT!**

## Monitored Keypoints

The system monitors these 6 body parts:
1. **Left Wrist** - Catches arm movements
2. **Right Wrist** - Catches arm movements
3. **Left Ankle** - Catches leg/foot movements
4. **Right Ankle** - Catches leg/foot movements
5. **Left Shoulder** - Catches torso rotation/leaning
6. **Right Shoulder** - Catches torso rotation/leaning

**Why these keypoints?**
- Easy to detect with pose estimation
- Cover major body parts
- Catch most types of movement (arms, legs, body sway)

## Smoothing & Stability

### Problem: Jittery Detection
Pose estimation isn't perfect - keypoint positions can "jitter" slightly even when the person is perfectly still. This could cause false positives.

### Solution: Exponential Moving Average (EMA)
The system applies smoothing to current poses before comparison:

```python
smoothed_position = last_position + alpha * (current_position - last_position)
```

Where:
- `alpha` = smoothing factor (0-1)
- Lower alpha = more smoothing = more stable
- Higher alpha = less smoothing = more responsive

**Default: `smoothing_alpha = 0.15`** (quite stable)

### Pose History
The system keeps a history of recent poses (default: 8 frames) to improve smoothing quality. This helps filter out noise and jitter.

## Confidence Filtering

Not all keypoints are equally visible in every frame. The system:
1. Checks confidence score for each keypoint
2. Only uses keypoints with confidence ≥ threshold (default: 0.5)
3. Ignores low-confidence detections that might be inaccurate

## Configuration Parameters

### `threshold_pixels` (Most Important)
**What it does:** Sets the base movement threshold in pixels.

**Higher value = Less sensitive** (more forgiving)
- **20px**: Very sensitive - catches small fidgeting
- **40px**: Moderate (default) - catches deliberate movements
- **60px**: Less sensitive - only larger movements
- **80px**: Very forgiving - only obvious movements

**Recommendation:** Start at 40px, adjust based on your camera setup and desired difficulty.

### `sensitivity_multiplier`
**What it does:** Scales the threshold without changing the base value.

**Lower value = Less sensitive**
- **1.0**: Full sensitivity (effective threshold = threshold_pixels)
- **0.8**: 80% sensitivity (default)
- **0.5**: 50% sensitivity (very forgiving)

**Formula:** `effective_threshold = threshold_pixels × sensitivity_multiplier`

### `smoothing_alpha`
**What it does:** Controls how much smoothing is applied.

**Lower value = More stable** (less sensitive to jitter)
- **0.1**: Very smooth (slow to respond)
- **0.15**: Moderate smoothing (default)
- **0.3**: Light smoothing (more responsive)
- **1.0**: No smoothing (instant response, more jitter)

**Recommendation:** Keep at 0.15 for best stability.

### `pose_history_size`
**What it does:** Number of past poses to keep for smoothing.

**Higher value = More stable**
- **5**: Basic smoothing
- **8**: Good smoothing (default)
- **12**: Very smooth

**Recommendation:** 8 provides good balance.

### `keypoint_confidence_threshold`
**What it does:** Minimum confidence to use a keypoint.

- **0.3**: Accept lower-quality detections
- **0.5**: Moderate quality (default)
- **0.7**: Only high-quality detections

**Recommendation:** 0.5 works well for most cases.

## Runtime Controls

During the game, you can adjust sensitivity:

### Keyboard Controls (Terminal)
- **`-` key**: Make LESS sensitive (+10px threshold)
- **`+` key**: Make MORE sensitive (-10px threshold)
- **`S` key**: Show current settings

### Control Range
- **Minimum threshold**: 10 pixels (very sensitive)
- **Maximum threshold**: 100 pixels (very forgiving)
- **Adjustment step**: 10 pixels per keypress

### Example Session
```
Initial: 40px threshold
Press '-' → 50px threshold (less sensitive)
Press '-' → 60px threshold (even less sensitive)
Press '+' → 50px threshold (more sensitive again)
```

## Typical Issues & Solutions

### Issue: Too many false positives (catching players who are still)
**Solution:** Make less sensitive
- Press `-` key during game (increases threshold)
- Or edit config: increase `threshold_pixels` to 50-60
- Or edit config: decrease `sensitivity_multiplier` to 0.6-0.7

### Issue: Not catching obvious movement
**Solution:** Make more sensitive
- Press `+` key during game (decreases threshold)
- Or edit config: decrease `threshold_pixels` to 20-30
- Or edit config: increase `sensitivity_multiplier` to 1.0

### Issue: Jittery detection (inconsistent)
**Solution:** Add more smoothing
- Edit config: decrease `smoothing_alpha` to 0.1
- Edit config: increase `pose_history_size` to 10-12

### Issue: Detection too slow to respond
**Solution:** Reduce smoothing
- Edit config: increase `smoothing_alpha` to 0.2-0.3
- Edit config: decrease `pose_history_size` to 5

## Technical Flow

```
1. RED LIGHT starts
   └─> Capture frozen poses for all players

2. Every frame during RED LIGHT:
   ├─> Get current poses from pipeline
   ├─> Filter keypoints by confidence (≥ 0.5)
   ├─> Apply smoothing to current poses (EMA)
   ├─> Compare to frozen poses
   │   └─> For each monitored keypoint:
   │       ├─> Calculate displacement (Euclidean distance)
   │       └─> Check: displacement > effective_threshold?
   └─> If ANY keypoint moved too much → CAUGHT!

3. Player caught:
   ├─> Add to caught_players list
   ├─> Transition to PENALTY state
   └─> Apply penalty (move back 2 rows)
```

## Debug Information

View current settings with `S` key during game:

```
=== Freeze Detection Settings ===
Threshold: 40.0px
Effective threshold: 32.0px
Sensitivity multiplier: 0.80x
Smoothing alpha: 0.15
Pose history size: 8
=================================
```

## Recommendations by Camera Setup

### High-Resolution Camera (1080p+, close to players)
- Higher threshold (50-60px) - small movements are magnified
- More smoothing (alpha: 0.1) - reduce jitter from high-res details

### Low-Resolution Camera (720p, far from players)
- Lower threshold (30-40px) - need to catch smaller movements
- Less smoothing (alpha: 0.2) - less jitter from lower resolution

### Noisy Camera (low light, compression artifacts)
- More smoothing (alpha: 0.1, history: 10-12)
- Higher confidence threshold (0.6-0.7)
- Slightly higher movement threshold (50px)

### Perfect Lighting, Good Camera
- Moderate settings (defaults work well)
- Can be more sensitive (30-40px threshold)
- Less smoothing needed (alpha: 0.2)

## Files

- **Implementation**: `freezing_phantoms/core/movement_detector.py`
- **Configuration**: `freezing_phantoms/config/game_config.yaml`
- **Usage**: `freezing_phantoms/main.py` (in `_check_for_movement` method)

