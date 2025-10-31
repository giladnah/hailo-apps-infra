# Movement Detector Refactoring Summary

## Date: October 31, 2025

## Problem
The velocity-based movement detector was producing false positives with incorrect velocity calculations. The main issues were:

1. **Transition Artifacts**: When transitioning from green light to red light, the velocity calculation looked back at a 0.3s window that included movement from the green light period, causing false detections with velocities like 1000+ px/s
2. **Complexity**: The code used a rolling time window approach with velocity calculations that was overly complex
3. **Noisy Input**: The input was noisy and required better smoothing

## Solution
Completely rewrote the movement detector with a simpler, more robust approach:

### New Approach: Displacement-Based Detection

Instead of calculating velocities over a time window, the new system:

1. **Smooths positions** using Exponential Moving Average (EMA)
2. **Stores reference pose** when red light starts (via `set_reference_pose()`)
3. **Bootstrap period** (1 second) to allow reference pose to stabilize
4. **Calculates displacement** from reference position
5. **Detects movement** when displacement exceeds threshold for consecutive frames

### Key Changes

#### MovementDetector class (`core/movement_detector.py`)
- Removed complex velocity calculation with rolling time windows
- Removed history storage (`pose_history_timestamped` deque)
- Added `set_reference_pose()` method to store frozen position when red light starts
- Added `calculate_displacement()` method to measure movement from reference
- Simplified data structures:
  - `smoothed_pose`: Current smoothed positions
  - `reference_pose`: Frozen reference positions
  - `frame_count`: Warm-up counter
  - `consecutive_violations`: Consecutive frame counter

#### Main Game Loop (`main.py`)
- Added state transition detection to call `set_reference_pose()` when red light starts
- Updated terminology from "velocity" to "displacement" in logs and UI
- Updated sensitivity controls to adjust displacement threshold (5-50 px instead of 50-300 px/s)
- Fixed stats display to show new metrics

### Parameters (Clean and Simple)
- **Displacement threshold**: 100 pixels (default, adjustable 20-300 px with +/- keys)
- **Smoothing alpha**: 0.3 (responsive but smooth)
- **Min consecutive frames**: 3 (strict detection)
- **Keypoint confidence**: 0.3 (accepts more keypoints for robustness)
- **Bootstrap duration**: 1.0 second (smoothing period after freeze starts)
- **Warm-up period**: 10 frames before detection starts
- **Red light duration**: 4-6 seconds (base), scales 1.0x-2.0x with progress (longer at later stages)

### Benefits
1. **No transition artifacts**: Reference pose is set when red light starts, so no contamination from previous movement
2. **Simpler code**: ~200 lines of code removed, easier to understand and maintain
3. **More robust**: Displacement-based detection is more stable than velocity calculations
4. **Better smoothing**: EMA smoothing handles noisy input well
5. **Clean API**: No backwards compatibility cruft, parameter names match their purpose

## Testing Recommendations
1. Test that movement detection works correctly during red light
2. Verify no false positives when player stands still (natural breathing/sway)
3. Test that small movements (< 100px) don't trigger detection
4. Test that large movements (> 100px) are caught reliably
5. Verify smooth transitions between game states
6. Test the +/- keys to adjust sensitivity (20-300 px range)

## Configuration
The displacement threshold can be adjusted in `config/game_config.yaml` or using the +/- keys during gameplay:
- **'S' key**: Shows current settings in terminal
- **'+' key**: Increase sensitivity (lower threshold by 10px, min 20px)
- **'-' key**: Decrease sensitivity (higher threshold by 10px, max 300px)
- Default: 100 pixels (very forgiving, only catches significant movements)

