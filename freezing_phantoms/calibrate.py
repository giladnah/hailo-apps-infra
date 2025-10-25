#!/usr/bin/env python3
"""
Standalone calibration tool for Freezing Phantoms.

Run this independently from the game to calibrate camera-projector alignment.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from calibration.calibration_tool import CalibrationTool


def main():
    """Standalone calibration entry point."""
    parser = argparse.ArgumentParser(
        description="Freezing Phantoms - Standalone Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run calibration with default camera
  python calibrate.py

  # Run calibration with specific camera
  python calibrate.py --video-source /dev/video0

  # Validate existing calibration
  python calibrate.py --validate

  # Specify custom output location
  python calibrate.py --output /path/to/calibration.npy
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "game_config.yaml",
        help="Path to configuration file (default: config/game_config.yaml)",
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default="/dev/video0",
        help="Video source: device path (/dev/videoX) or file path (default: /dev/video0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "config" / "calibration.npy",
        help="Output path for calibration matrix (default: config/calibration.npy)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing calibration instead of creating new one",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help="Minimum number of points required for calibration (default: 20)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FREEZING PHANTOMS - CAMERA-PROJECTOR CALIBRATION")
    print("=" * 70)
    print()

    # Validate mode
    if args.validate:
        if not args.output.exists():
            print(f"❌ Error: Calibration file not found at {args.output}")
            print(f"   Please run calibration first without --validate flag")
            sys.exit(1)

        print(f"📊 Loading calibration from: {args.output}")
        homography = np.load(args.output)

        print(f"🎥 Video source: {args.video_source}")
        print()

        tool = CalibrationTool(args.config)
        tool.validate_calibration(homography, args.video_source)

        print()
        print("✅ Validation complete")
        return

    # Calibration mode
    print("Configuration:")
    print(f"  📁 Config file: {args.config}")
    print(f"  🎥 Video source: {args.video_source}")
    print(f"  💾 Output file: {args.output}")
    print(f"  🎯 Min points: {args.min_points}")
    print()

    if not args.config.exists():
        print(f"❌ Error: Configuration file not found at {args.config}")
        sys.exit(1)

    tool = CalibrationTool(args.config)

    # Run calibration with retry logic
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"📸 Calibration attempt {attempt}/{max_attempts}")
        print()

        homography = tool.run_calibration(
            camera_source=args.video_source,
            output_path=args.output,
            min_required_points=args.min_points,
        )

        if homography is not None:
            print()
            print("=" * 70)
            print("✅ CALIBRATION SUCCESSFUL!")
            print("=" * 70)
            print(f"📊 Calibration matrix saved to: {args.output}")
            print()
            print("Next steps:")
            print("  1. Test calibration: python calibrate.py --validate")
            print("  2. Start the game: python main.py --video-source " + args.video_source)
            print()
            return

        if attempt < max_attempts:
            print()
            print(f"⚠️  Calibration failed. Retrying ({attempt}/{max_attempts})...")
            print("   Tips:")
            print("   - Ensure better lighting")
            print("   - Move camera closer or adjust angle")
            print("   - Ensure pattern is fully visible and in focus")
            print()
            input("Press ENTER to retry or Ctrl+C to cancel...")
            print()

    print()
    print("=" * 70)
    print("❌ CALIBRATION FAILED")
    print("=" * 70)
    print("Could not detect enough calibration points after", max_attempts, "attempts")
    print()
    print("Troubleshooting:")
    print("  • Check camera focus and lighting")
    print("  • Ensure calibration pattern is fully visible")
    print("  • Try adjusting camera position/angle")
    print("  • Verify projector brightness is sufficient")
    print()
    sys.exit(1)


if __name__ == "__main__":
    main()

