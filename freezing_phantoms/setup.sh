#!/bin/bash
# Setup script for Freezing Phantoms game

set -e

echo "=================================================="
echo "Freezing Phantoms - Setup Script"
echo "=================================================="
echo ""

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not running in a virtual environment"
    echo "   It's recommended to activate a virtual environment first."
    echo ""
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Python dependencies installed"
echo ""

# Create asset directories
echo "📁 Creating asset directories..."
mkdir -p assets/audio
mkdir -p assets/textures
mkdir -p config

echo "✅ Asset directories created"
echo ""

# Check for Hailo installation
echo "🔍 Checking for Hailo installation..."
if ! python -c "import hailo" 2>/dev/null; then
    echo "⚠️  Warning: Hailo Python bindings not found"
    echo "   Please ensure Hailo Apps Infrastructure is installed."
    echo "   Run: cd .. && ./install.sh"
else
    echo "✅ Hailo installation found"
fi
echo ""

# Check for camera
echo "🎥 Checking for camera devices..."
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✅ Camera devices found:"
    ls -l /dev/video*
else
    echo "⚠️  Warning: No camera devices found at /dev/video*"
fi
echo ""

# Check for GStreamer
echo "🎬 Checking for GStreamer..."
if command -v gst-launch-1.0 &> /dev/null; then
    echo "✅ GStreamer found"
else
    echo "⚠️  Warning: GStreamer not found"
    echo "   Please install GStreamer: sudo apt-get install gstreamer1.0-tools"
fi
echo ""

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Place audio files in assets/audio/ (see assets/audio/README.md)"
echo "2. Set up your camera and projector hardware"
echo "3. Run calibration: python main.py --run-calibration"
echo "4. Start the game: python main.py --video-source /dev/video0"
echo ""
echo "For more information, see README.md"
echo ""

