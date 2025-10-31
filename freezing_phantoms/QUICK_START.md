# Quick Start Guide - Freezing Phantoms

## For Users (Running the Game)

### Installation
```bash
cd freezing_phantoms
./setup.sh
```

### Running
```bash
# Run calibration first (one time)
python main.py --run-calibration --input /dev/video0

# Start the game
python main.py --input /dev/video0

# Debug mode with video display
python main.py --input /dev/video0 --show-video
```

## For Developers

### Initial Setup
```bash
# Install dependencies
make install-dev

# Set up pre-commit hooks
make setup-hooks
```

### Development Workflow
```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type-check

# Run tests
make test

# Run tests with coverage
make test-cov

# Run all checks (like CI)
make ci
```

### Adding New Features

1. Create a feature branch
2. Make changes
3. Add tests
4. Run `make format` and `make lint`
5. Run `make test`
6. Commit (pre-commit hooks run automatically)
7. Push and create PR

## New Features Available

### 1. Configuration Validation
```python
from freezing_phantoms.core.config_validator import load_and_validate_config

config = load_and_validate_config(Path("config/game_config.yaml"))
```

### 2. Performance Monitoring
```python
from freezing_phantoms.core.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_frame()
# ... game loop ...
print(monitor.get_report())
```

### 3. Better Error Handling
```python
from freezing_phantoms.core.exceptions import CalibrationError

try:
    calibrate()
except CalibrationError as e:
    print(f"Calibration failed: {e.message}")
```

### 4. Constants
```python
from freezing_phantoms.core.constants import (
    DEFAULT_VIDEO_DEVICE,
    DEFAULT_MOVEMENT_THRESHOLD_PX,
    Colors,
    KeyboardKeys,
)
```

## Common Commands

### Development
| Command            | Description                      |
| ------------------ | -------------------------------- |
| `make help`        | Show all available commands      |
| `make install`     | Install game dependencies        |
| `make install-dev` | Install development dependencies |
| `make format`      | Format code with black and isort |
| `make lint`        | Run linters (flake8)             |
| `make type-check`  | Run type checking (mypy)         |
| `make security`    | Run security checks (bandit)     |
| `make test`        | Run unit tests                   |
| `make test-cov`    | Run tests with coverage          |
| `make ci`          | Run all CI checks locally        |
| `make clean`       | Remove generated files           |

### Running
| Command          | Description         |
| ---------------- | ------------------- |
| `make run`       | Run the game        |
| `make run-debug` | Run with debug mode |
| `make calibrate` | Run calibration     |

## File Structure

```
freezing_phantoms/
├── core/                       # Core game components
│   ├── constants.py           # ⭐ NEW: Constants
│   ├── config_validator.py   # ⭐ NEW: Config validation
│   ├── exceptions.py          # ⭐ NEW: Error handling
│   ├── cli.py                 # ⭐ NEW: CLI utilities
│   ├── performance_monitor.py # ⭐ NEW: Performance tracking
│   ├── audio_manager.py
│   ├── game_state.py
│   ├── movement_detector.py
│   └── projection_renderer.py
├── pipeline/                   # AI pipeline
├── calibration/                # Calibration system
├── effects/                    # Visual effects
├── tests/                      # Unit tests
├── config/                     # Configuration
├── assets/                     # Game assets
├── .pre-commit-config.yaml    # ⭐ NEW: Pre-commit hooks
├── pyproject.toml             # ⭐ NEW: Tool config
├── Makefile                   # ⭐ NEW: Dev commands
├── requirements.txt           # Game dependencies
├── requirements-dev.txt       # ⭐ NEW: Dev dependencies
├── CONTRIBUTING.md            # ⭐ NEW: Dev guide
├── CHANGELOG.md               # ⭐ NEW: Version history
├── IMPROVEMENTS_SUMMARY.md    # ⭐ NEW: Improvements
├── REVIEW_SUMMARY.md          # ⭐ NEW: Review summary
├── QUICK_START.md             # ⭐ NEW: This file
└── main.py                    # Main entry point
```

## Documentation

- **README.md** - Main documentation
- **CONTRIBUTING.md** - Development guidelines
- **CHANGELOG.md** - Version history
- **IMPROVEMENTS_SUMMARY.md** - Detailed improvements
- **REVIEW_SUMMARY.md** - Code review summary
- **QUICK_START.md** - This file

## Getting Help

1. Check README.md for user documentation
2. Check CONTRIBUTING.md for development guidelines
3. Check existing code examples
4. Open an issue on GitHub

## Tips

### First Time Setup
```bash
cd freezing_phantoms
make install-dev
make setup-hooks
make test  # Verify everything works
```

### Before Committing
```bash
make format  # Format code
make lint    # Check for issues
make test    # Run tests
```

### Running CI Checks Locally
```bash
make ci  # Runs all checks that CI will run
```

### Performance Profiling
```bash
python main.py --input /dev/video0 --enable-metrics
```

## What Changed?

### Added ✅
- Constants management
- Configuration validation
- Performance monitoring
- Better error handling
- Pre-commit hooks
- CI/CD pipeline
- Makefile for commands
- Comprehensive documentation

### Unchanged ✨
- All existing functionality
- Game behavior
- Configuration format
- Command-line interface (enhanced, not changed)

## Next Steps

1. **Try it out**: Run `make test` to verify setup
2. **Read docs**: Check CONTRIBUTING.md for details
3. **Explore code**: See new utilities in `core/`
4. **Start developing**: Use `make` commands for workflow

---

**Need help?** Check the documentation files or open an issue!

