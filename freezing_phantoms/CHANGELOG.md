# Changelog

All notable changes to Freezing Phantoms will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-10-31

### Added

#### Code Quality & Development Tools
- **Pre-commit hooks** (.pre-commit-config.yaml) for automated code quality checks
- **CI/CD pipeline** (.github/workflows/ci.yml) for automated testing and linting
- **Makefile** with common development tasks (test, lint, format, etc.)
- **pyproject.toml** for tool configuration (black, isort, mypy, bandit, pytest)
- **Development requirements** (requirements-dev.txt) with testing and linting tools
- **Contributing guidelines** (CONTRIBUTING.md) with detailed development workflow
- **Gitignore file** for proper version control

#### Core Improvements
- **Constants module** (`core/constants.py`) - Centralized configuration constants
  - Eliminated hardcoded values throughout codebase
  - Type-safe constants using `Final` type hints
  - Organized by category (defaults, colors, exit codes, etc.)

- **Configuration validation** (`core/config_validator.py`)
  - Comprehensive YAML config validation
  - Schema validation with detailed error messages
  - File size and format checks
  - Default config generation utility

- **Performance monitoring** (`core/performance_monitor.py`)
  - Real-time FPS tracking
  - Component-level timing (pipeline, render, movement detection)
  - Frame drop detection and reporting
  - Performance metrics export

- **Enhanced error handling** (`core/exceptions.py`)
  - Custom exception hierarchy
  - Specific error types (CalibrationError, PipelineError, etc.)
  - Better error messages with context

- **CLI utilities** (`core/cli.py`)
  - Improved argument parsing
  - Better help messages and examples
  - Startup banners and user feedback

### Changed

#### Code Quality
- Improved type hints throughout codebase
- Enhanced docstrings (Google style)
- Better code organization and modularity
- Consistent import ordering with isort
- Code formatting with black (100 char line length)

#### Documentation
- More detailed README sections
- Added CONTRIBUTING.md for developers
- Added CHANGELOG.md for version tracking
- Improved inline code documentation

### Fixed
- Linting errors in existing code
- Import organization
- Type hint consistency
- Exception handling patterns

### Development Experience
- **Faster setup** with improved scripts
- **Easier testing** with pytest configuration
- **Better debugging** with enhanced logging
- **Automated checks** with pre-commit hooks
- **Clear contribution guidelines** for new developers

## [1.0.0] - 2024-XX-XX

### Initial Release
- AI-powered pose estimation and instance segmentation
- Projection mapping with camera-projector calibration
- Automated game state machine
- Movement detection with smoothing
- Dynamic visual effects (green light, red light, penalty)
- Audio system with pygame
- Board system with obstacles and ghosts
- Debug mode with video display
- Comprehensive configuration system
- Unit tests for core components

