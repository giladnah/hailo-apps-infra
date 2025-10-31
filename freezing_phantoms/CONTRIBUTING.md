# Contributing to Freezing Phantoms

Thank you for considering contributing to Freezing Phantoms! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hailo-apps-infra/freezing_phantoms
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install game dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code quality checks before each commit.

## Code Quality Standards

We maintain high code quality standards using automated tools.

### Code Formatting

- **Black**: Code formatter (line length: 100)
  ```bash
  black .
  ```

- **isort**: Import sorting
  ```bash
  isort .
  ```

### Linting

- **flake8**: Style guide enforcement
  ```bash
  flake8 .
  ```

- **mypy**: Type checking
  ```bash
  mypy . --ignore-missing-imports
  ```

- **bandit**: Security checks
  ```bash
  bandit -r . -c pyproject.toml
  ```

### Running All Checks

```bash
# Run pre-commit on all files
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_game_state.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use pytest fixtures for setup/teardown
- Aim for >80% code coverage

Example test:

```python
def test_game_state_transition():
    """Test that game state transitions correctly."""
    config = {...}
    state_machine = GameStateMachine(config)

    # Test initial state
    assert state_machine.current == GameState.COUNTDOWN

    # Test transition
    state_machine.update(5.0, None)
    assert state_machine.current == GameState.GREEN_LIGHT
```

## Code Style Guide

### Python Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for function signatures
- Use docstrings (Google style) for all public functions/classes
- Use f-strings for string formatting

### Docstring Format

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Short description of function.

    Longer description if needed, explaining what the function does,
    any algorithms used, etc.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: Description of when this error is raised.
    """
    pass
```

### Import Organization

Organize imports in the following order:
1. Standard library imports
2. Third-party imports
3. Local application imports

Use isort to automatically organize imports:
```bash
isort .
```

## Git Workflow

### Branches

- `main`: Stable release branch
- `dev`: Development branch
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches

### Commit Messages

Follow conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(calibration): add automatic retry on calibration failure

Add retry mechanism to calibration tool that automatically attempts
up to 3 calibrations if the first attempt fails or is rejected.

Closes #123
```

### Pull Request Process

1. Create a feature branch from `dev`
2. Make your changes with clear commit messages
3. Write/update tests as needed
4. Ensure all tests pass
5. Run pre-commit hooks and fix any issues
6. Update documentation if needed
7. Submit a pull request to `dev`
8. Address review comments

## Project Structure

```
freezing_phantoms/
├── core/               # Core game components
│   ├── audio_manager.py
│   ├── game_state.py
│   ├── movement_detector.py
│   ├── projection_renderer.py
│   ├── config_validator.py
│   ├── constants.py
│   ├── exceptions.py
│   ├── performance_monitor.py
│   └── cli.py
├── pipeline/           # AI pipeline components
│   ├── combined_pipeline.py
│   └── data_extractor.py
├── calibration/        # Calibration system
│   ├── calibration_tool.py
│   └── pattern_generator.py
├── effects/            # Visual effects
│   ├── green_light_effects.py
│   ├── red_light_effects.py
│   └── board_effects.py
├── tests/              # Unit tests
├── config/             # Configuration files
├── assets/             # Game assets
└── main.py             # Main entry point
```

## Adding New Features

### 1. Core Game Logic

Add new game logic to appropriate modules in `core/`:
- State management: `game_state.py`
- Movement detection: `movement_detector.py`
- Rendering: `projection_renderer.py`

### 2. Visual Effects

Add new effects to `effects/`:
- Green light effects: `green_light_effects.py`
- Red light effects: `red_light_effects.py`
- Board effects: `board_effects.py`

### 3. Configuration

Add new configuration options to:
1. `config/game_config.yaml` - Default values
2. `core/config_validator.py` - Validation logic
3. `core/constants.py` - Constants

### 4. Tests

Add tests for new features:
1. Create test file in `tests/`
2. Write unit tests
3. Ensure >80% coverage for new code

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Use type hints
- Add inline comments for complex logic

### User Documentation

Update `README.md` when:
- Adding new features
- Changing installation/setup process
- Changing command-line options
- Adding new dependencies

## Performance Considerations

- Profile code for performance bottlenecks
- Optimize rendering pipeline
- Minimize AI pipeline latency
- Keep frame processing under 33ms (30 FPS target)

Use the performance monitor:
```python
from freezing_phantoms.core.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_frame()
# ... game logic ...
print(monitor.get_report())
```

## Debugging

### Debug Mode

Run with debug mode to see video output:
```bash
python main.py --show-video --log-level DEBUG
```

### Performance Metrics

Enable performance metrics:
```bash
python main.py --enable-metrics
```

### Logging

Adjust log level for more details:
```bash
python main.py --log-level DEBUG
```

## Questions?

If you have questions or need help:
1. Check existing documentation
2. Look at existing code examples
3. Open an issue on GitHub
4. Contact the maintainers

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build something awesome together!

