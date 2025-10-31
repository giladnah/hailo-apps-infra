# Makefile for Freezing Phantoms development tasks

.PHONY: help install install-dev test lint format type-check security clean run calibrate docs

# Default target
help:
	@echo "Freezing Phantoms - Development Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install game dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make setup-hooks      Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make format           Format code with black and isort"
	@echo "  make lint             Run all linters"
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make security         Run security checks with bandit"
	@echo "  make test             Run unit tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Running:"
	@echo "  make run              Run the game (requires video device)"
	@echo "  make run-debug        Run in debug mode with video display"
	@echo "  make calibrate        Run calibration tool"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove generated files"
	@echo "  make docs             Generate documentation"
	@echo ""

# Installation
install:
	@echo "Installing game dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup-hooks:
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Code formatting
format:
	@echo "Formatting code with black..."
	black . --line-length 100
	@echo "Sorting imports with isort..."
	isort . --profile black --line-length 100
	@echo "✓ Code formatting complete"

# Linting
lint:
	@echo "Running flake8..."
	flake8 . --max-line-length=100 --extend-ignore=E203,W503 --exclude=__pycache__,*.pyc,.git,calibration_history
	@echo "✓ Linting complete"

# Type checking
type-check:
	@echo "Running mypy type checker..."
	mypy . --ignore-missing-imports --no-strict-optional
	@echo "✓ Type checking complete"

# Security checks
security:
	@echo "Running bandit security checks..."
	bandit -r . -c pyproject.toml
	@echo "✓ Security check complete"

# Testing
test:
	@echo "Running unit tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Run all quality checks
check-all: format lint type-check security test
	@echo "✓ All checks passed!"

# Running the game
run:
	@echo "Starting game..."
	python main.py --input /dev/video0

run-debug:
	@echo "Starting game in debug mode..."
	python main.py --input /dev/video0 --show-video --log-level DEBUG

calibrate:
	@echo "Running calibration..."
	python main.py --run-calibration --input /dev/video0

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation is in README.md and CONTRIBUTING.md"

# Cleaning
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "✓ Cleanup complete"

# CI simulation (run all checks like CI would)
ci: clean install-dev
	@echo "Running CI checks..."
	@make format
	@make lint
	@make type-check
	@make security
	@make test-cov
	@echo "✓ All CI checks passed!"

