"""
Custom exceptions for Freezing Phantoms.

Provides specific exception types for better error handling.
"""

from typing import Optional


class FreezingPhantomsError(Exception):
    """Base exception for all Freezing Phantoms errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        """
        Initialize error with message and optional details.

        Args:
            message: Error message.
            details: Additional details about the error.
        """
        self.message = message
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with details."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class CalibrationError(FreezingPhantomsError):
    """Raised when calibration fails."""


class PipelineError(FreezingPhantomsError):
    """Raised when pipeline initialization or execution fails."""


class ConfigurationError(FreezingPhantomsError):
    """Raised when configuration is invalid."""


class VideoSourceError(FreezingPhantomsError):
    """Raised when video source cannot be accessed."""


class HardwareError(FreezingPhantomsError):
    """Raised when hardware (Hailo device, camera, etc.) is not available."""


class RenderingError(FreezingPhantomsError):
    """Raised when rendering fails."""


class DataExtractionError(FreezingPhantomsError):
    """Raised when data extraction from AI pipeline fails."""


class AudioError(FreezingPhantomsError):
    """Raised when audio playback fails."""


def handle_error(error: Exception, logger, context: str = "") -> None:
    """
    Handle error with appropriate logging and user-friendly message.

    Args:
        error: Exception that occurred.
        logger: Logger instance.
        context: Context where error occurred.
    """
    error_type = type(error).__name__

    if isinstance(error, FreezingPhantomsError):
        logger.error(
            "%s in %s: %s",
            error_type,
            context,
            error.message
        )
        if error.details:
            logger.debug("Error details: %s", error.details)
    else:
        logger.error(
            "%s in %s: %s",
            error_type,
            context,
            str(error),
            exc_info=True
        )

