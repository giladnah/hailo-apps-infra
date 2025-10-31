"""
Performance monitoring utilities for Freezing Phantoms.

Tracks and reports performance metrics during gameplay.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.

    Attributes:
        fps: Frames per second.
        frame_time_ms: Average frame processing time in milliseconds.
        pipeline_latency_ms: AI pipeline latency in milliseconds.
        render_time_ms: Rendering time in milliseconds.
        movement_detection_time_ms: Movement detection time in milliseconds.
        total_frames: Total number of frames processed.
        dropped_frames: Number of dropped frames.
    """
    fps: float = 0.0
    frame_time_ms: float = 0.0
    pipeline_latency_ms: float = 0.0
    render_time_ms: float = 0.0
    movement_detection_time_ms: float = 0.0
    total_frames: int = 0
    dropped_frames: int = 0


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics during game execution.

    Provides real-time performance statistics and can generate reports.
    """

    def __init__(self, history_size: int = 60):
        """
        Initialize performance monitor.

        Args:
            history_size: Number of frames to keep in history for averaging.
        """
        self.history_size = history_size

        # Frame timing
        self.frame_times: deque = deque(maxlen=history_size)
        self.last_frame_time: float = time.time()

        # Component timings
        self.pipeline_times: deque = deque(maxlen=history_size)
        self.render_times: deque = deque(maxlen=history_size)
        self.movement_detection_times: deque = deque(maxlen=history_size)

        # Counters
        self.total_frames: int = 0
        self.dropped_frames: int = 0

        # Timing contexts
        self._current_timers: Dict[str, float] = {}

        # Last print time for print_metrics
        self._last_print_time: float = time.time()

        hailo_logger.debug("PerformanceMonitor initialized with history_size=%d", history_size)

    def start_frame(self) -> None:
        """Mark the start of a new frame."""
        current_time = time.time()
        if self.total_frames > 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)

        self.last_frame_time = current_time
        self.total_frames += 1

    def start_timer(self, name: str) -> None:
        """
        Start a named timer.

        Args:
            name: Timer name (e.g., 'pipeline', 'render', 'movement_detection').
        """
        self._current_timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record the duration.

        Args:
            name: Timer name.

        Returns:
            float: Elapsed time in seconds.
        """
        if name not in self._current_timers:
            hailo_logger.warning("Timer '%s' was not started", name)
            return 0.0

        elapsed = time.time() - self._current_timers[name]
        del self._current_timers[name]

        # Record based on timer name
        if name == "pipeline":
            self.pipeline_times.append(elapsed)
        elif name == "render":
            self.render_times.append(elapsed)
        elif name == "movement_detection":
            self.movement_detection_times.append(elapsed)

        return elapsed

    def record_dropped_frame(self) -> None:
        """Record that a frame was dropped."""
        self.dropped_frames += 1

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            PerformanceMetrics: Current performance statistics.
        """
        metrics = PerformanceMetrics()

        # Calculate FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            metrics.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            metrics.frame_time_ms = avg_frame_time * 1000

        # Calculate component times
        if self.pipeline_times:
            metrics.pipeline_latency_ms = (
                sum(self.pipeline_times) / len(self.pipeline_times) * 1000
            )

        if self.render_times:
            metrics.render_time_ms = (
                sum(self.render_times) / len(self.render_times) * 1000
            )

        if self.movement_detection_times:
            metrics.movement_detection_time_ms = (
                sum(self.movement_detection_times) / len(self.movement_detection_times) * 1000
            )

        # Set counters
        metrics.total_frames = self.total_frames
        metrics.dropped_frames = self.dropped_frames

        return metrics

    def get_fps(self) -> float:
        """
        Get current FPS.

        Returns:
            float: Current frames per second.
        """
        if not self.frame_times:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def print_metrics(self, interval_seconds: float = 5.0) -> None:
        """
        Print performance metrics if enough time has elapsed.

        Args:
            interval_seconds: Minimum interval between prints.
        """
        current_time = time.time()
        if current_time - self._last_print_time >= interval_seconds:
            metrics = self.get_metrics()
            self._print_metrics(metrics)
            self._last_print_time = current_time

    def _print_metrics(self, metrics: PerformanceMetrics) -> None:
        """
        Print performance metrics to console.

        Args:
            metrics: Performance metrics to print.
        """
        hailo_logger.info(
            "Performance Metrics: FPS=%.1f, Frame=%.1fms, "
            "Pipeline=%.1fms, Render=%.1fms, MoveDet=%.1fms, "
            "Frames=%d, Dropped=%d",
            metrics.fps,
            metrics.frame_time_ms,
            metrics.pipeline_latency_ms,
            metrics.render_time_ms,
            metrics.movement_detection_time_ms,
            metrics.total_frames,
            metrics.dropped_frames
        )

    def get_report(self) -> str:
        """
        Generate a detailed performance report.

        Returns:
            str: Formatted performance report.
        """
        metrics = self.get_metrics()

        report_lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            f"FPS:                    {metrics.fps:>10.1f}",
            f"Frame Time:             {metrics.frame_time_ms:>10.1f} ms",
            f"Pipeline Latency:       {metrics.pipeline_latency_ms:>10.1f} ms",
            f"Render Time:            {metrics.render_time_ms:>10.1f} ms",
            f"Movement Detection:     {metrics.movement_detection_time_ms:>10.1f} ms",
            "",
            f"Total Frames:           {metrics.total_frames:>10d}",
            f"Dropped Frames:         {metrics.dropped_frames:>10d}",
            f"Drop Rate:              {self._calculate_drop_rate():>10.2f}%",
            "=" * 60,
        ]

        return "\n".join(report_lines)

    def _calculate_drop_rate(self) -> float:
        """
        Calculate frame drop rate percentage.

        Returns:
            float: Drop rate as percentage.
        """
        if self.total_frames == 0:
            return 0.0
        return (self.dropped_frames / self.total_frames) * 100

    def reset(self) -> None:
        """Reset all metrics and counters."""
        self.frame_times.clear()
        self.pipeline_times.clear()
        self.render_times.clear()
        self.movement_detection_times.clear()
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_frame_time = time.time()
        self._current_timers.clear()
        hailo_logger.debug("PerformanceMonitor reset")

    def export_metrics(self) -> Dict:
        """
        Export metrics as a dictionary.

        Returns:
            Dict: Metrics dictionary for serialization.
        """
        metrics = self.get_metrics()
        return {
            "fps": metrics.fps,
            "frame_time_ms": metrics.frame_time_ms,
            "pipeline_latency_ms": metrics.pipeline_latency_ms,
            "render_time_ms": metrics.render_time_ms,
            "movement_detection_time_ms": metrics.movement_detection_time_ms,
            "total_frames": metrics.total_frames,
            "dropped_frames": metrics.dropped_frames,
            "drop_rate_percent": self._calculate_drop_rate(),
        }


class TimerContext:
    """
    Context manager for timing code blocks.

    Usage:
        with TimerContext(monitor, "pipeline"):
            # code to time
            pass
    """

    def __init__(self, monitor: PerformanceMonitor, timer_name: str):
        """
        Initialize timer context.

        Args:
            monitor: PerformanceMonitor instance.
            timer_name: Name of the timer.
        """
        self.monitor = monitor
        self.timer_name = timer_name

    def __enter__(self):
        """Start timer on context entry."""
        self.monitor.start_timer(self.timer_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer on context exit."""
        self.monitor.stop_timer(self.timer_name)
        return False

