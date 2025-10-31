"""
Hailo pipeline for pose estimation only.

Simplified pipeline for Freezing Phantoms game using only pose estimation at 720p.
"""

import os
import queue
import sys
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import setproctitle

from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import (
    HAILO_ARCH_KEY,
    POSE_ESTIMATION_PIPELINE,
    POSE_ESTIMATION_POSTPROCESS_FUNCTION,
    POSE_ESTIMATION_POSTPROCESS_SO_FILENAME,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    QUEUE,
    TRACKER_PIPELINE,
)
from freezing_phantoms.pipeline.data_extractor import DataExtractor

hailo_logger = get_logger(__name__)


class GameCallbackClass(app_callback_class):
    """
    Callback class for game pipeline with thread-safe data queue.

    Attributes:
        data_queue (queue.Queue): Thread-safe queue for player data.
        data_extractor (DataExtractor): Data extraction utility.
    """

    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize game callback class.

        Args:
            frame_width (int): Video frame width.
            frame_height (int): Video frame height.
        """
        super().__init__()
        self.data_queue = queue.Queue(maxsize=10)
        self.data_extractor = DataExtractor(frame_width, frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height
        hailo_logger.info(
            "GameCallbackClass initialized: %dx%d", frame_width, frame_height
        )

    def get_latest_data(self, timeout: float = 0.1):
        """
        Get latest player data from queue.

        Args:
            timeout (float): Timeout for queue get operation.

        Returns:
            dict: Player data or None if queue is empty.
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def game_pipeline_callback(pad, info, user_data: GameCallbackClass):  # noqa: ARG001
    """
    Callback function for combined pipeline.

    Extracts player data and puts it in the thread-safe queue.

    Args:
        pad: GStreamer pad.
        info: Probe info.
        user_data (GameCallbackClass): User data object.

    Returns:
        Gst.PadProbeReturn: OK to continue pipeline.
    """
    hailo_logger.debug("🎯 CALLBACK CALLED! Processing frame...")

    buffer = info.get_buffer()
    if buffer is None:
        hailo_logger.warning("Received None buffer")
        return Gst.PadProbeReturn.OK

    user_data.increment()
    hailo_logger.debug("Frame count: %d", user_data.frame_count)

    # Extract player data
    try:
        player_data = user_data.data_extractor.extract_players(buffer)
        hailo_logger.debug("Extracted %d players from callback", len(player_data.get('players', [])))

        # Put data in queue (non-blocking)
        if not user_data.data_queue.full():
            user_data.data_queue.put(player_data)
            hailo_logger.debug("✅ Data added to queue (size: %d)", user_data.data_queue.qsize())
        else:
            hailo_logger.debug("Data queue full, dropping frame data")

    except Exception as e:
        hailo_logger.error("Error extracting player data: %s", e)

    return Gst.PadProbeReturn.OK


class PoseEstimationPipeline(GStreamerApp):
    """
    Simplified pipeline running only pose estimation.

    This pipeline processes video frames at 720p to extract:
    - Person detection with tracking IDs
    - Pose keypoints (17 COCO keypoints)
    """

    def __init__(
        self,
        video_source: str = "/dev/video0",
        frame_width: int = 1280,
        frame_height: int = 720,
        frame_rate: int = 30,
        debug_mode: bool = False,
        hailo_options=None,
    ):
        """
        Initialize pose estimation pipeline.

        Args:
            video_source (str): Video source (device path or file).
            frame_width (int): Frame width (default 1280 for 720p).
            frame_height (int): Frame height (default 720 for 720p).
            frame_rate (int): Frame rate.
            debug_mode (bool): If True, show video output with AI overlays.
            hailo_options: Parsed Hailo standard options (arch, hef-path, etc.)
        """
        hailo_logger.info("Initializing PoseEstimationPipeline at %dx%d", frame_width, frame_height)

        # Store debug mode flag and options
        self.debug_mode = debug_mode
        self.custom_hailo_options = hailo_options

        # Create user data
        self.user_data = GameCallbackClass(frame_width, frame_height)

        # Create custom parser with our defaults
        parser = get_default_parser()

        # Build sys.argv from options
        argv_list = [
            "combined_pipeline",
            "--input", video_source,
        ]

        # Add custom options if provided
        if self.custom_hailo_options:
            if getattr(self.custom_hailo_options, 'arch', None):
                argv_list.extend(["--arch", self.custom_hailo_options.arch])
            if getattr(self.custom_hailo_options, 'hef_path', None):
                argv_list.extend(["--hef-path", self.custom_hailo_options.hef_path])
            if getattr(self.custom_hailo_options, 'disable_sync', False):
                argv_list.append("--disable-sync")
            if getattr(self.custom_hailo_options, 'show_fps', False):
                argv_list.append("--show-fps")
            if getattr(self.custom_hailo_options, 'dump_dot', False):
                argv_list.append("--dump-dot")
            if getattr(self.custom_hailo_options, 'disable_callback', False):
                argv_list.append("--disable-callback")
        else:
            # Default options when no custom options provided
            argv_list.append("--disable-sync")
            argv_list.append("--show-fps")

        # Temporarily override sys.argv for parser
        old_argv = sys.argv
        sys.argv = argv_list

        try:
            # Initialize base class (it will call parser.parse_args())
            super().__init__(parser, self.user_data)
        finally:
            # Restore original sys.argv
            sys.argv = old_argv

        # Set the callback function for the parent class
        self.app_callback = game_pipeline_callback

        # Video parameters
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_rate = frame_rate
        self.batch_size = 2

        # Detect architecture
        if self.options_menu.arch is None:
            arch = os.getenv(HAILO_ARCH_KEY, detect_hailo_arch())
            if not arch:
                hailo_logger.error("Could not detect Hailo architecture")
                raise ValueError("Could not auto-detect Hailo architecture")
            self.arch = arch
        else:
            self.arch = self.options_menu.arch

        hailo_logger.info("Using Hailo architecture: %s", self.arch)

        # Setup pose estimation resources
        if self.custom_hailo_options and getattr(self.custom_hailo_options, 'hef_path', None):
            self.hef_path = self.custom_hailo_options.hef_path
            hailo_logger.info("Using custom HEF path: %s", self.hef_path)
        else:
            self.hef_path = str(
                get_resource_path(POSE_ESTIMATION_PIPELINE, RESOURCES_MODELS_DIR_NAME)
            )
            hailo_logger.info("Using default HEF path: %s", self.hef_path)

        self.post_process_so = str(
            get_resource_path(
                POSE_ESTIMATION_PIPELINE,
                RESOURCES_SO_DIR_NAME,
                POSE_ESTIMATION_POSTPROCESS_SO_FILENAME,
            )
        )
        self.post_process_function = POSE_ESTIMATION_POSTPROCESS_FUNCTION

        hailo_logger.info("Post-process SO: %s", self.post_process_so)
        hailo_logger.info("Post-process function: %s", self.post_process_function)

        # Set process title
        setproctitle.setproctitle("Freezing Phantoms - Pose Pipeline")

        # Callback
        self.app_callback = game_pipeline_callback

        # Create pipeline
        self._create_game_pipeline()
        hailo_logger.info("Pose estimation pipeline created successfully")

        # Pipeline thread
        self._pipeline_thread = None
        self._running = False

    def _create_game_pipeline(self):
        """
        Create pipeline with optional display element.

        In debug mode, connects FPS callback to hailo_display.
        In game mode, uses fakesink without display.
        """
        from gi.repository import GLib

        hailo_logger.debug("Creating game pipeline...")
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        hailo_logger.debug("Pipeline string: %s", pipeline_string)

        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except Exception as e:
            hailo_logger.error("Error creating pipeline: %s", e)
            print(f"Error creating pipeline: {e}", file=sys.stderr)
            sys.exit(1)

        # Note: FPS callback not connected in debug mode
        # The hailooverlay element doesn't support fps-measurements signal
        # Video output with AI overlays will still work for debugging
        if self.debug_mode:
            hailo_logger.info("Debug mode: Video display enabled (FPS counter not available)")

        self.loop = GLib.MainLoop()
        hailo_logger.debug("Pipeline created successfully")

    def get_pipeline_string(self) -> str:
        """
        Build the pose estimation pipeline string.

        Returns:
            str: GStreamer pipeline string.
        """
        hailo_logger.debug("Building pose estimation pipeline string at %dx%d",
                          self.frame_width, self.frame_height)

        # Custom source pipeline WITHOUT horizontal flip (camera and projector on same axis)
        # Based on SOURCE_PIPELINE but removes videoflip element
        source = self._create_source_pipeline_no_flip()

        # Pose estimation inference
        infer = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size,
        )

        # Wrap inference pipeline to maintain resolution
        infer_wrapper = INFERENCE_PIPELINE_WRAPPER(infer)

        # Tracker (after pose detection)
        tracker = TRACKER_PIPELINE(class_id=0)

        # User callback (probe point) - identity element for callback attachment
        callback_queue = QUEUE(name="callback_q")
        identity_callback = "identity name=identity_callback"

        # Sink: display for debug mode, fakesink for game mode
        if self.debug_mode:
            sink = "hailooverlay name=hailo_overlay ! videoconvert ! xvimagesink name=hailo_display sync=false"
            hailo_logger.info("Debug mode: Using video display")
        else:
            sink = "fakesink name=game_sink sync=false"
            hailo_logger.info("Game mode: Using fakesink (no display)")

        # Combine all elements with wrapper
        pipeline_string = (
            f"{source} ! "
            f"{infer_wrapper} ! "
            f"{tracker} ! "
            f"{callback_queue} ! "
            f"{identity_callback} ! "
            f"{sink}"
        )

        hailo_logger.info("Pipeline configured: %dx%d @ %d fps",
                         self.frame_width, self.frame_height, self.frame_rate)
        hailo_logger.debug("Pipeline string: %s", pipeline_string)
        return pipeline_string

    def _create_source_pipeline_no_flip(self) -> str:
        """
        Create source pipeline WITHOUT horizontal flip.

        Camera and projector are on the same axis, so no flip is needed.
        This is a simplified version of SOURCE_PIPELINE without videoflip.

        Returns:
            str: GStreamer source pipeline string.
        """
        # Determine source type
        video_source = self.video_source
        if video_source.startswith('/dev/video'):
            # USB camera - use MJPEG compression for better performance at 720p
            width, height = 1280, 720  # Camera resolution (closest to our target)
            source_element = (
                f'v4l2src device={video_source} name=source ! '
                f'image/jpeg, framerate=30/1, width={width}, height={height} ! '
                f'{QUEUE(name="source_queue_decode")} ! '
                f'decodebin name=source_decodebin ! '
                # NOTE: videoflip removed - camera and projector on same axis
            )
        else:
            # File source or other
            source_element = (
                f'filesrc location="{video_source}" ! '
                f'{QUEUE(name="source_queue_decode")} ! '
                f'decodebin name=source_decodebin ! '
            )

        # FPS caps based on sync setting
        if self.sync:
            fps_caps = f"video/x-raw, framerate={self.frame_rate}/1"
        else:
            fps_caps = "video/x-raw"

        # Complete source pipeline
        source_pipeline = (
            f"{source_element} "
            f"{QUEUE(name='source_scale_q')} ! "
            f"videoscale name=source_videoscale n-threads=2 ! "
            f"{QUEUE(name='source_convert_q')} ! "
            f"videoconvert n-threads=3 name=source_convert qos=false ! "
            f"video/x-raw, pixel-aspect-ratio=1/1, format=RGB, "
            f"width={self.frame_width}, height={self.frame_height} ! "
            f'videorate name=source_videorate ! capsfilter name=source_fps_caps caps="{fps_caps}" '
        )

        hailo_logger.info("Created source pipeline WITHOUT horizontal flip (same-axis camera)")
        return source_pipeline

    def start(self) -> None:
        """Start the pipeline in a separate thread."""
        if self._running:
            hailo_logger.warning("Pipeline already running")
            return

        self._running = True
        self._pipeline_thread = threading.Thread(target=self.run, daemon=True)
        self._pipeline_thread.start()
        hailo_logger.info("Pipeline started in background thread")

    def stop(self) -> None:
        """Stop the pipeline."""
        if not self._running:
            return

        self._running = False
        self.shutdown()

        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=5.0)

        hailo_logger.info("Pipeline stopped")

    def get_latest_data(self, timeout: float = 0.1):
        """
        Get latest player data from pipeline.

        Args:
            timeout (float): Timeout for getting data.

        Returns:
            dict: Player data or None.
        """
        return self.user_data.get_latest_data(timeout)

