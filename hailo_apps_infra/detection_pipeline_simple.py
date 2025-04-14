# detection_pipeline_simple.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_default_parser,
    detect_hailo_arch,
)
# --- IMPORT GStreamerApp FROM MODIFIED FILE ---
# Ensure the path is correct if you placed the modified file elsewhere
# If gstreamer_app.py is in the same directory, this might need adjustment
# e.g., from .gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps_infra.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback
)
# --- END IMPORT MODIFICATION ---

# NOTE: Import these even if not directly used in this file,
# as GStreamerApp might depend on them now
from hailo_apps_infra.gstreamer_helper_pipelines import(
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    OVERLAY_PIPELINE, # Assuming this is defined in the helper pipelines
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    # VIDEO_STREAM_PIPELINE (Defined below) # Removed duplicate import
)


# --- NEW HELPER PIPELINE FOR STREAMING ---
def VIDEO_STREAM_PIPELINE(port=5004, host='127.0.0.1', bitrate=2048):
    """
    Creates a GStreamer pipeline string portion for encoding and streaming video over UDP.
    Args:
        port (int): UDP port number.
        host (str): Destination IP address.
        bitrate (int): Target bitrate for x264enc in kbps.
    Returns:
        str: GStreamer pipeline string fragment.
    """
    # Using x264enc with zerolatency tune. Adjust encoder/params as needed.
    # Hardware encoders (e.g., omxh264enc, v4l2h264enc, vaapih264enc) are preferable on embedded systems.
    # Example using omxh264enc (Raspberry Pi):
    # encoder = f'omxh264enc target-bitrate={bitrate*1000} control-rate=variable'
    # Example using vaapih264enc (Intel):
    # encoder = f'vaapih264enc rate-control=cbr bitrate={bitrate}' # May need caps negotiation
    encoder = f'x264enc tune=zerolatency bitrate={bitrate} speed-preset=ultrafast'
    return (f"videoconvert ! video/x-raw,format=I420 ! " # x264enc often prefers I420
            f"{encoder} ! video/x-h264,profile=baseline ! " # Add profile for better compatibility potentially
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={host} port={port} sync=false async=false")
# --- END NEW HELPER ---


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        # --- Add Argument BEFORE calling super().__init__ ---
        if parser is None:
            parser = get_default_parser()
        parser.add_argument(
            '--stream-video',
            action='store_true',
            help='Stream video over UDP instead of displaying locally.'
        )
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )
        # --- END Add Argument ---


        # Call the parent class constructor AFTER adding the argument
        super().__init__(parser, user_data)

        # Additional initialization code can be added here
        self.video_width = 640
        self.video_height = 640

        # Set Hailo parameters - these parameters should be set based on the model used
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45
        if self.options_menu.input is None:  # Setting up a new application-specific default video (overrides the default video set in the GStreamerApp constructor)
            self.video_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/example_640.mp4')
        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                # Default or raise error - let's default to hailo8l for broader compatibility
                print("Could not auto-detect Hailo architecture. Defaulting to hailo8l. Specify --arch if needed.")
                self.arch = "hailo8l"
                # raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            else:
                 self.arch = detected_arch
                 print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        # Set the HEF file path based on the arch
        elif self.arch == "hailo8":
            self.hef_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/yolov6n.hef')
        else:  # hailo8l, hailo15 etc.
            self.hef_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/yolov6n_h8l.hef')

        # Check if HEF exists
        if not os.path.isfile(self.hef_path):
             print(f"ERROR: HEF file not found at {self.hef_path}. Please check the path or provide one using --hef-path.")
             sys.exit(1)

        # Set the post-processing shared object file
        self.post_process_so = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/libyolo_hailortpp_postprocess.so')
        if not os.path.isfile(self.post_process_so):
             print(f"ERROR: Post-process SO file not found at {self.post_process_so}.")
             sys.exit(1)

        self.post_function_name = "filter"

        # User-defined label JSON file
        self.labels_json = self.options_menu.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle("Hailo Detection Simple App")

        # The call to create_pipeline is now at the end of super().__init__
        # self.create_pipeline() # Removed this redundant call

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,
                                          video_width=self.video_width, video_height=self.video_height,
                                          frame_rate=self.frame_rate, sync=self.sync,
                                          no_webcam_compression=True)

        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str)
        user_callback_pipeline = USER_CALLBACK_PIPELINE() # Contains identity_callback

        
        # --- CHOOSE OUTPUT BASED ON STREAM FLAG ---
        if self.stream_video:
            # Use the VIDEO_STREAM_PIPELINE helper function
            output_pipeline = (
                f'{OVERLAY_PIPELINE()} ! '
                f'{VIDEO_STREAM_PIPELINE(port=self.video_stream_port, host="127.0.0.1")}'
            )
        else:
            # Use original display pipeline if not streaming
            output_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
        # --- END CHOOSE OUTPUT ---

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline} ! '
            f'{user_callback_pipeline} ! ' # Callback happens before encoding/display
            f'{output_pipeline}'
        )
        print("--- GStreamer Pipeline ---")
        print(pipeline_string)
        print("--------------------------")
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    # The actual callback function (can still be dummy if UI doesn't need backend frames)
    app_callback = dummy_callback # Or your actual complex callback if needed for backend logic

    # Create the app instance (parser is created internally now)
    app = GStreamerDetectionApp(app_callback, user_data)
    app.create_pipeline() # Now called after __init__
    app.run()