# gstreamer_app.py
import multiprocessing
import setproctitle
import signal
import os
import gi
import threading
import sys
import cv2
import numpy as np
import time

# --- NEW IMPORTS ---
from flask import Flask, request, jsonify
import logging
# --- END NEW IMPORTS ---


gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
from hailo_apps_infra.gstreamer_helper_pipelines import get_source_type
from hailo_apps_infra.get_usb_camera import get_usb_video_devices

try:
    from picamera2 import Picamera2
except ImportError:
    pass # Available only on Pi OS

# -----------------------------------------------------------------------------------------------
# User-defined class - Potentially add methods Flask can call
# -----------------------------------------------------------------------------------------------
# A sample class to be used in the callback function
# This example allows to:
# 1. Count the number of frames
# 2. Setup a multiprocessing queue to pass the frame to the main thread (less useful for separate UI)
# Additional variables and functions can be added to this class as needed
class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.frame_queue = multiprocessing.Queue(maxsize=3) # This queue won't be used by the separate UI
        self.running = True
        # --- NEW ---
        # Example: Add a parameter the UI can control
        self.some_threshold = 0.5
        self._lock = threading.Lock() # Use a lock if modifying data from multiple threads (Flask + GStreamer callback)
        print(f"Initial user_data threshold: {self.some_threshold}")
        # --- END NEW ---

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        # This is less relevant for the separate UI process model
        # If use_frame is True AND not streaming, it might still be used for local display
        if self.use_frame and not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_frame(self):
        # This is less relevant for the separate UI process model
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        else:
            return None

    # --- NEW ---
    def update_threshold(self, new_value):
        with self._lock:
            try:
                val = float(new_value)
                if 0.0 <= val <= 1.0:
                    self.some_threshold = val
                    print(f"User data threshold updated to: {self.some_threshold}")
                    # Here you might trigger some action based on the new threshold
                    # e.g., update a property in a GStreamer element if applicable
                    return True, f"Threshold updated to {self.some_threshold}"
                else:
                    print(f"Invalid threshold value received: {val}. Must be between 0.0 and 1.0")
                    return False, "Value must be between 0.0 and 1.0"
            except (ValueError, TypeError) as e:
                print(f"Invalid threshold value type received: {new_value}. Error: {e}")
                return False, "Invalid value, must be a float"

    def get_threshold(self):
         with self._lock:
            return self.some_threshold
    # --- END NEW ---


def dummy_callback(pad, info, user_data):
    """
    A minimal dummy callback function that returns immediately.

    Args:
        pad: The GStreamer pad
        info: The probe info
        user_data: User-defined data passed to the callback

    Returns:
        Gst.PadProbeReturn.OK
    """
    # Example: Accessing user data within the callback
    # current_threshold = user_data.get_threshold()
    # buffer = info.get_buffer()
    # print(f"Callback - Frame {user_data.get_count()}, Threshold: {current_threshold}, Buffer PTS: {buffer.pts / Gst.MSECOND} ms")
    # user_data.increment()

    # If you need to pass frames to the separate display_user_data_frame process (when use_frame=True and not streaming):
    # if user_data.use_frame:
    #     buffer = info.get_buffer()
    #     caps = pad.get_current_caps()
    #     # Extract frame details and push to user_data.frame_queue if needed...

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Flask Command Server (NEW)
# -----------------------------------------------------------------------------------------------
flask_app = Flask(__name__)
# Disable Flask's default logging or configure it as needed
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # Only show errors, suppress startup messages
_gstreamer_app_instance = None # Global reference to access user_data/pipeline

@flask_app.route('/update_data', methods=['POST'])
def update_data():
    global _gstreamer_app_instance
    if not _gstreamer_app_instance or not hasattr(_gstreamer_app_instance, 'user_data'):
        return jsonify({"status": "error", "message": "GStreamer app or user_data not ready"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400

    response = {"status": "success", "updates": []}
    overall_success = True

    if 'threshold' in data:
        success, msg = _gstreamer_app_instance.user_data.update_threshold(data['threshold'])
        response["updates"].append({"parameter": "threshold", "success": success, "message": msg})
        if not success:
            overall_success = False

    # Add more parameters here as needed...
    # if 'some_other_param' in data:
    #   param_value = data['some_other_param']
    #   # success, msg = _gstreamer_app_instance.user_data.update_other_param(param_value)
    #   # response["updates"].append(...)
    #   # if not success: overall_success = False
    #   pass

    if not response["updates"]:
         return jsonify({"status": "error", "message": "No valid parameters found in request"}), 400

    status_code = 200 if overall_success else 400 # Return 400 if any update failed validation
    return jsonify(response), status_code

@flask_app.route('/control', methods=['POST'])
def control_pipeline():
    global _gstreamer_app_instance
    if not _gstreamer_app_instance or not hasattr(_gstreamer_app_instance, 'pipeline'):
        return jsonify({"status": "error", "message": "GStreamer app or pipeline not ready"}), 500

    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({"status": "error", "message": "Action not specified in JSON body"}), 400

    action = data['action']
    pipeline = _gstreamer_app_instance.pipeline

    if action == 'toggle_pause':
        # It's better to query state *just before* changing it
        current_state, pending_state = pipeline.get_state(Gst.CLOCK_TIME_NONE) # Timeout=0

        if pending_state != Gst.State.VOID_PENDING:
             print(f"Pipeline is transitioning state ({pending_state}), please wait.")
             return jsonify({"status": "error", "message": f"Pipeline busy transitioning to {pending_state}"}), 409 # Conflict

        if current_state == Gst.State.PLAYING:
             print("Pausing pipeline via API")
             ret = pipeline.set_state(Gst.State.PAUSED)
             if ret == Gst.StateChangeReturn.SUCCESS or ret == Gst.StateChangeReturn.ASYNC:
                return jsonify({"status": "success", "message": "Pipeline pausing"}), 200
             else:
                print(f"Error pausing pipeline: {ret}")
                return jsonify({"status": "error", "message": f"Failed to pause pipeline ({ret})"}), 500
        elif current_state == Gst.State.PAUSED:
             print("Playing pipeline via API")
             ret = pipeline.set_state(Gst.State.PLAYING)
             if ret == Gst.StateChangeReturn.SUCCESS or ret == Gst.StateChangeReturn.ASYNC:
                 return jsonify({"status": "success", "message": "Pipeline playing"}), 200
             else:
                 print(f"Error playing pipeline: {ret}")
                 return jsonify({"status": "error", "message": f"Failed to play pipeline ({ret})"}), 500
        else:
             return jsonify({"status": "error", "message": f"Cannot toggle pause from state {current_state}"}), 400
    elif action == 'stop':
        print("Stopping pipeline via API")
        _gstreamer_app_instance.shutdown() # Use the existing graceful shutdown
        return jsonify({"status": "success", "message": "Shutdown initiated"}), 200
    # Add more actions like 'set_property', etc.
    # elif action == 'set_property':
    #    if 'element' not in data or 'property' not in data or 'value' not in data:
    #        return jsonify({"status": "error", "message": "Missing element, property, or value for set_property"}), 400
    #    el = pipeline.get_by_name(data['element'])
    #    if el:
    #        try:
    #            # Need type conversion here based on property type!
    #            # Example: el.set_property(data['property'], float(data['value']))
    #            print(f"Setting {data['property']} on {data['element']} to {data['value']} (TYPE CONVERSION NEEDED!)")
    #            return jsonify({"status": "success", "message": "Property set (check type!)"}), 200
    #        except Exception as e:
    #            return jsonify({"status": "error", "message": f"Failed to set property: {e}"}), 500
    #    else:
    #        return jsonify({"status": "error", "message": f"Element '{data['element']}' not found"}), 404
    else:
        return jsonify({"status": "error", "message": f"Unknown action: {action}"}), 400


def run_flask_server(host='0.0.0.0', port=5005):
    print(f"Starting Flask command server on http://{host}:{port}")
    # Use 'daemon=True' to allow the main GStreamer app to exit even if this thread is running
    # Use 'use_reloader=False' because we are running in a thread
    # Use 'threaded=True' to handle multiple requests if needed, though might complicate GStreamer interactions
    flask_thread = threading.Thread(target=lambda: flask_app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    return flask_thread

# -----------------------------------------------------------------------------------------------
# GStreamerApp class Modifications
# -----------------------------------------------------------------------------------------------
class GStreamerApp:
    def __init__(self, parser, user_data: app_callback_class): # Changed args to parser
        global _gstreamer_app_instance # Store reference for Flask
        _gstreamer_app_instance = self

        # Set the process title
        setproctitle.setproctitle("Hailo Python App Backend") # Changed title

        # Create options menu (parser is now passed in)
        self.options_menu = parser.parse_args()

        # --- NEW ARGUMENT CHECK ---
        self.stream_video = self.options_menu.stream_video if hasattr(self.options_menu, 'stream_video') else False
        self.video_stream_port = 5004 # Default UDP port for video
        self.command_server_port = 5005 # Default Flask port
        # --- END NEW ARGUMENT CHECK ---


        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)

        # Initialize variables
        tappas_post_process_dir = os.environ.get('TAPPAS_POST_PROC_DIR', '')
        if tappas_post_process_dir == '':
            # Check common relative paths as fallback
            script_dir = os.path.dirname(os.path.abspath(__file__))
            common_paths = [
                os.path.join(script_dir, '../../../../tappas/postprocesses'), # If infra is inside hailo_apps
                 os.path.join(script_dir, '../postprocesses') # If infra is alongside postprocesses
            ]
            found = False
            for path in common_paths:
                 if os.path.isdir(path):
                      tappas_post_process_dir = os.path.abspath(path)
                      print(f"TAPPAS_POST_PROC_DIR not set, found postprocesses at: {tappas_post_process_dir}")
                      found = True
                      break
            if not found:
                print("ERROR: TAPPAS_POST_PROC_DIR environment variable is not set and common paths not found.")
                print("Please set it by sourcing setup_env.sh from the TAPPAS root directory.")
                exit(1)
        else:
            print(f"Using TAPPAS_POST_PROC_DIR: {tappas_post_process_dir}")

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.postprocess_dir = tappas_post_process_dir
        self.video_source = self.options_menu.input
        if self.video_source is None:
            # Use a default relative to *this* file's location
            self.video_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../resources/example.mp4')
            print(f"Input video not specified, using default: {self.video_source}")

        if self.video_source == 'usb':
            usb_devices = get_usb_video_devices() # Renamed variable
            if not usb_devices:
                print('Provided argument "--input" is set to "usb", however no available USB cameras found.')
                print('Please connect a camera or specify a different input method (e.g., /dev/video0).')
                exit(1)
            else:
                self.video_source = usb_devices[0] # Use the first found device
                print(f"Using USB camera: {self.video_source}")
        elif not os.path.exists(self.video_source) and not self.video_source.startswith(('http://', 'https://', 'rtsp://')):
             # Check if it's a device node that doesn't exist yet (less reliable)
             if not (self.video_source.startswith('/dev/video') or self.video_source == 'rpi'):
                print(f"ERROR: Input video source '{self.video_source}' not found or is not a valid URL/device.")
                exit(1)

        self.source_type = get_source_type(self.video_source)
        self.frame_rate = self.options_menu.frame_rate
        self.user_data = user_data
        self.pipeline = None
        self.loop = None
        self.threads = [] # For Flask server, Picamera, etc.
        self.error_occurred = False
        self.pipeline_latency = 300  # milliseconds

        # Modify video_sink based on streaming flag
        if self.stream_video:
            self.video_sink = None # Mark that we are not using a standard display sink
            print(f"Streaming video enabled. Outputting to UDP port {self.video_stream_port}")
        else:
            # Try env var or default to autovideosink
            self.video_sink = os.environ.get("DISPLAY_SINK", "autovideosink")
            print(f"Local display enabled using sink: {self.video_sink}")


        # Set Hailo parameters (placeholder, should be set by subclass)
        self.batch_size = 1
        self.video_width = 1280
        self.video_height = 720
        self.video_format = "RGB"
        self.hef_path = None
        self.app_callback = None # To be set by subclass

        # Set user data parameters
        user_data.use_frame = self.options_menu.use_frame if hasattr(self.options_menu, 'use_frame') else False # Note: use_frame is less useful with separate UI
        # Need to ensure the user_data instance passed in has this attribute

        self.sync = "false" if (self.options_menu.disable_sync or self.source_type != "file" or self.stream_video) else "true" # Force sync=false if streaming
        self.show_fps = self.options_menu.show_fps if hasattr(self.options_menu, 'show_fps') else False
        if self.stream_video:
            self.show_fps = False # Disable internal FPS display if streaming

        if self.options_menu.dump_dot if hasattr(self.options_menu, 'dump_dot') else False:
            dot_dir = os.path.join(os.getcwd(), "gst_dots")
            os.makedirs(dot_dir, exist_ok=True)
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = dot_dir
            print(f"GST_DEBUG_DUMP_DOT_DIR set to: {dot_dir}")


    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        # This callback is only connected if not self.stream_video and self.show_fps is True
        print(f"Local Display FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True # Keep signal connected

    def create_pipeline(self):
        # Initialize GStreamer
        Gst.init(None)

        pipeline_string = self.get_pipeline_string() # Call subclass implementation
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except GLib.Error as e: # More specific GStreamer error
            print(f"Error parsing pipeline: {e}", file=sys.stderr)
            self.error_occurred = True # Mark error
            self.shutdown() # Attempt shutdown
            sys.exit(1) # Exit immediately on parse failure
        except Exception as e:
            print(f"Unexpected error creating pipeline: {e}", file=sys.stderr)
            self.error_occurred = True
            self.shutdown()
            sys.exit(1)

        # Create a GLib Main Loop *after* Gst.init
        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream received.")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            element_name = message.src.get_name() if message.src else "Unknown Element"
            print(f"GStreamer Error from element {element_name}: {err}", file=sys.stderr)
            print(f"Debugging info: {debug}", file=sys.stderr)
            self.error_occurred = True
            self.shutdown()
        elif t == Gst.MessageType.WARNING:
             err, debug = message.parse_warning()
             element_name = message.src.get_name() if message.src else "Unknown Element"
             print(f"GStreamer Warning from element {element_name}: {err}", file=sys.stderr)
             print(f"Debugging info: {debug}", file=sys.stderr)
        # QOS
        elif t == Gst.MessageType.QOS:
            # Handle QoS message here
            # If lots of QoS messages are received, it may indicate that the pipeline is not able to keep up
            if not hasattr(self, 'qos_count'):
                self.qos_count = 0
            self.qos_count += 1
            if self.qos_count > 50 and self.qos_count % 10 == 0:
                # Check if src exists before getting name
                qos_element = message.src.get_name() if message.src else "Unknown Element"
                format_str, processed, dropped = message.parse_qos_stats()
                # Check if values exist
                proc_val = processed if processed is not None else 'N/A'
                drop_val = dropped if dropped is not None else 'N/A'

                print(f"\033[91mQoS Warning from {qos_element}: Processed={proc_val}, Dropped={drop_val}\033[0m")
                print(f"\033[91mLots of QoS messages ({self.qos_count}): pipeline may not be keeping up. Consider reducing frame rate ('--frame-rate') or complexity.\033[0m")

        return True # Keep watching bus


    def on_eos(self):
        if self.source_type == "file" and not self.stream_video: # Only rewind if it's a file and displaying locally
            print("Reached end of file.")
            # Option 1: Rewind and loop (if sync=true, this might work better)
            # print("Rewinding video file...")
            # self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0)
            # self.pipeline.set_state(Gst.State.PLAYING)

            # Option 2: Stop the application
            print("Stopping application.")
            self.shutdown()
        else:
            # For streams or network output, EOS usually means the source ended or disconnected
            print("Source stream ended or disconnected.")
            self.shutdown()


    def shutdown(self, signum=None, frame=None):
        if hasattr(self, 'shutting_down') and self.shutting_down:
             print("Shutdown already in progress...")
             # Force quit on second Ctrl-C if needed
             if signum == signal.SIGINT:
                  print("Forcing quit.")
                  signal.signal(signal.SIGINT, signal.SIG_DFL) # Restore default handler
                  os._exit(1) # Force exit
             return
        self.shutting_down = True

        print("Shutting down backend...")
        if signum == signal.SIGINT:
            print("Ctrl-C detected. Hit Ctrl-C again to force quit.")
            # Allow force quit on second Ctrl+C
            signal.signal(signal.SIGINT, signal.SIG_DFL)


        # Stop Flask server? (It's daemon, should stop automatically, but cleaner to signal if possible)
        # (No standard Flask way to stop from another thread easily)

        # Stop the GStreamer pipeline first
        if self.pipeline:
            print("Setting pipeline to NULL state...")
            self.pipeline.set_state(Gst.State.NULL)
            # Give some time for state change
            # GLib.usleep(200000) # 0.2 second delay - maybe not needed if loop.quit is used

        # Stop the GLib main loop
        if self.loop and self.loop.is_running():
            print("Quitting GLib main loop...")
            # Use idle_add to ensure quit is called from the main loop's context
            GLib.idle_add(self.loop.quit)
        else:
             # If loop not running (e.g., error during setup), exit directly
             print("Loop not running, exiting.")
             sys.exit(1 if self.error_occurred else 0)

    def update_fps_caps(self, new_fps=30, source_name='source'):
        """Updates the FPS by setting max-rate on videorate element directly"""
        if not self.pipeline:
             print("Pipeline not created yet, cannot update FPS.")
             return
        # Derive the videorate and capsfilter element names based on the source name
        videorate_name = f"{source_name}_videorate"
        capsfilter_name = f"{source_name}_fps_caps"

        # Get the videorate element
        videorate = self.pipeline.get_by_name(videorate_name)
        if videorate is None:
            print(f"Element {videorate_name} not found in the pipeline.")
            return

        # Print current properties for debugging
        try:
            current_max_rate = videorate.get_property("max-rate")
            print(f"Current videorate max-rate: {current_max_rate}")

            # Update the max-rate property directly
            videorate.set_property("max-rate", new_fps)

            # Verify the change
            updated_max_rate = videorate.get_property("max-rate")
            print(f"Updated videorate max-rate to: {updated_max_rate}")
        except Exception as e:
             print(f"Could not get/set property 'max-rate' on {videorate_name}: {e}")


        # Get the capsfilter element
        capsfilter = self.pipeline.get_by_name(capsfilter_name)
        if capsfilter:
            try:
                new_caps_str = f"video/x-raw, framerate={new_fps}/1"
                new_caps = Gst.Caps.from_string(new_caps_str)
                capsfilter.set_property("caps", new_caps)
                print(f"Updated capsfilter '{capsfilter_name}' caps to match new rate")
            except Exception as e:
                 print(f"Could not set property 'caps' on {capsfilter_name}: {e}")


        # Update frame_rate property (if used elsewhere)
        self.frame_rate = new_fps


    # Make get_pipeline_string abstract or ensure it's always overridden
    def get_pipeline_string(self):
        # This MUST be overridden by the child class (like GStreamerDetectionApp)
        raise NotImplementedError("get_pipeline_string must be implemented by subclass")

    def dump_dot_file(self, state_suffix=""):
        if not self.pipeline or not os.environ.get("GST_DEBUG_DUMP_DOT_DIR"):
             return False # Stop timeout if not possible

        print(f"Dumping dot file for state: {state_suffix}...")
        filename = f"pipeline_{state_suffix}"
        Gst.debug_bin_to_dot_file_with_ts(self.pipeline, Gst.DebugGraphDetails.ALL, filename)
        print(f"Dot file dumped: {filename}.dot")
        return True # Keep timeout running if needed, or return False to run once


    def run(self):
        if not self.pipeline or not self.loop:
             print("Error: Pipeline or MainLoop not initialized before run().", file=sys.stderr)
             sys.exit(1)

        # --- START COMMAND SERVER ---
        # Run Flask in a separate thread before starting the GStreamer loop
        flask_thread = run_flask_server(port=self.command_server_port)
        self.threads.append(flask_thread) # Keep track if needed
        # --- END COMMAND SERVER ---

        # Add a watch for messages on the pipeline's bus (existing)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)


        # Connect pad probe to the identity element (existing)
        if not (self.options_menu.disable_callback if hasattr(self.options_menu, 'disable_callback') else False):
            identity = self.pipeline.get_by_name("identity_callback")
            if identity is None:
                print("Warning: identity_callback element not found, add <identity name=identity_callback> in your pipeline where you want the callback to be called.")
            else:
                identity_pad = identity.get_static_pad("src")
                if not identity_pad:
                     print("Error: Could not get 'src' pad from identity_callback element.")
                else:
                     identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self.user_data)
                     print("Callback probe added to identity_callback.")

        # --- Modify Display Sink Handling ---
        if not self.stream_video and self.show_fps:
            # Only connect FPS measurement if using local display sink and show_fps is True
            hailo_display = self.pipeline.get_by_name("hailo_display")
            if hailo_display is None:
                print("Warning: hailo_display element not found. Cannot show FPS. Ensure sink is named 'hailo_display'.")
            else:
                 try:
                     # Check if the signal exists before connecting
                     if GObject.signal_lookup('fps-measurements', hailo_display.__class__):
                         hailo_display.connect("fps-measurements", self.on_fps_measurement)
                         print("Connected to fps-measurements signal on hailo_display.")
                     else:
                          print(f"Warning: Element 'hailo_display' ({hailo_display.__class__.__name__}) does not have 'fps-measurements' signal.")
                 except Exception as e:
                      print(f"Error connecting to fps-measurements signal: {e}")
        # --- End Modify Display Sink Handling ---


        # Disable QoS (existing)
        disable_qos(self.pipeline)

        # Start display subprocess (existing, but less useful now)
        display_process = None # Initialize
        if self.user_data.use_frame and not self.stream_video:
            # Only useful if also displaying locally and using the callback to push frames
            print("Starting separate frame display process (use_frame=True)")
            display_process = multiprocessing.Process(target=display_user_data_frame, args=(self.user_data,), daemon=True)
            display_process.start()
            self.threads.append(display_process) # Add to threads list


        # Start Picamera thread if needed (existing)
        if self.source_type == "rpi":
            print("Starting Picamera capture thread...")
            picam_thread = threading.Thread(target=picamera_thread, args=(self.pipeline, self.video_width, self.video_height, self.video_format), daemon=True)
            self.threads.append(picam_thread)
            picam_thread.start()


        # Dump dot file before PLAYING
        if os.environ.get("GST_DEBUG_DUMP_DOT_DIR"):
             GLib.idle_add(self.dump_dot_file, "paused") # Dump paused state

        # Set pipeline to PLAYING state
        print("Setting pipeline to PLAYING...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error: Unable to set the pipeline to the playing state.", file=sys.stderr)
            self.error_occurred = True
            self.shutdown()
            return # Exit run method
        elif ret == Gst.StateChangeReturn.ASYNC:
             print("Pipeline state change is asynchronous.")
             # You might need to wait for state change complete message on bus if required
        elif ret == Gst.StateChangeReturn.NO_PREROLL:
             print("Pipeline is live and does not generate data before PLAYING.")


        # Dump dot file shortly after PLAYING
        if os.environ.get("GST_DEBUG_DUMP_DOT_DIR"):
            # Use timeout_add_seconds for delayed dump
            GLib.timeout_add_seconds(3, self.dump_dot_file, "playing")


        # Run the GLib event loop (existing)
        print("Running GLib main loop...")
        try:
             self.loop.run()
        except KeyboardInterrupt:
             print("KeyboardInterrupt caught in main loop, initiating shutdown...")
             self.shutdown(signal.SIGINT) # Treat like Ctrl-C
        except Exception as e:
             print(f"Error during main loop execution: {e}", file=sys.stderr)
             self.error_occurred = True
             self.shutdown() # Attempt graceful shutdown

        # Clean up happens after loop.run() returns (usually triggered by loop.quit() in shutdown)
        print("GLib main loop has ended.")
        try:
            self.user_data.running = False # Stop display loop if it was running
            print("Waiting for background threads...")
            # Wait briefly for daemon threads (Flask, Picamera) - they should exit automatically
            # If non-daemon threads were used, they'd need explicit joining here.
            # For multiprocessing Process, terminate/join
            if display_process and display_process.is_alive():
                print("Terminating display process...")
                display_process.terminate()
                display_process.join(timeout=1) # Wait briefly for join

            # Ensure pipeline is NULL (might be redundant if shutdown did it)
            if self.pipeline:
                 self.pipeline.set_state(Gst.State.NULL)

        except Exception as e:
            print(f"Error during final cleanup: {e}", file=sys.stderr)
        finally:
            # Flask thread is daemon, should exit automatically
            if self.error_occurred:
                print("Exiting backend with error status.", file=sys.stderr)
                sys.exit(1)
            else:
                print("Exiting backend cleanly.")
                sys.exit(0)

# --- Picamera Thread ---
def picamera_thread(pipeline, video_width, video_height, video_format, picamera_config=None):
    appsrc = pipeline.get_by_name("app_source")
    if not appsrc:
         print("ERROR: 'app_source' element not found in pipeline for Picamera.")
         return
    # Ensure appsrc properties are set correctly *before* pushing buffers
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    # Set stream-type based on need (0=stream, 1=seekable, 2=seekable-stream)
    appsrc.set_property("stream-type", 0) # GST_APP_STREAM_TYPE_STREAM

    print("appsrc properties configured for Picamera.")
    # Initialize Picamera2
    try:
        with Picamera2() as picam2:
            if picamera_config is None:
                # Default configuration (example, adjust as needed)
                main = {'size': (1920, 1080), 'format': 'RGB888'} # Use a common sensor mode if possible
                lores_size = (video_width, video_height)
                # Ensure lores format matches expected GStreamer format if possible
                # Picamera often uses YUV formats internally more efficiently
                # Let's request RGB and convert later if needed, or request YUV if pipeline handles it
                lores_format = 'RGB888' # Match typical expectation, could be 'YUV420'
                controls = {'FrameRate': 30.0} # Use float for FrameRate control

                # Check available formats/modes if necessary
                # print("Available Picamera2 sensor modes:", picam2.sensor_modes)

                config = picam2.create_video_configuration(
                    main=main, # Main stream (preview or recording)
                    lores={'size': lores_size, 'format': lores_format}, # Low-res stream for appsrc
                    controls=controls
                 )
                print(f"Created Picamera2 config: {config}")
            else:
                config = picamera_config

            # Configure the camera with the created configuration
            picam2.configure(config)

            # Update GStreamer caps based on the *actual* 'lores' stream configuration
            actual_config = picam2.camera_configuration()
            lores_stream_config = actual_config['lores']
            # Map Picamera format string to GStreamer format string
            gst_format_map = {
                'RGB888': 'RGB',
                'BGR888': 'BGR',
                'YUV420': 'I420', # Common mapping
                'YUYV': 'YUY2',
                # Add other formats as needed
            }
            picam_format = lores_stream_config['format']
            gst_format = gst_format_map.get(picam_format, None)
            if gst_format is None:
                 print(f"ERROR: Unsupported Picamera format '{picam_format}' for GStreamer.")
                 return

            width, height = lores_stream_config['size']
            stride = lores_stream_config['stride'] # Important for correct buffer wrapping
            framerate = int(actual_config['controls']['FrameRate']) # Get actual framerate

            print(f"Actual Picamera lores stream: {width}x{height} @ {framerate}fps, Format: {picam_format} ({gst_format}), Stride: {stride}")

            # Set appsrc caps
            caps_str = (f"video/x-raw, format={gst_format}, width={width}, height={height}, "
                        f"framerate={framerate}/1, pixel-aspect-ratio=1/1")
            appsrc.set_property("caps", Gst.Caps.from_string(caps_str))
            print(f"Set appsrc caps: {caps_str}")

            picam2.start()
            print("Picamera started, starting frame capture loop.")
            frame_count = 0
            start_time = time.time()

            while True: # Loop should be controlled by main app shutdown
                # Capture into a NumPy array - 'lores' stream
                frame_data = picam2.capture_array('lores')

                if frame_data is None:
                    print("Warning: Failed to capture frame from Picamera.")
                    time.sleep(0.1) # Avoid busy-looping on error
                    continue

                # Ensure data is contiguous if needed (should be from capture_array)
                if not frame_data.flags['C_CONTIGUOUS']:
                     frame_data = np.ascontiguousarray(frame_data)

                # Create Gst.Buffer by wrapping the frame data
                # Use size=stride * height for potentially padded data
                buffer = Gst.Buffer.new_wrapped(frame_data.tobytes())

                # Set buffer PTS and duration (crucial for pipeline synchronization)
                # Use framerate from actual config
                buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, framerate)
                buffer.pts = frame_count * buffer_duration
                buffer.duration = buffer_duration
                buffer.offset = Gst.CLOCK_TIME_NONE # Usually not needed for live sources

                # Push the buffer to appsrc
                ret = appsrc.emit('push-buffer', buffer)
                if ret != Gst.FlowReturn.OK:
                    print(f"Failed to push buffer to appsrc: {ret}. Stopping Picamera thread.")
                    break # Exit loop on push error

                frame_count += 1
                # Optional: FPS calculation for this thread
                # if frame_count % 100 == 0:
                #     elapsed = time.time() - start_time
                #     print(f"Picamera Thread FPS: {frame_count / elapsed:.2f}")

    except NameError:
         print("Picamera2 not available on this system.")
    except Exception as e:
        print(f"Error in Picamera thread: {e}", file=sys.stderr)
    finally:
         # Cleanup, though Picamera2() context manager should handle closing
         print("Picamera thread finished.")


# --- QoS Disabler ---
def disable_qos(pipeline):
    """
    Iterate through all elements in the given GStreamer pipeline and set the qos property to False
    where applicable.
    When the 'qos' property is set to True, the element will measure the time it takes to process each buffer and will drop frames if latency is too high.
    We are running on long pipelines, so we want to disable this feature to avoid dropping frames.
    :param pipeline: A GStreamer pipeline object
    """
    # Ensure the pipeline is a Gst.Pipeline instance
    if not isinstance(pipeline, Gst.Pipeline):
        print("The provided object is not a GStreamer Pipeline")
        return

    print("Disabling QoS on pipeline elements...")
    # Iterate through all elements in the pipeline
    elements = pipeline.iterate_elements()
    while True:
        try:
            result, element = elements.next()
            if result != Gst.IteratorResult.OK:
                break # End of iteration

            # Check if the element has the 'qos' property using GObject introspection
            try:
                 qos_prop = GObject.find_property(element, 'qos')
                 if qos_prop and qos_prop.flags & GObject.ParamFlags.WRITABLE:
                     # Set the 'qos' property to False
                     element.set_property('qos', False)
                     # print(f"Set qos=False for {element.get_name()}")
            except TypeError:
                 # Some elements might not support GObject introspection fully
                 # print(f"Could not check/set qos property for {element.get_name()} (introspection error)")
                 pass
            except AttributeError:
                 # Some Gst elements might not be GObjects or lack list_properties
                 pass
        except Exception as e:
            print(f"Error iterating pipeline elements for QoS: {e}")
            break # Stop iteration on error
    print("Finished QoS disable attempt.")

# This function is used to display the user data frame from the queue
def display_user_data_frame(user_data: app_callback_class):
    print("Started separate display thread/process.")
    while getattr(user_data, 'running', True): # Check if running attribute exists
        frame = user_data.get_frame() # Blocking call if queue is empty
        if frame is not None:
            # Assuming frame is a NumPy array suitable for cv2.imshow
            try:
                cv2.imshow("User Frame Display", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): # Allow quitting with 'q'
                    break
            except Exception as e:
                print(f"Error displaying frame in display_user_data_frame: {e}")
                # Avoid continuous errors if frames are bad
                time.sleep(0.1)
        else:
             # If get_frame returns None, maybe the queue is empty and blocking isn't used,
             # or None signifies termination. Add a small sleep to prevent busy-waiting.
             time.sleep(0.01)

    print("Closing User Frame Display window.")
    cv2.destroyAllWindows()
    # Add extra waitKey calls to ensure window closes reliably on all platforms
    for _ in range(5):
        cv2.waitKey(1)
    print("Display thread/process finished.")