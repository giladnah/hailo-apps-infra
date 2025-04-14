# qt_ui_client.py
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Slot, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QIcon
import requests # For sending commands
import threading
import os
import signal # To handle Ctrl+C in the UI process too

# --- Configuration ---
GST_VIDEO_RECEIVE_PORT = 5004
BACKEND_COMMAND_URL = "http://127.0.0.1:5005" # Match Flask server port
TARGET_FPS = 30 # For UI updates (timer based), GStreamer drives actual frame arrival
REQUEST_TIMEOUT = 2.0 # Timeout for HTTP requests to backend (seconds)

# --- GStreamer Receiver Thread ---
class GStreamerReceiverThread(QThread):
    # Signal to emit frame (Pixmap) to the main UI thread
    new_frame_signal = Signal(QPixmap)
    # Signal to report errors
    error_signal = Signal(str)
    # Signal to indicate EOS
    eos_signal = Signal()

    def __init__(self, port=GST_VIDEO_RECEIVE_PORT, parent=None):
        super().__init__(parent)
        self.port = port
        self.pipeline = None
        self.loop = None
        self._running = True
        self.gst_context = GLib.MainContext() # Create a context for this thread's loop

    def _build_pipeline_string(self):
        # Ensure required GStreamer elements are installed:
        # gudev, rtph264depay, avdec_h264 (or other H.264 decoder like vaapidecode, nvdec etc.), videoconvert, appsink
        # Caps must match what the sender (rtph264pay) is sending
        caps = f"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264"
        # Using avdec_h264, replace with hardware decoder if available
        # e.g., vaapih264dec ! vaapipostproc (Intel), omxh264dec (RPi), nvh264dec (Nvidia)
        # Check decoder availability with `gst-inspect-1.0 avdec_h264` etc.
        decoder = "avdec_h264"
        # Add queue for buffering
        pipeline_str = (
            f"udpsrc port={self.port} caps=\"{caps}\" ! "
            f"queue ! " # Add buffer between network and decoder
            f"rtph264depay ! "
            f"queue ! " # Add buffer between depay and decoder
            f"{decoder} ! "
            f"videoconvert ! " # Convert to RGB for QImage
            f"video/x-raw,format=RGB ! "
            f"queue max-size-buffers=2 leaky=downstream ! " # Small queue before appsink
            f"appsink name=qtsink emit-signals=true sync=false max-buffers=2 drop=true" # emit-signals is key
        )
        print("--- UI Client GStreamer Pipeline ---")
        print(pipeline_str)
        print("------------------------------------")
        return pipeline_str

    def run(self):
        print("Starting GStreamer receiver thread...")
        # Acquire the context for this thread
        self.gst_context.push_thread_default()

        Gst.init(None)
        self.loop = GLib.MainLoop(context=self.gst_context) # Run loop in this thread's context

        try:
            self.pipeline = Gst.parse_launch(self._build_pipeline_string())
        except GLib.Error as e:
             print(f"Error parsing receiver pipeline: {e}")
             self.error_signal.emit(f"Failed to create GStreamer pipeline: {e}")
             self._running = False
             self.gst_context.pop_thread_default()
             return
        except Exception as e:
            print(f"Unexpected error creating receiver pipeline: {e}")
            self.error_signal.emit(f"Unexpected error creating pipeline: {e}")
            self._running = False
            self.gst_context.pop_thread_default()
            return

        # Get the appsink element
        appsink = self.pipeline.get_by_name("qtsink")
        if not appsink:
            self.error_signal.emit("Failed to get appsink element 'qtsink'")
            self._running = False
            self.gst_context.pop_thread_default()
            return

        # Connect the 'new-sample' signal
        appsink.connect("new-sample", self.on_new_sample, None) # Pass None as user_data

        # Connect bus messages for errors/EOS
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self.on_error)
        bus.connect("message::eos", self.on_eos)
        bus.connect("message::warning", self.on_warning)

        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
             self.error_signal.emit("Failed to set receiver pipeline to PLAYING state.")
             self._running = False
             self.gst_context.pop_thread_default()
             return

        print("Receiver pipeline playing...")

        # Run the GLib main loop (will block here)
        if self._running:
             try:
                 self.loop.run()
             except Exception as e:
                  print(f"Error in GStreamer thread main loop: {e}")
                  self.error_signal.emit(f"Runtime error in GStreamer loop: {e}")

        # --- Cleanup after loop exits ---
        print("GStreamer receiver thread loop finished.")
        if self.pipeline:
             print("Setting receiver pipeline to NULL state...")
             self.pipeline.set_state(Gst.State.NULL)
             self.pipeline = None

        # Release the context
        self.gst_context.pop_thread_default()
        print("GStreamer receiver thread finished cleanup.")


    def on_new_sample(self, sink, _user_data):
        # Pull the sample from the appsink
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            if not caps or not buf:
                return Gst.FlowReturn.ERROR # Invalid sample

            # Get width, height from caps structure
            structure = caps.get_structure(0)
            if not structure:
                 return Gst.FlowReturn.ERROR

            h = structure.get_value("height")
            w = structure.get_value("width")
            # format_str = structure.get_value("format") # Should be RGB

            # Map buffer to readable data
            result, mapinfo = buf.map(Gst.MapFlags.READ)
            if result:
                try:
                    # Create NumPy array from buffer data (no copy if possible)
                    # Assuming RGB format from pipeline
                    frame = np.ndarray((h, w, 3), buffer=mapinfo.data, dtype=np.uint8)

                    # IMPORTANT: We need to copy the frame data because the Qt signal
                    # might process it later in a different thread after the buffer is unmapped.
                    # Check if frame data is valid before copying
                    if frame.size == 0:
                        print("Warning: Received empty frame data.")
                        buf.unmap(mapinfo)
                        return Gst.FlowReturn.OK # Or drop?

                    frame_copy = frame.copy()

                    # Convert NumPy array to QImage (no copy needed here)
                    # Ensure stride is correct (w * 3 for RGB)
                    qt_image = QImage(frame_copy.data, w, h, w * 3, QImage.Format_RGB888)

                    # Convert QImage to QPixmap (this might involve a copy depending on Qt version/platform)
                    qt_pixmap = QPixmap.fromImage(qt_image)

                    # Emit the signal with the pixmap
                    self.new_frame_signal.emit(qt_pixmap)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                finally:
                    # Unmap the buffer in a finally block
                    buf.unmap(mapinfo)

            else:
                 print("Warning: Failed to map GStreamer buffer.")


            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

    def on_error(self, bus, message):
        err, debug = message.parse_error()
        element_name = message.src.get_name() if message.src else "Unknown Element"
        error_str = f"GStreamer Error from {element_name}: {err}\nDebug: {debug}"
        print(error_str, file=sys.stderr)
        self.error_signal.emit(error_str)
        # Don't stop the thread immediately, let the main UI decide or handle reconnection perhaps
        # self.stop()

    def on_warning(self, bus, message):
        warn, debug = message.parse_warning()
        element_name = message.src.get_name() if message.src else "Unknown Element"
        print(f"GStreamer Warning from {element_name}: {warn}\nDebug: {debug}", file=sys.stderr)

    def on_eos(self, bus, message):
        print("GStreamer Receiver: End-Of-Stream received from pipeline.")
        self.eos_signal.emit()
        # Don't stop thread here automatically, EOS might just mean backend stopped streaming
        # self.stop()

    def stop(self):
        print("Stopping GStreamer receiver thread...")
        if not self._running:
            print("Receiver thread already stopping/stopped.")
            return

        self._running = False

        # Schedule the loop quit from the loop's context thread
        if self.loop and self.loop.is_running():
            self.gst_context.invoke(self.loop.quit) # Use invoke for thread safety
            print("Scheduled GLib loop quit.")
        else:
             print("GLib loop not running or not initialized.")


# --- Main UI Widget ---
class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_label = QLabel("Waiting for video stream...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #333; color: white;")
        # Make label expand and keep aspect ratio (optional, but good for resizing)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(320, 240) # Start with a reasonable minimum size

        # --- UI Elements ---
        self.status_label = QLabel("Status: Initializing...")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100) # Represent 0.0 to 1.0
        self.threshold_slider.setValue(50) # Default 0.5
        self.threshold_label = QLabel(f"Threshold: {self.threshold_slider.value()/100.0:.2f}")
        self.pause_button = QPushButton("Pause/Play Backend")
        self.reconnect_button = QPushButton("Reconnect Stream")
        self.reconnect_button.setEnabled(False) # Enable on error/EOS

        # --- Layout ---
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Threshold:"))
        control_layout.addWidget(self.threshold_slider)
        control_layout.addWidget(self.threshold_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reconnect_button)
        button_layout.addStretch()


        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, 1) # Video label takes expanding space
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # --- GStreamer Thread Setup ---
        self.gst_thread = None # Initialize later
        self.start_gst_thread()

        # --- Connections ---
        self.threshold_slider.valueChanged.connect(self.slider_changed)
        self.pause_button.clicked.connect(self.toggle_backend_pause)
        self.reconnect_button.clicked.connect(self.start_gst_thread)


    def start_gst_thread(self):
        if self.gst_thread and self.gst_thread.isRunning():
             print("GStreamer thread already running.")
             return

        print("Starting new GStreamer receiver thread...")
        self.gst_thread = GStreamerReceiverThread()
        self.gst_thread.new_frame_signal.connect(self.update_video_label)
        self.gst_thread.error_signal.connect(self.handle_gst_error)
        self.gst_thread.eos_signal.connect(self.handle_gst_eos)
        self.gst_thread.finished.connect(self.gst_thread_finished) # Connect finished signal
        self.gst_thread.start()
        self.status_label.setText("Status: Connecting to stream...")
        self.video_label.setText("Connecting...")
        self.video_label.setStyleSheet("background-color: #333; color: white;")
        self.reconnect_button.setEnabled(False) # Disable while connecting


    @Slot(QPixmap)
    def update_video_label(self, pixmap):
        if not self.video_label.isVisible():
            return # Don't update if widget isn't visible

        # Scale pixmap to fit label while maintaining aspect ratio
        # Use Qt.KeepAspectRatioByExpanding for filling, or Qt.KeepAspectRatio for fitting within bounds
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        if "Connecting" in self.status_label.text() or "Error" in self.status_label.text() or "Stopped" in self.status_label.text():
             self.status_label.setText("Status: Receiving video")
             self.reconnect_button.setEnabled(False) # Disable reconnect if receiving


    @Slot(int)
    def slider_changed(self, value):
        float_value = value / 100.0
        self.threshold_label.setText(f"Threshold: {float_value:.2f}")
        # Send command to backend
        try:
            payload = {'threshold': float_value}
            # Run in a separate thread to avoid blocking UI? For quick requests, maybe not needed.
            response = requests.post(f"{BACKEND_COMMAND_URL}/update_data", json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            result = response.json()
            # print(f"Backend response: {result}")
            # Check response status if backend provides it
            update_status = "OK"
            if result and result.get('updates'):
                for update in result['updates']:
                    if not update.get('success'):
                        update_status = f"Failed ({update.get('message', 'Unknown error')})"
                        break
            self.status_label.setText(f"Status: Threshold update sent ({update_status})")
        except requests.exceptions.Timeout:
            error_msg = f"Error: Command timed out after {REQUEST_TIMEOUT}s"
            print(error_msg, file=sys.stderr)
            self.status_label.setText(f"Status: {error_msg}")
        except requests.exceptions.RequestException as e:
            error_msg = f"Error sending command: {e}"
            print(error_msg, file=sys.stderr)
            self.status_label.setText(f"Status: Error - {error_msg}")
            # Optionally show a message box:
            # QMessageBox.warning(self, "Command Error", f"Could not send threshold update:\n{e}")

    @Slot()
    def toggle_backend_pause(self):
         try:
            payload = {'action': 'toggle_pause'}
            response = requests.post(f"{BACKEND_COMMAND_URL}/control", json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            print(f"Backend control response: {result}")
            self.status_label.setText(f"Status: Toggle Pause/Play sent ({result.get('message', 'No message')})")
         except requests.exceptions.Timeout:
            error_msg = f"Error: Command timed out after {REQUEST_TIMEOUT}s"
            print(error_msg, file=sys.stderr)
            self.status_label.setText(f"Status: {error_msg}")
         except requests.exceptions.RequestException as e:
            error_msg = f"Error sending command: {e}"
            print(error_msg, file=sys.stderr)
            self.status_label.setText(f"Status: Error - {error_msg}")

    @Slot(str)
    def handle_gst_error(self, error_message):
        print(f"handle_gst_error called: {error_message}")
        self.video_label.setText(f"GStreamer Error:\n{error_message}\n\nCheck backend and network.\nTry reconnecting.")
        self.video_label.setStyleSheet("background-color: darkred; color: white;")
        self.status_label.setText(f"Status: GStreamer Error - Check Logs")
        # Don't show modal messagebox, as it blocks UI, just update status
        # QMessageBox.critical(self, "GStreamer Error", error_message)
        self.reconnect_button.setEnabled(True) # Enable reconnect on error

    @Slot()
    def handle_gst_eos(self):
         print("handle_gst_eos called.")
         self.video_label.setText("Video stream ended (EOS).\nBackend might have stopped.\nTry reconnecting.")
         self.video_label.setStyleSheet("background-color: #555; color: white;")
         self.status_label.setText("Status: Stream Ended (EOS)")
         self.reconnect_button.setEnabled(True) # Enable reconnect on EOS


    @Slot()
    def gst_thread_finished(self):
        print("GStreamer receiver thread has finished signal received.")
        if "Error" not in self.status_label.text() and "EOS" not in self.status_label.text(): # Don't overwrite error/EOS messages
             self.status_label.setText("Status: GStreamer thread stopped.")
             self.video_label.setText("Video stream stopped.")
             self.video_label.setStyleSheet("background-color: #555; color: white;")
        self.reconnect_button.setEnabled(True) # Enable reconnect when thread stops


    def stop_gst_thread(self):
         if self.gst_thread and self.gst_thread.isRunning():
             print("Requesting GStreamer thread stop...")
             self.gst_thread.stop() # Signal GStreamer thread to stop
             print("Waiting for GStreamer thread to finish...")
             if not self.gst_thread.wait(3000): # Wait up to 3 seconds
                 print("Warning: GStreamer thread did not stop gracefully. Forcing termination.")
                 self.gst_thread.terminate() # Use terminate as last resort
                 self.gst_thread.wait() # Wait after terminate
             print("GStreamer thread stopped.")
         self.gst_thread = None


    def closeEvent(self, event):
        print("Closing UI application...")
        self.stop_gst_thread()
        event.accept()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hailo Detection UI Client")
        # Set window icon (optional)
        # self.setWindowIcon(QIcon("path/to/your/icon.png"))
        self.central_widget = VideoWidget(self)
        self.setCentralWidget(self.central_widget)
        self.resize(740, 640) # Initial size


# --- Main execution ---
if __name__ == "__main__":
    # Function to handle Ctrl+C gracefully for the Qt app
    def signal_handler(sig, frame):
        print("\nCtrl+C detected in UI. Shutting down...")
        QApplication.quit()

    # Set GST_DEBUG env variable if needed, e.g., "GST_DEBUG=udpsrc:4,rtph264depay:4,avdec_h264:2,appsink:4"
    # os.environ["GST_DEBUG"] = "3" # General level 3 debug
    # os.environ["GST_DEBUG_FILE"] = "gst_client.log" # Log to file

    # Install signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Required for signals/slots to work across threads with GStreamer/GLib
    GObject.threads_init()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start timer to periodically process signals when main loop might be busy
    timer = QTimer()
    timer.start(500)  # Milliseconds
    timer.timeout.connect(lambda: None) # Necessary to keep Python interpreter running

    sys.exit(app.exec())