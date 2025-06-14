"""
NEXARIS Cognitive Load Estimator (NCLE)

GUI Module

Builds and manages the user interface using PyQt or similar.
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTabWidget, QTextEdit, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# Import other NCLE modules (placeholders for now, will need actual integration)
# from . import task_simulation
# from . import tracking
# from . import face_analysis
# from . import scoring

class NCLEMainWindow(QMainWindow):
    """Main window for the NCLE application."""
    def __init__(self, app_context=None):
        super().__init__()
        self.app_context = app_context # To hold references to other modules (task_manager, etc.)
        self.setWindowTitle("NEXARIS Cognitive Load Estimator (NCLE)")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height

        self._init_ui()
        self._connect_signals()
        self._start_timers()

    def _init_ui(self):
        """Initialize the main UI components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Top Bar (Status/Controls) ---
        self.top_bar_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.status_label.setFont(QFont("Arial", 10))
        self.start_button = QPushButton("Start Session")
        self.stop_button = QPushButton("Stop Session")
        self.stop_button.setEnabled(False)
        self.top_bar_layout.addWidget(self.status_label, stretch=1)
        self.top_bar_layout.addWidget(self.start_button)
        self.top_bar_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(self.top_bar_layout)

        # --- Main Content Area (Tabs) ---
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # Tab 1: Dashboard / Real-time View
        self.dashboard_tab = QWidget()
        self.dashboard_layout = QVBoxLayout(self.dashboard_tab)
        self.cognitive_load_display = QLabel("Cognitive Load: --%")
        self.cognitive_load_display.setFont(QFont("Arial", 24, QFont.Bold))
        self.cognitive_load_display.setAlignment(Qt.AlignCenter)
        self.dashboard_layout.addWidget(self.cognitive_load_display)
        # TODO: Add graphs for real-time data (e.g., using Matplotlib or PyQtGraph)
        self.tab_widget.addTab(self.dashboard_tab, "Dashboard")

        # Tab 2: Task View
        self.task_tab = QWidget()
        self.task_layout = QVBoxLayout(self.task_tab)
        self.task_description_label = QLabel("Current Task: None")
        self.task_question_area = QTextEdit("Task questions will appear here.")
        self.task_question_area.setReadOnly(True)
        self.task_answer_button = QPushButton("Submit Answer / Next") # Context-dependent
        self.task_layout.addWidget(self.task_description_label)
        self.task_layout.addWidget(self.task_question_area, stretch=1)
        self.task_layout.addWidget(self.task_answer_button)
        self.tab_widget.addTab(self.task_tab, "Task")

        # Tab 3: Webcam View (Face Analysis)
        self.webcam_tab = QWidget()
        self.webcam_layout = QVBoxLayout(self.webcam_tab)
        self.webcam_feed_label = QLabel("Webcam feed will appear here.") # Placeholder
        self.webcam_feed_label.setAlignment(Qt.AlignCenter)
        self.webcam_feed_label.setFrameShape(QFrame.Box)
        self.webcam_feed_label.setMinimumSize(640, 480)
        self.emotion_label = QLabel("Detected Emotion: N/A")
        self.webcam_layout.addWidget(self.webcam_feed_label, stretch=1)
        self.webcam_layout.addWidget(self.emotion_label)
        self.tab_widget.addTab(self.webcam_tab, "Face Analysis")

        # Tab 4: Logs
        self.log_tab = QWidget()
        self.log_layout = QVBoxLayout(self.log_tab)
        self.log_display_area = QTextEdit("Application logs will appear here...")
        self.log_display_area.setReadOnly(True)
        self.log_display_area.setFont(QFont("Courier New", 9))
        self.log_layout.addWidget(self.log_display_area)
        self.tab_widget.addTab(self.log_tab, "Logs")

        # --- Bottom Bar (Summary/Info) ---
        self.bottom_bar_layout = QHBoxLayout()
        self.session_time_label = QLabel("Session Time: 00:00:00")
        self.bottom_bar_layout.addWidget(self.session_time_label)
        self.main_layout.addLayout(self.bottom_bar_layout)

    def _connect_signals(self):
        """Connect UI element signals to handler methods."""
        self.start_button.clicked.connect(self.start_session)
        self.stop_button.clicked.connect(self.stop_session)
        # TODO: Connect signals from app_context modules to update UI elements
        # e.g., self.app_context.facial_analyzer.frame_ready.connect(self.update_webcam_feed)
        # e.g., self.app_context.score_calculator.score_updated.connect(self.update_cognitive_load_display)

    def _start_timers(self):
        """Start timers for UI updates or periodic checks."""
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_ui_elements)
        self.ui_update_timer.start(100) # Update 10 times per second

        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self.update_session_time)
        self.session_start_time = None

    def start_session(self):
        """Handler for starting a new cognitive load estimation session."""
        self.status_label.setText("Status: Session Active")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_message("Session started.")

        # Start backend processes (these are placeholders)
        if self.app_context:
            # self.app_context.task_manager.start_task("default_task_id") # Example
            # self.app_context.behavior_tracker.start_tracking()
            # self.app_context.facial_analyzer.start_analysis()
            pass
        
        self.session_start_time = QTimer.elapsed(QApplication.instance()) # Using QTimer for elapsed time
        self.session_timer.start(1000) # Update session time every second
        self.update_session_time() # Initial update

    def stop_session(self):
        """Handler for stopping the current session."""
        self.status_label.setText("Status: Idle")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Session stopped.")

        if self.app_context:
            # self.app_context.behavior_tracker.stop_tracking()
            # self.app_context.facial_analyzer.stop_analysis()
            pass
        
        self.session_timer.stop()

    def update_ui_elements(self):
        """Periodically update UI elements based on backend data."""
        # This method would poll or react to signals from backend modules
        if not self.app_context or not self.stop_button.isEnabled(): # Only update if session is active
            return

        # Example: Update cognitive load display
        # current_load = self.app_context.score_calculator.get_current_score()
        # self.cognitive_load_display.setText(f"Cognitive Load: {current_load:.1f}%")

        # Example: Update emotion display
        # current_emotion = self.app_context.facial_analyzer.get_current_emotion()
        # self.emotion_label.setText(f"Detected Emotion: {current_emotion}")
        
        # Example: Update webcam feed (if using QPixmap)
        # frame_pixmap = self.app_context.facial_analyzer.get_display_frame_pixmap()
        # if frame_pixmap:
        #    self.webcam_feed_label.setPixmap(frame_pixmap.scaled(
        #        self.webcam_feed_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        #    ))
        pass

    def update_session_time(self):
        """Update the displayed session time."""
        if self.session_start_time is not None and self.stop_button.isEnabled():
            elapsed_ms = QTimer.elapsed(QApplication.instance()) - self.session_start_time
            seconds = int((elapsed_ms / 1000) % 60)
            minutes = int((elapsed_ms / (1000 * 60)) % 60)
            hours = int((elapsed_ms / (1000 * 60 * 60)) % 24)
            self.session_time_label.setText(f"Session Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.session_time_label.setText("Session Time: 00:00:00")

    def log_message(self, message):
        """Append a message to the log display area."""
        timestamp = QTimer.elapsed(QApplication.instance()) # Or use datetime
        self.log_display_area.append(f"[{timestamp/1000.0:.2f}s] {message}")

    def closeEvent(self, event):
        """Handle window close event."""
        # Ensure background processes are stopped cleanly
        self.stop_session() 
        # if self.app_context and hasattr(self.app_context.facial_analyzer, '__del__'):
        #    self.app_context.facial_analyzer.__del__() # Explicitly call if needed for camera release
        super().closeEvent(event)

class GUI:
    """Main GUI application class to encapsulate PyQt app setup."""
    def __init__(self, app_context=None):
        self.app = QApplication.instance() # Check if an instance already exists
        if not self.app:
            self.app = QApplication(sys.argv)
        self.main_window = NCLEMainWindow(app_context)

    def start(self):
        """Show the main window and start the Qt application event loop."""
        self.main_window.show()
        # sys.exit(self.app.exec_()) # This should be called from main.py typically
        self.app.exec_()

if __name__ == '__main__':
    # This allows running gui.py directly for testing UI components
    # In the full application, main.py would instantiate and run the GUI.
    
    # Mock app_context for standalone testing
    class MockAppContext:
        pass
        # task_manager = None # Initialize with mock objects if needed
        # behavior_tracker = None
        # facial_analyzer = None
        # score_calculator = None

    # context = MockAppContext()
    # ncle_gui = GUI(app_context=context)
    ncle_gui = GUI() # No context for simple standalone test
    ncle_gui.start()