#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Window for NEXARIS Cognitive Load Estimator

This module provides the main application window and UI components
for the NEXARIS Cognitive Load Estimator.
"""

import os
import sys
import logging
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QSplitter, QFrame,
    QMessageBox, QFileDialog, QStatusBar, QAction, QToolBar,
    QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSlot, QSettings
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPalette

# Import core components
from ..core.data_manager import DataManager
from ..core.task_simulator import TaskSimulator
from ..core.behavior_tracker import BehaviorTracker
from ..core.facial_analyzer import FacialAnalyzer
from ..core.cognitive_load_calculator import CognitiveLoadCalculator
from ..core.eeg_integration import EEGIntegration

# Import UI components
from .dashboard_widget import DashboardWidget
from .task_widget import TaskWidget
from .settings_widget import SettingsWidget
from .visualization_widget import VisualizationWidget
from .camera_widget import CameraWidget
from .live_analysis_panel import LiveAnalysisPanel # Added import

# Import utilities
from ..utils.logging_utils import get_logger
from ..utils.config_utils import load_config, save_config
import psutil


class MainWindow(QMainWindow):
    """
    Main application window for NEXARIS Cognitive Load Estimator
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initialize the main window
        
        Args:
            config: Application configuration dictionary
            data_manager: Data manager instance
        """
        super().__init__()
        self.config = config
        self.data_manager = data_manager
        self.logger = get_logger(__name__)
        self.csv_writer = None
        self.csv_file = None
        self.csv_headers_written = False
        
        # Initialize UI components
        self.setup_ui()
        
        # Initialize core components
        self.setup_core_components()
        
        # Connect signals and slots
        self.connect_signals()
        
        # Set up timers
        self.setup_timers()
        
        # Set up status bar
        self.setup_status_bar()
        
        # Load window state
        self.load_window_state()
        
        self.logger.info("Main window initialized")

        # CPU Usage Monitor
        system_monitoring_config = self.config.get('system_monitoring', {})
        self.cpu_usage_threshold = system_monitoring_config.get('cpu_warning_threshold', 80)
        self.cpu_check_interval_ms = system_monitoring_config.get('cpu_check_interval', 5000)
        self.cpu_monitor_timer = QTimer(self)
        self.cpu_monitor_timer.timeout.connect(self._check_cpu_usage)
        if system_monitoring_config.get('enable_cpu_monitoring', True):
            self.cpu_monitor_timer.start(self.cpu_check_interval_ms)
            self.logger.info(f"CPU usage monitoring started. Threshold: {self.cpu_usage_threshold}%, Interval: {self.cpu_check_interval_ms}ms")
        else:
            self.logger.info("CPU usage monitoring is disabled in config.")
    
    def setup_ui(self) -> None:
        """
        Set up the user interface
        """
        # Set window properties
        self.setWindowTitle("NEXARIS Cognitive Load Estimator")
        self.setMinimumSize(1024, 768)
        
        # Set window icon if available
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'nexaris_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Changed to QHBoxLayout to accommodate the splitter
        self.main_layout = QHBoxLayout(self.central_widget) 
        self.main_layout.setContentsMargins(5, 5, 5, 5) # Reduced margins for tighter layout
        self.main_layout.setSpacing(5) # Reduced spacing

        # Create Live Analysis Panel
        self.live_analysis_panel = LiveAnalysisPanel(self.config, self)

        # Create Splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.live_analysis_panel)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        # self.main_layout.addWidget(self.tab_widget) # Tab widget now goes into splitter
        self.splitter.addWidget(self.tab_widget)
        self.main_layout.addWidget(self.splitter)

        # Set initial splitter sizes (adjust as needed)
        # Give more space to the tab_widget initially
        total_width = self.width() if self.width() > 0 else 1024 # Use default if not yet sized
        panel_width = int(total_width * 0.3) # 30% for the live panel
        self.splitter.setSizes([panel_width, total_width - panel_width])
        
        # Create dashboard tab
        self.dashboard_widget = DashboardWidget(self.config)
        self.tab_widget.addTab(self.dashboard_widget, "Dashboard")
        
        # Create task tab
        self.task_widget = TaskWidget(self.config)
        self.tab_widget.addTab(self.task_widget, "Task Simulator")
        
        # Create visualization tab
        self.visualization_widget = VisualizationWidget(self.data_manager, self.config)
        self.tab_widget.addTab(self.visualization_widget, "Visualizations")
        
        # Create settings tab
        self.settings_widget = SettingsWidget(self.config)
        self.tab_widget.addTab(self.settings_widget, "Settings")
        
        # Create camera widget (not in tabs)
        self.camera_widget = CameraWidget()
        self.camera_widget.setVisible(False)  # Hidden by default
        
        # Create menu bar
        self.setup_menu_bar()
        
        # Create tool bar
        self.setup_tool_bar()
        
        # Apply theme
        self.apply_theme()

    def _setup_csv_logging(self):
        """
        Sets up CSV logging for the new session.
        Creates a /logs directory if it doesn't exist.
        """
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
                self.logger.info(f"Created logs directory: {logs_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = self.data_manager.get_current_session_id() or f"session_{timestamp}"
            # Sanitize session_id for filename
            safe_session_id = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in str(session_id))
            
            filename = os.path.join(logs_dir, f"{safe_session_id}.csv")
            
            # Close existing CSV file if open
            self._close_csv_logger()

            self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_headers_written = False # Reset for new file
            self.logger.info(f"Opened CSV log file: {filename}")
            # Example: Write headers immediately if known, or defer to first log_event_to_csv call
            # self.log_event_to_csv({}, write_header_if_needed=True) # To write headers if predefined

        except Exception as e:
            self.logger.error(f"Failed to set up CSV logging: {e}")
            QMessageBox.warning(self, "Logging Error", f"Could not start CSV logging: {e}")
            if self.csv_file:
                self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def log_event_to_csv(self, event_data: Dict[str, Any]):
        """
        Logs an event (a dictionary) to the CSV file.
        Writes headers if this is the first write operation to the file.
        Ensures 'timestamp' is always the first column if present.
        """
        if not self.csv_writer or not self.csv_file:
            # self.logger.warning("CSV logging not set up, cannot log event.") # Can be noisy
            return

        try:
            # Ensure standard columns like timestamp are present
            if 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now().isoformat()
            
            if not self.csv_headers_written:
                # Order headers: 'timestamp' first, then others alphabetically for consistency
                headers = sorted(list(event_data.keys()))
                if 'timestamp' in headers: # Move timestamp to the front
                    headers.remove('timestamp')
                    headers.insert(0, 'timestamp')
                self.csv_writer.writerow(headers)
                self.csv_headers_written = True
            
            # Ensure row values are in the same order as headers written
            # This requires knowing the header order established above.
            # For simplicity, let's re-fetch headers if dynamic or ensure event_data always has all keys
            # A more robust way is to store the headers list and use it to order values.
            # For now, assuming keys in event_data are consistent after first write or using a fixed set.
            # If headers are truly dynamic per event, this needs more complex handling.
            
            # Let's assume for now that the first event_data dictates the headers for the session.
            # If subsequent events have different keys, they might not align or cause errors.
            # A better approach: define expected headers at _setup_csv_logging or have a fixed schema.
            
            # Simplified: write values in order of current event_data keys (matches header if first call)
            # This might lead to misaligned columns if keys change order/number later.
            # A robust solution would be to get current headers from self.csv_headers list (if stored)
            # and write event_data.get(header, '') for each header.
            
            # For now, let's assume the first call to log_event_to_csv defines the headers,
            # and subsequent calls should provide data for those headers.
            # If a key is missing in a subsequent event, it will write an empty string.
            # If a new key appears, it won't be logged unless headers are re-written (not ideal).

            # Let's refine: Store headers once written and use them for subsequent writes.
            if not hasattr(self, '_csv_column_headers') or not self._csv_column_headers:
                 # This block should ideally only run if self.csv_headers_written was just set to True
                temp_headers = sorted(list(event_data.keys()))
                if 'timestamp' in temp_headers:
                    temp_headers.remove('timestamp')
                    temp_headers.insert(0, 'timestamp')
                self._csv_column_headers = temp_headers # Store the headers

            row_values = [event_data.get(header, '') for header in self._csv_column_headers]
            self.csv_writer.writerow(row_values)
            self.csv_file.flush() # Ensure data is written to disk periodically
        except Exception as e:
            self.logger.error(f"Error writing to CSV log: {e}")

    def _close_csv_logger(self):
        """
        Closes the CSV file if it's open.
        """
        if self.csv_file:
            try:
                self.csv_file.close()
                self.logger.info("CSV log file closed.")
            except Exception as e:
                self.logger.error(f"Error closing CSV log file: {e}")
            finally:
                self.csv_file = None
                self.csv_writer = None
                self.csv_headers_written = False
                if hasattr(self, '_csv_column_headers'):
                    delattr(self, '_csv_column_headers')
    
    def setup_menu_bar(self) -> None:
        """
        Set up the menu bar
        """
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # New session action
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self.on_new_session) # Connected to existing on_new_session
        file_menu.addAction(new_session_action)
        
        # Open session action
        open_session_action = QAction("&Open Session", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self.on_open_session)
        file_menu.addAction(open_session_action)
        
        # Save session action
        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self.on_save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        # Export data action
        export_data_action = QAction("&Export Data", self)
        export_data_action.setShortcut("Ctrl+E")
        export_data_action.triggered.connect(self.on_export_data)
        file_menu.addAction(export_data_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close) # self.close will trigger closeEvent
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        # Toggle camera action
        self.toggle_camera_action = QAction("&Camera View", self)
        self.toggle_camera_action.setCheckable(True)
        self.toggle_camera_action.setChecked(False)
        self.toggle_camera_action.triggered.connect(self.on_toggle_camera)
        view_menu.addAction(self.toggle_camera_action)
        
        # Toggle fullscreen action
        self.toggle_fullscreen_action = QAction("&Fullscreen", self)
        self.toggle_fullscreen_action.setCheckable(True)
        self.toggle_fullscreen_action.setChecked(False)
        self.toggle_fullscreen_action.setShortcut("F11")
        self.toggle_fullscreen_action.triggered.connect(self.on_toggle_fullscreen)
        view_menu.addAction(self.toggle_fullscreen_action)
        
        # Task menu
        task_menu = self.menuBar().addMenu("&Task")
        
        # Start task action
        start_task_action = QAction("&Start Task", self)
        start_task_action.setShortcut("F5")
        start_task_action.triggered.connect(self.on_start_task)
        task_menu.addAction(start_task_action)
        
        # Stop task action
        stop_task_action = QAction("S&top Task", self)
        stop_task_action.setShortcut("F6")
        stop_task_action.triggered.connect(self.on_stop_task)
        task_menu.addAction(stop_task_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
    
    def setup_tool_bar(self) -> None:
        """
        Set up the tool bar
        """
        self.tool_bar = QToolBar("Main Toolbar")
        self.tool_bar.setIconSize(QSize(24, 24))
        self.addToolBar(self.tool_bar)
        
        # Add actions to toolbar
        # New session action
        new_session_action = QAction("New Session", self)
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'new_session.png')
        if os.path.exists(icon_path):
            new_session_action.setIcon(QIcon(icon_path))
        new_session_action.triggered.connect(self.on_new_session)
        self.tool_bar.addAction(new_session_action)
        
        # Start task action
        start_task_action = QAction("Start Task", self)
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'start_task.png')
        if os.path.exists(icon_path):
            start_task_action.setIcon(QIcon(icon_path))
        start_task_action.triggered.connect(self.on_start_task)
        self.tool_bar.addAction(start_task_action)
        
        # Stop task action
        stop_task_action = QAction("Stop Task", self)
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'stop_task.png')
        if os.path.exists(icon_path):
            stop_task_action.setIcon(QIcon(icon_path))
        stop_task_action.triggered.connect(self.on_stop_task)
        self.tool_bar.addAction(stop_task_action)
        
        self.tool_bar.addSeparator()
        
        # Toggle camera action
        toggle_camera_action = QAction("Camera View", self)
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'camera.png')
        if os.path.exists(icon_path):
            toggle_camera_action.setIcon(QIcon(icon_path))
        toggle_camera_action.setCheckable(True)
        toggle_camera_action.setChecked(False)
        toggle_camera_action.triggered.connect(self.on_toggle_camera)
        self.tool_bar.addAction(toggle_camera_action)
    
    def setup_status_bar(self) -> None:
        """
        Set up the status bar
        """
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add status labels
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Add cognitive load indicator
        self.load_indicator_label = QLabel("Cognitive Load:")
        self.status_bar.addPermanentWidget(self.load_indicator_label)
        
        self.load_value_label = QLabel("N/A")
        self.load_value_label.setMinimumWidth(80)
        self.load_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_bar.addPermanentWidget(self.load_value_label)
        
        # Add EEG status indicator
        self.eeg_status_label = QLabel("EEG: Not Connected")
        self.status_bar.addPermanentWidget(self.eeg_status_label)
    
    def setup_core_components(self) -> None:
        """
        Set up core application components
        """
        # Create task simulator
        self.task_simulator = TaskSimulator(self.config)
        
        # Create behavior tracker
        self.behavior_tracker = BehaviorTracker(self.config)
        
        # Create facial analyzer
        self.facial_analyzer = FacialAnalyzer(self.config)
        
        # Initialize Cognitive Load Calculator
        self.cognitive_load_calculator = CognitiveLoadCalculator(self.config, csv_logger_callback=self.log_event_to_csv)
        self.logger.info("Cognitive Load Calculator initialized in MainWindow with CSV logging")
        
        # Create EEG integration
        self.eeg_integration = EEGIntegration(self.config)
        
        # Install event filter for behavior tracking
        self.behavior_tracker.install_event_filter(self)
        
        # Pass components to widgets
        self.dashboard_widget.set_components(
            self.task_simulator,
            self.behavior_tracker,
            self.facial_analyzer,
            self.cognitive_load_calculator,
            self.eeg_integration
        )
        
        self.task_widget.set_components(
            self.task_simulator,
            self.behavior_tracker
        )
        
        self.visualization_widget.set_components(
            self.data_manager,
            self.cognitive_load_calculator
        )
        
        self.camera_widget.set_components(
            self.facial_analyzer
        )

        # Pass components to LiveAnalysisPanel
        self.live_analysis_panel.set_core_components(
            self.task_simulator,
            self.behavior_tracker,
            self.facial_analyzer,
            self.cognitive_load_calculator
        )
        
        # Register data callbacks
        self.behavior_tracker.register_data_callback(self.on_behavior_data)
        self.facial_analyzer.register_data_callback(self.on_facial_data)
        self.cognitive_load_calculator.register_data_callback(self.on_cognitive_load_data)
        self.eeg_integration.register_data_callback(self.on_eeg_data)
        
        # Start facial analysis if enabled
        if self.config.get('facial_analysis', {}).get('enabled', True):
            self.facial_analyzer.start_analysis()
        
        # Connect to EEG device if enabled
        if self.config.get('advanced_features', {}).get('eeg', {}).get('enabled', False):
            self.eeg_integration.connect()
    
    def connect_signals(self) -> None:
        """
        Connect signals and slots
        """
        # Connect task simulator signals
        self.task_simulator.task_started.connect(self.on_task_started)
        self.task_simulator.task_completed.connect(self.on_task_completed)
        self.task_simulator.task_progress.connect(self.on_task_progress)
        
        # Connect behavior tracker signals
        self.behavior_tracker.hesitation_detected.connect(self.on_hesitation_detected)
        
        # Connect facial analyzer signals
        self.facial_analyzer.face_detected.connect(self.on_face_detected)
        self.facial_analyzer.emotion_detected.connect(self.on_emotion_detected)
        self.facial_analyzer.frame_processed.connect(self.camera_widget.update_frame)
        
        # Connect cognitive load calculator signals
        self.cognitive_load_calculator.load_updated.connect(self.on_load_updated)
        self.cognitive_load_calculator.threshold_exceeded.connect(self.on_threshold_exceeded)
        
        # Connect EEG integration signals
        self.eeg_integration.connection_status_changed.connect(self.on_eeg_connection_status)
        self.eeg_integration.cognitive_load_update.connect(self.on_eeg_load_update)
        
        # Connect settings widget signals
        self.settings_widget.settings_changed.connect(self.on_settings_changed)
        
        # Connect tab widget signals
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Connect LiveAnalysisPanel signals
        self.live_analysis_panel.start_task_requested.connect(self.on_start_task)
        self.live_analysis_panel.stop_task_requested.connect(self.on_stop_task)

        # Ensure live panel buttons reflect main task state from toolbar/menu actions
        # And also when task starts/stops from task_widget directly
        self.task_simulator.task_started.connect(self.live_analysis_panel.on_task_started) # Connect directly
        self.task_simulator.task_completed.connect(self.live_analysis_panel.on_task_stopped_or_completed)
        if hasattr(self.task_simulator, 'task_aborted'): # If task_aborted signal exists
            self.task_simulator.task_aborted.connect(self.live_analysis_panel.on_task_stopped_or_completed)
    
    def setup_timers(self) -> None:
        """
        Set up application timers
        """
        # Create update timer for UI refresh
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # 10 Hz refresh rate
    
    def _check_cpu_usage(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            if cpu_percent > self.cpu_usage_threshold:
                self.logger.warning(f"High CPU usage detected: {cpu_percent}% (Threshold: {self.cpu_usage_threshold}%)")
                QMessageBox.warning(self, "High CPU Usage",
                                    f"CPU usage is at {cpu_percent}%, which is above the threshold of {self.cpu_usage_threshold}%.\n"
                                    "Consider closing other applications or reducing workload.")
                # Potential actions: self.reduce_system_load()
            
            # If the check was successful (did not raise an exception), reset the fail count.
            if hasattr(self, '_cpu_check_fail_count') and self._cpu_check_fail_count > 0:
                self.logger.debug("CPU usage check successful, resetting fail count.")
                self._cpu_check_fail_count = 0
                
        except Exception as e:
            self.logger.error(f"Error checking CPU usage: {e}", exc_info=True)
            if not hasattr(self, '_cpu_check_fail_count'):
                self._cpu_check_fail_count = 0
            self._cpu_check_fail_count += 1
            self.logger.warning(f"CPU usage check failed ({self._cpu_check_fail_count} consecutive failures).")
            if self._cpu_check_fail_count > 3:
                if self.cpu_monitor_timer.isActive():
                    self.cpu_monitor_timer.stop()
                    self.logger.error("Stopping CPU monitor due to repeated errors checking CPU usage.")

    # def reduce_system_load(self):
    #     """ Placeholder for actions to reduce system load, e.g., lower FPS """
    #     self.logger.info("Attempting to reduce system load due to high CPU usage...")
    #     if self.facial_analyzer and self.facial_analyzer.is_analyzing():
    #         # Example: try to reduce facial analysis processing frequency or complexity
    #         # This would require methods in FacialAnalyzer to adjust its parameters
    #         self.logger.info("Suggesting reduction in facial analysis intensity (not yet implemented).")
    #     # Add other load reduction strategies here

    def apply_theme(self) -> None:
        """
        Apply the application theme
        """
        # Get theme configuration
        ui_config = self.config.get('ui', {})
        theme = ui_config.get('theme', 'light')
        
        if theme == 'dark':
            # Set dark theme
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            QApplication.setPalette(palette)
        else:
            # Set light theme (default)
            QApplication.setPalette(QApplication.style().standardPalette())
        
        # Set font
        font_family = ui_config.get('font_family', 'Segoe UI')
        font_size = ui_config.get('font_size', 10)
        font = QFont(font_family, font_size)
        QApplication.setFont(font)
    
    def load_window_state(self) -> None:
        """
        Load window state from settings
        """
        settings = QSettings("NEXARIS", "Cognitive Load Estimator")
        
        # Restore window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def save_window_state(self) -> None:
        """
        Save window state to settings
        """
        settings = QSettings("NEXARIS", "Cognitive Load Estimator")
        
        # Save window geometry
        settings.setValue("geometry", self.saveGeometry())
        
        # Save window state
        settings.setValue("windowState", self.saveState())
    
    def update_ui(self) -> None:
        """
        Update UI components
        """
        # Update dashboard
        self.dashboard_widget.update_ui()
        
        # Update task widget if active
        if self.tab_widget.currentWidget() == self.task_widget:
            self.task_widget.update_ui()
        
        # Update visualization widget if active
        if self.tab_widget.currentWidget() == self.visualization_widget:
            self.visualization_widget.update_ui()
        
        # Update camera widget if visible (this is the pop-out one)
        if self.camera_widget.isVisible():
            self.camera_widget.update_ui()
        
        # LiveAnalysisPanel updates itself via its own timer and signal connections
        # No explicit call needed here unless specific cross-updates are required.
    
    @pyqtSlot()
    def on_new_session(self) -> None:
        """
        Handle new session creation. This includes resetting data and setting up logging.
        """
        # Confirm if there's an active session
        if self.data_manager.is_session_active():
            reply = QMessageBox.question(
                self, "New Session",
                "This will end the current session. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # End current session
            self.data_manager.end_current_session()
        
        # Start new session
        session_id = self.data_manager.start_new_session()
        
        # Update status
        self.status_label.setText(f"Session {session_id} started")
        
        # Reset components
        self.behavior_tracker.reset_tracking_data()
        self.cognitive_load_calculator.reset_tracking_data()
        
        # Start behavior tracking
        self.behavior_tracker.start_tracking()
        
        # Update dashboard
        self.dashboard_widget.on_new_session(session_id)
        
        self.logger.info("New session started")
        # Setup CSV logging for the new session
        self._setup_csv_logging()
        QMessageBox.information(self, "New Session", "New session started, data reset, and CSV logging initiated.")
    
    @pyqtSlot()
    def on_open_session(self) -> None:
        """
        Handle open session action
        """
        # Get list of available sessions
        sessions = self.data_manager.list_sessions()
        
        if not sessions:
            QMessageBox.information(
                self, "Open Session",
                "No saved sessions found."
            )
            return
        
        # TODO: Implement session selection dialog
        # For now, just show a message
        QMessageBox.information(
            self, "Open Session",
            "Session loading not implemented yet."
        )
    
    @pyqtSlot()
    def on_save_session(self) -> None:
        """
        Handle save session action
        """
        if not self.data_manager.is_session_active():
            QMessageBox.information(
                self, "Save Session",
                "No active session to save."
            )
            return
        
        # Save current session
        session_id = self.data_manager.get_current_session_id()
        success = self.data_manager.save_session(session_id)
        
        if success:
            self.status_label.setText(f"Session {session_id} saved")
            self.logger.info(f"Session saved: {session_id}")
        else:
            QMessageBox.warning(
                self, "Save Session",
                "Failed to save session."
            )
    
    @pyqtSlot()
    def on_export_data(self) -> None:
        """
        Handle export data action
        """
        if not self.data_manager.is_session_active():
            QMessageBox.information(
                self, "Export Data",
                "No active session to export."
            )
            return
        
        # Get export file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Session Data", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Export session data
        session_id = self.data_manager.get_current_session_id()
        success = self.data_manager.export_session_data(session_id, file_path)
        
        if success:
            self.status_label.setText(f"Session data exported to {file_path}")
            self.logger.info(f"Session data exported: {session_id} to {file_path}")
        else:
            QMessageBox.warning(
                self, "Export Data",
                "Failed to export session data."
            )
    
    @pyqtSlot(bool)
    def on_toggle_camera(self, checked: bool) -> None:
        """
        Handle toggle camera action
        
        Args:
            checked: Whether the action is checked
        """
        self.camera_widget.setVisible(checked)
        self.toggle_camera_action.setChecked(checked)
        
        # Update camera widget position
        if checked:
            # Position camera widget in bottom right corner
            desktop = QApplication.desktop()
            screen_rect = desktop.availableGeometry(self)
            camera_rect = self.camera_widget.frameGeometry()
            camera_rect.moveBottomRight(screen_rect.bottomRight())
            self.camera_widget.move(camera_rect.topLeft())
            
            # Ensure camera widget is on top
            self.camera_widget.raise_()
    
    @pyqtSlot(bool)
    def on_toggle_fullscreen(self, checked: bool) -> None:
        """
        Handle toggle fullscreen action
        
        Args:
            checked: Whether the action is checked
        """
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
    
    @pyqtSlot()
    def on_start_task(self) -> None:
        """
        Handle start task action
        """
        # Switch to task tab
        self.tab_widget.setCurrentWidget(self.task_widget)
        
        # Start task
        self.task_widget.start_task()
    
    @pyqtSlot()
    def on_stop_task(self) -> None:
        """
        Handle stop task action
        """
        # Stop task
        self.task_widget.stop_task()
    
    @pyqtSlot()
    def on_about(self) -> None:
        """
        Handle about action
        """
        QMessageBox.about(
            self, "About NEXARIS Cognitive Load Estimator",
            "<h1>NEXARIS Cognitive Load Estimator</h1>"
            "<p>Version 1.0</p>"
            "<p>A tool for estimating cognitive load during tasks.</p>"
            "<p>© 2023 NEXARIS</p>"
        )
    
    @pyqtSlot(dict)
    def on_task_started(self, task_data: Dict[str, Any]) -> None:
        """
        Handle task started signal
        
        Args:
            task_data: Task data dictionary
        """
        # Get task ID from task data
        task_id = task_data.get('id', 'unknown')
        
        # Update status
        self.status_label.setText(f"Task {task_id} started")
        
        # Record task start in data manager
        if self.data_manager.is_session_active():
            session_id = self.data_manager.get_current_session_id()
            self.data_manager.start_task(session_id, task_id)
        
        self.logger.info(f"Task started: {task_id}")
    
    @pyqtSlot(dict)
    def on_task_completed(self, task_data: Dict[str, Any]) -> None:
        """
        Handle task completed signal
        
        Args:
            task_data: Task data dictionary
        """
        # Get task ID and results from task data
        task_id = task_data.get('id', 'unknown')
        results = task_data.get('results', {})
        
        # Update status
        self.status_label.setText(f"Task {task_id} completed")
        
        # Record task completion in data manager
        if self.data_manager.is_session_active():
            session_id = self.data_manager.get_current_session_id()
            self.data_manager.end_task(session_id, task_id, results)
        
        # Get behavior metrics
        behavior_metrics = self.behavior_tracker.get_metrics()
        
        # Update cognitive load calculator with performance metrics
        performance_metrics_for_calc = {
            'accuracy': results.get('accuracy', 1.0),
            'response_time': results.get('avg_response_time', 0.0),
            'completion_rate': results.get('completion_rate', 1.0),
            'difficulty': results.get('difficulty', 0.5)
        }
        # The new calculator expects a combined call
        # self.cognitive_load_calculator.update_performance_metrics(performance_metrics_for_calc)
        # Instead, we'll rely on the calculator's internal logic or a combined update method if available
        # For now, let's assume the calculator is updated elsewhere or this is sufficient
        # If CognitiveLoadCalculator has a method to directly take task results, use that.
        # For now, this part remains as is, but might need adjustment based on CognitiveLoadCalculator's API.
        # The new `calculate_combined_cognitive_load` in the calculator might be triggered by data callbacks.
        self.cognitive_load_calculator.update_performance_metrics(performance_metrics_for_calc)

        
        # Show results dialog
        self.show_task_results(task_id, results, behavior_metrics)
        
        self.logger.info(f"Task completed: {task_id}")
    
    @pyqtSlot(int, int)
    def on_task_progress(self, current: int, total: int) -> None:
        """
        Handle task progress signal
        
        Args:
            current: Current progress value
            total: Total progress value
        """
        # Calculate progress percentage and remaining time
        progress = current / total if total > 0 else 0.0
        
        # Update status
        self.status_label.setText(
            f"Task in progress: {progress:.0%} ({current}/{total})"
        )
    
    @pyqtSlot(float)
    def on_hesitation_detected(self, duration: float) -> None:
        """
        Handle hesitation detected signal
        
        Args:
            duration: Hesitation duration in seconds
        """
        self.logger.debug(f"Hesitation detected: {duration:.2f} seconds")
    
    @pyqtSlot(bool)
    def on_face_detected(self, detected: bool) -> None:
        """
        Handle face detected signal
        
        Args:
            detected: Whether a face is detected
        """
        # Update camera widget
        self.camera_widget.set_face_detected(detected)
    
    @pyqtSlot(str, float)
    def on_emotion_detected(self, emotion: str, confidence: float) -> None:
        """
        Handle emotion detected signal
        
        Args:
            emotion: Detected emotion
            confidence: Detection confidence
        """
        # Update camera widget
        self.camera_widget.set_emotion(emotion, confidence)
    
    @pyqtSlot(float, dict)
    def on_load_updated(self, load: float, components: Dict[str, float]) -> None:
        """
        Handle cognitive load updated signal
        
        Args:
            load: Cognitive load value (0.0 to 1.0)
            components: Component breakdown
        """
        # Update load value label
        self.load_value_label.setText(f"{load:.2f}")
        
        # Set color based on load level
        if load < 0.4:
            self.load_value_label.setStyleSheet("color: green;")
        elif load < 0.7:
            self.load_value_label.setStyleSheet("color: orange;")
        else:
            self.load_value_label.setStyleSheet("color: red;")
        
        # Update dashboard
        self.dashboard_widget.update_cognitive_load(load, components)
    
    @pyqtSlot(float, float)
    def on_threshold_exceeded(self, load: float, threshold: float) -> None:
        """
        Handle cognitive load threshold exceeded signal
        
        Args:
            load: Cognitive load value (0.0 to 1.0)
            threshold: Threshold value
        """
        # Show notification if not in task
        if not self.task_simulator.is_task_active():
            self.status_label.setText(
                f"High cognitive load detected: {load:.2f} (threshold: {threshold:.2f})"
            )
    
    @pyqtSlot(bool, str)
    def on_eeg_connection_status(self, connected: bool, message: str) -> None:
        """
        Handle EEG connection status signal
        
        Args:
            connected: Whether EEG is connected
            message: Status message
        """
        # Update EEG status label
        if connected:
            self.eeg_status_label.setText(f"EEG: {message}")
            self.eeg_status_label.setStyleSheet("color: green;")
        else:
            self.eeg_status_label.setText(f"EEG: {message}")
            self.eeg_status_label.setStyleSheet("color: gray;")
    
    @pyqtSlot(float)
    def on_eeg_load_update(self, load: float) -> None:
        """
        Handle EEG cognitive load update signal
        
        Args:
            load: Cognitive load value from EEG (0.0 to 1.0)
        """
        # Update cognitive load calculator with EEG metrics
        eeg_metrics = {
            'cognitive_load': load,
            'signal_quality': self.eeg_integration.signal_quality
        }
        self.cognitive_load_calculator.update_eeg_metrics(eeg_metrics)
    
    @pyqtSlot(dict)
    def on_behavior_data(self, data: Dict[str, Any]) -> None:
        """
        Handle behavior data callback
        
        Args:
            data: Behavior data
        """
        # Record behavior data in data manager
        if self.data_manager.is_session_active():
            session_id = self.data_manager.get_current_session_id()
            self.data_manager.record_behavior_data(session_id, data)
        
        # Update cognitive load calculator with behavior metrics
        if data['data_type'] == 'mouse_move' or \
           data['data_type'] == 'mouse_click' or \
           data['data_type'] == 'key_press' or \
           data['data_type'] == 'hesitation':
            behavior_metrics = self.behavior_tracker.get_metrics()
            self.cognitive_load_calculator.update_behavior_metrics(behavior_metrics)
    
    @pyqtSlot(dict)
    def on_facial_data(self, data: Dict[str, Any]) -> None:
        """
        Handle facial data callback
        
        Args:
            data: Facial data
        """
        # Record facial data in data manager
        if self.data_manager.is_session_active():
            session_id = self.data_manager.get_current_session_id()
            self.data_manager.record_facial_data(session_id, data)
        
        # Update cognitive load calculator with facial metrics
        if data['data_type'] == 'facial_analysis':
            facial_metrics = self.facial_analyzer.get_metrics()
            self.cognitive_load_calculator.update_facial_metrics(facial_metrics)
    
    @pyqtSlot(dict)
    def on_cognitive_load_data(self, data: Dict[str, Any]) -> None:
        """
        Handle cognitive load data callback
        
        Args:
            data: Cognitive load data
        """
        # Record cognitive load data in data manager
        if self.data_manager.is_session_active():
            session_id = self.data_manager.get_current_session_id()
            self.data_manager.record_cognitive_load_data(session_id, data)
    
    @pyqtSlot(dict)
    def on_eeg_data(self, data: Dict[str, Any]) -> None:
        """
        Handle EEG data callback
        
        Args:
            data: EEG data
        """
        # Record EEG data in data manager
        if self.data_manager.is_session_active():
            session_id = self.data_manager.get_current_session_id()
            self.data_manager.record_eeg_data(session_id, data)
    
    @pyqtSlot(dict)
    def on_settings_changed(self, new_settings: Dict[str, Any]) -> None:
        """
        Handle settings changed signal
        
        Args:
            new_settings: New settings from the settings widget
        """
        self.logger.debug(f"on_settings_changed called with: {new_settings}")

        # Store old values for comparison. These are from self.config BEFORE it's updated.
        old_fa_config = self.config.get('facial_analysis', {})
        old_fa_enabled = old_fa_config.get('enabled', True)
        
        old_eeg_config = self.config.get('advanced_features', {}).get('eeg', {})
        old_eeg_enabled = old_eeg_config.get('enabled', False)
        
        old_sm_config = self.config.get('system_monitoring', {})
        old_sm_enabled = old_sm_config.get('enable_cpu_monitoring', True)
        old_cpu_threshold = old_sm_config.get('cpu_warning_threshold', 80)
        old_cpu_interval = old_sm_config.get('cpu_check_interval', 5000)

        # Update main configuration object with the new settings
        self.config.update(new_settings)
        
        # Save configuration to disk
        try:
            save_config(self.config)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}", exc_info=True)
            QMessageBox.warning(self, "Settings Error", f"Could not save configuration: {e}")
        
        # Apply UI theme (might depend on new config)
        self.apply_theme()
        
        # Apply changes to components based on the new self.config
        
        # Facial Analysis
        current_fa_config = self.config.get('facial_analysis', {}) # Gets from updated self.config
        new_fa_enabled_from_current_config = current_fa_config.get('enabled', True)
        
        if new_fa_enabled_from_current_config != old_fa_enabled:
            if new_fa_enabled_from_current_config:
                if hasattr(self.facial_analyzer, 'start_analysis'): # Check if method exists
                    self.logger.info("Facial analysis enabled by settings change. Starting analysis.")
                    self.facial_analyzer.start_analysis()
            else:
                if hasattr(self.facial_analyzer, 'stop_analysis'): # Check if method exists
                    self.logger.info("Facial analysis disabled by settings change. Stopping analysis.")
                    self.facial_analyzer.stop_analysis()
        
        # EEG Integration
        current_eeg_config = self.config.get('advanced_features', {}).get('eeg', {})
        new_eeg_enabled_from_current_config = current_eeg_config.get('enabled', False)
        if new_eeg_enabled_from_current_config != old_eeg_enabled:
            if new_eeg_enabled_from_current_config:
                if hasattr(self.eeg_integration, 'connect'): # Check if method exists
                    self.logger.info("EEG integration enabled by settings change. Connecting.")
                    self.eeg_integration.connect()
            else:
                if hasattr(self.eeg_integration, 'disconnect'): # Check if method exists
                    self.logger.info("EEG integration disabled by settings change. Disconnecting.")
                    self.eeg_integration.disconnect()

        # System Monitoring (CPU)
        current_sm_config = self.config.get('system_monitoring', {})
        new_sm_enabled_from_current_config = current_sm_config.get('enable_cpu_monitoring', True)
        new_cpu_threshold_from_current_config = current_sm_config.get('cpu_warning_threshold', 80)
        new_cpu_interval_from_current_config = current_sm_config.get('cpu_check_interval', 5000)

        config_changed = (
            new_sm_enabled_from_current_config != old_sm_enabled or
            new_cpu_threshold_from_current_config != old_cpu_threshold or
            new_cpu_interval_from_current_config != old_cpu_interval
        )

        if config_changed:
            self.cpu_usage_threshold = new_cpu_threshold_from_current_config
            self.cpu_check_interval_ms = new_cpu_interval_from_current_config
            
            if new_sm_enabled_from_current_config:
                timer_needs_restart = (
                    not self.cpu_monitor_timer.isActive() or
                    old_sm_enabled != new_sm_enabled_from_current_config or
                    old_cpu_interval != new_cpu_interval_from_current_config
                )
                if timer_needs_restart:
                    if self.cpu_monitor_timer.isActive():
                        self.cpu_monitor_timer.stop()
                    self.cpu_monitor_timer.start(self.cpu_check_interval_ms)
                    self.logger.info(f"CPU usage monitoring started/updated. Threshold: {self.cpu_usage_threshold}%, Interval: {self.cpu_check_interval_ms}ms")
            else: # Not enabled
                if self.cpu_monitor_timer.isActive():
                    self.cpu_monitor_timer.stop()
                    self.logger.info("CPU usage monitoring stopped by settings change.")
        
        self.logger.info("Settings updated and applied.")
    
    @pyqtSlot(int)
    def on_tab_changed(self, index: int) -> None:
        """
        Handle tab changed signal
        
        Args:
            index: New tab index
        """
        # Update UI based on current tab
        current_widget = self.tab_widget.widget(index)
        
        if current_widget == self.task_widget:
            # Update task widget
            self.task_widget.update_ui()
        elif current_widget == self.visualization_widget:
            # Update visualization widget
            self.visualization_widget.update_ui()
    
    def show_task_results(self, task_id: str, results: Dict[str, Any], behavior_metrics: Dict[str, Any]) -> None:
        """
        Show task results dialog
        
        Args:
            task_id: Task ID
            results: Task results
            behavior_metrics: Behavior metrics
        """
        # Get cognitive load
        cognitive_load = self.cognitive_load_calculator.get_current_load()
        
        # Create message
        message = f"<h2>Task Results</h2>"
        message += f"<p><b>Task:</b> {task_id}</p>"
        message += f"<p><b>Accuracy:</b> {results.get('accuracy', 0.0):.0%}</p>"
        message += f"<p><b>Average Response Time:</b> {results.get('avg_response_time', 0.0):.2f} seconds</p>"
        message += f"<p><b>Completion Rate:</b> {results.get('completion_rate', 0.0):.0%}</p>"
        message += f"<p><b>Mouse Distance:</b> {behavior_metrics.get('mouse_distance', 0.0):.0f} pixels</p>"
        message += f"<p><b>Click Count:</b> {behavior_metrics.get('click_count', 0)}</p>"
        message += f"<p><b>Keypress Count:</b> {behavior_metrics.get('keypress_count', 0)}</p>"
        message += f"<p><b>Hesitation Count:</b> {behavior_metrics.get('hesitation_count', 0)}</p>"
        message += f"<p><b>Cognitive Load:</b> {cognitive_load.get('smoothed_ecls', 0.0):.2f}</p>"
        message += f"<p><b>Load Level:</b> {cognitive_load.get('load_level', 'unknown').replace('_', ' ').title()}</p>"
        
        # Show dialog
        QMessageBox.information(self, "Task Results", message)
    
    def closeEvent(self, event) -> None:
        """
        Handle window close event. Ensures components are stopped and state is saved.
        
        Args:
            event: Close event
        """
        self.logger.info("Initiating shutdown sequence...")
        
        # Stop core components (ensure threads are closed, etc.)
        try:
            if hasattr(self, 'facial_analyzer') and self.facial_analyzer and hasattr(self.facial_analyzer, 'is_analyzing') and self.facial_analyzer.is_analyzing():
                self.logger.info("Stopping facial analyzer...")
                if hasattr(self.facial_analyzer, 'stop_analysis'):
                    self.facial_analyzer.stop_analysis()
        except Exception as e:
            self.logger.error(f"Error stopping facial analyzer: {e}", exc_info=True)

        try:
            if hasattr(self, 'behavior_tracker') and self.behavior_tracker and hasattr(self.behavior_tracker, 'is_tracking') and self.behavior_tracker.is_tracking():
                self.logger.info("Stopping behavior tracker...")
                if hasattr(self.behavior_tracker, 'stop_tracking'):
                    self.behavior_tracker.stop_tracking()
        except Exception as e:
            self.logger.error(f"Error stopping behavior tracker: {e}", exc_info=True)

        try:
            if hasattr(self, 'eeg_integration') and self.eeg_integration and hasattr(self.eeg_integration, 'is_active') and self.eeg_integration.is_active():
                self.logger.info("Stopping EEG integration...")
                if hasattr(self.eeg_integration, 'stop'):
                    self.eeg_integration.stop()
        except Exception as e:
            self.logger.error(f"Error stopping EEG integration: {e}", exc_info=True)

        try:
            if hasattr(self, 'task_simulator') and self.task_simulator and hasattr(self.task_simulator, 'is_running') and self.task_simulator.is_running():
                self.logger.info("Stopping task simulator...")
                if hasattr(self.task_simulator, 'stop_task'):
                    self.task_simulator.stop_task()
        except Exception as e:
            self.logger.error(f"Error stopping task simulator: {e}", exc_info=True)

        try:
            if hasattr(self, 'cpu_monitor_timer') and self.cpu_monitor_timer.isActive():
                self.logger.info("Stopping CPU monitor timer...")
                self.cpu_monitor_timer.stop()
        except Exception as e:
            self.logger.error(f"Error stopping CPU monitor timer: {e}", exc_info=True)
            
        # Close CSV logger
        try:
            self._close_csv_logger()
        except Exception as e:
            self.logger.error(f"Error closing CSV logger: {e}", exc_info=True)

        # Save window state
        try:
            self.save_window_state()
        except Exception as e:
            self.logger.error(f"Error saving window state: {e}", exc_info=True)
        
        self.logger.info("Main window closed and resources released.")
        super().closeEvent(event)