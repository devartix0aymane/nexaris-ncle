#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Settings Widget for NEXARIS Cognitive Load Estimator

This module provides the settings widget for configuring the application.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox, QSlider, QScrollArea, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSettings
from PyQt5.QtGui import QFont

# Import utilities
from ..utils.logging_utils import get_logger
from ..utils.config_utils import save_config


class SettingsWidget(QWidget):
    """
    Widget for application settings
    """
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config.copy()  # Make a copy to avoid modifying the original
        self.logger = get_logger(__name__)
        
        # Create layout
        self.setup_ui()
        
        # Load settings
        self.load_settings()
    
    def setup_ui(self) -> None:
        """
        Set up the user interface
        """
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create scroll area for settings
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Create scroll content widget
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(20)
        
        # Create settings tabs
        self.tab_widget = QTabWidget()
        self.scroll_layout.addWidget(self.tab_widget)
        
        # Create general settings tab
        self.general_tab = QWidget()
        self.tab_widget.addTab(self.general_tab, "General")
        self.setup_general_tab()
        
        # Create UI settings tab
        self.ui_tab = QWidget()
        self.tab_widget.addTab(self.ui_tab, "User Interface")
        self.setup_ui_tab()
        
        # Create task settings tab
        self.task_tab = QWidget()
        self.tab_widget.addTab(self.task_tab, "Task Simulation")
        self.setup_task_tab()
        
        # Create tracking settings tab
        self.tracking_tab = QWidget()
        self.tab_widget.addTab(self.tracking_tab, "Behavior Tracking")
        self.setup_tracking_tab()
        
        # Create facial analysis settings tab
        self.facial_tab = QWidget()
        self.tab_widget.addTab(self.facial_tab, "Facial Analysis")
        self.setup_facial_tab()
        
        # Create scoring settings tab
        self.scoring_tab = QWidget()
        self.tab_widget.addTab(self.scoring_tab, "Cognitive Load Scoring")
        self.setup_scoring_tab()
        
        # Create advanced settings tab
        self.advanced_tab = QWidget()
        self.tab_widget.addTab(self.advanced_tab, "Advanced Features")
        self.setup_advanced_tab()
        
        # Create data settings tab
        self.data_tab = QWidget()
        self.tab_widget.addTab(self.data_tab, "Data Storage")
        self.setup_data_tab()
        
        # Set scroll area widget
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)
        
        # Create button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(10)
        
        # Create spacer
        self.button_layout.addStretch()
        
        # Create reset button
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        self.button_layout.addWidget(self.reset_button)
        
        # Create apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_settings)
        self.button_layout.addWidget(self.apply_button)
        
        self.main_layout.addLayout(self.button_layout)
    
    def setup_general_tab(self) -> None:
        """
        Set up the general settings tab
        """
        # Create layout
        self.general_layout = QVBoxLayout(self.general_tab)
        self.general_layout.setContentsMargins(10, 10, 10, 10)
        self.general_layout.setSpacing(20)
        
        # Create application settings group
        self.app_group = QGroupBox("Application Settings")
        self.app_layout = QFormLayout(self.app_group)
        
        # Application name
        self.app_name_edit = QLineEdit()
        self.app_layout.addRow("Application Name:", self.app_name_edit)
        
        # Application version
        self.app_version_edit = QLineEdit()
        self.app_version_edit.setReadOnly(True)
        self.app_layout.addRow("Version:", self.app_version_edit)
        
        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.app_layout.addRow("Log Level:", self.log_level_combo)
        
        # Auto-save settings
        self.auto_save_check = QCheckBox("Auto-save settings on exit")
        self.app_layout.addRow("", self.auto_save_check)
        
        self.general_layout.addWidget(self.app_group)
        
        # Create session settings group
        self.session_group = QGroupBox("Session Settings")
        self.session_layout = QFormLayout(self.session_group)
        
        # Auto-start session
        self.auto_start_session_check = QCheckBox("Auto-start session on launch")
        self.session_layout.addRow("", self.auto_start_session_check)
        
        # Auto-save session
        self.auto_save_session_check = QCheckBox("Auto-save session data")
        self.session_layout.addRow("", self.auto_save_session_check)
        
        # Session save interval
        self.session_save_interval_spin = QSpinBox()
        self.session_save_interval_spin.setRange(1, 60)
        self.session_save_interval_spin.setSuffix(" minutes")
        self.session_layout.addRow("Auto-save Interval:", self.session_save_interval_spin)
        
        self.general_layout.addWidget(self.session_group)
        
        # Add spacer
        self.general_layout.addStretch()
    
    def setup_ui_tab(self) -> None:
        """
        Set up the UI settings tab
        """
        # Create layout
        self.ui_layout = QVBoxLayout(self.ui_tab)
        self.ui_layout.setContentsMargins(10, 10, 10, 10)
        self.ui_layout.setSpacing(20)
        
        # Create appearance group
        self.appearance_group = QGroupBox("Appearance")
        self.appearance_layout = QFormLayout(self.appearance_group)
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.appearance_layout.addRow("Theme:", self.theme_combo)
        
        # Font family
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(["System Default", "Segoe UI", "Arial", "Helvetica", "Times New Roman"])
        self.appearance_layout.addRow("Font Family:", self.font_family_combo)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setSuffix(" pt")
        self.appearance_layout.addRow("Font Size:", self.font_size_spin)
        
        self.ui_layout.addWidget(self.appearance_group)
        
        # Create dashboard group
        self.dashboard_group = QGroupBox("Dashboard")
        self.dashboard_layout = QFormLayout(self.dashboard_group)
        
        # Show gauge
        self.show_gauge_check = QCheckBox("Show cognitive load gauge")
        self.dashboard_layout.addRow("", self.show_gauge_check)
        
        # Show time series
        self.show_time_series_check = QCheckBox("Show time series chart")
        self.dashboard_layout.addRow("", self.show_time_series_check)
        
        # Show component breakdown
        self.show_components_check = QCheckBox("Show component breakdown chart")
        self.dashboard_layout.addRow("", self.show_components_check)
        
        # Chart update interval
        self.chart_update_interval_spin = QSpinBox()
        self.chart_update_interval_spin.setRange(100, 5000)
        self.chart_update_interval_spin.setSingleStep(100)
        self.chart_update_interval_spin.setSuffix(" ms")
        self.dashboard_layout.addRow("Chart Update Interval:", self.chart_update_interval_spin)
        
        self.ui_layout.addWidget(self.dashboard_group)
        
        # Create visualization group
        self.visualization_group = QGroupBox("Visualizations")
        self.visualization_layout = QFormLayout(self.visualization_group)
        
        # Chart style
        self.chart_style_combo = QComboBox()
        self.chart_style_combo.addItems(["Default", "ggplot", "bmh", "dark_background", "seaborn"])
        self.visualization_layout.addRow("Chart Style:", self.chart_style_combo)
        
        # Color map
        self.color_map_combo = QComboBox()
        self.color_map_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])
        self.visualization_layout.addRow("Color Map:", self.color_map_combo)
        
        self.ui_layout.addWidget(self.visualization_group)
        
        # Add spacer
        self.ui_layout.addStretch()
    
    def setup_task_tab(self) -> None:
        """
        Set up the task simulation settings tab
        """
        # Create layout
        self.task_layout = QVBoxLayout(self.task_tab)
        self.task_layout.setContentsMargins(10, 10, 10, 10)
        self.task_layout.setSpacing(20)
        
        # Create task settings group
        self.task_settings_group = QGroupBox("Task Settings")
        self.task_settings_layout = QFormLayout(self.task_settings_group)
        
        # Default task type
        self.default_task_type_combo = QComboBox()
        self.default_task_type_combo.addItems(["Question Set", "Alert Simulation"])
        self.task_settings_layout.addRow("Default Task Type:", self.default_task_type_combo)
        
        # Default question set
        self.default_question_set_combo = QComboBox()
        self.default_question_set_combo.addItems(["General Knowledge", "Cybersecurity", "Alert Triage"])
        self.task_settings_layout.addRow("Default Question Set:", self.default_question_set_combo)
        
        # Default difficulty
        self.default_difficulty_combo = QComboBox()
        self.default_difficulty_combo.addItems(["Easy", "Medium", "Hard"])
        self.task_settings_layout.addRow("Default Difficulty:", self.default_difficulty_combo)
        
        # Default time limit
        self.default_time_limit_spin = QSpinBox()
        self.default_time_limit_spin.setRange(30, 600)
        self.default_time_limit_spin.setSingleStep(30)
        self.default_time_limit_spin.setSuffix(" seconds")
        self.task_settings_layout.addRow("Default Time Limit:", self.default_time_limit_spin)
        
        # Default item count
        self.default_item_count_spin = QSpinBox()
        self.default_item_count_spin.setRange(1, 20)
        self.task_settings_layout.addRow("Default Item Count:", self.default_item_count_spin)
        
        self.task_layout.addWidget(self.task_settings_group)
        
        # Create question settings group
        self.question_settings_group = QGroupBox("Question Settings")
        self.question_settings_layout = QFormLayout(self.question_settings_group)
        
        # Custom question set path
        self.custom_question_set_edit = QLineEdit()
        self.custom_question_set_layout = QHBoxLayout()
        self.custom_question_set_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_question_set_layout.setSpacing(5)
        self.custom_question_set_layout.addWidget(self.custom_question_set_edit)
        
        self.custom_question_set_button = QPushButton("Browse...")
        self.custom_question_set_button.clicked.connect(self.browse_custom_question_set)
        self.custom_question_set_layout.addWidget(self.custom_question_set_button)
        
        self.question_settings_layout.addRow("Custom Question Set:", self.custom_question_set_layout)
        
        # Show correct answers
        self.show_correct_answers_check = QCheckBox("Show correct answers after submission")
        self.question_settings_layout.addRow("", self.show_correct_answers_check)
        
        # Randomize question order
        self.randomize_questions_check = QCheckBox("Randomize question order")
        self.question_settings_layout.addRow("", self.randomize_questions_check)
        
        self.task_layout.addWidget(self.question_settings_group)
        
        # Create alert settings group
        self.alert_settings_group = QGroupBox("Alert Settings")
        self.alert_settings_layout = QFormLayout(self.alert_settings_group)
        
        # Custom alert set path
        self.custom_alert_set_edit = QLineEdit()
        self.custom_alert_set_layout = QHBoxLayout()
        self.custom_alert_set_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_alert_set_layout.setSpacing(5)
        self.custom_alert_set_layout.addWidget(self.custom_alert_set_edit)
        
        self.custom_alert_set_button = QPushButton("Browse...")
        self.custom_alert_set_button.clicked.connect(self.browse_custom_alert_set)
        self.custom_alert_set_layout.addWidget(self.custom_alert_set_button)
        
        self.alert_settings_layout.addRow("Custom Alert Set:", self.custom_alert_set_layout)
        
        # Randomize alert order
        self.randomize_alerts_check = QCheckBox("Randomize alert order")
        self.alert_settings_layout.addRow("", self.randomize_alerts_check)
        
        # Require notes
        self.require_notes_check = QCheckBox("Require notes for alert actions")
        self.alert_settings_layout.addRow("", self.require_notes_check)
        
        self.task_layout.addWidget(self.alert_settings_group)
        
        # Add spacer
        self.task_layout.addStretch()
    
    def setup_tracking_tab(self) -> None:
        """
        Set up the behavior tracking settings tab
        """
        # Create layout
        self.tracking_layout = QVBoxLayout(self.tracking_tab)
        self.tracking_layout.setContentsMargins(10, 10, 10, 10)
        self.tracking_layout.setSpacing(20)
        
        # Create tracking settings group
        self.tracking_settings_group = QGroupBox("Tracking Settings")
        self.tracking_settings_layout = QFormLayout(self.tracking_settings_group)
        
        # Enable tracking
        self.enable_tracking_check = QCheckBox("Enable behavior tracking")
        self.tracking_settings_layout.addRow("", self.enable_tracking_check)
        
        # Track mouse movement
        self.track_mouse_movement_check = QCheckBox("Track mouse movement")
        self.tracking_settings_layout.addRow("", self.track_mouse_movement_check)
        
        # Track mouse clicks
        self.track_mouse_clicks_check = QCheckBox("Track mouse clicks")
        self.tracking_settings_layout.addRow("", self.track_mouse_clicks_check)
        
        # Track key presses
        self.track_key_presses_check = QCheckBox("Track key presses")
        self.tracking_settings_layout.addRow("", self.track_key_presses_check)
        
        # Track hesitation
        self.track_hesitation_check = QCheckBox("Track hesitation")
        self.tracking_settings_layout.addRow("", self.track_hesitation_check)
        
        self.tracking_layout.addWidget(self.tracking_settings_group)
        
        # Create hesitation settings group
        self.hesitation_settings_group = QGroupBox("Hesitation Settings")
        self.hesitation_settings_layout = QFormLayout(self.hesitation_settings_group)
        
        # Hesitation threshold
        self.hesitation_threshold_spin = QDoubleSpinBox()
        self.hesitation_threshold_spin.setRange(0.5, 10.0)
        self.hesitation_threshold_spin.setSingleStep(0.1)
        self.hesitation_threshold_spin.setDecimals(1)
        self.hesitation_threshold_spin.setSuffix(" seconds")
        self.hesitation_settings_layout.addRow("Hesitation Threshold:", self.hesitation_threshold_spin)
        
        # Minimum hesitation duration
        self.min_hesitation_duration_spin = QDoubleSpinBox()
        self.min_hesitation_duration_spin.setRange(0.5, 5.0)
        self.min_hesitation_duration_spin.setSingleStep(0.1)
        self.min_hesitation_duration_spin.setDecimals(1)
        self.min_hesitation_duration_spin.setSuffix(" seconds")
        self.hesitation_settings_layout.addRow("Minimum Hesitation Duration:", self.min_hesitation_duration_spin)
        
        # Check interval
        self.check_interval_spin = QSpinBox()
        self.check_interval_spin.setRange(50, 1000)
        self.check_interval_spin.setSingleStep(50)
        self.check_interval_spin.setSuffix(" ms")
        self.hesitation_settings_layout.addRow("Check Interval:", self.check_interval_spin)
        
        self.tracking_layout.addWidget(self.hesitation_settings_group)
        
        # Create data collection group
        self.data_collection_group = QGroupBox("Data Collection")
        self.data_collection_layout = QFormLayout(self.data_collection_group)
        
        # Sampling rate
        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(1, 100)
        self.sampling_rate_spin.setSuffix(" Hz")
        self.data_collection_layout.addRow("Sampling Rate:", self.sampling_rate_spin)
        
        # Buffer size
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(10, 1000)
        self.buffer_size_spin.setSingleStep(10)
        self.buffer_size_spin.setSuffix(" samples")
        self.data_collection_layout.addRow("Buffer Size:", self.buffer_size_spin)
        
        self.tracking_layout.addWidget(self.data_collection_group)
        
        # Add spacer
        self.tracking_layout.addStretch()
    
    def setup_facial_tab(self) -> None:
        """
        Set up the facial analysis settings tab
        """
        # Create layout
        self.facial_layout = QVBoxLayout(self.facial_tab)
        self.facial_layout.setContentsMargins(10, 10, 10, 10)
        self.facial_layout.setSpacing(20)
        
        # Create facial analysis settings group
        self.facial_settings_group = QGroupBox("Facial Analysis Settings")
        self.facial_settings_layout = QFormLayout(self.facial_settings_group)
        
        # Enable facial analysis
        self.enable_facial_analysis_check = QCheckBox("Enable facial analysis")
        self.facial_settings_layout.addRow("", self.enable_facial_analysis_check)
        
        # Camera device
        self.camera_device_spin = QSpinBox()
        self.camera_device_spin.setRange(0, 10)
        self.facial_settings_layout.addRow("Camera Device:", self.camera_device_spin)
        
        # Frame rate
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setRange(1, 30)
        self.frame_rate_spin.setSuffix(" fps")
        self.facial_settings_layout.addRow("Frame Rate:", self.frame_rate_spin)
        
        # Resolution
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["320x240", "640x480", "800x600", "1280x720"])
        self.facial_settings_layout.addRow("Resolution:", self.resolution_combo)
        
        self.facial_layout.addWidget(self.facial_settings_group)
        
        # Create detection settings group
        self.detection_settings_group = QGroupBox("Detection Settings")
        self.detection_settings_layout = QFormLayout(self.detection_settings_group)
        
        # Detection method
        self.detection_method_combo = QComboBox()
        self.detection_method_combo.addItems(["Haar Cascade", "DNN"])
        self.detection_settings_layout.addRow("Detection Method:", self.detection_method_combo)
        
        # Scale factor
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(1.01, 2.0)
        self.scale_factor_spin.setSingleStep(0.01)
        self.scale_factor_spin.setDecimals(2)
        self.detection_settings_layout.addRow("Scale Factor:", self.scale_factor_spin)
        
        # Minimum neighbors
        self.min_neighbors_spin = QSpinBox()
        self.min_neighbors_spin.setRange(1, 10)
        self.detection_settings_layout.addRow("Minimum Neighbors:", self.min_neighbors_spin)
        
        # Minimum size
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 100)
        self.min_size_spin.setSuffix(" px")
        self.detection_settings_layout.addRow("Minimum Size:", self.min_size_spin)
        
        self.facial_layout.addWidget(self.detection_settings_group)
        
        # Create emotion analysis group
        self.emotion_settings_group = QGroupBox("Emotion Analysis Settings")
        self.emotion_settings_layout = QFormLayout(self.emotion_settings_group)
        
        # Enable emotion analysis
        self.enable_emotion_analysis_check = QCheckBox("Enable emotion analysis")
        self.emotion_settings_layout.addRow("", self.enable_emotion_analysis_check)
        
        # Emotion model path
        self.emotion_model_edit = QLineEdit()
        self.emotion_model_layout = QHBoxLayout()
        self.emotion_model_layout.setContentsMargins(0, 0, 0, 0)
        self.emotion_model_layout.setSpacing(5)
        self.emotion_model_layout.addWidget(self.emotion_model_edit)
        
        self.emotion_model_button = QPushButton("Browse...")
        self.emotion_model_button.clicked.connect(self.browse_emotion_model)
        self.emotion_model_layout.addWidget(self.emotion_model_button)
        
        self.emotion_settings_layout.addRow("Emotion Model:", self.emotion_model_layout)
        
        # Confidence threshold
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.0, 1.0)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.setDecimals(2)
        self.emotion_settings_layout.addRow("Confidence Threshold:", self.confidence_threshold_spin)
        
        self.facial_layout.addWidget(self.emotion_settings_group)
        
        # Create display settings group
        self.display_settings_group = QGroupBox("Display Settings")
        self.display_settings_layout = QFormLayout(self.display_settings_group)
        
        # Show camera feed
        self.show_camera_feed_check = QCheckBox("Show camera feed")
        self.display_settings_layout.addRow("", self.show_camera_feed_check)
        
        # Show face detection
        self.show_face_detection_check = QCheckBox("Show face detection boxes")
        self.display_settings_layout.addRow("", self.show_face_detection_check)
        
        # Show emotion labels
        self.show_emotion_labels_check = QCheckBox("Show emotion labels")
        self.display_settings_layout.addRow("", self.show_emotion_labels_check)
        
        # Save frames
        self.save_frames_check = QCheckBox("Save frames")
        self.display_settings_layout.addRow("", self.save_frames_check)
        
        # Save interval
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(1, 60)
        self.save_interval_spin.setSuffix(" seconds")
        self.display_settings_layout.addRow("Save Interval:", self.save_interval_spin)
        
        self.facial_layout.addWidget(self.display_settings_group)
        
        # Add spacer
        self.facial_layout.addStretch()
    
    def setup_scoring_tab(self) -> None:
        """
        Set up the cognitive load scoring settings tab
        """
        # Create layout
        self.scoring_layout = QVBoxLayout(self.scoring_tab)
        self.scoring_layout.setContentsMargins(10, 10, 10, 10)
        self.scoring_layout.setSpacing(20)
        
        # Create scoring settings group
        self.scoring_settings_group = QGroupBox("Scoring Settings")
        self.scoring_settings_layout = QFormLayout(self.scoring_settings_group)
        
        # Scoring method
        self.scoring_method_combo = QComboBox()
        self.scoring_method_combo.addItems(["Weighted Sum", "Machine Learning"])
        self.scoring_settings_layout.addRow("Scoring Method:", self.scoring_method_combo)
        
        # Update interval
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(100, 5000)
        self.update_interval_spin.setSingleStep(100)
        self.update_interval_spin.setSuffix(" ms")
        self.scoring_settings_layout.addRow("Update Interval:", self.update_interval_spin)
        
        # Smoothing factor
        self.smoothing_factor_spin = QDoubleSpinBox()
        self.smoothing_factor_spin.setRange(0.0, 1.0)
        self.smoothing_factor_spin.setSingleStep(0.05)
        self.smoothing_factor_spin.setDecimals(2)
        self.scoring_settings_layout.addRow("Smoothing Factor:", self.smoothing_factor_spin)
        
        self.scoring_layout.addWidget(self.scoring_settings_group)
        
        # Create component weights group
        self.weights_group = QGroupBox("Component Weights")
        self.weights_layout = QFormLayout(self.weights_group)
        
        # Behavior weight
        self.behavior_weight_spin = QDoubleSpinBox()
        self.behavior_weight_spin.setRange(0.0, 1.0)
        self.behavior_weight_spin.setSingleStep(0.05)
        self.behavior_weight_spin.setDecimals(2)
        self.weights_layout.addRow("Behavior Weight:", self.behavior_weight_spin)
        
        # Facial weight
        self.facial_weight_spin = QDoubleSpinBox()
        self.facial_weight_spin.setRange(0.0, 1.0)
        self.facial_weight_spin.setSingleStep(0.05)
        self.facial_weight_spin.setDecimals(2)
        self.weights_layout.addRow("Facial Weight:", self.facial_weight_spin)
        
        # Performance weight
        self.performance_weight_spin = QDoubleSpinBox()
        self.performance_weight_spin.setRange(0.0, 1.0)
        self.performance_weight_spin.setSingleStep(0.05)
        self.performance_weight_spin.setDecimals(2)
        self.weights_layout.addRow("Performance Weight:", self.performance_weight_spin)
        
        # EEG weight
        self.eeg_weight_spin = QDoubleSpinBox()
        self.eeg_weight_spin.setRange(0.0, 1.0)
        self.eeg_weight_spin.setSingleStep(0.05)
        self.eeg_weight_spin.setDecimals(2)
        self.weights_layout.addRow("EEG Weight:", self.eeg_weight_spin)
        
        self.scoring_layout.addWidget(self.weights_group)
        
        # Create threshold settings group
        self.threshold_group = QGroupBox("Threshold Settings")
        self.threshold_layout = QFormLayout(self.threshold_group)
        
        # Low threshold
        self.low_threshold_spin = QDoubleSpinBox()
        self.low_threshold_spin.setRange(0.0, 1.0)
        self.low_threshold_spin.setSingleStep(0.05)
        self.low_threshold_spin.setDecimals(2)
        self.threshold_layout.addRow("Low Threshold:", self.low_threshold_spin)
        
        # High threshold
        self.high_threshold_spin = QDoubleSpinBox()
        self.high_threshold_spin.setRange(0.0, 1.0)
        self.high_threshold_spin.setSingleStep(0.05)
        self.high_threshold_spin.setDecimals(2)
        self.threshold_layout.addRow("High Threshold:", self.high_threshold_spin)
        
        # Enable alerts
        self.enable_alerts_check = QCheckBox("Enable threshold alerts")
        self.threshold_layout.addRow("", self.enable_alerts_check)
        
        self.scoring_layout.addWidget(self.threshold_group)
        
        # Create machine learning group
        self.ml_group = QGroupBox("Machine Learning Settings")
        self.ml_layout = QFormLayout(self.ml_group)
        
        # ML model path
        self.ml_model_edit = QLineEdit()
        self.ml_model_layout = QHBoxLayout()
        self.ml_model_layout.setContentsMargins(0, 0, 0, 0)
        self.ml_model_layout.setSpacing(5)
        self.ml_model_layout.addWidget(self.ml_model_edit)
        
        self.ml_model_button = QPushButton("Browse...")
        self.ml_model_button.clicked.connect(self.browse_ml_model)
        self.ml_model_layout.addWidget(self.ml_model_button)
        
        self.ml_layout.addRow("ML Model:", self.ml_model_layout)
        
        # Enable model training
        self.enable_training_check = QCheckBox("Enable model training")
        self.ml_layout.addRow("", self.enable_training_check)
        
        # Training interval
        self.training_interval_spin = QSpinBox()
        self.training_interval_spin.setRange(1, 60)
        self.training_interval_spin.setSuffix(" minutes")
        self.ml_layout.addRow("Training Interval:", self.training_interval_spin)
        
        self.scoring_layout.addWidget(self.ml_group)
        
        # Add spacer
        self.scoring_layout.addStretch()
    
    def setup_advanced_tab(self) -> None:
        """
        Set up the advanced features settings tab
        """
        # Create layout
        self.advanced_layout = QVBoxLayout(self.advanced_tab)
        self.advanced_layout.setContentsMargins(10, 10, 10, 10)
        self.advanced_layout.setSpacing(20)
        
        # Create machine learning group
        self.advanced_ml_group = QGroupBox("Machine Learning")
        self.advanced_ml_layout = QFormLayout(self.advanced_ml_group)
        
        # Enable ML
        self.enable_ml_check = QCheckBox("Enable machine learning features")
        self.advanced_ml_layout.addRow("", self.enable_ml_check)
        
        # ML framework
        self.ml_framework_combo = QComboBox()
        self.ml_framework_combo.addItems(["scikit-learn", "TensorFlow"])
        self.advanced_ml_layout.addRow("ML Framework:", self.ml_framework_combo)
        
        # Model type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Random Forest", "Neural Network", "SVM", "Gradient Boosting"])
        self.advanced_ml_layout.addRow("Model Type:", self.model_type_combo)
        
        self.advanced_layout.addWidget(self.advanced_ml_group)
        
        # Create EEG group
        self.eeg_group = QGroupBox("EEG Integration")
        self.eeg_layout = QFormLayout(self.eeg_group)
        
        # Enable EEG
        self.enable_eeg_check = QCheckBox("Enable EEG integration")
        self.eeg_layout.addRow("", self.enable_eeg_check)
        
        # EEG device
        self.eeg_device_combo = QComboBox()
        self.eeg_device_combo.addItems(["OpenBCI", "Muse", "Emotiv", "NeuroSky", "Other"])
        self.eeg_layout.addRow("EEG Device:", self.eeg_device_combo)
        
        # Board ID
        self.board_id_spin = QSpinBox()
        self.board_id_spin.setRange(0, 10)
        self.eeg_layout.addRow("Board ID:", self.board_id_spin)
        
        # Serial port
        self.serial_port_edit = QLineEdit()
        self.eeg_layout.addRow("Serial Port:", self.serial_port_edit)
        
        # Sampling rate
        self.eeg_sampling_rate_combo = QComboBox()
        self.eeg_sampling_rate_combo.addItems(["125 Hz", "250 Hz", "500 Hz", "1000 Hz"])
        self.eeg_layout.addRow("Sampling Rate:", self.eeg_sampling_rate_combo)
        
        self.advanced_layout.addWidget(self.eeg_group)
        
        # Create API integration group
        self.api_group = QGroupBox("API Integration")
        self.api_layout = QFormLayout(self.api_group)
        
        # Enable API
        self.enable_api_check = QCheckBox("Enable API integration")
        self.api_layout.addRow("", self.enable_api_check)
        
        # API endpoint
        self.api_endpoint_edit = QLineEdit()
        self.api_layout.addRow("API Endpoint:", self.api_endpoint_edit)
        
        # API key
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_layout.addRow("API Key:", self.api_key_edit)
        
        # API update interval
        self.api_update_interval_spin = QSpinBox()
        self.api_update_interval_spin.setRange(1, 60)
        self.api_update_interval_spin.setSuffix(" seconds")
        self.api_layout.addRow("Update Interval:", self.api_update_interval_spin)
        
        self.advanced_layout.addWidget(self.api_group)
        
        # Add spacer
        self.advanced_layout.addStretch()
    
    def setup_data_tab(self) -> None:
        """
        Set up the data storage settings tab
        """
        # Create layout
        self.data_layout = QVBoxLayout(self.data_tab)
        self.data_layout.setContentsMargins(10, 10, 10, 10)
        self.data_layout.setSpacing(20)
        
        # Create storage settings group
        self.storage_settings_group = QGroupBox("Storage Settings")
        self.storage_settings_layout = QFormLayout(self.storage_settings_group)
        
        # Data directory
        self.data_dir_edit = QLineEdit()
        self.data_dir_layout = QHBoxLayout()
        self.data_dir_layout.setContentsMargins(0, 0, 0, 0)
        self.data_dir_layout.setSpacing(5)
        self.data_dir_layout.addWidget(self.data_dir_edit)
        
        self.data_dir_button = QPushButton("Browse...")
        self.data_dir_button.clicked.connect(self.browse_data_dir)
        self.data_dir_layout.addWidget(self.data_dir_button)
        
        self.storage_settings_layout.addRow("Data Directory:", self.data_dir_layout)
        
        # Storage format
        self.storage_format_combo = QComboBox()
        self.storage_format_combo.addItems(["JSON", "CSV", "SQLite", "HDF5"])
        self.storage_settings_layout.addRow("Storage Format:", self.storage_format_combo)
        
        # Compression
        self.compression_check = QCheckBox("Enable compression")
        self.storage_settings_layout.addRow("", self.compression_check)
        
        # Encryption
        self.encryption_check = QCheckBox("Enable encryption")
        self.storage_settings_layout.addRow("", self.encryption_check)
        
        self.data_layout.addWidget(self.storage_settings_group)
        
        # Create data retention group
        self.retention_group = QGroupBox("Data Retention")
        self.retention_layout = QFormLayout(self.retention_group)
        
        # Auto-cleanup
        self.auto_cleanup_check = QCheckBox("Enable automatic cleanup")
        self.retention_layout.addRow("", self.auto_cleanup_check)
        
        # Retention period
        self.retention_period_spin = QSpinBox()
        self.retention_period_spin.setRange(1, 365)
        self.retention_period_spin.setSuffix(" days")
        self.retention_layout.addRow("Retention Period:", self.retention_period_spin)
        
        # Max storage size
        self.max_storage_spin = QSpinBox()
        self.max_storage_spin.setRange(100, 10000)
        self.max_storage_spin.setSingleStep(100)
        self.max_storage_spin.setSuffix(" MB")
        self.retention_layout.addRow("Maximum Storage Size:", self.max_storage_spin)
        
        self.data_layout.addWidget(self.retention_group)
        
        # Create export settings group
        self.export_group = QGroupBox("Export Settings")
        self.export_layout = QFormLayout(self.export_group)
        
        # Default export format
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["CSV", "JSON", "Excel", "HDF5"])
        self.export_layout.addRow("Default Export Format:", self.export_format_combo)
        
        # Include metadata
        self.include_metadata_check = QCheckBox("Include metadata")
        self.export_layout.addRow("", self.include_metadata_check)
        
        # Include timestamps
        self.include_timestamps_check = QCheckBox("Include timestamps")
        self.export_layout.addRow("", self.include_timestamps_check)
        
        self.data_layout.addWidget(self.export_group)
        
        # Add spacer
        self.data_layout.addStretch()
    
    def load_settings(self) -> None:
        """
        Load settings from configuration
        """
        # General settings
        app_config = self.config.get('app', {})
        self.app_name_edit.setText(app_config.get('name', 'NEXARIS Cognitive Load Estimator'))
        self.app_version_edit.setText(app_config.get('version', '1.0.0'))
        self.log_level_combo.setCurrentText(app_config.get('log_level', 'INFO'))
        self.auto_save_check.setChecked(app_config.get('auto_save_settings', True))
        
        session_config = self.config.get('session', {})
        self.auto_start_session_check.setChecked(session_config.get('auto_start', False))
        self.auto_save_session_check.setChecked(session_config.get('auto_save', True))
        self.session_save_interval_spin.setValue(session_config.get('save_interval', 5))
        
        # UI settings
        ui_config = self.config.get('ui', {})
        self.theme_combo.setCurrentText(ui_config.get('theme', 'Light').title())
        self.font_family_combo.setCurrentText(ui_config.get('font_family', 'System Default'))
        self.font_size_spin.setValue(ui_config.get('font_size', 10))
        
        dashboard_config = ui_config.get('dashboard', {})
        self.show_gauge_check.setChecked(dashboard_config.get('show_gauge', True))
        self.show_time_series_check.setChecked(dashboard_config.get('show_time_series', True))
        self.show_components_check.setChecked(dashboard_config.get('show_components', True))
        self.chart_update_interval_spin.setValue(dashboard_config.get('update_interval', 500))
        
        visualization_config = ui_config.get('visualization', {})
        self.chart_style_combo.setCurrentText(visualization_config.get('chart_style', 'Default'))
        self.color_map_combo.setCurrentText(visualization_config.get('color_map', 'viridis'))
        
        # Task settings
        task_config = self.config.get('task', {})
        self.default_task_type_combo.setCurrentText(task_config.get('default_type', 'Question Set'))
        self.default_question_set_combo.setCurrentText(task_config.get('default_question_set', 'General Knowledge'))
        self.default_difficulty_combo.setCurrentText(task_config.get('default_difficulty', 'Medium').title())
        self.default_time_limit_spin.setValue(task_config.get('default_time_limit', 180))
        self.default_item_count_spin.setValue(task_config.get('default_item_count', 5))
        
        question_config = task_config.get('question', {})
        self.custom_question_set_edit.setText(question_config.get('custom_set_path', ''))
        self.show_correct_answers_check.setChecked(question_config.get('show_correct_answers', True))
        self.randomize_questions_check.setChecked(question_config.get('randomize', True))
        
        alert_config = task_config.get('alert', {})
        self.custom_alert_set_edit.setText(alert_config.get('custom_set_path', ''))
        self.randomize_alerts_check.setChecked(alert_config.get('randomize', True))
        self.require_notes_check.setChecked(alert_config.get('require_notes', False))
        
        # Tracking settings
        tracking_config = self.config.get('tracking', {})
        self.enable_tracking_check.setChecked(tracking_config.get('enabled', True))
        self.track_mouse_movement_check.setChecked(tracking_config.get('track_mouse_movement', True))
        self.track_mouse_clicks_check.setChecked(tracking_config.get('track_mouse_clicks', True))
        self.track_key_presses_check.setChecked(tracking_config.get('track_key_presses', True))
        self.track_hesitation_check.setChecked(tracking_config.get('track_hesitation', True))
        
        hesitation_config = tracking_config.get('hesitation', {})
        self.hesitation_threshold_spin.setValue(hesitation_config.get('threshold', 2.0))
        self.min_hesitation_duration_spin.setValue(hesitation_config.get('min_duration', 1.0))
        self.check_interval_spin.setValue(hesitation_config.get('check_interval', 200))
        
        data_collection_config = tracking_config.get('data_collection', {})
        self.sampling_rate_spin.setValue(data_collection_config.get('sampling_rate', 10))
        self.buffer_size_spin.setValue(data_collection_config.get('buffer_size', 100))
        
        # Facial analysis settings
        facial_config = self.config.get('facial_analysis', {})
        self.enable_facial_analysis_check.setChecked(facial_config.get('enabled', True))
        self.camera_device_spin.setValue(facial_config.get('camera_device', 0))
        self.frame_rate_spin.setValue(facial_config.get('frame_rate', 15))
        self.resolution_combo.setCurrentText(facial_config.get('resolution', '640x480'))
        
        detection_config = facial_config.get('detection', {})
        self.detection_method_combo.setCurrentText(detection_config.get('method', 'Haar Cascade'))
        self.scale_factor_spin.setValue(detection_config.get('scale_factor', 1.1))
        self.min_neighbors_spin.setValue(detection_config.get('min_neighbors', 5))
        self.min_size_spin.setValue(detection_config.get('min_size', 30))
        
        emotion_config = facial_config.get('emotion', {})
        self.enable_emotion_analysis_check.setChecked(emotion_config.get('enabled', True))
        self.emotion_model_edit.setText(emotion_config.get('model_path', ''))
        self.confidence_threshold_spin.setValue(emotion_config.get('confidence_threshold', 0.5))
        
        display_config = facial_config.get('display', {})
        self.show_camera_feed_check.setChecked(display_config.get('show_camera_feed', True))
        self.show_face_detection_check.setChecked(display_config.get('show_face_detection', True))
        self.show_emotion_labels_check.setChecked(display_config.get('show_emotion_labels', True))
        self.save_frames_check.setChecked(display_config.get('save_frames', False))
        self.save_interval_spin.setValue(display_config.get('save_interval', 5))
        
        # Scoring settings
        scoring_config = self.config.get('scoring', {})
        self.scoring_method_combo.setCurrentText(scoring_config.get('method', 'Weighted Sum'))
        self.update_interval_spin.setValue(scoring_config.get('update_interval', 500))
        self.smoothing_factor_spin.setValue(scoring_config.get('smoothing_factor', 0.3))
        
        weights_config = scoring_config.get('weights', {})
        self.behavior_weight_spin.setValue(weights_config.get('behavior', 0.3))
        self.facial_weight_spin.setValue(weights_config.get('facial', 0.3))
        self.performance_weight_spin.setValue(weights_config.get('performance', 0.3))
        self.eeg_weight_spin.setValue(weights_config.get('eeg', 0.1))
        
        threshold_config = scoring_config.get('thresholds', {})
        self.low_threshold_spin.setValue(threshold_config.get('low', 0.4))
        self.high_threshold_spin.setValue(threshold_config.get('high', 0.7))
        self.enable_alerts_check.setChecked(threshold_config.get('enable_alerts', True))
        
        ml_config = scoring_config.get('ml', {})
        self.ml_model_edit.setText(ml_config.get('model_path', ''))
        self.enable_training_check.setChecked(ml_config.get('enable_training', False))
        self.training_interval_spin.setValue(ml_config.get('training_interval', 10))
        
        # Advanced settings
        advanced_config = self.config.get('advanced_features', {})
        
        ml_advanced_config = advanced_config.get('ml', {})
        self.enable_ml_check.setChecked(ml_advanced_config.get('enabled', False))
        self.ml_framework_combo.setCurrentText(ml_advanced_config.get('framework', 'scikit-learn'))
        self.model_type_combo.setCurrentText(ml_advanced_config.get('model_type', 'Random Forest'))
        
        eeg_config = advanced_config.get('eeg', {})
        self.enable_eeg_check.setChecked(eeg_config.get('enabled', False))
        self.eeg_device_combo.setCurrentText(eeg_config.get('device', 'OpenBCI'))
        self.board_id_spin.setValue(eeg_config.get('board_id', 0))
        self.serial_port_edit.setText(eeg_config.get('serial_port', ''))
        self.eeg_sampling_rate_combo.setCurrentText(f"{eeg_config.get('sampling_rate', 250)} Hz")
        
        api_config = advanced_config.get('api', {})
        self.enable_api_check.setChecked(api_config.get('enabled', False))
        self.api_endpoint_edit.setText(api_config.get('endpoint', ''))
        self.api_key_edit.setText(api_config.get('api_key', ''))
        self.api_update_interval_spin.setValue(api_config.get('update_interval', 5))
        
        # Data settings
        data_config = self.config.get('data', {})
        self.data_dir_edit.setText(data_config.get('data_dir', ''))
        self.storage_format_combo.setCurrentText(data_config.get('storage_format', 'JSON'))
        self.compression_check.setChecked(data_config.get('compression', False))
        self.encryption_check.setChecked(data_config.get('encryption', False))
        
        retention_config = data_config.get('retention', {})
        self.auto_cleanup_check.setChecked(retention_config.get('auto_cleanup', False))
        self.retention_period_spin.setValue(retention_config.get('retention_period', 30))
        self.max_storage_spin.setValue(retention_config.get('max_storage', 1000))
        
        export_config = data_config.get('export', {})
        self.export_format_combo.setCurrentText(export_config.get('default_format', 'CSV'))
        self.include_metadata_check.setChecked(export_config.get('include_metadata', True))
        self.include_timestamps_check.setChecked(export_config.get('include_timestamps', True))
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get settings from UI controls
        
        Returns:
            Dictionary of settings
        """
        settings = {}
        
        # General settings
        settings['app'] = {
            'name': self.app_name_edit.text(),
            'version': self.app_version_edit.text(),
            'log_level': self.log_level_combo.currentText(),
            'auto_save_settings': self.auto_save_check.isChecked()
        }
        
        settings['session'] = {
            'auto_start': self.auto_start_session_check.isChecked(),
            'auto_save': self.auto_save_session_check.isChecked(),
            'save_interval': self.session_save_interval_spin.value()
        }
        
        # UI settings
        settings['ui'] = {
            'theme': self.theme_combo.currentText().lower(),
            'font_family': self.font_family_combo.currentText(),
            'font_size': self.font_size_spin.value(),
            'dashboard': {
                'show_gauge': self.show_gauge_check.isChecked(),
                'show_time_series': self.show_time_series_check.isChecked(),
                'show_components': self.show_components_check.isChecked(),
                'update_interval': self.chart_update_interval_spin.value()
            },
            'visualization': {
                'chart_style': self.chart_style_combo.currentText(),
                'color_map': self.color_map_combo.currentText()
            }
        }
        
        # Task settings
        settings['task'] = {
            'default_type': self.default_task_type_combo.currentText(),
            'default_question_set': self.default_question_set_combo.currentText(),
            'default_difficulty': self.default_difficulty_combo.currentText().lower(),
            'default_time_limit': self.default_time_limit_spin.value(),
            'default_item_count': self.default_item_count_spin.value(),
            'question': {
                'custom_set_path': self.custom_question_set_edit.text(),
                'show_correct_answers': self.show_correct_answers_check.isChecked(),
                'randomize': self.randomize_questions_check.isChecked()
            },
            'alert': {
                'custom_set_path': self.custom_alert_set_edit.text(),
                'randomize': self.randomize_alerts_check.isChecked(),
                'require_notes': self.require_notes_check.isChecked()
            }
        }
        
        # Tracking settings
        settings['tracking'] = {
            'enabled': self.enable_tracking_check.isChecked(),
            'track_mouse_movement': self.track_mouse_movement_check.isChecked(),
            'track_mouse_clicks': self.track_mouse_clicks_check.isChecked(),
            'track_key_presses': self.track_key_presses_check.isChecked(),
            'track_hesitation': self.track_hesitation_check.isChecked(),
            'hesitation': {
                'threshold': self.hesitation_threshold_spin.value(),
                'min_duration': self.min_hesitation_duration_spin.value(),
                'check_interval': self.check_interval_spin.value()
            },
            'data_collection': {
                'sampling_rate': self.sampling_rate_spin.value(),
                'buffer_size': self.buffer_size_spin.value()
            }
        }
        
        # Facial analysis settings
        settings['facial_analysis'] = {
            'enabled': self.enable_facial_analysis_check.isChecked(),
            'camera_device': self.camera_device_spin.value(),
            'frame_rate': self.frame_rate_spin.value(),
            'resolution': self.resolution_combo.currentText(),
            'detection': {
                'method': self.detection_method_combo.currentText(),
                'scale_factor': self.scale_factor_spin.value(),
                'min_neighbors': self.min_neighbors_spin.value(),
                'min_size': self.min_size_spin.value()
            },
            'emotion': {
                'enabled': self.enable_emotion_analysis_check.isChecked(),
                'model_path': self.emotion_model_edit.text(),
                'confidence_threshold': self.confidence_threshold_spin.value()
            },
            'display': {
                'show_camera_feed': self.show_camera_feed_check.isChecked(),
                'show_face_detection': self.show_face_detection_check.isChecked(),
                'show_emotion_labels': self.show_emotion_labels_check.isChecked(),
                'save_frames': self.save_frames_check.isChecked(),
                'save_interval': self.save_interval_spin.value()
            }
        }
        
        # Scoring settings
        settings['scoring'] = {
            'method': self.scoring_method_combo.currentText(),
            'update_interval': self.update_interval_spin.value(),
            'smoothing_factor': self.smoothing_factor_spin.value(),
            'weights': {
                'behavior': self.behavior_weight_spin.value(),
                'facial': self.facial_weight_spin.value(),
                'performance': self.performance_weight_spin.value(),
                'eeg': self.eeg_weight_spin.value()
            },
            'thresholds': {
                'low': self.low_threshold_spin.value(),
                'high': self.high_threshold_spin.value(),
                'enable_alerts': self.enable_alerts_check.isChecked()
            },
            'ml': {
                'model_path': self.ml_model_edit.text(),
                'enable_training': self.enable_training_check.isChecked(),
                'training_interval': self.training_interval_spin.value()
            }
        }
        
        # Advanced settings
        settings['advanced_features'] = {
            'ml': {
                'enabled': self.enable_ml_check.isChecked(),
                'framework': self.ml_framework_combo.currentText(),
                'model_type': self.model_type_combo.currentText()
            },
            'eeg': {
                'enabled': self.enable_eeg_check.isChecked(),
                'device': self.eeg_device_combo.currentText(),
                'board_id': self.board_id_spin.value(),
                'serial_port': self.serial_port_edit.text(),
                'sampling_rate': int(self.eeg_sampling_rate_combo.currentText().split()[0])
            },
            'api': {
                'enabled': self.enable_api_check.isChecked(),
                'endpoint': self.api_endpoint_edit.text(),
                'api_key': self.api_key_edit.text(),
                'update_interval': self.api_update_interval_spin.value()
            }
        }
        
        # Data settings
        settings['data'] = {
            'data_dir': self.data_dir_edit.text(),
            'storage_format': self.storage_format_combo.currentText(),
            'compression': self.compression_check.isChecked(),
            'encryption': self.encryption_check.isChecked(),
            'retention': {
                'auto_cleanup': self.auto_cleanup_check.isChecked(),
                'retention_period': self.retention_period_spin.value(),
                'max_storage': self.max_storage_spin.value()
            },
            'export': {
                'default_format': self.export_format_combo.currentText(),
                'include_metadata': self.include_metadata_check.isChecked(),
                'include_timestamps': self.include_timestamps_check.isChecked()
            }
        }
        
        return settings
    
    def apply_settings(self) -> None:
        """
        Apply settings and emit settings_changed signal
        """
        # Get settings from UI
        settings = self.get_settings()
        
        # Update config
        self.config = settings
        
        # Save settings
        save_config(settings)
        
        # Emit settings changed signal
        self.settings_changed.emit(settings)
        
        # Show success message
        QMessageBox.information(self, "Settings Applied", "Settings have been applied successfully.")
    
    def reset_settings(self) -> None:
        """
        Reset settings to defaults
        """
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset config to defaults
            from ..utils.config_utils import get_default_config
            self.config = get_default_config()
            
            # Load settings
            self.load_settings()
            
            # Show success message
            QMessageBox.information(self, "Settings Reset", "Settings have been reset to defaults.")
    
    def browse_custom_question_set(self) -> None:
        """
        Browse for custom question set file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Question Set",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.custom_question_set_edit.setText(file_path)
    
    def browse_custom_alert_set(self) -> None:
        """
        Browse for custom alert set file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Alert Set",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.custom_alert_set_edit.setText(file_path)
    
    def browse_emotion_model(self) -> None:
        """
        Browse for emotion model file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Emotion Model",
            "",
            "ONNX Models (*.onnx);;All Files (*)"
        )
        
        if file_path:
            self.emotion_model_edit.setText(file_path)
    
    def browse_ml_model(self) -> None:
        """
        Browse for ML model file
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ML Model",
            "",
            "Pickle Files (*.pkl);;HDF5 Files (*.h5);;SavedModel Files (*.pb);;All Files (*)"
        )
        
        if file_path:
            self.ml_model_edit.setText(file_path)
    
    def browse_data_dir(self) -> None:
        """
        Browse for data directory
        """
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            ""
        )
        
        if dir_path:
            self.data_dir_edit.setText(dir_path)