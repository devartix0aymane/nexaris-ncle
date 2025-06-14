#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live Analysis Panel for NEXARIS Cognitive Load Estimator

This module provides a dedicated panel to display real-time task performance,
behavioral metrics, facial emotion, and cognitive load.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout,
    QProgressBar, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon

from ..core.task_simulator import TaskSimulator
from ..core.behavior_tracker import BehaviorTracker
from ..core.facial_analyzer import FacialAnalyzer
from ..core.cognitive_load_calculator import CognitiveLoadCalculator
from .camera_widget import CameraWidget # For live facial emotion display

class LiveAnalysisPanel(QWidget):
    """
    A widget to display live analysis data including task info, biometrics, and load score.
    """
    start_task_requested = pyqtSignal()
    stop_task_requested = pyqtSignal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._task_simulator: Optional[TaskSimulator] = None
        self._behavior_tracker: Optional[BehaviorTracker] = None
        self._facial_analyzer: Optional[FacialAnalyzer] = None
        self._cognitive_load_calculator: Optional[CognitiveLoadCalculator] = None

        self.current_task_name = "N/A"
        self.elapsed_time_s = 0

        self._init_ui()
        self._connect_signals()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_live_data)
        self.update_timer.start(100) # Update 10 times per second for timer

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5,5,5,5)
        main_layout.setSpacing(10)

        # --- Control Buttons ---
        control_group = QGroupBox("Session Control")
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Task")
        self.start_button.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'start_task.png')))
        self.start_button.setIconSize(QSize(16,16))
        self.stop_button = QPushButton("Stop Task")
        self.stop_button.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', 'stop_task.png')))
        self.stop_button.setIconSize(QSize(16,16))
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # --- Task Information ---
        task_info_group = QGroupBox("Task Information")
        task_info_layout = QGridLayout()
        task_info_layout.addWidget(QLabel("Current Task:"), 0, 0)
        self.task_name_label = QLabel("N/A")
        self.task_name_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        task_info_layout.addWidget(self.task_name_label, 0, 1)
        task_info_layout.addWidget(QLabel("Timer:"), 1, 0)
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        task_info_layout.addWidget(self.timer_label, 1, 1)
        task_info_group.setLayout(task_info_layout)
        main_layout.addWidget(task_info_group)

        # --- Behavioral Metrics ---
        behavior_group = QGroupBox("Behavioral Metrics")
        behavior_layout = QGridLayout()
        behavior_layout.addWidget(QLabel("Mouse Clicks:"), 0, 0)
        self.clicks_label = QLabel("0")
        behavior_layout.addWidget(self.clicks_label, 0, 1)
        behavior_layout.addWidget(QLabel("Hesitation Events:"), 1, 0) # Changed from delay to count
        self.hesitation_label = QLabel("0 events")
        behavior_layout.addWidget(self.hesitation_label, 1, 1)
        behavior_group.setLayout(behavior_layout)
        main_layout.addWidget(behavior_group)

        # --- Facial Emotion ---
        emotion_group = QGroupBox("Facial Emotion")
        emotion_layout = QVBoxLayout()
        self.emotion_camera_widget = CameraWidget() # Use existing CameraWidget
        self.emotion_camera_widget.setMinimumHeight(150)
        self.emotion_label = QLabel("Emotion: N/A (Confidence: N/A)")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        emotion_layout.addWidget(self.emotion_camera_widget)
        emotion_layout.addWidget(self.emotion_label)
        emotion_group.setLayout(emotion_layout)
        main_layout.addWidget(emotion_group)

        # --- Cognitive Load Score ---
        load_group = QGroupBox("Cognitive Load")
        load_layout = QVBoxLayout()
        self.load_score_label = QLabel("Score: N/A")
        self.load_score_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.load_score_label.setAlignment(Qt.AlignCenter)
        self.load_gauge = QProgressBar()
        self.load_gauge.setRange(0, 100)
        self.load_gauge.setValue(0)
        self.load_gauge.setTextVisible(True)
        self.load_gauge.setFormat("%v / 100")
        load_layout.addWidget(self.load_score_label)
        load_layout.addWidget(self.load_gauge)
        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group)

        main_layout.addStretch(1)

    def _connect_signals(self):
        self.start_button.clicked.connect(self._on_start_task_clicked)
        self.stop_button.clicked.connect(self._on_stop_task_clicked)

    def set_core_components(self, 
                            task_simulator: TaskSimulator,
                            behavior_tracker: BehaviorTracker,
                            facial_analyzer: FacialAnalyzer,
                            cognitive_load_calculator: CognitiveLoadCalculator):
        self._task_simulator = task_simulator
        self._behavior_tracker = behavior_tracker
        self._facial_analyzer = facial_analyzer
        self._cognitive_load_calculator = cognitive_load_calculator

        # Connect signals from core components
        if self._task_simulator:
            self._task_simulator.task_started.connect(self.on_task_started)
            self._task_simulator.task_completed.connect(self.on_task_stopped_or_completed)

        if self._behavior_tracker:
            # We'll poll metrics in _update_live_data or connect to a summary signal if available
            pass 

        if self._facial_analyzer:
            self._facial_analyzer.emotion_detected.connect(self.update_emotion_display)
            self._facial_analyzer.frame_processed.connect(self.emotion_camera_widget.update_frame)
            self.emotion_camera_widget.set_components(self._facial_analyzer) # Pass analyzer to camera widget
            if self.config.get('facial_analysis', {}).get('enabled', True) and not self._facial_analyzer.is_analyzing:
                 self._facial_analyzer.start_analysis() # Ensure it's running
            self.emotion_camera_widget.start_camera() # Start camera for this panel

        if self._cognitive_load_calculator:
            self._cognitive_load_calculator.load_updated.connect(self.update_cognitive_load_display)

    def _on_start_task_clicked(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.elapsed_time_s = 0
        self.timer_label.setText("00:00:00")
        self.start_task_requested.emit()

    def _on_stop_task_clicked(self):
        self.stop_button.setEnabled(False)
        # Start button re-enabled by on_task_stopped_or_completed
        self.stop_task_requested.emit()

    def on_task_started(self, task_data: dict):
        self.current_task_name = task_data.get('name', task_data.get('id', 'Unknown Task'))
        self.task_name_label.setText(self.current_task_name)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.elapsed_time_s = 0

    def on_task_stopped_or_completed(self, _=None): # Argument can vary
        self.current_task_name = "N/A"
        self.task_name_label.setText(self.current_task_name)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # self.elapsed_time_s = 0 # Keep timer value until next start

    def _update_live_data(self):
        # Update Timer
        if self.stop_button.isEnabled(): # Only update if task is running
            self.elapsed_time_s += self.update_timer.interval() / 1000.0
        hours = int(self.elapsed_time_s // 3600)
        minutes = int((self.elapsed_time_s % 3600) // 60)
        seconds = int(self.elapsed_time_s % 60)
        self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        # Update Behavioral Metrics
        if self._behavior_tracker:
            metrics = self._behavior_tracker.get_metrics() # Polling for simplicity
            self.clicks_label.setText(str(metrics.get('click_count', 0)))
            self.hesitation_label.setText(f"{metrics.get('hesitation_count',0)} events ({metrics.get('total_hesitation_time', 0.0):.2f}s)")

    def update_emotion_display(self, emotion: str, confidence: float):
        self.emotion_label.setText(f"Emotion: {emotion.capitalize()} (Confidence: {confidence:.2f})")
        # Camera frame is updated by direct signal from FacialAnalyzer to CameraWidget

    def update_cognitive_load_display(self, load_score: float, details: dict):
        # Assuming load_score is 0-100 from the updated CognitiveLoadCalculator
        explanation_text = details.get('explanation', 'No details available.')
        self.load_score_label.setText(f"Score: {load_score:.1f}/100")
        self.load_gauge.setValue(int(load_score))
        self.load_gauge.setToolTip(explanation_text)
        self.load_score_label.setToolTip(explanation_text)

        # Change gauge color based on load
        if load_score <= 30:
            self.load_gauge.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; } QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; }") # Green
        elif load_score <= 70:
            self.load_gauge.setStyleSheet("QProgressBar::chunk { background-color: #FFC107; } QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; }") # Amber
        else:
            self.load_gauge.setStyleSheet("QProgressBar::chunk { background-color: #F44336; } QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; }") # Red

    def closeEvent(self, event):
        self.emotion_camera_widget.stop_camera() # Ensure camera stops
        self.update_timer.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    # This is for testing the panel independently
    app = QApplication(sys.argv)
    # Mock config and components for testing
    mock_config = {
        'facial_analysis': {'enabled': True, 'camera_id': 0},
        'ui': {'theme': 'light'}
    }
    panel = LiveAnalysisPanel(mock_config)
    panel.setWindowTitle("Live Analysis Panel Test")
    panel.show()
    sys.exit(app.exec_())