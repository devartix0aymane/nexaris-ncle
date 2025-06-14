#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard Widget for NEXARIS Cognitive Load Estimator

This module provides the dashboard widget for displaying cognitive load metrics
and summary information.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QSplitter, QGroupBox,
    QProgressBar, QSpacerItem, QSizePolicy, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap

# Matplotlib imports for charts
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

# Import utilities
from ..utils.logging_utils import get_logger


class CognitiveLoadGauge(QWidget):
    """
    Custom gauge widget for displaying cognitive load
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Create axes
        self.axes = self.figure.add_subplot(111, polar=True)
        self.axes.set_theta_zero_location('N')
        self.axes.set_theta_direction(-1)
        self.axes.set_rlim(0, 1)
        self.axes.set_rticks([])
        self.axes.set_xticks([])
        
        # Create gauge
        self.update_gauge(0.0)
    
    def update_gauge(self, value: float) -> None:
        """
        Update gauge value
        
        Args:
            value: Gauge value (0.0 to 1.0)
        """
        self.value = max(0.0, min(1.0, value))
        
        # Clear axes
        self.axes.clear()
        
        # Set up gauge properties
        self.axes.set_theta_zero_location('N')
        self.axes.set_theta_direction(-1)
        self.axes.set_rlim(0, 1)
        self.axes.set_rticks([])
        self.axes.set_xticks([])
        
        # Create background
        theta = np.linspace(0, 2*np.pi, 100)
        radii = np.ones_like(theta) * 0.9
        self.axes.plot(theta, radii, color='lightgray', linewidth=20, alpha=0.3)
        
        # Create value arc
        theta = np.linspace(0, 2*np.pi * self.value, 100)
        radii = np.ones_like(theta) * 0.9
        
        # Color based on value
        if self.value < 0.4:
            color = 'green'
        elif self.value < 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        self.axes.plot(theta, radii, color=color, linewidth=20, alpha=0.8)
        
        # Add value text
        self.axes.text(0, 0, f"{self.value:.2f}", 
                      ha='center', va='center', fontsize=24, 
                      fontweight='bold', color=color)
        
        # Add labels
        self.axes.text(0, -1.2, "Cognitive Load", 
                      ha='center', va='center', fontsize=12)
        
        # Redraw canvas
        self.canvas.draw()


class TimeSeriesChart(QWidget):
    """
    Time series chart widget for displaying cognitive load over time
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.timestamps = []
        self.max_points = 3600  # Store up to 1 hour of data (1 point/sec)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Create axes
        self.axes = self.figure.add_subplot(111)
        self.axes.set_ylim(0, 1)
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Cognitive Load')
        self.axes.set_title('Cognitive Load Over Time')
        
        # Create line
        self.line, = self.axes.plot([], [], 'b-')
        
        # Add threshold lines
        self.axes.axhline(y=0.4, color='g', linestyle='--', alpha=0.5)
        self.axes.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
        
        # Create legend
        self.axes.legend(['ECLS', 'Low Threshold', 'High Threshold'])
        
        # Adjust layout
        self.figure.tight_layout()

    def export_chart(self, file_path: str) -> None:
        """
        Export the current chart to an image file.

        Args:
            file_path: The path to save the image file.
        """
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Chart exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting chart: {e}")
    
    def add_value(self, value: float) -> None:
        """
        Add a new value to the chart
        
        Args:
            value: New value to add
        """
        # Add value and timestamp
        self.values.append(value)
        self.timestamps.append(time.time())
        
        # Limit number of points
        if len(self.values) > self.max_points:
            self.values.pop(0)
            self.timestamps.pop(0)
        
        # Update line data
        if self.timestamps:
            # Convert timestamps to relative time in seconds
            relative_times = [t - self.timestamps[0] for t in self.timestamps]
            self.line.set_data(relative_times, self.values)
            
            # Update x-axis limits
            self.axes.set_xlim(0, max(relative_times))
            
            # Redraw canvas
            self.canvas.draw()


class ComponentBreakdownChart(QWidget):
    """
    Component breakdown chart widget for displaying cognitive load components
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.components = {
            'behavior': 0.0,
            'facial': 0.0,
            'performance': 0.0,
            'eeg': 0.0
        }
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Create axes
        self.axes = self.figure.add_subplot(111)
        self.axes.set_ylim(0, 1)
        self.axes.set_title('Cognitive Load Components')
        
        # Create bars
        self.update_chart()
        
        # Adjust layout
        self.figure.tight_layout()
    
    def update_chart(self) -> None:
        """
        Update the component breakdown chart
        """
        # Clear axes
        self.axes.clear()
        
        # Get component names and values
        names = list(self.components.keys())
        values = list(self.components.values())
        
        # Create bars
        bars = self.axes.bar(names, values, color=['blue', 'green', 'orange', 'purple'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}',
                          ha='center', va='bottom')
        
        # Set y-axis limit
        self.axes.set_ylim(0, 1)
        
        # Set title
        self.axes.set_title('Cognitive Load Components')
        
        # Redraw canvas
        self.canvas.draw()
    
    def update_components(self, components: Dict[str, float]) -> None:
        """
        Update component values
        
        Args:
            components: Component values
        """
        # Update components
        for key, value in components.items():
            if key in self.components:
                self.components[key] = value
        
        # Update chart
        self.update_chart()


class MetricsPanel(QWidget):
    """
    Panel for displaying metrics
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Create metric labels
        self.create_metric_labels()
    
    def create_metric_labels(self) -> None:
        """
        Create metric labels
        """
        # Create header labels
        header_font = QFont()
        header_font.setBold(True)
        
        # Behavior metrics
        behavior_label = QLabel("Behavior Metrics")
        behavior_label.setFont(header_font)
        self.layout.addWidget(behavior_label, 0, 0, 1, 2)
        
        self.layout.addWidget(QLabel("Mouse Distance:"), 1, 0)
        self.mouse_distance_label = QLabel("0 px")
        self.layout.addWidget(self.mouse_distance_label, 1, 1)
        
        self.layout.addWidget(QLabel("Click Count:"), 2, 0)
        self.click_count_label = QLabel("0")
        self.layout.addWidget(self.click_count_label, 2, 1)
        
        self.layout.addWidget(QLabel("Keypress Count:"), 3, 0)
        self.keypress_count_label = QLabel("0")
        self.layout.addWidget(self.keypress_count_label, 3, 1)
        
        self.layout.addWidget(QLabel("Hesitation Count:"), 4, 0)
        self.hesitation_count_label = QLabel("0")
        self.layout.addWidget(self.hesitation_count_label, 4, 1)
        
        # Facial metrics
        facial_label = QLabel("Facial Metrics")
        facial_label.setFont(header_font)
        self.layout.addWidget(facial_label, 0, 2, 1, 2)
        
        self.layout.addWidget(QLabel("Confusion:"), 1, 2)
        self.confusion_label = QLabel("0.00")
        self.layout.addWidget(self.confusion_label, 1, 3)
        
        self.layout.addWidget(QLabel("Frustration:"), 2, 2)
        self.frustration_label = QLabel("0.00")
        self.layout.addWidget(self.frustration_label, 2, 3)
        
        self.layout.addWidget(QLabel("Concentration:"), 3, 2)
        self.concentration_label = QLabel("0.00")
        self.layout.addWidget(self.concentration_label, 3, 3)
        
        self.layout.addWidget(QLabel("Face Detected:"), 4, 2)
        self.face_detected_label = QLabel("No")
        self.layout.addWidget(self.face_detected_label, 4, 3)
        
        # Performance metrics
        performance_label = QLabel("Performance Metrics")
        performance_label.setFont(header_font)
        self.layout.addWidget(performance_label, 0, 4, 1, 2)
        
        self.layout.addWidget(QLabel("Accuracy:"), 1, 4)
        self.accuracy_label = QLabel("N/A")
        self.layout.addWidget(self.accuracy_label, 1, 5)
        
        self.layout.addWidget(QLabel("Response Time:"), 2, 4)
        self.response_time_label = QLabel("N/A")
        self.layout.addWidget(self.response_time_label, 2, 5)
        
        self.layout.addWidget(QLabel("Completion Rate:"), 3, 4)
        self.completion_rate_label = QLabel("N/A")
        self.layout.addWidget(self.completion_rate_label, 3, 5)
        
        self.layout.addWidget(QLabel("Task Difficulty:"), 4, 4)
        self.difficulty_label = QLabel("N/A")
        self.layout.addWidget(self.difficulty_label, 4, 5)
        
        # EEG metrics
        eeg_label = QLabel("EEG Metrics")
        eeg_label.setFont(header_font)
        self.layout.addWidget(eeg_label, 0, 6, 1, 2)
        
        self.layout.addWidget(QLabel("Signal Quality:"), 1, 6)
        self.signal_quality_label = QLabel("N/A")
        self.layout.addWidget(self.signal_quality_label, 1, 7)
        
        self.layout.addWidget(QLabel("Alpha Power:"), 2, 6)
        self.alpha_power_label = QLabel("N/A")
        self.layout.addWidget(self.alpha_power_label, 2, 7)
        
        self.layout.addWidget(QLabel("Beta Power:"), 3, 6)
        self.beta_power_label = QLabel("N/A")
        self.layout.addWidget(self.beta_power_label, 3, 7)
        
        self.layout.addWidget(QLabel("Theta Power:"), 4, 6)
        self.theta_power_label = QLabel("N/A")
        self.layout.addWidget(self.theta_power_label, 4, 7)
    
    def update_behavior_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update behavior metrics
        
        Args:
            metrics: Behavior metrics
        """
        self.mouse_distance_label.setText(f"{metrics.get('mouse_distance', 0):.0f} px")
        self.click_count_label.setText(f"{metrics.get('click_count', 0)}")
        self.keypress_count_label.setText(f"{metrics.get('keypress_count', 0)}")
        self.hesitation_count_label.setText(f"{metrics.get('hesitation_count', 0)}")
    
    def update_facial_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update facial metrics
        
        Args:
            metrics: Facial metrics
        """
        self.confusion_label.setText(f"{metrics.get('confusion', 0.0):.2f}")
        self.frustration_label.setText(f"{metrics.get('frustration', 0.0):.2f}")
        self.concentration_label.setText(f"{metrics.get('concentration', 0.0):.2f}")
        self.face_detected_label.setText("Yes" if metrics.get('face_detected', False) else "No")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics
        
        Args:
            metrics: Performance metrics
        """
        if metrics.get('accuracy') is not None:
            self.accuracy_label.setText(f"{metrics.get('accuracy', 0.0):.0%}")
        else:
            self.accuracy_label.setText("N/A")
        
        if metrics.get('response_time') is not None:
            self.response_time_label.setText(f"{metrics.get('response_time', 0.0):.2f} s")
        else:
            self.response_time_label.setText("N/A")
        
        if metrics.get('completion_rate') is not None:
            self.completion_rate_label.setText(f"{metrics.get('completion_rate', 0.0):.0%}")
        else:
            self.completion_rate_label.setText("N/A")
        
        if metrics.get('difficulty') is not None:
            # Handle difficulty as a string value, not a float
            difficulty = metrics.get('difficulty', 'N/A')
            self.difficulty_label.setText(f"{difficulty}")
        else:
            self.difficulty_label.setText("N/A")
    
    def update_eeg_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update EEG metrics
        
        Args:
            metrics: EEG metrics
        """
        if metrics.get('signal_quality') is not None:
            self.signal_quality_label.setText(f"{metrics.get('signal_quality', 0.0):.2f}")
        else:
            self.signal_quality_label.setText("N/A")
        
        if metrics.get('alpha_power') is not None:
            self.alpha_power_label.setText(f"{metrics.get('alpha_power', 0.0):.2f}")
        else:
            self.alpha_power_label.setText("N/A")
        
        if metrics.get('beta_power') is not None:
            self.beta_power_label.setText(f"{metrics.get('beta_power', 0.0):.2f}")
        else:
            self.beta_power_label.setText("N/A")
        
        if metrics.get('theta_power') is not None:
            self.theta_power_label.setText(f"{metrics.get('theta_power', 0.0):.2f}")
        else:
            self.theta_power_label.setText("N/A")


class SessionInfoPanel(QWidget):
    """
    Panel for displaying session information
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Create session info group
        self.session_group = QGroupBox("Session Information")
        self.session_layout = QFormLayout(self.session_group)
        
        # Create session labels
        self.session_id_label = QLabel("N/A")
        self.session_layout.addRow("Session ID:", self.session_id_label)
        
        self.session_start_label = QLabel("N/A")
        self.session_layout.addRow("Start Time:", self.session_start_label)
        
        self.session_duration_label = QLabel("N/A")
        self.session_layout.addRow("Duration:", self.session_duration_label)
        
        self.task_count_label = QLabel("0")
        self.session_layout.addRow("Tasks Completed:", self.task_count_label)
        
        self.layout.addWidget(self.session_group)
        
        # Create cognitive load group
        self.load_group = QGroupBox("Cognitive Load Summary")
        self.load_layout = QFormLayout(self.load_group)
        
        # Create load labels
        self.current_load_label = QLabel("0.00")
        self.load_layout.addRow("Current Load:", self.current_load_label)
        
        self.average_load_label = QLabel("0.00")
        self.load_layout.addRow("Average Load:", self.average_load_label)
        
        self.peak_load_label = QLabel("0.00")
        self.load_layout.addRow("Peak Load:", self.peak_load_label)
        
        self.load_level_label = QLabel("Low")
        self.load_layout.addRow("Load Level:", self.load_level_label)
        
        self.layout.addWidget(self.load_group)
        
        # Add spacer
        self.layout.addStretch()
    
    def update_session_info(self, session_info: Dict[str, Any]) -> None:
        """
        Update session information
        
        Args:
            session_info: Session information
        """
        self.session_id_label.setText(session_info.get('session_id', 'N/A'))
        
        start_time = session_info.get('start_time')
        if start_time:
            self.session_start_label.setText(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        else:
            self.session_start_label.setText('N/A')
        
        duration = session_info.get('duration')
        if duration is not None:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.session_duration_label.setText(f"{minutes:02d}:{seconds:02d}")
        else:
            self.session_duration_label.setText('N/A')
        
        self.task_count_label.setText(str(session_info.get('task_count', 0)))
    
    def update_load_summary(self, load_summary: Dict[str, Any]) -> None:
        """
        Update cognitive load summary
        
        Args:
            load_summary: Cognitive load summary
        """
        self.current_load_label.setText(f"{load_summary.get('current_load', 0.0):.2f}")
        self.average_load_label.setText(f"{load_summary.get('average_load', 0.0):.2f}")
        self.peak_load_label.setText(f"{load_summary.get('peak_load', 0.0):.2f}")
        
        load_level = load_summary.get('load_level', 'low')
        self.load_level_label.setText(load_level.replace('_', ' ').title())
        
        # Set color based on load level
        if load_level == 'low':
            self.load_level_label.setStyleSheet("color: green;")
        elif load_level == 'moderate':
            self.load_level_label.setStyleSheet("color: orange;")
        elif load_level == 'high':
            self.load_level_label.setStyleSheet("color: red;")
        else:
            self.load_level_label.setStyleSheet("")


class DashboardWidget(QWidget):
    """
    Dashboard widget for displaying cognitive load metrics and summary information
    """
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.task_simulator = None
        self.behavior_tracker = None
        self.facial_analyzer = None
        self.cognitive_load_calculator = None
        self.eeg_integration = None
        
        # Initialize data
        self.cognitive_load_values = []
        self.cognitive_load_timestamps = []
        self.max_history_points = 100
        
        # Create layout
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """
        Set up the user interface
        """
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create top section
        self.top_layout = QHBoxLayout()
        
        # Create gauge
        self.gauge = CognitiveLoadGauge()
        self.top_layout.addWidget(self.gauge)
        
        # Create session info panel
        self.session_info_panel = SessionInfoPanel()
        self.top_layout.addWidget(self.session_info_panel)
        
        self.main_layout.addLayout(self.top_layout)
        
        # Create middle section
        self.middle_layout = QHBoxLayout()
        
        # Create time series chart
        self.time_series_chart = TimeSeriesChart()
        self.middle_layout.addWidget(self.time_series_chart, 2)
        
        # Create component breakdown chart
        self.component_chart = ComponentBreakdownChart()
        self.middle_layout.addWidget(self.component_chart, 1)
        
        self.main_layout.addLayout(self.middle_layout)
        
        # Create bottom section
        self.metrics_panel = MetricsPanel()
        self.main_layout.addWidget(self.metrics_panel)
        
        # Create button layout
        self.button_layout = QHBoxLayout()
        
        # Create new session button
        self.new_session_button = QPushButton("New Session")
        self.new_session_button.clicked.connect(self.on_new_session_clicked)
        self.button_layout.addWidget(self.new_session_button)
        
        # Create start task button
        self.start_task_button = QPushButton("Start Task")
        self.start_task_button.clicked.connect(self.on_start_task_clicked)
        self.button_layout.addWidget(self.start_task_button)
        
        # Create export data button
        self.export_data_button = QPushButton("Export Data")
        self.export_data_button.clicked.connect(self.on_export_data_clicked)
        self.button_layout.addWidget(self.export_data_button)

        # Create export session graph button
        self.export_session_graph_button = QPushButton("Export Session Graph")
        self.export_session_graph_button.clicked.connect(self.on_export_session_graph_clicked)
        self.button_layout.addWidget(self.export_session_graph_button)
        
        self.main_layout.addLayout(self.button_layout)
    
    def set_components(self, task_simulator, behavior_tracker, facial_analyzer, 
                      cognitive_load_calculator, eeg_integration) -> None:
        """
        Set core components
        
        Args:
            task_simulator: Task simulator instance
            behavior_tracker: Behavior tracker instance
            facial_analyzer: Facial analyzer instance
            cognitive_load_calculator: Cognitive load calculator instance
            eeg_integration: EEG integration instance
        """
        self.task_simulator = task_simulator
        self.behavior_tracker = behavior_tracker
        self.facial_analyzer = facial_analyzer
        self.cognitive_load_calculator = cognitive_load_calculator
        self.eeg_integration = eeg_integration
    
    def update_ui(self) -> None:
        """
        Update UI components
        """
        if not self.cognitive_load_calculator:
            return
        
        # Get current cognitive load
        cognitive_load = self.cognitive_load_calculator.get_current_load()
        current_load = cognitive_load.get('smoothed_ecls', 0.0)
        
        # Update gauge
        self.gauge.update_gauge(current_load)
        
        # Update time series chart
        self.time_series_chart.add_value(current_load)
        
        # Update component breakdown chart
        components = {
            'behavior': cognitive_load.get('behavior_component', 0.0),
            'facial': cognitive_load.get('facial_component', 0.0),
            'performance': cognitive_load.get('performance_component', 0.0),
            'eeg': cognitive_load.get('eeg_component', 0.0)
        }
        self.component_chart.update_components(components)
        
        # Update metrics panel
        if self.behavior_tracker:
            behavior_metrics = self.behavior_tracker.get_metrics()
            self.metrics_panel.update_behavior_metrics(behavior_metrics)
        
        if self.facial_analyzer:
            facial_metrics = self.facial_analyzer.get_metrics()
            self.metrics_panel.update_facial_metrics(facial_metrics)
        
        if self.task_simulator:
            performance_metrics = self.task_simulator.get_performance_metrics()
            self.metrics_panel.update_performance_metrics(performance_metrics)
        
        if self.eeg_integration:
            eeg_metrics = self.eeg_integration.get_metrics()
            self.metrics_panel.update_eeg_metrics(eeg_metrics)
        
        # Update load summary
        load_summary = {
            'current_load': current_load,
            'average_load': cognitive_load.get('average_ecls', 0.0),
            'peak_load': cognitive_load.get('peak_ecls', 0.0),
            'load_level': cognitive_load.get('load_level', 'low')
        }
        self.session_info_panel.update_load_summary(load_summary)
    
    def update_cognitive_load(self, load: float, components: Dict[str, float]) -> None:
        """
        Update cognitive load display
        
        Args:
            load: Cognitive load value (0.0 to 1.0)
            components: Component breakdown
        """
        # Update gauge
        self.gauge.update_gauge(load)
        
        # Update time series chart
        self.time_series_chart.add_value(load)
        
        # Update component breakdown chart
        component_values = {
            'behavior': components.get('behavior_component', 0.0),
            'facial': components.get('facial_component', 0.0),
            'performance': components.get('performance_component', 0.0),
            'eeg': components.get('eeg_component', 0.0)
        }
        self.component_chart.update_components(component_values)
    
    def on_new_session(self, session_id: str) -> None:
        """
        Handle new session event
        
        Args:
            session_id: Session ID
        """
        # Update session info
        session_info = {
            'session_id': session_id,
            'start_time': time.time(),
            'duration': 0,
            'task_count': 0
        }
        self.session_info_panel.update_session_info(session_info)
        
        # Reset charts
        self.time_series_chart.values = []
        self.time_series_chart.timestamps = []
        self.component_chart.update_components({
            'behavior': 0.0,
            'facial': 0.0,
            'performance': 0.0,
            'eeg': 0.0
        })
    
    @pyqtSlot()
    def on_new_session_clicked(self) -> None:
        """
        Handle new session button click
        """
        # Emit signal to parent
        parent = self.parent()
        while parent:
            if hasattr(parent, 'on_new_session'):
                parent.on_new_session()
                break
            parent = parent.parent()

    @pyqtSlot()
    def on_export_session_graph_clicked(self) -> None:
        """
        Handle export session graph button click.
        Exports the time series chart of cognitive load.
        """
        if not self.time_series_chart:
            self.logger.warning("Time series chart is not available for export.")
            return

        # Ensure assets directory exists
        assets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')
        os.makedirs(assets_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        session_id_part = "unknown_session"
        if self.cognitive_load_calculator and hasattr(self.cognitive_load_calculator, 'session_id') and self.cognitive_load_calculator.session_id:
            session_id_part = self.cognitive_load_calculator.session_id
        elif hasattr(self, 'session_info_panel') and self.session_info_panel.session_id_label.text() != "N/A":
            session_id_part = self.session_info_panel.session_id_label.text()
        
        file_name = f"cognitive_load_session_{session_id_part}_{timestamp}.png"
        file_path = os.path.join(assets_dir, file_name)

        self.time_series_chart.export_chart(file_path)
        # Optionally, inform the user via a dialog or status bar message
        # For example, using QMessageBox:
        # msg_box = QMessageBox()
        # msg_box.setIcon(QMessageBox.Information)
        # msg_box.setText(f"Session graph exported to: {file_path}")
        # msg_box.setWindowTitle("Export Successful")
        # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.exec_()
    
    @pyqtSlot()
    def on_start_task_clicked(self) -> None:
        """
        Handle start task button click
        """
        # Emit signal to parent
        parent = self.parent()
        while parent:
            if hasattr(parent, 'on_start_task'):
                parent.on_start_task()
                break
            parent = parent.parent()
    
    @pyqtSlot()
    def on_export_data_clicked(self) -> None:
        """
        Handle export data button click
        """
        # Emit signal to parent
        parent = self.parent()
        while parent:
            if hasattr(parent, 'on_export_data'):
                parent.on_export_data()
                break
            parent = parent.parent()