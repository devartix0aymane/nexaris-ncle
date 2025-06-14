#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Widget for NEXARIS Cognitive Load Estimator

This module provides the visualization widget for displaying cognitive load data
in various chart formats.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# PyQt imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox, QSlider, QScrollArea, QFrame,
    QSizePolicy, QDateTimeEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QDateTime
from PyQt5.QtGui import QFont

# Matplotlib imports for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Import utilities
from ..utils.logging_utils import get_logger
from ..core.data_manager import DataManager


class MatplotlibCanvas(FigureCanvas):
    """
    Matplotlib canvas for embedding plots in PyQt
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100, style='default'):
        # Set the style
        if style != 'default':
            plt.style.use(style)
        
        # Create figure and axes
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # Initialize canvas
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Set canvas properties
        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
    
    def clear(self):
        """
        Clear the figure
        """
        self.axes.clear()
        self.draw()


class VisualizationWidget(QWidget):
    """
    Widget for visualizing cognitive load data
    """
    def __init__(self, data_manager: DataManager, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize data
        self.current_session_id = None
        self.session_data = None
        self.cognitive_load_calculator = None
        
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
        
        # Create controls layout
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(10)
        
        # Create session selection
        self.session_label = QLabel("Session:")
        self.controls_layout.addWidget(self.session_label)
        
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self.on_session_changed)
        self.controls_layout.addWidget(self.session_combo)
        
        # Create time range selection
        self.time_range_label = QLabel("Time Range:")
        self.controls_layout.addWidget(self.time_range_label)
        
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.dateTimeChanged.connect(self.on_time_range_changed)
        self.controls_layout.addWidget(self.start_time_edit)
        
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_time_edit.setCalendarPopup(True)
        self.end_time_edit.dateTimeChanged.connect(self.on_time_range_changed)
        self.controls_layout.addWidget(self.end_time_edit)
        
        # Create visualization type selection
        self.visualization_label = QLabel("Visualization:")
        self.controls_layout.addWidget(self.visualization_label)
        
        self.visualization_combo = QComboBox()
        self.visualization_combo.addItems([
            "Time Series",
            "Component Breakdown",
            "Heatmap",
            "Histogram",
            "Scatter Plot",
            "Box Plot"
        ])
        self.visualization_combo.currentIndexChanged.connect(self.on_visualization_changed)
        self.controls_layout.addWidget(self.visualization_combo)
        
        # Create export button
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.on_export_clicked)
        self.controls_layout.addWidget(self.export_button)
        
        self.main_layout.addLayout(self.controls_layout)
        
        # Create tab widget for different visualizations
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create time series tab
        self.time_series_tab = QWidget()
        self.tab_widget.addTab(self.time_series_tab, "Time Series")
        self.setup_time_series_tab()
        
        # Create component breakdown tab
        self.component_tab = QWidget()
        self.tab_widget.addTab(self.component_tab, "Component Breakdown")
        self.setup_component_tab()
        
        # Create heatmap tab
        self.heatmap_tab = QWidget()
        self.tab_widget.addTab(self.heatmap_tab, "Heatmap")
        self.setup_heatmap_tab()
        
        # Create histogram tab
        self.histogram_tab = QWidget()
        self.tab_widget.addTab(self.histogram_tab, "Histogram")
        self.setup_histogram_tab()
        
        # Create scatter plot tab
        self.scatter_tab = QWidget()
        self.tab_widget.addTab(self.scatter_tab, "Scatter Plot")
        self.setup_scatter_tab()
        
        # Create box plot tab
        self.box_tab = QWidget()
        self.tab_widget.addTab(self.box_tab, "Box Plot")
        self.setup_box_tab()
        
        # Connect tab changed signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Load sessions
        self.load_sessions()
    
    def setup_time_series_tab(self) -> None:
        """
        Set up the time series tab
        """
        # Create layout
        self.time_series_layout = QVBoxLayout(self.time_series_tab)
        self.time_series_layout.setContentsMargins(0, 0, 0, 0)
        self.time_series_layout.setSpacing(10)
        
        # Create controls layout
        self.time_series_controls_layout = QHBoxLayout()
        self.time_series_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.time_series_controls_layout.setSpacing(10)
        
        # Create metric selection
        self.time_series_metric_label = QLabel("Metrics:")
        self.time_series_controls_layout.addWidget(self.time_series_metric_label)
        
        self.time_series_metric_combo = QComboBox()
        self.time_series_metric_combo.addItems([
            "Cognitive Load",
            "Behavior Score",
            "Facial Score",
            "Performance Score",
            "EEG Score"
        ])
        self.time_series_metric_combo.currentIndexChanged.connect(self.update_time_series)
        self.time_series_controls_layout.addWidget(self.time_series_metric_combo)
        
        # Create smoothing option
        self.time_series_smoothing_check = QCheckBox("Smoothing")
        self.time_series_smoothing_check.setChecked(True)
        self.time_series_smoothing_check.stateChanged.connect(self.update_time_series)
        self.time_series_controls_layout.addWidget(self.time_series_smoothing_check)
        
        # Create smoothing factor
        self.time_series_smoothing_label = QLabel("Smoothing Factor:")
        self.time_series_controls_layout.addWidget(self.time_series_smoothing_label)
        
        self.time_series_smoothing_spin = QDoubleSpinBox()
        self.time_series_smoothing_spin.setRange(0.0, 1.0)
        self.time_series_smoothing_spin.setSingleStep(0.05)
        self.time_series_smoothing_spin.setValue(0.3)
        self.time_series_smoothing_spin.setDecimals(2)
        self.time_series_smoothing_spin.valueChanged.connect(self.update_time_series)
        self.time_series_controls_layout.addWidget(self.time_series_smoothing_spin)
        
        # Create show thresholds option
        self.time_series_thresholds_check = QCheckBox("Show Thresholds")
        self.time_series_thresholds_check.setChecked(True)
        self.time_series_thresholds_check.stateChanged.connect(self.update_time_series)
        self.time_series_controls_layout.addWidget(self.time_series_thresholds_check)
        
        # Add spacer
        self.time_series_controls_layout.addStretch()
        
        self.time_series_layout.addLayout(self.time_series_controls_layout)
        
        # Create matplotlib canvas
        self.time_series_canvas = MatplotlibCanvas(
            parent=self.time_series_tab,
            width=5,
            height=4,
            dpi=100,
            style=self.config.get('ui', {}).get('visualization', {}).get('chart_style', 'default')
        )
        self.time_series_layout.addWidget(self.time_series_canvas)
        
        # Create matplotlib toolbar
        self.time_series_toolbar = NavigationToolbar(self.time_series_canvas, self.time_series_tab)
        self.time_series_layout.addWidget(self.time_series_toolbar)
    
    def setup_component_tab(self) -> None:
        """
        Set up the component breakdown tab
        """
        # Create layout
        self.component_layout = QVBoxLayout(self.component_tab)
        self.component_layout.setContentsMargins(0, 0, 0, 0)
        self.component_layout.setSpacing(10)
        
        # Create controls layout
        self.component_controls_layout = QHBoxLayout()
        self.component_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.component_controls_layout.setSpacing(10)
        
        # Create chart type selection
        self.component_chart_label = QLabel("Chart Type:")
        self.component_controls_layout.addWidget(self.component_chart_label)
        
        self.component_chart_combo = QComboBox()
        self.component_chart_combo.addItems(["Stacked Area", "Line", "Bar"])
        self.component_chart_combo.currentIndexChanged.connect(self.update_component_breakdown)
        self.component_controls_layout.addWidget(self.component_chart_combo)
        
        # Create normalization option
        self.component_normalize_check = QCheckBox("Normalize")
        self.component_normalize_check.setChecked(False)
        self.component_normalize_check.stateChanged.connect(self.update_component_breakdown)
        self.component_controls_layout.addWidget(self.component_normalize_check)
        
        # Add spacer
        self.component_controls_layout.addStretch()
        
        self.component_layout.addLayout(self.component_controls_layout)
        
        # Create matplotlib canvas
        self.component_canvas = MatplotlibCanvas(
            parent=self.component_tab,
            width=5,
            height=4,
            dpi=100,
            style=self.config.get('ui', {}).get('visualization', {}).get('chart_style', 'default')
        )
        self.component_layout.addWidget(self.component_canvas)
        
        # Create matplotlib toolbar
        self.component_toolbar = NavigationToolbar(self.component_canvas, self.component_tab)
        self.component_layout.addWidget(self.component_toolbar)
    
    def setup_heatmap_tab(self) -> None:
        """
        Set up the heatmap tab
        """
        # Create layout
        self.heatmap_layout = QVBoxLayout(self.heatmap_tab)
        self.heatmap_layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap_layout.setSpacing(10)
        
        # Create controls layout
        self.heatmap_controls_layout = QHBoxLayout()
        self.heatmap_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.heatmap_controls_layout.setSpacing(10)
        
        # Create metric selection
        self.heatmap_metric_label = QLabel("Metric:")
        self.heatmap_controls_layout.addWidget(self.heatmap_metric_label)
        
        self.heatmap_metric_combo = QComboBox()
        self.heatmap_metric_combo.addItems([
            "Cognitive Load",
            "Behavior Score",
            "Facial Score",
            "Performance Score",
            "EEG Score"
        ])
        self.heatmap_metric_combo.currentIndexChanged.connect(self.update_heatmap)
        self.heatmap_controls_layout.addWidget(self.heatmap_metric_combo)
        
        # Create color map selection
        self.heatmap_cmap_label = QLabel("Color Map:")
        self.heatmap_controls_layout.addWidget(self.heatmap_cmap_label)
        
        self.heatmap_cmap_combo = QComboBox()
        self.heatmap_cmap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdBu_r", "RdYlGn_r", "YlOrRd"
        ])
        self.heatmap_cmap_combo.setCurrentText(
            self.config.get('ui', {}).get('visualization', {}).get('color_map', 'viridis')
        )
        self.heatmap_cmap_combo.currentIndexChanged.connect(self.update_heatmap)
        self.heatmap_controls_layout.addWidget(self.heatmap_cmap_combo)
        
        # Create time bins selection
        self.heatmap_bins_label = QLabel("Time Bins:")
        self.heatmap_controls_layout.addWidget(self.heatmap_bins_label)
        
        self.heatmap_bins_spin = QSpinBox()
        self.heatmap_bins_spin.setRange(5, 100)
        self.heatmap_bins_spin.setValue(20)
        self.heatmap_bins_spin.valueChanged.connect(self.update_heatmap)
        self.heatmap_controls_layout.addWidget(self.heatmap_bins_spin)
        
        # Add spacer
        self.heatmap_controls_layout.addStretch()
        
        self.heatmap_layout.addLayout(self.heatmap_controls_layout)
        
        # Create matplotlib canvas
        self.heatmap_canvas = MatplotlibCanvas(
            parent=self.heatmap_tab,
            width=5,
            height=4,
            dpi=100,
            style=self.config.get('ui', {}).get('visualization', {}).get('chart_style', 'default')
        )
        self.heatmap_layout.addWidget(self.heatmap_canvas)
        
        # Create matplotlib toolbar
        self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, self.heatmap_tab)
        self.heatmap_layout.addWidget(self.heatmap_toolbar)
    
    def setup_histogram_tab(self) -> None:
        """
        Set up the histogram tab
        """
        # Create layout
        self.histogram_layout = QVBoxLayout(self.histogram_tab)
        self.histogram_layout.setContentsMargins(0, 0, 0, 0)
        self.histogram_layout.setSpacing(10)
        
        # Create controls layout
        self.histogram_controls_layout = QHBoxLayout()
        self.histogram_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.histogram_controls_layout.setSpacing(10)
        
        # Create metric selection
        self.histogram_metric_label = QLabel("Metric:")
        self.histogram_controls_layout.addWidget(self.histogram_metric_label)
        
        self.histogram_metric_combo = QComboBox()
        self.histogram_metric_combo.addItems([
            "Cognitive Load",
            "Behavior Score",
            "Facial Score",
            "Performance Score",
            "EEG Score",
            "Response Time",
            "Mouse Movement",
            "Click Count",
            "Hesitation Count"
        ])
        self.histogram_metric_combo.currentIndexChanged.connect(self.update_histogram)
        self.histogram_controls_layout.addWidget(self.histogram_metric_combo)
        
        # Create bins selection
        self.histogram_bins_label = QLabel("Bins:")
        self.histogram_controls_layout.addWidget(self.histogram_bins_label)
        
        self.histogram_bins_spin = QSpinBox()
        self.histogram_bins_spin.setRange(5, 100)
        self.histogram_bins_spin.setValue(20)
        self.histogram_bins_spin.valueChanged.connect(self.update_histogram)
        self.histogram_controls_layout.addWidget(self.histogram_bins_spin)
        
        # Create KDE option
        self.histogram_kde_check = QCheckBox("Show KDE")
        self.histogram_kde_check.setChecked(True)
        self.histogram_kde_check.stateChanged.connect(self.update_histogram)
        self.histogram_controls_layout.addWidget(self.histogram_kde_check)
        
        # Add spacer
        self.histogram_controls_layout.addStretch()
        
        self.histogram_layout.addLayout(self.histogram_controls_layout)
        
        # Create matplotlib canvas
        self.histogram_canvas = MatplotlibCanvas(
            parent=self.histogram_tab,
            width=5,
            height=4,
            dpi=100,
            style=self.config.get('ui', {}).get('visualization', {}).get('chart_style', 'default')
        )
        self.histogram_layout.addWidget(self.histogram_canvas)
        
        # Create matplotlib toolbar
        self.histogram_toolbar = NavigationToolbar(self.histogram_canvas, self.histogram_tab)
        self.histogram_layout.addWidget(self.histogram_toolbar)
    
    def setup_scatter_tab(self) -> None:
        """
        Set up the scatter plot tab
        """
        # Create layout
        self.scatter_layout = QVBoxLayout(self.scatter_tab)
        self.scatter_layout.setContentsMargins(0, 0, 0, 0)
        self.scatter_layout.setSpacing(10)
        
        # Create controls layout
        self.scatter_controls_layout = QHBoxLayout()
        self.scatter_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.scatter_controls_layout.setSpacing(10)
        
        # Create x-axis metric selection
        self.scatter_x_label = QLabel("X-Axis:")
        self.scatter_controls_layout.addWidget(self.scatter_x_label)
        
        self.scatter_x_combo = QComboBox()
        self.scatter_x_combo.addItems([
            "Cognitive Load",
            "Behavior Score",
            "Facial Score",
            "Performance Score",
            "EEG Score",
            "Response Time",
            "Mouse Movement",
            "Click Count",
            "Hesitation Count"
        ])
        self.scatter_x_combo.setCurrentText("Behavior Score")
        self.scatter_x_combo.currentIndexChanged.connect(self.update_scatter)
        self.scatter_controls_layout.addWidget(self.scatter_x_combo)
        
        # Create y-axis metric selection
        self.scatter_y_label = QLabel("Y-Axis:")
        self.scatter_controls_layout.addWidget(self.scatter_y_label)
        
        self.scatter_y_combo = QComboBox()
        self.scatter_y_combo.addItems([
            "Cognitive Load",
            "Behavior Score",
            "Facial Score",
            "Performance Score",
            "EEG Score",
            "Response Time",
            "Mouse Movement",
            "Click Count",
            "Hesitation Count"
        ])
        self.scatter_y_combo.setCurrentText("Cognitive Load")
        self.scatter_y_combo.currentIndexChanged.connect(self.update_scatter)
        self.scatter_controls_layout.addWidget(self.scatter_y_combo)
        
        # Create regression line option
        self.scatter_regression_check = QCheckBox("Show Regression Line")
        self.scatter_regression_check.setChecked(True)
        self.scatter_regression_check.stateChanged.connect(self.update_scatter)
        self.scatter_controls_layout.addWidget(self.scatter_regression_check)
        
        # Add spacer
        self.scatter_controls_layout.addStretch()
        
        self.scatter_layout.addLayout(self.scatter_controls_layout)
        
        # Create matplotlib canvas
        self.scatter_canvas = MatplotlibCanvas(
            parent=self.scatter_tab,
            width=5,
            height=4,
            dpi=100,
            style=self.config.get('ui', {}).get('visualization', {}).get('chart_style', 'default')
        )
        self.scatter_layout.addWidget(self.scatter_canvas)
        
        # Create matplotlib toolbar
        self.scatter_toolbar = NavigationToolbar(self.scatter_canvas, self.scatter_tab)
        self.scatter_layout.addWidget(self.scatter_toolbar)
    
    def setup_box_tab(self) -> None:
        """
        Set up the box plot tab
        """
        # Create layout
        self.box_layout = QVBoxLayout(self.box_tab)
        self.box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_layout.setSpacing(10)
        
        # Create controls layout
        self.box_controls_layout = QHBoxLayout()
        self.box_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.box_controls_layout.setSpacing(10)
        
        # Create metric selection
        self.box_metric_label = QLabel("Metrics:")
        self.box_controls_layout.addWidget(self.box_metric_label)
        
        self.box_metric_combo = QComboBox()
        self.box_metric_combo.addItems([
            "All Scores",
            "Cognitive Load",
            "Behavior Score",
            "Facial Score",
            "Performance Score",
            "EEG Score"
        ])
        self.box_metric_combo.currentIndexChanged.connect(self.update_box_plot)
        self.box_controls_layout.addWidget(self.box_metric_combo)
        
        # Create grouping selection
        self.box_group_label = QLabel("Group By:")
        self.box_controls_layout.addWidget(self.box_group_label)
        
        self.box_group_combo = QComboBox()
        self.box_group_combo.addItems(["None", "Task Type", "Difficulty"])
        self.box_group_combo.currentIndexChanged.connect(self.update_box_plot)
        self.box_controls_layout.addWidget(self.box_group_combo)
        
        # Create violin plot option
        self.box_violin_check = QCheckBox("Violin Plot")
        self.box_violin_check.setChecked(False)
        self.box_violin_check.stateChanged.connect(self.update_box_plot)
        self.box_controls_layout.addWidget(self.box_violin_check)
        
        # Add spacer
        self.box_controls_layout.addStretch()
        
        self.box_layout.addLayout(self.box_controls_layout)
        
        # Create matplotlib canvas
        self.box_canvas = MatplotlibCanvas(
            parent=self.box_tab,
            width=5,
            height=4,
            dpi=100,
            style=self.config.get('ui', {}).get('visualization', {}).get('chart_style', 'default')
        )
        self.box_layout.addWidget(self.box_canvas)
        
        # Create matplotlib toolbar
        self.box_toolbar = NavigationToolbar(self.box_canvas, self.box_tab)
        self.box_layout.addWidget(self.box_toolbar)
    
    def set_components(self, data_manager, cognitive_load_calculator) -> None:
        """Set core components
        
        Args:
            data_manager: Data manager instance
            cognitive_load_calculator: Cognitive load calculator instance
        """
        self.data_manager = data_manager
        self.cognitive_load_calculator = cognitive_load_calculator
        
        # Load sessions
        self.load_sessions()
    
    def load_sessions(self) -> None:
        """
        Load available sessions
        """
        # Clear session combo
        self.session_combo.clear()
        
        # Get sessions from data manager
        sessions = self.data_manager.list_sessions()
        
        if not sessions:
            self.session_combo.addItem("No sessions available")
            self.session_combo.setEnabled(False)
            return
        
        # Add sessions to combo box
        for session_id, session_info in sessions.items():
            session_name = f"{session_info.get('name', 'Session')} ({session_id})"
            self.session_combo.addItem(session_name, session_id)
        
        # Enable session combo
        self.session_combo.setEnabled(True)
        
        # Select first session
        if self.session_combo.count() > 0:
            self.session_combo.setCurrentIndex(0)
    
    def load_session_data(self, session_id: str) -> None:
        """
        Load data for the selected session
        
        Args:
            session_id: Session ID to load
        """
        # Load session data
        self.session_data = self.data_manager.load_session(session_id)
        
        if not self.session_data:
            self.logger.warning(f"Failed to load session data for {session_id}")
            return
        
        # Set current session ID
        self.current_session_id = session_id
        
        # Set time range
        start_time = self.session_data.get('start_time')
        end_time = self.session_data.get('end_time')
        
        if start_time and end_time:
            start_dt = QDateTime.fromString(start_time, Qt.ISODate)
            end_dt = QDateTime.fromString(end_time, Qt.ISODate)
            
            self.start_time_edit.setDateTime(start_dt)
            self.end_time_edit.setDateTime(end_dt)
        
        # Update current visualization
        self.update_current_visualization()
    
    def update_current_visualization(self) -> None:
        """
        Update the current visualization based on the selected tab
        """
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Time Series
            self.update_time_series()
        elif current_tab == 1:  # Component Breakdown
            self.update_component_breakdown()
        elif current_tab == 2:  # Heatmap
            self.update_heatmap()
        elif current_tab == 3:  # Histogram
            self.update_histogram()
        elif current_tab == 4:  # Scatter Plot
            self.update_scatter()
        elif current_tab == 5:  # Box Plot
            self.update_box_plot()
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        Get filtered data based on time range
        
        Returns:
            Filtered DataFrame
        """
        if not self.session_data or 'cognitive_load_data' not in self.session_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.session_data['cognitive_load_data'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time range
        start_time = self.start_time_edit.dateTime().toString(Qt.ISODate)
        end_time = self.end_time_edit.dateTime().toString(Qt.ISODate)
        
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        
        return df
    
    def update_time_series(self) -> None:
        """
        Update the time series visualization
        """
        # Get filtered data
        df = self.get_filtered_data()
        
        if df.empty:
            self.time_series_canvas.clear()
            return
        
        # Get selected metric
        metric = self.time_series_metric_combo.currentText()
        metric_map = {
            "Cognitive Load": "cognitive_load",
            "Behavior Score": "behavior_score",
            "Facial Score": "facial_score",
            "Performance Score": "performance_score",
            "EEG Score": "eeg_score"
        }
        
        metric_key = metric_map.get(metric, "cognitive_load")
        
        # Apply smoothing if enabled
        if self.time_series_smoothing_check.isChecked():
            alpha = self.time_series_smoothing_spin.value()
            df[f"{metric_key}_smooth"] = df[metric_key].ewm(alpha=alpha).mean()
            plot_key = f"{metric_key}_smooth"
        else:
            plot_key = metric_key
        
        # Clear the figure
        self.time_series_canvas.axes.clear()
        
        # Plot the data
        self.time_series_canvas.axes.plot(
            df['timestamp'],
            df[plot_key],
            label=metric,
            color='blue',
            linewidth=2
        )
        
        # Add thresholds if enabled
        if self.time_series_thresholds_check.isChecked() and metric_key == "cognitive_load":
            # Get thresholds from config
            low_threshold = self.config.get('scoring', {}).get('thresholds', {}).get('low', 0.4)
            high_threshold = self.config.get('scoring', {}).get('thresholds', {}).get('high', 0.7)
            
            # Plot thresholds
            self.time_series_canvas.axes.axhline(
                y=low_threshold,
                color='green',
                linestyle='--',
                alpha=0.7,
                label=f"Low Threshold ({low_threshold})"
            )
            
            self.time_series_canvas.axes.axhline(
                y=high_threshold,
                color='red',
                linestyle='--',
                alpha=0.7,
                label=f"High Threshold ({high_threshold})"
            )
        
        # Set labels and title
        self.time_series_canvas.axes.set_xlabel('Time')
        self.time_series_canvas.axes.set_ylabel(metric)
        self.time_series_canvas.axes.set_title(f"{metric} Over Time")
        
        # Format x-axis
        self.time_series_canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Add legend
        self.time_series_canvas.axes.legend()
        
        # Add grid
        self.time_series_canvas.axes.grid(True, alpha=0.3)
        
        # Adjust layout
        self.time_series_canvas.fig.tight_layout()
        
        # Draw the canvas
        self.time_series_canvas.draw()
    
    def update_component_breakdown(self) -> None:
        """
        Update the component breakdown visualization
        """
        # Get filtered data
        df = self.get_filtered_data()
        
        if df.empty:
            self.component_canvas.clear()
            return
        
        # Get chart type
        chart_type = self.component_chart_combo.currentText()
        
        # Get normalization option
        normalize = self.component_normalize_check.isChecked()
        
        # Clear the figure
        self.component_canvas.axes.clear()
        
        # Extract component scores
        components = ['behavior_score', 'facial_score', 'performance_score', 'eeg_score']
        component_labels = ['Behavior', 'Facial', 'Performance', 'EEG']
        
        # Normalize if requested
        if normalize:
            for component in components:
                if df[component].max() > 0:
                    df[f"{component}_norm"] = df[component] / df[component].max()
                else:
                    df[f"{component}_norm"] = df[component]
            
            plot_components = [f"{component}_norm" for component in components]
        else:
            plot_components = components
        
        # Plot based on chart type
        if chart_type == "Stacked Area":
            self.component_canvas.axes.stackplot(
                df['timestamp'],
                [df[component] for component in plot_components],
                labels=component_labels,
                alpha=0.7
            )
        elif chart_type == "Line":
            for i, component in enumerate(plot_components):
                self.component_canvas.axes.plot(
                    df['timestamp'],
                    df[component],
                    label=component_labels[i]
                )
        elif chart_type == "Bar":
            # Resample to reduce number of bars
            resampled = df.set_index('timestamp').resample('1min').mean().reset_index()
            
            # Set up bar positions
            bar_width = 0.2
            index = np.arange(len(resampled))
            
            for i, component in enumerate(plot_components):
                self.component_canvas.axes.bar(
                    index + i * bar_width,
                    resampled[component],
                    bar_width,
                    label=component_labels[i]
                )
            
            # Set x-ticks
            self.component_canvas.axes.set_xticks(index + bar_width * (len(components) - 1) / 2)
            self.component_canvas.axes.set_xticklabels(
                [d.strftime('%H:%M:%S') for d in resampled['timestamp']],
                rotation=45
            )
        
        # Set labels and title
        self.component_canvas.axes.set_xlabel('Time')
        
        if normalize:
            self.component_canvas.axes.set_ylabel('Normalized Score')
            self.component_canvas.axes.set_title('Normalized Component Breakdown')
        else:
            self.component_canvas.axes.set_ylabel('Score')
            self.component_canvas.axes.set_title('Component Breakdown')
        
        # Format x-axis for line and stacked area
        if chart_type != "Bar":
            self.component_canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Add legend
        self.component_canvas.axes.legend()
        
        # Add grid
        self.component_canvas.axes.grid(True, alpha=0.3)
        
        # Adjust layout
        self.component_canvas.fig.tight_layout()
        
        # Draw the canvas
        self.component_canvas.draw()
    
    def update_heatmap(self) -> None:
        """
        Update the heatmap visualization
        """
        # Get filtered data
        df = self.get_filtered_data()
        
        if df.empty:
            self.heatmap_canvas.clear()
            return
        
        # Get selected metric
        metric = self.heatmap_metric_combo.currentText()
        metric_map = {
            "Cognitive Load": "cognitive_load",
            "Behavior Score": "behavior_score",
            "Facial Score": "facial_score",
            "Performance Score": "performance_score",
            "EEG Score": "eeg_score"
        }
        
        metric_key = metric_map.get(metric, "cognitive_load")
        
        # Get color map
        cmap = self.heatmap_cmap_combo.currentText()
        
        # Get time bins
        time_bins = self.heatmap_bins_spin.value()
        
        # Clear the figure
        self.heatmap_canvas.axes.clear()
        
        # Create time bins
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        time_range = end_time - start_time
        bin_size = time_range / time_bins
        
        # Create bin edges
        bin_edges = [start_time + i * bin_size for i in range(time_bins + 1)]
        
        # Assign bins to data points
        df['time_bin'] = pd.cut(df['timestamp'], bins=bin_edges, labels=False)
        
        # Create heatmap data
        heatmap_data = df.groupby('time_bin')[metric_key].mean().values.reshape(-1, 1)
        
        # Create time labels
        time_labels = [start_time + i * bin_size for i in range(time_bins)]
        time_labels = [t.strftime('%H:%M:%S') for t in time_labels]
        
        # Plot heatmap
        im = self.heatmap_canvas.axes.imshow(
            heatmap_data,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = self.heatmap_canvas.fig.colorbar(im, ax=self.heatmap_canvas.axes)
        cbar.set_label(metric)
        
        # Set labels and title
        self.heatmap_canvas.axes.set_xlabel('Metric Value')
        self.heatmap_canvas.axes.set_ylabel('Time')
        self.heatmap_canvas.axes.set_title(f"{metric} Heatmap")
        
        # Set y-ticks
        self.heatmap_canvas.axes.set_yticks(np.arange(len(time_labels)))
        self.heatmap_canvas.axes.set_yticklabels(time_labels)
        
        # Hide x-ticks
        self.heatmap_canvas.axes.set_xticks([])
        
        # Adjust layout
        self.heatmap_canvas.fig.tight_layout()
        
        # Draw the canvas
        self.heatmap_canvas.draw()
    
    def update_histogram(self) -> None:
        """
        Update the histogram visualization
        """
        # Get filtered data
        df = self.get_filtered_data()
        
        if df.empty:
            self.histogram_canvas.clear()
            return
        
        # Get selected metric
        metric = self.histogram_metric_combo.currentText()
        metric_map = {
            "Cognitive Load": "cognitive_load",
            "Behavior Score": "behavior_score",
            "Facial Score": "facial_score",
            "Performance Score": "performance_score",
            "EEG Score": "eeg_score",
            "Response Time": "response_time",
            "Mouse Movement": "mouse_movement",
            "Click Count": "click_count",
            "Hesitation Count": "hesitation_count"
        }
        
        metric_key = metric_map.get(metric, "cognitive_load")
        
        # Get bins
        bins = self.histogram_bins_spin.value()
        
        # Get KDE option
        kde = self.histogram_kde_check.isChecked()
        
        # Clear the figure
        self.histogram_canvas.axes.clear()
        
        # Plot histogram
        sns.histplot(
            data=df,
            x=metric_key,
            bins=bins,
            kde=kde,
            ax=self.histogram_canvas.axes
        )
        
        # Set labels and title
        self.histogram_canvas.axes.set_xlabel(metric)
        self.histogram_canvas.axes.set_ylabel('Frequency')
        self.histogram_canvas.axes.set_title(f"{metric} Distribution")
        
        # Add grid
        self.histogram_canvas.axes.grid(True, alpha=0.3)
        
        # Adjust layout
        self.histogram_canvas.fig.tight_layout()
        
        # Draw the canvas
        self.histogram_canvas.draw()
    
    def update_scatter(self) -> None:
        """
        Update the scatter plot visualization
        """
        # Get filtered data
        df = self.get_filtered_data()
        
        if df.empty:
            self.scatter_canvas.clear()
            return
        
        # Get selected metrics
        x_metric = self.scatter_x_combo.currentText()
        y_metric = self.scatter_y_combo.currentText()
        
        metric_map = {
            "Cognitive Load": "cognitive_load",
            "Behavior Score": "behavior_score",
            "Facial Score": "facial_score",
            "Performance Score": "performance_score",
            "EEG Score": "eeg_score",
            "Response Time": "response_time",
            "Mouse Movement": "mouse_movement",
            "Click Count": "click_count",
            "Hesitation Count": "hesitation_count"
        }
        
        x_key = metric_map.get(x_metric, "behavior_score")
        y_key = metric_map.get(y_metric, "cognitive_load")
        
        # Get regression line option
        regression = self.scatter_regression_check.isChecked()
        
        # Clear the figure
        self.scatter_canvas.axes.clear()
        
        # Plot scatter plot
        if regression:
            sns.regplot(
                data=df,
                x=x_key,
                y=y_key,
                ax=self.scatter_canvas.axes,
                scatter_kws={'alpha': 0.5}
            )
        else:
            sns.scatterplot(
                data=df,
                x=x_key,
                y=y_key,
                ax=self.scatter_canvas.axes,
                alpha=0.5
            )
        
        # Set labels and title
        self.scatter_canvas.axes.set_xlabel(x_metric)
        self.scatter_canvas.axes.set_ylabel(y_metric)
        self.scatter_canvas.axes.set_title(f"{y_metric} vs {x_metric}")
        
        # Add grid
        self.scatter_canvas.axes.grid(True, alpha=0.3)
        
        # Adjust layout
        self.scatter_canvas.fig.tight_layout()
        
        # Draw the canvas
        self.scatter_canvas.draw()
    
    def update_box_plot(self) -> None:
        """
        Update the box plot visualization
        """
        # Get filtered data
        df = self.get_filtered_data()
        
        if df.empty:
            self.box_canvas.clear()
            return
        
        # Get selected metric
        metric = self.box_metric_combo.currentText()
        
        # Get grouping
        grouping = self.box_group_combo.currentText()
        
        # Get violin plot option
        violin = self.box_violin_check.isChecked()
        
        # Clear the figure
        self.box_canvas.axes.clear()
        
        # Prepare data for plotting
        if metric == "All Scores":
            # Melt the dataframe to get all scores in one column
            plot_df = df.melt(
                id_vars=['timestamp'],
                value_vars=['behavior_score', 'facial_score', 'performance_score', 'eeg_score'],
                var_name='component',
                value_name='score'
            )
            
            # Map component names
            component_map = {
                'behavior_score': 'Behavior',
                'facial_score': 'Facial',
                'performance_score': 'Performance',
                'eeg_score': 'EEG'
            }
            
            plot_df['component'] = plot_df['component'].map(component_map)
            
            # Plot
            if violin:
                sns.violinplot(
                    data=plot_df,
                    x='component',
                    y='score',
                    ax=self.box_canvas.axes
                )
            else:
                sns.boxplot(
                    data=plot_df,
                    x='component',
                    y='score',
                    ax=self.box_canvas.axes
                )
            
            # Set labels and title
            self.box_canvas.axes.set_xlabel('Component')
            self.box_canvas.axes.set_ylabel('Score')
            self.box_canvas.axes.set_title('Component Score Distribution')
        else:
            # Get metric key
            metric_map = {
                "Cognitive Load": "cognitive_load",
                "Behavior Score": "behavior_score",
                "Facial Score": "facial_score",
                "Performance Score": "performance_score",
                "EEG Score": "eeg_score"
            }
            
            metric_key = metric_map.get(metric, "cognitive_load")
            
            # Handle grouping
            if grouping == "None":
                # Simple box plot without grouping
                if violin:
                    sns.violinplot(
                        data=df,
                        y=metric_key,
                        ax=self.box_canvas.axes
                    )
                else:
                    sns.boxplot(
                        data=df,
                        y=metric_key,
                        ax=self.box_canvas.axes
                    )
                
                # Set labels and title
                self.box_canvas.axes.set_xlabel('')
                self.box_canvas.axes.set_ylabel(metric)
                self.box_canvas.axes.set_title(f"{metric} Distribution")
            else:
                # Group by task type or difficulty
                group_key = 'task_type' if grouping == 'Task Type' else 'difficulty'
                
                # Check if grouping key exists in data
                if group_key in df.columns:
                    if violin:
                        sns.violinplot(
                            data=df,
                            x=group_key,
                            y=metric_key,
                            ax=self.box_canvas.axes
                        )
                    else:
                        sns.boxplot(
                            data=df,
                            x=group_key,
                            y=metric_key,
                            ax=self.box_canvas.axes
                        )
                    
                    # Set labels and title
                    self.box_canvas.axes.set_xlabel(grouping)
                    self.box_canvas.axes.set_ylabel(metric)
                    self.box_canvas.axes.set_title(f"{metric} Distribution by {grouping}")
                else:
                    # Fallback to simple box plot
                    if violin:
                        sns.violinplot(
                            data=df,
                            y=metric_key,
                            ax=self.box_canvas.axes
                        )
                    else:
                        sns.boxplot(
                            data=df,
                            y=metric_key,
                            ax=self.box_canvas.axes
                        )
                    
                    # Set labels and title
                    self.box_canvas.axes.set_xlabel('')
                    self.box_canvas.axes.set_ylabel(metric)
                    self.box_canvas.axes.set_title(f"{metric} Distribution")
        
        # Add grid
        self.box_canvas.axes.grid(True, alpha=0.3)
        
        # Adjust layout
        self.box_canvas.fig.tight_layout()
        
        # Draw the canvas
        self.box_canvas.draw()
    
    @pyqtSlot(int)
    def on_session_changed(self, index: int) -> None:
        """
        Handle session selection change
        
        Args:
            index: Index of the selected session
        """
        if index < 0:
            return
        
        # Get session ID
        session_id = self.session_combo.itemData(index)
        
        if not session_id:
            return
        
        # Load session data
        self.load_session_data(session_id)
    
    @pyqtSlot()
    def on_time_range_changed(self) -> None:
        """
        Handle time range change
        """
        # Update current visualization
        self.update_current_visualization()
    
    @pyqtSlot(int)
    def on_visualization_changed(self, index: int) -> None:
        """
        Handle visualization type change
        
        Args:
            index: Index of the selected visualization
        """
        # Set tab index to match visualization
        self.tab_widget.setCurrentIndex(index)
    
    @pyqtSlot(int)
    def on_tab_changed(self, index: int) -> None:
        """
        Handle tab change
        
        Args:
            index: Index of the selected tab
        """
        # Set visualization combo to match tab
        self.visualization_combo.setCurrentIndex(index)
        
        # Update visualization
        self.update_current_visualization()
    
    @pyqtSlot()
    def on_export_clicked(self) -> None:
        """
        Handle export button click
        """
        if not self.session_data or not self.current_session_id:
            QMessageBox.warning(self, "Export Error", "No session data available to export.")
            return
        
        # Get export format from config
        default_format = self.config.get('data', {}).get('export', {}).get('default_format', 'CSV')
        
        # Create file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session Data",
            f"session_{self.current_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            f"{default_format} Files (*.{default_format.lower()});;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Export data
        try:
            self.data_manager.export_session(self.current_session_id, file_path)
            QMessageBox.information(self, "Export Successful", f"Session data exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export session data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export session data: {e}")
    
    def update_ui(self) -> None:
        """
        Update the UI components
        """
        # This method is called by MainWindow.update_ui()
        # Update visualizations if needed
        if self.current_session_id and self.isVisible():
            # Refresh data if needed
            pass