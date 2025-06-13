#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI utilities for NEXARIS Cognitive Load Estimator

This module provides functions for creating and managing UI elements.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List

# Import PyQt5 for UI components
from PyQt5.QtWidgets import QWidget, QApplication, QStyleFactory
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QSize

# Import matplotlib for charts
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import PyQtChart for gauges and charts
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QPieSeries

# Set up logging
logger = logging.getLogger(__name__)


def apply_theme(widget: QWidget, theme: str = "light") -> None:
    """
    Apply a theme to a widget and its children
    
    Args:
        widget: The widget to apply the theme to
        theme: The theme to apply ("light" or "dark")
    """
    app = QApplication.instance()
    
    if theme == "dark":
        # Set the application style
        app.setStyle(QStyleFactory.create("Fusion"))
        
        # Create a dark palette
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
        
        # Apply the palette
        app.setPalette(palette)
        
        # Set stylesheet for additional customization
        app.setStyleSheet("""
            QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background-color: #353535; color: #ffffff; padding: 5px; }
            QTabBar::tab:selected { background-color: #444; }
            QComboBox { background-color: #353535; color: #ffffff; border: 1px solid #777; }
            QComboBox QAbstractItemView { background-color: #353535; color: #ffffff; }
            QLineEdit { background-color: #353535; color: #ffffff; border: 1px solid #777; }
            QPushButton { background-color: #353535; color: #ffffff; border: 1px solid #777; padding: 5px; }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #2a82da; }
            QProgressBar { border: 1px solid #777; background-color: #353535; color: #ffffff; }
            QProgressBar::chunk { background-color: #2a82da; }
        """)
    else:  # Light theme
        # Set the application style
        app.setStyle(QStyleFactory.create("Fusion"))
        
        # Create a light palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 120, 215))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # Apply the palette
        app.setPalette(palette)
        
        # Set stylesheet for additional customization
        app.setStyleSheet("""
            QToolTip { color: #000000; background-color: #ffffff; border: 1px solid #000000; }
            QTabWidget::pane { border: 1px solid #c0c0c0; }
            QTabBar::tab { background-color: #f0f0f0; color: #000000; padding: 5px; }
            QTabBar::tab:selected { background-color: #ffffff; }
            QComboBox { background-color: #ffffff; color: #000000; border: 1px solid #c0c0c0; }
            QComboBox QAbstractItemView { background-color: #ffffff; color: #000000; }
            QLineEdit { background-color: #ffffff; color: #000000; border: 1px solid #c0c0c0; }
            QPushButton { background-color: #f0f0f0; color: #000000; border: 1px solid #c0c0c0; padding: 5px; }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #0078d7; color: #ffffff; }
            QProgressBar { border: 1px solid #c0c0c0; background-color: #ffffff; color: #000000; }
            QProgressBar::chunk { background-color: #0078d7; }
        """)
    
    logger.info(f"Applied {theme} theme to application")


def create_gauge(title: str, min_value: float, max_value: float, initial_value: float = 0.0) -> QChartView:
    """
    Create a gauge chart for displaying a single value
    
    Args:
        title: The title of the gauge
        min_value: The minimum value of the gauge
        max_value: The maximum value of the gauge
        initial_value: The initial value of the gauge
        
    Returns:
        A QChartView containing the gauge
    """
    # Create a pie series for the gauge
    series = QPieSeries()
    
    # Calculate the value as a percentage
    value_percent = (initial_value - min_value) / (max_value - min_value) * 100
    
    # Add slices for the value and the remaining space
    value_slice = series.append(f"{title}: {initial_value:.2f}", value_percent)
    empty_slice = series.append("", 100 - value_percent)
    
    # Set the colors and other properties
    value_slice.setBrush(QColor(0, 120, 215))  # Blue for the value
    empty_slice.setBrush(QColor(240, 240, 240))  # Light gray for the empty space
    
    # Create a chart and add the series
    chart = QChart()
    chart.addSeries(series)
    chart.setTitle(title)
    chart.legend().hide()
    
    # Create a chart view
    chart_view = QChartView(chart)
    chart_view.setRenderHint(chart_view.Antialiasing)
    
    return chart_view


def update_gauge(gauge: QChartView, value: float, min_value: float, max_value: float) -> None:
    """
    Update the value of a gauge chart
    
    Args:
        gauge: The gauge chart to update
        value: The new value
        min_value: The minimum value of the gauge
        max_value: The maximum value of the gauge
    """
    # Get the chart and series
    chart = gauge.chart()
    series = chart.series()[0]
    
    # Calculate the value as a percentage
    value_percent = (value - min_value) / (max_value - min_value) * 100
    
    # Update the slices
    series.slices()[0].setValue(value_percent)
    series.slices()[0].setLabel(f"{chart.title()}: {value:.2f}")
    series.slices()[1].setValue(100 - value_percent)


def create_chart(title: str, x_label: str, y_label: str) -> QChartView:
    """
    Create a line chart for displaying time series data
    
    Args:
        title: The title of the chart
        x_label: The label for the x-axis
        y_label: The label for the y-axis
        
    Returns:
        A QChartView containing the chart
    """
    # Create a series for the data
    series = QLineSeries()
    
    # Create a chart and add the series
    chart = QChart()
    chart.addSeries(series)
    chart.setTitle(title)
    
    # Create axes
    x_axis = QValueAxis()
    x_axis.setTitleText(x_label)
    y_axis = QValueAxis()
    y_axis.setTitleText(y_label)
    
    # Add axes to the chart
    chart.addAxis(x_axis, Qt.AlignBottom)
    chart.addAxis(y_axis, Qt.AlignLeft)
    
    # Attach the series to the axes
    series.attachAxis(x_axis)
    series.attachAxis(y_axis)
    
    # Create a chart view
    chart_view = QChartView(chart)
    chart_view.setRenderHint(chart_view.Antialiasing)
    
    return chart_view


def update_chart(chart_view: QChartView, x_data: List[float], y_data: List[float]) -> None:
    """
    Update the data in a line chart
    
    Args:
        chart_view: The chart view to update
        x_data: The x-axis data points
        y_data: The y-axis data points
    """
    # Get the chart and series
    chart = chart_view.chart()
    series = chart.series()[0]
    
    # Clear the existing data
    series.clear()
    
    # Add the new data points
    for x, y in zip(x_data, y_data):
        series.append(x, y)
    
    # Update the axes ranges
    x_axis = chart.axes(Qt.Horizontal)[0]
    y_axis = chart.axes(Qt.Vertical)[0]
    
    if x_data:
        x_axis.setRange(min(x_data), max(x_data))
    if y_data:
        y_min = min(y_data)
        y_max = max(y_data)
        y_range = y_max - y_min
        y_axis.setRange(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)


class MatplotlibCanvas(FigureCanvas):
    """
    A canvas for embedding Matplotlib figures in PyQt5 widgets
    """
    
    def __init__(self, width: int = 5, height: int = 4, dpi: int = 100):
        """
        Initialize the canvas
        
        Args:
            width: The width of the figure in inches
            height: The height of the figure in inches
            dpi: The resolution of the figure in dots per inch
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        
        # Set up the figure
        self.fig.tight_layout()


def create_matplotlib_chart(parent: QWidget, width: int = 5, height: int = 4, dpi: int = 100) -> MatplotlibCanvas:
    """
    Create a Matplotlib chart for advanced visualizations
    
    Args:
        parent: The parent widget
        width: The width of the figure in inches
        height: The height of the figure in inches
        dpi: The resolution of the figure in dots per inch
        
    Returns:
        A MatplotlibCanvas containing the chart
    """
    return MatplotlibCanvas(width, height, dpi)


def create_time_series_chart(canvas: MatplotlibCanvas, x_data: List[float], y_data: List[float], 
                            title: str, x_label: str, y_label: str, color: str = 'blue') -> None:
    """
    Create a time series chart using Matplotlib
    
    Args:
        canvas: The MatplotlibCanvas to draw on
        x_data: The x-axis data points
        y_data: The y-axis data points
        title: The title of the chart
        x_label: The label for the x-axis
        y_label: The label for the y-axis
        color: The color of the line
    """
    # Clear the axes
    canvas.axes.clear()
    
    # Plot the data
    canvas.axes.plot(x_data, y_data, color=color, linewidth=2)
    
    # Set the title and labels
    canvas.axes.set_title(title)
    canvas.axes.set_xlabel(x_label)
    canvas.axes.set_ylabel(y_label)
    
    # Add a grid
    canvas.axes.grid(True, linestyle='--', alpha=0.7)
    
    # Update the canvas
    canvas.fig.tight_layout()
    canvas.draw()


def create_bar_chart(canvas: MatplotlibCanvas, categories: List[str], values: List[float], 
                    title: str, x_label: str, y_label: str, color: str = 'blue') -> None:
    """
    Create a bar chart using Matplotlib
    
    Args:
        canvas: The MatplotlibCanvas to draw on
        categories: The categories for the bars
        values: The values for the bars
        title: The title of the chart
        x_label: The label for the x-axis
        y_label: The label for the y-axis
        color: The color of the bars
    """
    # Clear the axes
    canvas.axes.clear()
    
    # Plot the data
    canvas.axes.bar(categories, values, color=color, alpha=0.7)
    
    # Set the title and labels
    canvas.axes.set_title(title)
    canvas.axes.set_xlabel(x_label)
    canvas.axes.set_ylabel(y_label)
    
    # Add a grid
    canvas.axes.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Rotate the x-axis labels if there are many categories
    if len(categories) > 5:
        canvas.axes.set_xticklabels(categories, rotation=45, ha='right')
    
    # Update the canvas
    canvas.fig.tight_layout()
    canvas.draw()


def create_pie_chart(canvas: MatplotlibCanvas, categories: List[str], values: List[float], 
                    title: str, colors: List[str] = None) -> None:
    """
    Create a pie chart using Matplotlib
    
    Args:
        canvas: The MatplotlibCanvas to draw on
        categories: The categories for the pie slices
        values: The values for the pie slices
        title: The title of the chart
        colors: The colors for the pie slices
    """
    # Clear the axes
    canvas.axes.clear()
    
    # Plot the data
    canvas.axes.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
    
    # Set the title
    canvas.axes.set_title(title)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    canvas.axes.axis('equal')
    
    # Update the canvas
    canvas.fig.tight_layout()
    canvas.draw()


def create_heatmap(canvas: MatplotlibCanvas, data: List[List[float]], row_labels: List[str], 
                  col_labels: List[str], title: str, cmap: str = 'viridis') -> None:
    """
    Create a heatmap using Matplotlib
    
    Args:
        canvas: The MatplotlibCanvas to draw on
        data: The 2D data for the heatmap
        row_labels: The labels for the rows
        col_labels: The labels for the columns
        title: The title of the chart
        cmap: The colormap to use
    """
    # Clear the axes
    canvas.axes.clear()
    
    # Plot the data
    im = canvas.axes.imshow(data, cmap=cmap)
    
    # Set the title
    canvas.axes.set_title(title)
    
    # Set the tick labels
    canvas.axes.set_xticks(range(len(col_labels)))
    canvas.axes.set_yticks(range(len(row_labels)))
    canvas.axes.set_xticklabels(col_labels)
    canvas.axes.set_yticklabels(row_labels)
    
    # Rotate the x-axis labels if there are many columns
    if len(col_labels) > 5:
        canvas.axes.set_xticklabels(col_labels, rotation=45, ha='right')
    
    # Add a colorbar
    canvas.fig.colorbar(im, ax=canvas.axes)
    
    # Update the canvas
    canvas.fig.tight_layout()
    canvas.draw()


def set_font_size(widget: QWidget, size: int) -> None:
    """
    Set the font size for a widget and its children
    
    Args:
        widget: The widget to set the font size for
        size: The font size in points
    """
    font = widget.font()
    font.setPointSize(size)
    widget.setFont(font)
    
    # Apply to all child widgets
    for child in widget.findChildren(QWidget):
        child_font = child.font()
        child_font.setPointSize(size)
        child.setFont(child_font)


def get_color_for_value(value: float, min_value: float = 0.0, max_value: float = 1.0, 
                       low_color: QColor = QColor(0, 255, 0), mid_color: QColor = QColor(255, 255, 0), 
                       high_color: QColor = QColor(255, 0, 0)) -> QColor:
    """
    Get a color for a value based on a gradient
    
    Args:
        value: The value to get the color for
        min_value: The minimum value of the range
        max_value: The maximum value of the range
        low_color: The color for the minimum value
        mid_color: The color for the middle value
        high_color: The color for the maximum value
        
    Returns:
        A QColor representing the value
    """
    # Normalize the value to the range [0, 1]
    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
    
    # Determine which gradient to use
    if normalized <= 0.5:
        # Interpolate between low and mid colors
        t = normalized * 2  # Scale to [0, 1]
        r = int(low_color.red() + t * (mid_color.red() - low_color.red()))
        g = int(low_color.green() + t * (mid_color.green() - low_color.green()))
        b = int(low_color.blue() + t * (mid_color.blue() - low_color.blue()))
    else:
        # Interpolate between mid and high colors
        t = (normalized - 0.5) * 2  # Scale to [0, 1]
        r = int(mid_color.red() + t * (high_color.red() - mid_color.red()))
        g = int(mid_color.green() + t * (high_color.green() - mid_color.green()))
        b = int(mid_color.blue() + t * (high_color.blue() - mid_color.blue()))
    
    return QColor(r, g, b)