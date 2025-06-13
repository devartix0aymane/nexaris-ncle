#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User Interface Package for NEXARIS Cognitive Load Estimator

This package contains all UI components for the application.
"""

__all__ = [
    'main_window',
    'dashboard_widget',
    'task_widget',
    'settings_widget',
    'visualization_widget'
]

from .main_window import MainWindow
from .dashboard_widget import DashboardWidget
from .task_widget import TaskWidget, QuestionWidget, AlertWidget
from .settings_widget import SettingsWidget
from .visualization_widget import VisualizationWidget