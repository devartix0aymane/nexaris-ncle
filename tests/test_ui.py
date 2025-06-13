#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for NEXARIS UI components

This module contains tests for the UI components of the NEXARIS Cognitive Load Estimator.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PyQt5 for UI testing
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Import UI components
from src.ui.dashboard_widget import DashboardWidget
from src.ui.task_widget import TaskWidget, QuestionWidget, AlertWidget
from src.ui.settings_widget import SettingsWidget
from src.ui.visualization_widget import VisualizationWidget

# Import core components for mocking
from src.core.data_manager import DataManager
from src.core.task_simulator import TaskSimulator
from src.core.behavior_tracker import BehaviorTracker
from src.core.facial_analyzer import FacialAnalyzer
from src.core.cognitive_load_calculator import CognitiveLoadCalculator
from src.core.eeg_integration import EEGIntegration

# Create QApplication instance for testing
app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestDashboardWidget(unittest.TestCase):
    """Test cases for DashboardWidget"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock objects for dependencies
        self.data_manager = MagicMock(spec=DataManager)
        self.task_simulator = MagicMock(spec=TaskSimulator)
        self.cognitive_load_calculator = MagicMock(spec=CognitiveLoadCalculator)
        self.cognitive_load_calculator.cognitive_load = 0.5
        self.cognitive_load_calculator.behavior_score = 0.4
        self.cognitive_load_calculator.facial_score = 0.6
        self.cognitive_load_calculator.performance_score = 0.5
        self.cognitive_load_calculator.eeg_score = 0.3
        
        # Create the widget
        self.widget = DashboardWidget(
            self.data_manager,
            self.task_simulator,
            self.cognitive_load_calculator
        )
    
    def test_update_cognitive_load_gauge(self):
        """Test updating the cognitive load gauge"""
        self.widget.update_cognitive_load_gauge(0.75)
        self.assertEqual(self.widget.gauge_value, 0.75)
    
    def test_update_metrics_panel(self):
        """Test updating the metrics panel"""
        self.widget.update_metrics_panel(0.6, 0.7, 0.5, 0.4)
        self.assertEqual(self.widget.behavior_value, 0.6)
        self.assertEqual(self.widget.facial_value, 0.7)
        self.assertEqual(self.widget.performance_value, 0.5)
        self.assertEqual(self.widget.eeg_value, 0.4)
    
    def test_new_session_button(self):
        """Test new session button click"""
        # Mock the data manager's start_session method
        self.data_manager.start_session.return_value = "test_session_id"
        
        # Simulate button click
        QTest.mouseClick(self.widget.new_session_button, Qt.LeftButton)
        
        # Check that start_session was called
        self.data_manager.start_session.assert_called_once()
    
    def test_export_data_button(self):
        """Test export data button click"""
        # Mock the data manager's export_session method
        self.data_manager.export_session.return_value = True
        
        # Simulate button click
        QTest.mouseClick(self.widget.export_button, Qt.LeftButton)
        
        # Check that export_session was called
        self.data_manager.export_session.assert_called_once()


class TestTaskWidget(unittest.TestCase):
    """Test cases for TaskWidget"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock objects for dependencies
        self.task_simulator = MagicMock(spec=TaskSimulator)
        self.behavior_tracker = MagicMock(spec=BehaviorTracker)
        
        # Create the widget
        self.widget = TaskWidget(self.task_simulator, self.behavior_tracker)
    
    def test_start_task_button(self):
        """Test start task button click"""
        # Set up the task simulator mock
        self.task_simulator.task_active = False
        
        # Simulate button click
        QTest.mouseClick(self.widget.start_button, Qt.LeftButton)
        
        # Check that start_task was called
        self.task_simulator.start_task.assert_called_once()
    
    def test_stop_task_button(self):
        """Test stop task button click"""
        # Set up the task simulator mock
        self.task_simulator.task_active = True
        
        # Simulate button click
        QTest.mouseClick(self.widget.stop_button, Qt.LeftButton)
        
        # Check that end_task was called
        self.task_simulator.end_task.assert_called_once()


class TestQuestionWidget(unittest.TestCase):
    """Test cases for QuestionWidget"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock objects for dependencies
        self.task_simulator = MagicMock(spec=TaskSimulator)
        
        # Create the widget
        self.widget = QuestionWidget(self.task_simulator)
    
    def test_display_question(self):
        """Test displaying a question"""
        # Create a test question
        question = {
            "id": "q1",
            "text": "Test question",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "difficulty": "medium",
            "time_limit": 30
        }
        
        # Display the question
        self.widget.display_question(question)
        
        # Check that the question is displayed correctly
        self.assertEqual(self.widget.question_id, "q1")
        self.assertEqual(self.widget.question_label.text(), "Test question")
        self.assertEqual(len(self.widget.option_buttons), 4)
    
    def test_submit_answer(self):
        """Test submitting an answer"""
        # Set up a question first
        question = {
            "id": "q1",
            "text": "Test question",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "difficulty": "medium",
            "time_limit": 30
        }
        self.widget.display_question(question)
        
        # Select an answer
        self.widget.selected_option = "B"
        
        # Submit the answer
        self.widget.submit_answer()
        
        # Check that record_answer was called with the correct parameters
        self.task_simulator.record_answer.assert_called_once_with("q1", "B")


class TestSettingsWidget(unittest.TestCase):
    """Test cases for SettingsWidget"""
    
    def setUp(self):
        """Set up test environment"""
        # Create the widget
        self.widget = SettingsWidget()
    
    def test_load_settings(self):
        """Test loading settings"""
        # Create mock settings
        settings = {
            "ui": {
                "theme": "dark",
                "font_size": 12
            },
            "task": {
                "default_duration": 300,
                "default_type": "question_set"
            },
            "behavior": {
                "enabled": True,
                "mouse_tracking": True,
                "keyboard_tracking": True
            },
            "facial": {
                "enabled": True,
                "camera_index": 0,
                "detection_model": "haarcascade"
            },
            "scoring": {
                "method": "weighted_sum",
                "update_interval": 1.0,
                "smoothing_factor": 0.3
            },
            "advanced": {
                "ml_enabled": False,
                "eeg_enabled": False
            },
            "data": {
                "storage_path": "data/sessions",
                "format": "json",
                "compression": False
            }
        }
        
        # Load the settings
        with patch('src.utils.config_utils.load_config', return_value=settings):
            self.widget.load_settings()
        
        # Check that the settings are loaded correctly
        self.assertEqual(self.widget.get_settings()["ui"]["theme"], "dark")
        self.assertEqual(self.widget.get_settings()["task"]["default_duration"], 300)
        self.assertEqual(self.widget.get_settings()["behavior"]["enabled"], True)


class TestVisualizationWidget(unittest.TestCase):
    """Test cases for VisualizationWidget"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock objects for dependencies
        self.data_manager = MagicMock(spec=DataManager)
        
        # Create the widget
        self.widget = VisualizationWidget(self.data_manager)
    
    def test_session_selection(self):
        """Test session selection"""
        # Mock the data manager's get_session_list method
        self.data_manager.get_session_list.return_value = [
            {"id": "session1", "name": "Session 1", "date": "2023-01-01"},
            {"id": "session2", "name": "Session 2", "date": "2023-01-02"}
        ]
        
        # Update the session list
        self.widget.update_session_list()
        
        # Check that the session combo box is populated correctly
        self.assertEqual(self.widget.session_combo.count(), 2)
    
    def test_visualization_type_selection(self):
        """Test visualization type selection"""
        # Select a visualization type
        index = self.widget.visualization_type_combo.findText("Time Series")
        self.widget.visualization_type_combo.setCurrentIndex(index)
        
        # Check that the visualization type is set correctly
        self.assertEqual(self.widget.visualization_type_combo.currentText(), "Time Series")


if __name__ == '__main__':
    unittest.main()