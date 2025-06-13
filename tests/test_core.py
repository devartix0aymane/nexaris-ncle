#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for NEXARIS core components

This module contains tests for the core components of the NEXARIS Cognitive Load Estimator.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.data_manager import DataManager
from src.core.task_simulator import TaskSimulator
from src.core.behavior_tracker import BehaviorTracker
from src.core.facial_analyzer import FacialAnalyzer
from src.core.cognitive_load_calculator import CognitiveLoadCalculator
from src.utils.config_utils import load_config, get_default_config


class TestDataManager(unittest.TestCase):
    """Test cases for DataManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = get_default_config()
        self.config['data']['storage_path'] = self.temp_dir.name
        self.data_manager = DataManager(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_start_session(self):
        """Test starting a new session"""
        session_id = self.data_manager.start_session("Test Session")
        self.assertIsNotNone(session_id)
        self.assertTrue(self.data_manager.current_session_active)
        self.assertEqual(self.data_manager.current_session_id, session_id)
    
    def test_end_session(self):
        """Test ending a session"""
        session_id = self.data_manager.start_session("Test Session")
        self.data_manager.end_session()
        self.assertFalse(self.data_manager.current_session_active)
        self.assertIsNone(self.data_manager.current_session_id)
    
    def test_save_load_session(self):
        """Test saving and loading a session"""
        # Start a session and add some data
        session_id = self.data_manager.start_session("Test Session")
        self.data_manager.record_cognitive_load(0.5, 0.3, 0.4, 0.6, 0.2)
        self.data_manager.end_session()
        
        # Load the session
        session_data = self.data_manager.load_session(session_id)
        self.assertIsNotNone(session_data)
        self.assertEqual(session_data['name'], "Test Session")
        self.assertTrue('cognitive_load_data' in session_data)
        self.assertEqual(len(session_data['cognitive_load_data']), 1)
        self.assertEqual(session_data['cognitive_load_data'][0]['cognitive_load'], 0.5)


class TestTaskSimulator(unittest.TestCase):
    """Test cases for TaskSimulator"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = get_default_config()
        self.task_simulator = TaskSimulator(self.config)
    
    def test_load_question_set(self):
        """Test loading a question set"""
        # Create a mock question set
        question_set = {
            "name": "Test Questions",
            "description": "Test question set",
            "difficulty": "medium",
            "questions": [
                {
                    "id": "q1",
                    "text": "Test question 1",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "difficulty": "easy",
                    "time_limit": 10
                }
            ]
        }
        
        # Mock the file loading
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(question_set))):
            self.task_simulator.load_question_set("test")
        
        self.assertEqual(len(self.task_simulator.questions), 1)
        self.assertEqual(self.task_simulator.questions[0]['id'], "q1")
    
    def test_start_task(self):
        """Test starting a task"""
        # Create a mock question set
        self.task_simulator.questions = [
            {
                "id": "q1",
                "text": "Test question 1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "difficulty": "easy",
                "time_limit": 10
            }
        ]
        
        self.task_simulator.start_task("question_set", 60)
        self.assertTrue(self.task_simulator.task_active)
        self.assertEqual(self.task_simulator.task_type, "question_set")
        self.assertEqual(self.task_simulator.total_time, 60)
    
    def test_end_task(self):
        """Test ending a task"""
        # Start a task first
        self.task_simulator.questions = [
            {
                "id": "q1",
                "text": "Test question 1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "difficulty": "easy",
                "time_limit": 10
            }
        ]
        
        self.task_simulator.start_task("question_set", 60)
        self.task_simulator.end_task()
        self.assertFalse(self.task_simulator.task_active)


class TestBehaviorTracker(unittest.TestCase):
    """Test cases for BehaviorTracker"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = get_default_config()
        self.behavior_tracker = BehaviorTracker(self.config)
    
    def test_start_tracking(self):
        """Test starting behavior tracking"""
        self.behavior_tracker.start_tracking()
        self.assertTrue(self.behavior_tracker.tracking_active)
    
    def test_stop_tracking(self):
        """Test stopping behavior tracking"""
        self.behavior_tracker.start_tracking()
        self.behavior_tracker.stop_tracking()
        self.assertFalse(self.behavior_tracker.tracking_active)
    
    def test_record_mouse_movement(self):
        """Test recording mouse movement"""
        self.behavior_tracker.start_tracking()
        self.behavior_tracker.record_mouse_movement(100, 100)
        self.assertEqual(self.behavior_tracker.mouse_position, (100, 100))
    
    def test_record_click(self):
        """Test recording mouse clicks"""
        self.behavior_tracker.start_tracking()
        initial_clicks = self.behavior_tracker.click_count
        self.behavior_tracker.record_click(100, 100)
        self.assertEqual(self.behavior_tracker.click_count, initial_clicks + 1)


class TestCognitiveLoadCalculator(unittest.TestCase):
    """Test cases for CognitiveLoadCalculator"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = get_default_config()
        self.calculator = CognitiveLoadCalculator(self.config)
    
    def test_update_from_behavior(self):
        """Test updating cognitive load from behavior data"""
        initial_score = self.calculator.cognitive_load
        self.calculator.update_from_behavior(0.8, 10, 5, 2)
        self.assertNotEqual(self.calculator.cognitive_load, initial_score)
        self.assertNotEqual(self.calculator.behavior_score, 0)
    
    def test_update_from_facial(self):
        """Test updating cognitive load from facial data"""
        initial_score = self.calculator.cognitive_load
        self.calculator.update_from_facial(0.7, 0.3, 0.5)
        self.assertNotEqual(self.calculator.cognitive_load, initial_score)
        self.assertNotEqual(self.calculator.facial_score, 0)
    
    def test_update_from_performance(self):
        """Test updating cognitive load from performance data"""
        initial_score = self.calculator.cognitive_load
        self.calculator.update_from_performance(0.6, 2.5)
        self.assertNotEqual(self.calculator.cognitive_load, initial_score)
        self.assertNotEqual(self.calculator.performance_score, 0)
    
    def test_calculate_cognitive_load(self):
        """Test calculating cognitive load"""
        # Set component scores
        self.calculator.behavior_score = 0.7
        self.calculator.facial_score = 0.6
        self.calculator.performance_score = 0.8
        self.calculator.eeg_score = 0.5
        
        # Calculate cognitive load
        self.calculator.calculate_cognitive_load()
        
        # Check that the cognitive load is calculated correctly
        expected_load = (
            0.7 * self.calculator.weights['behavior'] +
            0.6 * self.calculator.weights['facial'] +
            0.8 * self.calculator.weights['performance'] +
            0.5 * self.calculator.weights['eeg']
        )
        self.assertAlmostEqual(self.calculator.cognitive_load, expected_load, places=4)


if __name__ == '__main__':
    unittest.main()