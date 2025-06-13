#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for NEXARIS utility functions

This module contains tests for the utility functions of the NEXARIS Cognitive Load Estimator.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from src.utils.config_utils import load_config, save_config, get_default_config
from src.utils.ui_utils import apply_theme, create_gauge, create_chart
from src.utils.ml_utils import load_model, save_model, predict


class TestConfigUtils(unittest.TestCase):
    """Test cases for configuration utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.json")
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        config = get_default_config()
        self.assertIsNotNone(config)
        self.assertTrue("app" in config)
        self.assertTrue("ui" in config)
        self.assertTrue("task" in config)
        self.assertTrue("behavior" in config)
        self.assertTrue("facial" in config)
        self.assertTrue("scoring" in config)
        self.assertTrue("advanced" in config)
        self.assertTrue("data" in config)
    
    def test_save_load_config(self):
        """Test saving and loading configuration"""
        # Create a test configuration
        test_config = {
            "app": {
                "name": "NEXARIS Test",
                "version": "0.1.0"
            },
            "ui": {
                "theme": "dark",
                "font_size": 12
            }
        }
        
        # Save the configuration
        save_config(test_config, self.config_path)
        
        # Load the configuration
        loaded_config = load_config(self.config_path)
        
        # Check that the loaded configuration matches the saved configuration
        self.assertEqual(loaded_config["app"]["name"], test_config["app"]["name"])
        self.assertEqual(loaded_config["ui"]["theme"], test_config["ui"]["theme"])
    
    def test_load_nonexistent_config(self):
        """Test loading a nonexistent configuration file"""
        # Try to load a nonexistent configuration file
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent.json")


class TestUIUtils(unittest.TestCase):
    """Test cases for UI utilities"""
    
    def test_apply_theme(self):
        """Test applying a theme"""
        # Create a mock widget
        widget = MagicMock()
        
        # Apply a theme
        with patch('PyQt5.QtWidgets.QApplication.setStyle') as mock_set_style:
            apply_theme(widget, "dark")
            mock_set_style.assert_called_once()
    
    def test_create_gauge(self):
        """Test creating a gauge widget"""
        # Create a gauge
        gauge = create_gauge("Test Gauge", 0, 100, 50)
        
        # Check that the gauge is created correctly
        self.assertIsNotNone(gauge)
        self.assertEqual(gauge.minimum(), 0)
        self.assertEqual(gauge.maximum(), 100)
        self.assertEqual(gauge.value(), 50)
    
    def test_create_chart(self):
        """Test creating a chart widget"""
        # Create a chart
        chart = create_chart("Test Chart", "X Axis", "Y Axis")
        
        # Check that the chart is created correctly
        self.assertIsNotNone(chart)
        self.assertEqual(chart.title().text(), "Test Chart")
        self.assertEqual(chart.axisX().titleText(), "X Axis")
        self.assertEqual(chart.axisY().titleText(), "Y Axis")


class TestMLUtils(unittest.TestCase):
    """Test cases for machine learning utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "model.pkl")
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_save_load_model(self):
        """Test saving and loading a model"""
        # Create a mock model
        model = MagicMock()
        model.predict.return_value = [0.5]
        
        # Save the model
        with patch('pickle.dump') as mock_dump:
            save_model(model, self.model_path)
            mock_dump.assert_called_once()
        
        # Load the model
        with patch('pickle.load', return_value=model) as mock_load:
            loaded_model = load_model(self.model_path)
            mock_load.assert_called_once()
        
        # Check that the loaded model works correctly
        self.assertEqual(loaded_model.predict([1, 2, 3, 4]), [0.5])
    
    def test_predict(self):
        """Test making predictions with a model"""
        # Create a mock model
        model = MagicMock()
        model.predict.return_value = [0.7]
        
        # Make a prediction
        result = predict(model, [1, 2, 3, 4])
        
        # Check that the prediction is correct
        self.assertEqual(result, 0.7)
        model.predict.assert_called_once()


if __name__ == '__main__':
    unittest.main()