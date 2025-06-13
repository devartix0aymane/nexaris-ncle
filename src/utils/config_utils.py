#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration utilities for NEXARIS Cognitive Load Estimator

This module provides functions for loading, saving, and managing
application configuration settings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration path
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'config.json'

# Default configuration settings
DEFAULT_CONFIG = {
    "application": {
        "name": "NEXARIS Cognitive Load Estimator",
        "version": "0.1.0",
        "debug": False
    },
    "ui": {
        "theme": "dark",
        "window_width": 1200,
        "window_height": 800,
        "refresh_rate": 30  # Hz
    },
    "task": {
        "default_duration": 300,  # seconds
        "difficulty_levels": ["easy", "medium", "hard"],
        "default_difficulty": "medium",
        "question_sets": ["general", "cybersecurity", "alert_triage"],
        "default_question_set": "cybersecurity"
    },
    "tracking": {
        "mouse_tracking": True,
        "keyboard_tracking": True,
        "facial_analysis": False,  # Disabled by default
        "sampling_rate": 10,  # Hz
        "save_raw_data": True
    },
    "facial_analysis": {
        "enabled": False,
        "camera_index": 0,
        "detection_interval": 500,  # ms
        "emotions_to_track": ["neutral", "happy", "sad", "angry", "frustrated", "confused", "focused"]
    },
    "scoring": {
        "algorithm": "standard",  # Options: standard, advanced, ml
        "weights": {
            "response_time": 0.3,
            "mouse_movement": 0.2,
            "click_frequency": 0.15,
            "hesitation": 0.25,
            "facial_emotion": 0.1
        },
        "normalization": "z-score"  # Options: z-score, min-max, none
    },
    "advanced": {
        "ml_model_path": "",  # Path to custom ML model if used
        "eeg_enabled": False,
        "eeg_device": "",
        "eeg_sampling_rate": 250  # Hz
    },
    "data": {
        "storage_path": str(Path(__file__).resolve().parents[2] / 'data'),
        "backup_interval": 300,  # seconds
        "max_session_history": 50
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or create default if not exists
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Dict containing configuration settings
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If config file doesn't exist, create it with defaults
    if not config_path.exists():
        logging.info(f"Configuration file not found at {config_path}. Creating default configuration.")
        save_config(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG
    
    # Load existing configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logging.info(f"Configuration loaded from {config_path}")
            
            # Update with any missing default values
            updated_config = update_missing_config(config, DEFAULT_CONFIG)
            if updated_config != config:
                logging.info("Updated configuration with new default values")
                save_config(updated_config, config_path)
                
            return updated_config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        logging.warning("Falling back to default configuration")
        return DEFAULT_CONFIG


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save configuration file (optional)
        
    Returns:
        True if successful, False otherwise
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")
        return False


def update_missing_config(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update configuration with missing default values
    
    Args:
        config: Current configuration dictionary
        default_config: Default configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    result = config.copy()
    
    for key, value in default_config.items():
        if key not in result:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = update_missing_config(result[key], value)
    
    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using a dot-separated path
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the configuration value (e.g., 'ui.theme')
        default: Default value to return if key not found
        
    Returns:
        Configuration value or default if not found
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default