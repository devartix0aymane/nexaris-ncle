#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEXARIS Cognitive Load Estimator (NCLE)
Main application entry point

This module initializes and runs the NCLE application, setting up the UI,
core components, and system configuration.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

# Import core application components
from src.ui.main_window import MainWindow
from src.core.data_manager import DataManager
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging

# PyQt imports
from PyQt5.QtWidgets import QApplication


def setup_environment():
    """Set up the application environment, directories, and configuration."""
    # Create necessary directories if they don't exist
    dirs = ['data', 'config', 'logs']
    for directory in dirs:
        os.makedirs(Path(__file__).resolve().parent / directory, exist_ok=True)
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = load_config()
    
    return config


def main():
    """Main application entry point."""
    # Set up environment and configuration
    config = setup_environment()
    
    # Initialize the Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("NEXARIS Cognitive Load Estimator")
    app.setOrganizationName("NEXARIS")
    
    # Initialize the data manager
    data_manager = DataManager(config)
    
    # Create and show the main window
    main_window = MainWindow(config, data_manager)
    main_window.show()
    
    # Start the application event loop
    logging.info("NCLE application started")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()