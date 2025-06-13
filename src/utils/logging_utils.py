#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging utilities for NEXARIS Cognitive Load Estimator

This module provides functions for setting up and managing application logging.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default log directory
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / 'logs'


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up application logging with file and console handlers
    
    Args:
        log_dir: Directory to store log files (optional)
        level: Logging level (default: INFO)
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    else:
        log_dir = Path(log_dir)
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ncle_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:  
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and optional level
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Optional logging level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    return logger


def set_log_level(level: int) -> None:
    """
    Set the log level for all handlers
    
    Args:
        level: New logging level
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.setLevel(level)
    
    logging.info(f"Log level set to {logging.getLevelName(level)}")


def create_session_log(session_id: str) -> logging.Logger:
    """
    Create a session-specific logger for tracking a user session
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Session-specific logger
    """
    # Create session log directory
    session_log_dir = DEFAULT_LOG_DIR / 'sessions'
    session_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create session log file
    log_file = session_log_dir / f"session_{session_id}.log"
    
    # Create session logger
    logger = logging.getLogger(f"session.{session_id}")
    logger.setLevel(logging.INFO)
    
    # Create handler and formatter
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    logger.info(f"Session {session_id} started")
    return logger