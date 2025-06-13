#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Behavior Tracker for NEXARIS Cognitive Load Estimator

This module tracks user behavior during tasks, including mouse movements,
clicks, keyboard input, and hesitation patterns.
"""

import time
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports for event handling
from PyQt5.QtCore import QObject, pyqtSignal, QPoint, Qt

# Import utilities
from ..utils.logging_utils import get_logger


class BehaviorTracker(QObject):
    """
    Tracks user behavior during cognitive tasks
    """
    # Define signals for behavior events
    mouse_moved = pyqtSignal(QPoint)
    mouse_clicked = pyqtSignal(QPoint, int)  # Position, button
    key_pressed = pyqtSignal(int)  # Key code
    hesitation_detected = pyqtSignal(float)  # Duration in seconds
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the behavior tracker
        
        Args:
            config: Application configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get tracking configuration
        self.tracking_config = config.get('tracking', {})
        self.mouse_tracking = self.tracking_config.get('mouse_tracking', True)
        self.keyboard_tracking = self.tracking_config.get('keyboard_tracking', True)
        self.sampling_rate = self.tracking_config.get('sampling_rate', 10)  # Hz
        self.save_raw_data = self.tracking_config.get('save_raw_data', True)
        
        # Initialize tracking data
        self.reset_tracking_data()
        
        # Set up tracking state
        self.is_tracking = False
        self.tracking_thread = None
        self.last_activity_time = None
        self.hesitation_threshold = 2.0  # seconds
        
        # Callbacks
        self.data_callbacks = []
        
        self.logger.info("Behavior Tracker initialized")
    
    def reset_tracking_data(self) -> None:
        """
        Reset all tracking data
        """
        self.mouse_positions = []
        self.mouse_clicks = []
        self.key_presses = []
        self.hesitations = []
        self.activity_timestamps = []
        
        # Metrics
        self.total_mouse_distance = 0.0
        self.click_count = 0
        self.keypress_count = 0
        self.hesitation_count = 0
        self.total_hesitation_time = 0.0
        
        # Last known position
        self.last_mouse_pos = None
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to receive behavior data
        
        Args:
            callback: Function that takes a behavior data dictionary as argument
        """
        self.data_callbacks.append(callback)
        self.logger.debug("Registered data callback")
    
    def _notify_data_callbacks(self, data: Dict[str, Any]) -> None:
        """
        Notify all registered callbacks with behavior data
        
        Args:
            data: Behavior data dictionary
        """
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in behavior data callback: {e}")
    
    def start_tracking(self) -> None:
        """
        Start tracking user behavior
        """
        if self.is_tracking:
            self.logger.warning("Behavior tracking is already active")
            return
        
        # Reset tracking data
        self.reset_tracking_data()
        
        # Set up tracking state
        self.is_tracking = True
        self.last_activity_time = time.time()
        
        # Connect signals to handlers
        self.mouse_moved.connect(self._handle_mouse_move)
        self.mouse_clicked.connect(self._handle_mouse_click)
        self.key_pressed.connect(self._handle_key_press)
        
        # Start tracking thread for periodic checks
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        self.logger.info("Behavior tracking started")
    
    def stop_tracking(self) -> Dict[str, Any]:
        """
        Stop tracking user behavior and return metrics
        
        Returns:
            Dictionary of behavior metrics
        """
        if not self.is_tracking:
            self.logger.warning("Behavior tracking is not active")
            return self.get_metrics()
        
        # Stop tracking
        self.is_tracking = False
        
        # Disconnect signals
        self.mouse_moved.disconnect(self._handle_mouse_move)
        self.mouse_clicked.disconnect(self._handle_mouse_click)
        self.key_pressed.disconnect(self._handle_key_press)
        
        # Wait for tracking thread to finish
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
        
        self.logger.info("Behavior tracking stopped")
        
        # Return final metrics
        return self.get_metrics()
    
    def _tracking_loop(self) -> None:
        """
        Background thread for periodic behavior checks
        """
        while self.is_tracking:
            # Check for hesitation (inactivity)
            self._check_hesitation()
            
            # Sleep according to sampling rate
            time.sleep(1.0 / self.sampling_rate)
    
    def _check_hesitation(self) -> None:
        """
        Check for user hesitation (inactivity)
        """
        if not self.last_activity_time:
            return
        
        current_time = time.time()
        inactive_time = current_time - self.last_activity_time
        
        # If inactive for longer than threshold, record hesitation
        if inactive_time > self.hesitation_threshold:
            self.hesitation_detected.emit(inactive_time)
            self._handle_hesitation(inactive_time)
            
            # Reset last activity time to avoid multiple hesitation events
            self.last_activity_time = current_time
    
    def _handle_mouse_move(self, pos: QPoint) -> None:
        """
        Handle mouse movement event
        
        Args:
            pos: New mouse position
        """
        if not self.is_tracking or not self.mouse_tracking:
            return
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        self.last_activity_time = time.time()
        
        # Calculate distance if we have a previous position
        if self.last_mouse_pos is not None:
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()
            distance = np.sqrt(dx*dx + dy*dy)
            self.total_mouse_distance += distance
        
        # Update last position
        self.last_mouse_pos = pos
        
        # Record data if enabled
        if self.save_raw_data:
            self.mouse_positions.append({
                'timestamp': timestamp,
                'x': pos.x(),
                'y': pos.y()
            })
        
        # Record activity timestamp
        self.activity_timestamps.append(timestamp)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'mouse_move',
            'timestamp': timestamp,
            'position': {'x': pos.x(), 'y': pos.y()},
            'total_distance': self.total_mouse_distance
        })
    
    def _handle_mouse_click(self, pos: QPoint, button: int) -> None:
        """
        Handle mouse click event
        
        Args:
            pos: Mouse position
            button: Mouse button (Qt.LeftButton, Qt.RightButton, etc.)
        """
        if not self.is_tracking or not self.mouse_tracking:
            return
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        self.last_activity_time = time.time()
        
        # Update metrics
        self.click_count += 1
        
        # Record data if enabled
        if self.save_raw_data:
            self.mouse_clicks.append({
                'timestamp': timestamp,
                'x': pos.x(),
                'y': pos.y(),
                'button': button
            })
        
        # Record activity timestamp
        self.activity_timestamps.append(timestamp)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'mouse_click',
            'timestamp': timestamp,
            'position': {'x': pos.x(), 'y': pos.y()},
            'button': button,
            'click_count': self.click_count
        })
    
    def _handle_key_press(self, key_code: int) -> None:
        """
        Handle key press event
        
        Args:
            key_code: Key code
        """
        if not self.is_tracking or not self.keyboard_tracking:
            return
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        self.last_activity_time = time.time()
        
        # Update metrics
        self.keypress_count += 1
        
        # Record data if enabled
        if self.save_raw_data:
            self.key_presses.append({
                'timestamp': timestamp,
                'key_code': key_code
            })
        
        # Record activity timestamp
        self.activity_timestamps.append(timestamp)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'key_press',
            'timestamp': timestamp,
            'key_code': key_code,
            'keypress_count': self.keypress_count
        })
    
    def _handle_hesitation(self, duration: float) -> None:
        """
        Handle detected hesitation
        
        Args:
            duration: Hesitation duration in seconds
        """
        if not self.is_tracking:
            return
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        
        # Update metrics
        self.hesitation_count += 1
        self.total_hesitation_time += duration
        
        # Record data if enabled
        if self.save_raw_data:
            self.hesitations.append({
                'timestamp': timestamp,
                'duration': duration
            })
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'hesitation',
            'timestamp': timestamp,
            'duration': duration,
            'hesitation_count': self.hesitation_count,
            'total_hesitation_time': self.total_hesitation_time
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current behavior metrics
        
        Returns:
            Dictionary of behavior metrics
        """
        # Calculate additional metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'mouse_distance': self.total_mouse_distance,
            'click_count': self.click_count,
            'keypress_count': self.keypress_count,
            'hesitation_count': self.hesitation_count,
            'total_hesitation_time': self.total_hesitation_time,
            'avg_hesitation_time': self.total_hesitation_time / max(1, self.hesitation_count)
        }
        
        # Calculate click rate (clicks per minute)
        if self.activity_timestamps and len(self.activity_timestamps) >= 2:
            start_time = datetime.fromisoformat(self.activity_timestamps[0])
            end_time = datetime.fromisoformat(self.activity_timestamps[-1])
            duration_minutes = (end_time - start_time).total_seconds() / 60.0
            if duration_minutes > 0:
                metrics['click_rate'] = self.click_count / duration_minutes
                metrics['keypress_rate'] = self.keypress_count / duration_minutes
        
        return metrics
    
    def get_raw_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get raw tracking data
        
        Returns:
            Dictionary of raw tracking data lists
        """
        return {
            'mouse_positions': self.mouse_positions,
            'mouse_clicks': self.mouse_clicks,
            'key_presses': self.key_presses,
            'hesitations': self.hesitations
        }
    
    def install_event_filter(self, widget) -> None:
        """
        Install an event filter on a widget to track events
        
        Args:
            widget: QWidget to track events for
        """
        class BehaviorEventFilter(QObject):
            def __init__(self, tracker):
                super().__init__()
                self.tracker = tracker
            
            def eventFilter(self, obj, event):
                # Track mouse movement
                if event.type() == Qt.MouseMove and self.tracker.mouse_tracking:
                    self.tracker.mouse_moved.emit(event.pos())
                
                # Track mouse clicks
                elif event.type() == Qt.MouseButtonPress and self.tracker.mouse_tracking:
                    self.tracker.mouse_clicked.emit(event.pos(), event.button())
                
                # Track key presses
                elif event.type() == Qt.KeyPress and self.tracker.keyboard_tracking:
                    self.tracker.key_pressed.emit(event.key())
                
                # Pass event to original handler
                return False
        
        # Create and install event filter
        event_filter = BehaviorEventFilter(self)
        widget.installEventFilter(event_filter)
        
        # Store reference to prevent garbage collection
        self._event_filter = event_filter
        
        self.logger.debug(f"Installed event filter on widget {widget}")