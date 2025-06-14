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
import csv
import os

# PyQt imports for event handling
from PyQt5.QtCore import QObject, pyqtSignal, QPoint, Qt, QEvent

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
        self.idle_time_threshold = 5.0 # seconds for idle time

        # CSV logging setup
        self.csv_writer = None
        self.csv_file = None
        self.csv_file_path = os.path.join(self.config.get('data_dir', 'data'), 'behavior_log.csv')
        os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
        
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
        self.total_idle_time = 0.0
        self.last_click_time = None
        
        # Last known position
        self.last_mouse_pos = None
        self.last_mouse_time = None
    
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

        # Initialize CSV logging
        try:
            self.csv_file = open(self.csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Write header
            self.csv_writer.writerow([
                'timestamp', 'event_type', 'x', 'y', 'button', 'key_code',
                'duration', 'mouse_speed', 'click_delay', 'idle_time'
            ])
        except IOError as e:
            self.logger.error(f"Failed to open CSV file for writing: {e}")
            self.csv_writer = None # Ensure writer is None if file opening failed
        
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

        # Close CSV file
        if self.csv_file:
            try:
                self.csv_file.close()
                self.logger.info(f"Behavior log saved to {self.csv_file_path}")
            except IOError as e:
                self.logger.error(f"Failed to close CSV file: {e}")
            self.csv_file = None
            self.csv_writer = None
        
        self.logger.info("Behavior tracking stopped")
        
        # Return final metrics
        return self.get_metrics()
    
    def _tracking_loop(self) -> None:
        """
        Background thread for periodic behavior checks
        """
        while self.is_tracking:
            current_time = time.time()
            # Check for hesitation (inactivity)
            self._check_hesitation(current_time)
            # Check for idle time
            self._check_idle_time(current_time)
            
            # Sleep according to sampling rate
            time.sleep(1.0 / self.sampling_rate)
    
    def _check_hesitation(self, current_time: float) -> None:
        """
        Check for user hesitation (inactivity between specific actions like clicks or key presses)
        Args:
            current_time: The current time from the tracking loop.
        """
        if not self.last_activity_time:
            return
        
        inactive_time = current_time - self.last_activity_time
        
        # If inactive for longer than threshold, record hesitation
        # This is a general inactivity check, might be refined based on task context
        if inactive_time > self.hesitation_threshold:
            # We only log hesitation if it's not already part of a longer idle period
            if inactive_time < self.idle_time_threshold:
                self.hesitation_detected.emit(inactive_time)
                self._handle_hesitation(inactive_time, current_time)
            
            # Reset last activity time to avoid multiple hesitation events for the same period
            # but only if it's not a longer idle period, which is handled by _check_idle_time
            if inactive_time < self.idle_time_threshold:
                 self.last_activity_time = current_time

    def _check_idle_time(self, current_time: float) -> None:
        """
        Check for user idle time (longer periods of inactivity).
        Args:
            current_time: The current time from the tracking loop.
        """
        if not self.last_activity_time:
            return

        idle_duration = current_time - self.last_activity_time
        if idle_duration > self.idle_time_threshold:
            self.total_idle_time += (1.0 / self.sampling_rate) # Add the sampling interval to idle time
            # Log idle event to CSV if needed, or just accumulate total_idle_time
            if self.csv_writer:
                try:
                    self.csv_writer.writerow([
                        datetime.now().isoformat(), 'idle', '', '', '', '',
                        round(1.0 / self.sampling_rate, 3), '', '', round(idle_duration, 3)
                    ])
                except Exception as e:
                    self.logger.error(f"Error writing idle event to CSV: {e}")
            # No specific signal for continuous idle, but could be added if needed
            # self.last_activity_time is NOT reset here, so idle_duration continues to grow
            # until an activity occurs.
    
    def _handle_mouse_move(self, pos: QPoint) -> None:
        """
        Handle mouse movement event
        
        Args:
            pos: New mouse position
        """
        if not self.is_tracking or not self.mouse_tracking:
            return
        
        current_event_time = time.time()
        timestamp = datetime.now().isoformat()
        self.last_activity_time = current_event_time
        
        mouse_speed = 0.0
        # Calculate distance and speed if we have a previous position and time
        if self.last_mouse_pos is not None and self.last_mouse_time is not None:
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()
            distance = np.sqrt(dx*dx + dy*dy)
            self.total_mouse_distance += distance
            time_delta = current_event_time - self.last_mouse_time
            if time_delta > 0:
                mouse_speed = distance / time_delta # pixels per second
        
        # Update last position and time
        self.last_mouse_pos = pos
        self.last_mouse_time = current_event_time
        
        # Record data if enabled
        if self.save_raw_data:
            self.mouse_positions.append({
                'timestamp': timestamp,
                'x': pos.x(),
                'y': pos.y(),
                'speed': mouse_speed
            })
        
        # Log to CSV
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    timestamp, 'mouse_move', pos.x(), pos.y(), '', '',
                    '', round(mouse_speed, 2), '', ''
                ])
            except Exception as e:
                self.logger.error(f"Error writing mouse_move to CSV: {e}")

        # Record activity timestamp
        self.activity_timestamps.append(timestamp)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'mouse_move',
            'timestamp': timestamp,
            'position': {'x': pos.x(), 'y': pos.y()},
            'speed': mouse_speed,
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
        
        current_event_time = time.time()
        timestamp = datetime.now().isoformat()
        self.last_activity_time = current_event_time
        
        click_delay = 0.0
        if self.last_click_time is not None:
            click_delay = current_event_time - self.last_click_time
        self.last_click_time = current_event_time

        # Update metrics
        self.click_count += 1
        
        # Record data if enabled
        if self.save_raw_data:
            self.mouse_clicks.append({
                'timestamp': timestamp,
                'x': pos.x(),
                'y': pos.y(),
                'button': button,
                'delay_from_last_click': click_delay
            })
        
        # Log to CSV
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    timestamp, 'mouse_click', pos.x(), pos.y(), button, '',
                    '', '', round(click_delay, 3), ''
                ])
            except Exception as e:
                self.logger.error(f"Error writing mouse_click to CSV: {e}")

        # Record activity timestamp
        self.activity_timestamps.append(timestamp)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'mouse_click',
            'timestamp': timestamp,
            'position': {'x': pos.x(), 'y': pos.y()},
            'button': button,
            'click_delay': click_delay,
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
        
        current_event_time = time.time()
        timestamp = datetime.now().isoformat()
        self.last_activity_time = current_event_time
        
        # Update metrics
        self.keypress_count += 1
        
        # Record data if enabled
        if self.save_raw_data:
            self.key_presses.append({
                'timestamp': timestamp,
                'key_code': key_code
            })
        
        # Log to CSV
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    timestamp, 'key_press', '', '', '', key_code,
                    '', '', '', ''
                ])
            except Exception as e:
                self.logger.error(f"Error writing key_press to CSV: {e}")

        # Record activity timestamp
        self.activity_timestamps.append(timestamp)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'key_press',
            'timestamp': timestamp,
            'key_code': key_code,
            'keypress_count': self.keypress_count
        })
    
    def _handle_hesitation(self, duration: float, event_time: float) -> None:
        """
        Handle detected hesitation
        
        Args:
            duration: Hesitation duration in seconds
            event_time: The time the hesitation was detected by the loop
        """
        if not self.is_tracking:
            return
        
        # Record timestamp (use event_time for more accuracy with the loop)
        timestamp = datetime.fromtimestamp(event_time).isoformat()
        # self.last_activity_time is updated by _check_hesitation if this is not an idle period
        
        # Update metrics
        self.hesitation_count += 1
        self.total_hesitation_time += duration
        
        # Record data if enabled
        if self.save_raw_data:
            self.hesitations.append({
                'timestamp': timestamp,
                'duration': duration
            })
        
        # Log to CSV
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    timestamp, 'hesitation', '', '', '', '',
                    round(duration, 3), '', '', ''
                ])
            except Exception as e:
                self.logger.error(f"Error writing hesitation to CSV: {e}")

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
            'avg_hesitation_time': self.total_hesitation_time / max(1, self.hesitation_count),
            'total_idle_time': self.total_idle_time
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
                if event.type() == QEvent.MouseMove and self.tracker.mouse_tracking:
                    self.tracker.mouse_moved.emit(event.pos())
                
                # Track mouse clicks
                elif event.type() == QEvent.MouseButtonPress and self.tracker.mouse_tracking:
                    self.tracker.mouse_clicked.emit(event.pos(), event.button())
                
                # Track key presses
                elif event.type() == QEvent.KeyPress and self.tracker.keyboard_tracking:
                    self.tracker.key_pressed.emit(event.key())
                
                # Pass event to original handler
                return False
        
        # Create and install event filter
        event_filter = BehaviorEventFilter(self)
        widget.installEventFilter(event_filter)
        
        # Store reference to prevent garbage collection
        self._event_filter = event_filter
        
        self.logger.debug(f"Installed event filter on widget {widget}")