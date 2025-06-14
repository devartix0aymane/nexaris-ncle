#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Widget for NEXARIS Cognitive Load Estimator

This module provides the camera widget for displaying video feed
and facial analysis results.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QGroupBox, QFormLayout, QSizePolicy,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont

# Import utilities
from ..utils.logging_utils import get_logger
from datetime import datetime # Added for timestamp


class CameraWidget(QWidget):
    """
    Widget for displaying camera feed and facial analysis
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.camera_active = False
        self.camera_id = 0
        self.facial_analysis_enabled = True
        
        # Initialize camera
        self.cap = None
        
        # Set up UI
        self.init_ui()
        
    def init_ui(self):
        """
        Initialize the user interface
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Camera feed display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        self.camera_label.setText("Camera feed not available")
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Camera selection
        camera_group = QGroupBox("Camera Settings")
        camera_form = QFormLayout(camera_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Default Camera", 0)
        for i in range(1, 5):  # Add some potential camera options
            self.camera_combo.addItem(f"Camera {i}", i)
        camera_form.addRow("Select Camera:", self.camera_combo)
        
        # Camera control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        camera_form.addRow("", button_layout)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.facial_checkbox = QCheckBox("Enable Facial Analysis")
        self.facial_checkbox.setChecked(True)
        analysis_layout.addWidget(self.facial_checkbox)
        
        # Add widgets to controls layout
        controls_layout.addWidget(camera_group)
        controls_layout.addWidget(analysis_group)
        
        # Add all components to main layout
        main_layout.addWidget(self.camera_label)
        main_layout.addLayout(controls_layout)
        
        # Connect signals
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.camera_combo.currentIndexChanged.connect(self.select_camera)
        self.facial_checkbox.toggled.connect(self.toggle_facial_analysis)
        
        # Labels for displaying facial analysis results
        self.emotion_label = QLabel("Emotion: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        self.timestamp_label = QLabel("Timestamp: N/A")
        font = QFont()
        font.setPointSize(10)
        self.emotion_label.setFont(font)
        self.confidence_label.setFont(font)
        self.timestamp_label.setFont(font)

        analysis_results_layout = QVBoxLayout()
        analysis_results_layout.addWidget(self.emotion_label)
        analysis_results_layout.addWidget(self.confidence_label)
        analysis_results_layout.addWidget(self.timestamp_label)
        analysis_group.setLayout(analysis_results_layout) # Add to analysis_group

        # Timer for updating camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def start_camera(self):
        """
        Start the camera feed and facial analysis if enabled.
        Implements webcam fallback if the initial camera fails.
        """
        try:
            if hasattr(self, 'facial_analyzer') and self.facial_analyzer is not None and self.facial_analysis_enabled:
                # FacialAnalyzer will handle its own camera capture if it's used
                if not self.facial_analyzer.is_analyzing:
                    success = self.facial_analyzer.start_analysis() # This now uses facial_analyzer's camera_index
                    if not success:
                        QMessageBox.warning(self, "Analysis Error", "Failed to start facial analysis.")
                        return
                    self.logger.info("Facial analysis started via CameraWidget.")
                self.camera_active = True # Indicates widget's camera view is active
                # Timer is not strictly needed if FacialAnalyzer pushes frames, but can be a fallback or for UI updates
                # self.timer.start(30) # FacialAnalyzer's frame_processed signal will drive updates
            else:
                # Standard camera start if no facial analyzer or it's disabled
                camera_indices_to_try = [self.camera_id] + [i for i in range(5) if i != self.camera_id] # Prioritize selected, then try others
                cap_opened = False
                for cam_idx in camera_indices_to_try:
                    try:
                        self.cap = cv2.VideoCapture(cam_idx)
                        if self.cap and self.cap.isOpened():
                            self.camera_id = cam_idx # Update to the working camera_id
                            self.camera_combo.setCurrentIndex(self.camera_combo.findData(cam_idx))
                            self.logger.info(f"Successfully opened camera {self.camera_id}")
                            cap_opened = True
                            break
                        else:
                            if self.cap: # Release if opened but not valid
                                self.cap.release()
                            self.logger.warning(f"Failed to open camera {cam_idx}. Trying next.")
                    except Exception as e_cam:
                        self.logger.error(f"Exception trying to open camera {cam_idx}: {str(e_cam)}")
                        if self.cap: self.cap.release()
                
                if not cap_opened:
                    self.logger.error(f"Failed to open any camera after trying indices: {camera_indices_to_try}")
                    QMessageBox.warning(self, "Camera Error", "Failed to open any available camera.")
                    return
                self.camera_active = True
                self.timer.start(30)  # Update at 30ms intervals (approx. 33 fps)

            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            
            self.logger.info(f"Camera view started (Camera ID: {self.camera_id if not (hasattr(self, 'facial_analyzer') and self.facial_analyzer and self.facial_analysis_enabled) else self.facial_analyzer.camera_index})")
        except Exception as e:
            self.logger.error(f"Error starting camera/analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Could not start camera/analysis: {e}")
            
    def stop_camera(self):
        """
        Stop the camera feed and facial analysis if active.
        Ensures webcam is released properly.
        """
        if self.camera_active:
            self.timer.stop()
            if hasattr(self, 'facial_analyzer') and self.facial_analyzer is not None and self.facial_analyzer.is_analyzing:
                self.facial_analyzer.stop_analysis()
                self.logger.info("Facial analysis stopped via CameraWidget.")
            
            if self.cap is not None:
                try:
                    self.cap.release()
                    self.logger.info("Camera released successfully.")
                except Exception as e_release:
                    self.logger.error(f"Error releasing camera: {str(e_release)}")
                finally:
                    self.cap = None
            
            self.camera_active = False
            
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.camera_combo.setEnabled(True)
            self.camera_label.setText("Camera feed not available")
            self.emotion_label.setText("Emotion: N/A")
            self.confidence_label.setText("Confidence: N/A")
            self.timestamp_label.setText("Timestamp: N/A")
            
            self.logger.info("Camera view stopped")
            
    def select_camera(self, index):
        """
        Select a different camera
        """
        self.camera_id = self.camera_combo.itemData(index)
        self.logger.info(f"Selected camera {self.camera_id}")
        
    def toggle_facial_analysis(self, enabled):
        """
        Toggle facial analysis on/off
        """
        self.facial_analysis_enabled = enabled
        self.logger.info(f"Facial analysis {'enabled' if enabled else 'disabled'}")
    
    def set_components(self, facial_analyzer) -> None:
        """
        Set core components
        
        Args:
            facial_analyzer: Facial analyzer instance
        """
        self.facial_analyzer = facial_analyzer
        if self.facial_analyzer:
            self.facial_analyzer.emotion_detected.connect(self.update_emotion_display)
            self.facial_analyzer.frame_processed.connect(self.display_processed_frame)
            # Assuming FacialAnalyzer will emit a signal with timestamp, or we get it from frame_processed
            # For now, we'll update timestamp when a frame is processed.
            self.logger.info("Connected FacialAnalyzer signals to CameraWidget slots.")
        
    def update_frame(self):
        """
        Update the camera frame
        """
        if self.cap is None or not self.camera_active:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            return
            
        # If facial analysis is enabled and analyzer is set, it will process and emit frame_processed
        # So, we don't call self.process_frame(frame) directly here anymore if facial_analyzer is active.
        # The FacialAnalyzer's _analysis_loop will call _process_frame which emits frame_processed.

        if not (self.facial_analysis_enabled and hasattr(self, 'facial_analyzer') and self.facial_analyzer is not None and self.facial_analyzer.is_analyzing):
            # If facial analysis is not active, display raw frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(pixmap)
            self.timestamp_label.setText(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        # else: The display_processed_frame slot will handle displaying the frame from FacialAnalyzer
        
        # Convert frame to QPixmap and display (This part is moved to display_processed_frame or handled above)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(pixmap)
        
    def process_frame(self, frame):
        """
        Process the frame for facial analysis
        
        Args:
            frame: The camera frame to process
            
        Returns:
            Processed frame with analysis overlays
        """
        # This is a placeholder for actual facial analysis
        # In a real implementation, this would detect faces, landmarks, etc.
        
        # Draw a simple rectangle as a placeholder
        h, w = frame.shape[:2]
        cv2.putText(frame, "Facial Analysis Placeholder", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    @pyqtSlot(object)
    def display_processed_frame(self, frame: np.ndarray):
        """
        Slot to display the processed frame from FacialAnalyzer.
        """
        if frame is None:
            return
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(pixmap)
            self.timestamp_label.setText(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        except Exception as e:
            self.logger.error(f"Error displaying processed frame: {e}")

    @pyqtSlot(str, float)
    def update_emotion_display(self, emotion: str, confidence: float):
        """
        Slot to update emotion and confidence labels.
        """
        self.emotion_label.setText(f"Emotion: {emotion}")
        self.confidence_label.setText(f"Confidence: {confidence:.2f}")
        
    def closeEvent(self, event):
        """
        Handle widget close event
        """
        self.stop_camera()
        super().closeEvent(event)
    
    def update_ui(self):
        """
        Update the UI components
        """
        # This method is called by MainWindow.update_ui()
        # Currently, the camera widget updates itself via the timer
        # No additional updates needed here
        pass