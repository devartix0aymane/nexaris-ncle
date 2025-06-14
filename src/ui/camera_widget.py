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
        
        # Timer for updating camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def start_camera(self):
        """
        Start the camera feed
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                QMessageBox.warning(self, "Camera Error", f"Failed to open camera {self.camera_id}")
                return
                
            self.camera_active = True
            self.timer.start(30)  # Update at 30ms intervals (approx. 33 fps)
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            
            self.logger.info(f"Started camera {self.camera_id}")
        except Exception as e:
            self.logger.error(f"Error starting camera: {str(e)}")
            
    def stop_camera(self):
        """
        Stop the camera feed
        """
        if self.camera_active:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.camera_active = False
            
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.camera_combo.setEnabled(True)
            self.camera_label.setText("Camera feed not available")
            
            self.logger.info("Stopped camera")
            
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
            
        # Process frame if facial analysis is enabled
        if self.facial_analysis_enabled:
            # Placeholder for facial analysis
            # In a real implementation, this would call facial_analyzer methods
            frame = self.process_frame(frame)
            
        # Convert frame to QPixmap and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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