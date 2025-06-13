#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Facial Analyzer for NEXARIS Cognitive Load Estimator

This module uses OpenCV to analyze facial expressions and detect
emotions that may indicate cognitive load, such as frustration,
concentration, and confusion.
"""

import os
import time
import threading
import logging
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports for signals
from PyQt5.QtCore import QObject, pyqtSignal

# Import utilities
from ..utils.logging_utils import get_logger


class FacialAnalyzer(QObject):
    """
    Analyzes facial expressions to detect emotions and cognitive load indicators
    """
    # Define signals for facial analysis events
    face_detected = pyqtSignal(bool)  # Face detected or lost
    emotion_detected = pyqtSignal(str, float)  # Emotion type, confidence
    cognitive_load_update = pyqtSignal(float)  # Estimated load from facial cues
    frame_processed = pyqtSignal(object)  # Processed frame (for display)
    
    # Emotion types
    EMOTIONS = [
        'neutral', 'happy', 'sad', 'surprise', 
        'anger', 'disgust', 'fear', 'contempt',
        'confusion', 'frustration', 'concentration'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the facial analyzer
        
        Args:
            config: Application configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get facial analysis configuration
        self.facial_config = config.get('facial_analysis', {})
        self.enabled = self.facial_config.get('enabled', True)
        self.camera_index = self.facial_config.get('camera_index', 0)
        self.frame_rate = self.facial_config.get('frame_rate', 5)  # FPS
        self.display_video = self.facial_config.get('display_video', True)
        self.save_frames = self.facial_config.get('save_frames', False)
        self.save_interval = self.facial_config.get('save_interval', 5)  # seconds
        self.detection_confidence = self.facial_config.get('detection_confidence', 0.7)
        
        # Initialize OpenCV components
        self.face_cascade = None
        self.emotion_model = None
        self.cap = None
        
        # Initialize tracking data
        self.reset_tracking_data()
        
        # Set up tracking state
        self.is_analyzing = False
        self.analysis_thread = None
        self.last_save_time = None
        
        # Callbacks
        self.data_callbacks = []
        
        # Load models if enabled
        if self.enabled:
            self._load_models()
        
        self.logger.info("Facial Analyzer initialized")
    
    def _load_models(self) -> None:
        """
        Load OpenCV models for face detection and emotion recognition
        """
        try:
            # Load face detection model
            self.face_cascade = cv2.CascadeClassifier()
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not self.face_cascade.load(face_cascade_path):
                self.logger.error(f"Error loading face cascade from {face_cascade_path}")
                self.enabled = False
                return
            
            # Check if we have DNN face detector available (better accuracy)
            try:
                # Try to load DNN face detector model
                model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
                face_model_path = os.path.join(model_path, 'opencv_face_detector.pbtxt')
                face_weights_path = os.path.join(model_path, 'opencv_face_detector_uint8.pb')
                
                if os.path.exists(face_weights_path) and os.path.exists(face_model_path):
                    self.face_net = cv2.dnn.readNet(face_weights_path, face_model_path)
                    self.logger.info("Loaded DNN face detector model")
                else:
                    self.face_net = None
                    self.logger.info("DNN face detector model not found, using Haar cascade")
            except Exception as e:
                self.face_net = None
                self.logger.warning(f"Could not load DNN face detector: {e}")
            
            # Try to load emotion recognition model if available
            try:
                emotion_model_path = os.path.join(model_path, 'emotion-ferplus-8.onnx')
                if os.path.exists(emotion_model_path):
                    self.emotion_model = cv2.dnn.readNet(emotion_model_path)
                    self.logger.info("Loaded emotion recognition model")
                else:
                    self.emotion_model = None
                    self.logger.info("Emotion recognition model not found")
            except Exception as e:
                self.emotion_model = None
                self.logger.warning(f"Could not load emotion recognition model: {e}")
            
            self.logger.info("OpenCV models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading OpenCV models: {e}")
            self.enabled = False
    
    def reset_tracking_data(self) -> None:
        """
        Reset all tracking data
        """
        self.face_detections = []
        self.emotion_detections = []
        self.cognitive_load_estimates = []
        
        # Metrics
        self.face_detection_rate = 0.0
        self.dominant_emotion = 'neutral'
        self.emotion_confidence = 0.0
        self.current_cognitive_load = 0.0
        
        # Current frame data
        self.current_frame = None
        self.current_faces = []
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to receive facial analysis data
        
        Args:
            callback: Function that takes a facial data dictionary as argument
        """
        self.data_callbacks.append(callback)
        self.logger.debug("Registered data callback")
    
    def _notify_data_callbacks(self, data: Dict[str, Any]) -> None:
        """
        Notify all registered callbacks with facial analysis data
        
        Args:
            data: Facial analysis data dictionary
        """
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in facial data callback: {e}")
    
    def start_analysis(self) -> bool:
        """
        Start facial analysis
        
        Returns:
            True if analysis started successfully, False otherwise
        """
        if not self.enabled:
            self.logger.warning("Facial analysis is not enabled")
            return False
        
        if self.is_analyzing:
            self.logger.warning("Facial analysis is already active")
            return True
        
        # Reset tracking data
        self.reset_tracking_data()
        
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.logger.error(f"Could not open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False
        
        # Set up analysis state
        self.is_analyzing = True
        self.last_save_time = time.time()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self.logger.info("Facial analysis started")
        return True
    
    def stop_analysis(self) -> Dict[str, Any]:
        """
        Stop facial analysis and return metrics
        
        Returns:
            Dictionary of facial analysis metrics
        """
        if not self.is_analyzing:
            self.logger.warning("Facial analysis is not active")
            return self.get_metrics()
        
        # Stop analysis
        self.is_analyzing = False
        
        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        self.logger.info("Facial analysis stopped")
        
        # Return final metrics
        return self.get_metrics()
    
    def _analysis_loop(self) -> None:
        """
        Background thread for facial analysis
        """
        frame_interval = 1.0 / self.frame_rate
        
        while self.is_analyzing and self.cap and self.cap.isOpened():
            loop_start = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to capture frame from camera")
                break
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Process frame
            self._process_frame(frame)
            
            # Save frame if enabled
            current_time = time.time()
            if self.save_frames and (current_time - self.last_save_time) >= self.save_interval:
                self._save_frame(frame)
                self.last_save_time = current_time
            
            # Calculate sleep time to maintain frame rate
            processing_time = time.time() - loop_start
            sleep_time = max(0, frame_interval - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Process a video frame for facial analysis
        
        Args:
            frame: OpenCV frame to process
        """
        timestamp = datetime.now().isoformat()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        
        # Detect faces using DNN if available, otherwise use Haar cascade
        if hasattr(self, 'face_net') and self.face_net is not None:
            # Use DNN face detector (more accurate)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            frame_height, frame_width = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.detection_confidence:
                    x1 = int(detections[0, 0, i, 3] * frame_width)
                    y1 = int(detections[0, 0, i, 4] * frame_height)
                    x2 = int(detections[0, 0, i, 5] * frame_width)
                    y2 = int(detections[0, 0, i, 6] * frame_height)
                    
                    # Ensure coordinates are within frame boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)
                    
                    faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        else:
            # Use Haar cascade face detector
            face_rects = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in face_rects:
                faces.append((x, y, w, h, 1.0))  # Confidence is always 1.0 for Haar
        
        # Update face detection status
        face_detected = len(faces) > 0
        self.face_detected.emit(face_detected)
        
        # Store current faces
        self.current_faces = faces
        
        # Record face detection
        self.face_detections.append({
            'timestamp': timestamp,
            'detected': face_detected,
            'count': len(faces)
        })
        
        # Update face detection rate
        if len(self.face_detections) > 0:
            detections = [d['detected'] for d in self.face_detections]
            self.face_detection_rate = sum(detections) / len(detections)
        
        # Process each detected face
        for i, (x, y, w, h, conf) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Analyze emotions if model is available
            emotions = {}
            if self.emotion_model is not None:
                emotions = self._analyze_emotions(face_roi)
                
                # Get dominant emotion
                if emotions:
                    self.dominant_emotion, self.emotion_confidence = max(
                        emotions.items(), key=lambda x: x[1]
                    )
                    
                    # Emit emotion detected signal
                    self.emotion_detected.emit(
                        self.dominant_emotion, self.emotion_confidence
                    )
                    
                    # Record emotion detection
                    self.emotion_detections.append({
                        'timestamp': timestamp,
                        'emotions': emotions,
                        'dominant': self.dominant_emotion,
                        'confidence': self.emotion_confidence
                    })
                    
                    # Add emotion text to frame
                    emotion_text = f"{self.dominant_emotion}: {self.emotion_confidence:.2f}"
                    cv2.putText(
                        frame, emotion_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
            
            # Estimate cognitive load from facial cues
            cognitive_load = self._estimate_cognitive_load(emotions)
            self.current_cognitive_load = cognitive_load
            
            # Emit cognitive load update signal
            self.cognitive_load_update.emit(cognitive_load)
            
            # Record cognitive load estimate
            self.cognitive_load_estimates.append({
                'timestamp': timestamp,
                'cognitive_load': cognitive_load
            })
            
            # Add cognitive load text to frame
            load_text = f"Cognitive Load: {cognitive_load:.2f}"
            cv2.putText(
                frame, load_text, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        # Emit processed frame signal
        self.frame_processed.emit(frame)
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'facial_analysis',
            'timestamp': timestamp,
            'face_detected': face_detected,
            'face_count': len(faces),
            'dominant_emotion': self.dominant_emotion,
            'emotion_confidence': self.emotion_confidence,
            'cognitive_load': self.current_cognitive_load
        })
    
    def _analyze_emotions(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Analyze emotions in a face region
        
        Args:
            face_roi: Face region of interest (grayscale)
        
        Returns:
            Dictionary of emotion probabilities
        """
        try:
            # Resize face to expected size
            face = cv2.resize(face_roi, (64, 64))
            
            # Normalize and prepare input blob
            blob = cv2.dnn.blobFromImage(
                face, 1.0, (64, 64), [0], True, False
            )
            
            # Set input and forward pass
            self.emotion_model.setInput(blob)
            output = self.emotion_model.forward()
            
            # Convert output to probabilities
            probabilities = np.exp(output) / np.sum(np.exp(output))
            
            # Map to emotion labels (depends on model)
            # This is for FER+ ONNX model which has 8 emotions
            emotion_labels = [
                'neutral', 'happy', 'surprise', 'sad',
                'anger', 'disgust', 'fear', 'contempt'
            ]
            
            # Create emotion dictionary
            emotions = {}
            for i, label in enumerate(emotion_labels):
                if i < probabilities.shape[1]:
                    emotions[label] = float(probabilities[0, i])
            
            # Add derived emotions based on combinations
            # Confusion: mix of surprise and sadness/anger
            if 'surprise' in emotions and ('sad' in emotions or 'anger' in emotions):
                surprise = emotions['surprise']
                negative = max(emotions.get('sad', 0), emotions.get('anger', 0))
                emotions['confusion'] = (surprise + negative) / 2
            
            # Frustration: mix of anger and disgust
            if 'anger' in emotions and 'disgust' in emotions:
                emotions['frustration'] = (emotions['anger'] + emotions['disgust']) / 2
            
            # Concentration: high neutral with low happiness
            if 'neutral' in emotions and emotions['neutral'] > 0.5:
                happiness = emotions.get('happy', 0)
                if happiness < 0.3:
                    emotions['concentration'] = emotions['neutral'] * (1 - happiness)
            
            return emotions
        except Exception as e:
            self.logger.error(f"Error analyzing emotions: {e}")
            return {}
    
    def _estimate_cognitive_load(self, emotions: Dict[str, float]) -> float:
        """
        Estimate cognitive load from facial emotions
        
        Args:
            emotions: Dictionary of emotion probabilities
        
        Returns:
            Estimated cognitive load (0.0 to 1.0)
        """
        if not emotions:
            return 0.5  # Default value when no emotions detected
        
        # Define cognitive load weights for each emotion
        # Higher values indicate higher cognitive load
        load_weights = {
            'neutral': 0.3,
            'happy': 0.2,
            'sad': 0.6,
            'surprise': 0.5,
            'anger': 0.7,
            'disgust': 0.6,
            'fear': 0.7,
            'contempt': 0.5,
            'confusion': 0.8,
            'frustration': 0.9,
            'concentration': 0.7
        }
        
        # Calculate weighted average of emotions
        total_weight = 0.0
        weighted_sum = 0.0
        
        for emotion, probability in emotions.items():
            if emotion in load_weights:
                weight = load_weights[emotion]
                weighted_sum += probability * weight
                total_weight += probability
        
        # Normalize result
        if total_weight > 0:
            return min(1.0, max(0.0, weighted_sum / total_weight))
        else:
            return 0.5
    
    def _save_frame(self, frame: np.ndarray) -> None:
        """
        Save the current frame to disk
        
        Args:
            frame: Frame to save
        """
        try:
            # Create frames directory if it doesn't exist
            frames_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = os.path.join(frames_dir, f"frame_{timestamp}.jpg")
            
            # Save frame
            cv2.imwrite(filename, frame)
            self.logger.debug(f"Saved frame to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current facial analysis metrics
        
        Returns:
            Dictionary of facial analysis metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'face_detection_rate': self.face_detection_rate,
            'dominant_emotion': self.dominant_emotion,
            'emotion_confidence': self.emotion_confidence,
            'cognitive_load': self.current_cognitive_load
        }
        
        return metrics
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current processed frame
        
        Returns:
            Current processed frame or None if not available
        """
        return self.current_frame
    
    def is_face_detected(self) -> bool:
        """
        Check if a face is currently detected
        
        Returns:
            True if a face is detected, False otherwise
        """
        return len(self.current_faces) > 0