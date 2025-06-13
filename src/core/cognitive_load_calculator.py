#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cognitive Load Calculator for NEXARIS Cognitive Load Estimator

This module calculates the Estimated Cognitive Load Score (ECLS)
based on behavioral metrics, facial analysis, and task performance.
"""

import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports for signals
from PyQt5.QtCore import QObject, pyqtSignal

# Import utilities
from ..utils.logging_utils import get_logger
from ..utils.ml_utils import check_ml_dependencies, load_model, predict_cognitive_load


class CognitiveLoadCalculator(QObject):
    """
    Calculates the Estimated Cognitive Load Score (ECLS) based on multiple inputs
    """
    # Define signals for cognitive load events
    load_updated = pyqtSignal(float, dict)  # ECLS score, component breakdown
    threshold_exceeded = pyqtSignal(float, float)  # ECLS score, threshold
    
    # Load level classifications
    LOAD_LEVELS = {
        'very_low': (0.0, 0.2),
        'low': (0.2, 0.4),
        'moderate': (0.4, 0.6),
        'high': (0.6, 0.8),
        'very_high': (0.8, 1.0)
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cognitive load calculator
        
        Args:
            config: Application configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get scoring configuration
        self.scoring_config = config.get('scoring', {})
        self.update_interval = self.scoring_config.get('update_interval', 1.0)  # seconds
        self.smoothing_factor = self.scoring_config.get('smoothing_factor', 0.3)
        self.threshold = self.scoring_config.get('threshold', 0.7)
        
        # Get component weights
        self.weights = self.scoring_config.get('weights', {})
        self.behavior_weight = self.weights.get('behavior', 0.4)
        self.facial_weight = self.weights.get('facial', 0.3)
        self.performance_weight = self.weights.get('performance', 0.3)
        self.eeg_weight = self.weights.get('eeg', 0.0)  # Default to 0 if not available
        
        # Advanced features configuration
        self.advanced_config = config.get('advanced_features', {})
        self.use_ml = self.advanced_config.get('use_ml_model', False)
        self.ml_model_path = self.advanced_config.get('ml_model_path', '')
        self.use_eeg = self.advanced_config.get('use_eeg', False)
        
        # Initialize ML model if enabled
        self.ml_model = None
        if self.use_ml and check_ml_dependencies():
            try:
                self.ml_model = load_model(self.ml_model_path)
                self.logger.info(f"Loaded ML model from {self.ml_model_path}")
            except Exception as e:
                self.logger.error(f"Error loading ML model: {e}")
                self.use_ml = False
        
        # Initialize tracking data
        self.reset_tracking_data()
        
        # Callbacks
        self.data_callbacks = []
        
        self.logger.info("Cognitive Load Calculator initialized")
    
    def reset_tracking_data(self) -> None:
        """
        Reset all tracking data
        """
        self.cognitive_load_history = []
        self.component_scores = {
            'behavior': 0.0,
            'facial': 0.0,
            'performance': 0.0,
            'eeg': 0.0
        }
        self.current_ecls = 0.5  # Start at moderate load
        self.smoothed_ecls = 0.5
        self.last_update_time = None
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to receive cognitive load data
        
        Args:
            callback: Function that takes a cognitive load data dictionary as argument
        """
        self.data_callbacks.append(callback)
        self.logger.debug("Registered data callback")
    
    def _notify_data_callbacks(self, data: Dict[str, Any]) -> None:
        """
        Notify all registered callbacks with cognitive load data
        
        Args:
            data: Cognitive load data dictionary
        """
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in cognitive load data callback: {e}")
    
    def update_behavior_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update behavior component of cognitive load calculation
        
        Args:
            metrics: Dictionary of behavior metrics
        """
        # Extract relevant metrics
        mouse_distance = metrics.get('mouse_distance', 0.0)
        click_count = metrics.get('click_count', 0)
        keypress_count = metrics.get('keypress_count', 0)
        hesitation_count = metrics.get('hesitation_count', 0)
        total_hesitation_time = metrics.get('total_hesitation_time', 0.0)
        
        # Calculate behavior score components
        activity_score = self._calculate_activity_score(mouse_distance, click_count, keypress_count)
        hesitation_score = self._calculate_hesitation_score(hesitation_count, total_hesitation_time)
        
        # Combine components into overall behavior score
        behavior_score = 0.6 * hesitation_score + 0.4 * activity_score
        
        # Update component score
        self.component_scores['behavior'] = behavior_score
        
        # Update overall cognitive load
        self._update_cognitive_load()
    
    def update_facial_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update facial component of cognitive load calculation
        
        Args:
            metrics: Dictionary of facial analysis metrics
        """
        # Extract relevant metrics
        face_detection_rate = metrics.get('face_detection_rate', 0.0)
        dominant_emotion = metrics.get('dominant_emotion', 'neutral')
        emotion_confidence = metrics.get('emotion_confidence', 0.0)
        cognitive_load = metrics.get('cognitive_load', 0.5)
        
        # If no face is detected consistently, reduce the weight of facial analysis
        if face_detection_rate < 0.5:
            self.facial_weight = self.weights.get('facial', 0.3) * face_detection_rate
            # Redistribute weight to other components
            remaining_weight = self.weights.get('facial', 0.3) * (1 - face_detection_rate)
            self.behavior_weight += remaining_weight * 0.6
            self.performance_weight += remaining_weight * 0.4
        else:
            # Reset weights to original values
            self.behavior_weight = self.weights.get('behavior', 0.4)
            self.facial_weight = self.weights.get('facial', 0.3)
            self.performance_weight = self.weights.get('performance', 0.3)
        
        # Update component score
        self.component_scores['facial'] = cognitive_load
        
        # Update overall cognitive load
        self._update_cognitive_load()
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update performance component of cognitive load calculation
        
        Args:
            metrics: Dictionary of task performance metrics
        """
        # Extract relevant metrics
        accuracy = metrics.get('accuracy', 1.0)
        response_time = metrics.get('response_time', 0.0)
        completion_rate = metrics.get('completion_rate', 1.0)
        difficulty = metrics.get('difficulty', 0.5)
        
        # Calculate performance score
        # Higher response time and lower accuracy indicate higher cognitive load
        time_factor = min(1.0, response_time / 10.0)  # Normalize to 0-1 range
        accuracy_factor = 1.0 - accuracy  # Invert accuracy (lower is higher load)
        completion_factor = 1.0 - completion_rate  # Invert completion rate
        
        # Combine factors with difficulty
        performance_score = (
            0.4 * time_factor + 
            0.3 * accuracy_factor + 
            0.2 * completion_factor + 
            0.1 * difficulty
        )
        
        # Update component score
        self.component_scores['performance'] = performance_score
        
        # Update overall cognitive load
        self._update_cognitive_load()
    
    def update_eeg_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update EEG component of cognitive load calculation
        
        Args:
            metrics: Dictionary of EEG metrics
        """
        if not self.use_eeg:
            return
        
        # Extract relevant metrics
        cognitive_load = metrics.get('cognitive_load', 0.5)
        signal_quality = metrics.get('signal_quality', 0.0)
        
        # Adjust EEG weight based on signal quality
        if signal_quality > 0.7:  # Good signal quality
            self.eeg_weight = self.weights.get('eeg', 0.0)
        else:  # Poor signal quality, reduce weight
            self.eeg_weight = self.weights.get('eeg', 0.0) * signal_quality
        
        # Update component score
        self.component_scores['eeg'] = cognitive_load
        
        # Update overall cognitive load
        self._update_cognitive_load()
    
    def _calculate_activity_score(self, mouse_distance: float, click_count: int, keypress_count: int) -> float:
        """
        Calculate activity score based on mouse and keyboard activity
        
        Args:
            mouse_distance: Total mouse movement distance
            click_count: Number of mouse clicks
            keypress_count: Number of key presses
        
        Returns:
            Activity score (0.0 to 1.0)
        """
        # Normalize metrics
        # These thresholds may need adjustment based on typical usage patterns
        norm_distance = min(1.0, mouse_distance / 5000.0)
        norm_clicks = min(1.0, click_count / 50.0)
        norm_keypresses = min(1.0, keypress_count / 100.0)
        
        # Calculate activity score
        # Higher activity can indicate higher engagement but also potentially higher load
        activity_score = 0.4 * norm_distance + 0.3 * norm_clicks + 0.3 * norm_keypresses
        
        return activity_score
    
    def _calculate_hesitation_score(self, hesitation_count: int, total_hesitation_time: float) -> float:
        """
        Calculate hesitation score based on hesitation patterns
        
        Args:
            hesitation_count: Number of hesitations
            total_hesitation_time: Total time spent hesitating (seconds)
        
        Returns:
            Hesitation score (0.0 to 1.0)
        """
        # Normalize metrics
        norm_count = min(1.0, hesitation_count / 20.0)
        norm_time = min(1.0, total_hesitation_time / 60.0)  # Normalize to 1 minute
        
        # Calculate hesitation score
        # More hesitation indicates higher cognitive load
        hesitation_score = 0.4 * norm_count + 0.6 * norm_time
        
        return hesitation_score
    
    def _update_cognitive_load(self) -> None:
        """
        Update the overall Estimated Cognitive Load Score (ECLS)
        """
        current_time = time.time()
        
        # Check if it's time to update
        if self.last_update_time is not None and \
           (current_time - self.last_update_time) < self.update_interval:
            return
        
        self.last_update_time = current_time
        timestamp = datetime.now().isoformat()
        
        # Calculate weighted sum of component scores
        if self.use_ml and self.ml_model is not None:
            # Use ML model to predict cognitive load
            features = {
                'behavior_score': self.component_scores['behavior'],
                'facial_score': self.component_scores['facial'],
                'performance_score': self.component_scores['performance'],
                'eeg_score': self.component_scores['eeg']
            }
            
            try:
                self.current_ecls = predict_cognitive_load(self.ml_model, features)
            except Exception as e:
                self.logger.error(f"Error predicting cognitive load with ML model: {e}")
                # Fall back to weighted sum
                self.current_ecls = self._calculate_weighted_sum()
        else:
            # Use weighted sum of components
            self.current_ecls = self._calculate_weighted_sum()
        
        # Apply exponential smoothing
        if len(self.cognitive_load_history) > 0:
            self.smoothed_ecls = (self.smoothing_factor * self.current_ecls + 
                                 (1 - self.smoothing_factor) * self.smoothed_ecls)
        else:
            self.smoothed_ecls = self.current_ecls
        
        # Record cognitive load
        self.cognitive_load_history.append({
            'timestamp': timestamp,
            'raw_ecls': self.current_ecls,
            'smoothed_ecls': self.smoothed_ecls,
            'components': self.component_scores.copy()
        })
        
        # Check if threshold is exceeded
        if self.smoothed_ecls > self.threshold:
            self.threshold_exceeded.emit(self.smoothed_ecls, self.threshold)
        
        # Emit load updated signal
        self.load_updated.emit(self.smoothed_ecls, self.component_scores.copy())
        
        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'cognitive_load',
            'timestamp': timestamp,
            'raw_ecls': self.current_ecls,
            'smoothed_ecls': self.smoothed_ecls,
            'components': self.component_scores.copy(),
            'load_level': self.get_load_level(self.smoothed_ecls)
        })
    
    def _calculate_weighted_sum(self) -> float:
        """
        Calculate weighted sum of component scores
        
        Returns:
            Weighted sum (0.0 to 1.0)
        """
        # Normalize weights
        total_weight = (self.behavior_weight + self.facial_weight + 
                       self.performance_weight + self.eeg_weight)
        
        if total_weight == 0:
            return 0.5  # Default value if no weights
        
        # Calculate normalized weights
        norm_behavior_weight = self.behavior_weight / total_weight
        norm_facial_weight = self.facial_weight / total_weight
        norm_performance_weight = self.performance_weight / total_weight
        norm_eeg_weight = self.eeg_weight / total_weight
        
        # Calculate weighted sum
        weighted_sum = (
            norm_behavior_weight * self.component_scores['behavior'] +
            norm_facial_weight * self.component_scores['facial'] +
            norm_performance_weight * self.component_scores['performance'] +
            norm_eeg_weight * self.component_scores['eeg']
        )
        
        return weighted_sum
    
    def get_load_level(self, ecls: float) -> str:
        """
        Get the cognitive load level classification
        
        Args:
            ecls: Estimated Cognitive Load Score
        
        Returns:
            Load level classification
        """
        for level, (min_val, max_val) in self.LOAD_LEVELS.items():
            if min_val <= ecls < max_val:
                return level
        
        # Default to very_high if outside range
        return 'very_high'
    
    def get_current_load(self) -> Dict[str, Any]:
        """
        Get the current cognitive load data
        
        Returns:
            Dictionary with current cognitive load data
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'raw_ecls': self.current_ecls,
            'smoothed_ecls': self.smoothed_ecls,
            'components': self.component_scores.copy(),
            'load_level': self.get_load_level(self.smoothed_ecls)
        }
    
    def get_load_history(self) -> List[Dict[str, Any]]:
        """
        Get the cognitive load history
        
        Returns:
            List of cognitive load history entries
        """
        return self.cognitive_load_history