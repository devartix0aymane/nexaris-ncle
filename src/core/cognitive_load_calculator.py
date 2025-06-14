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
    
    # Load level classifications (0-100 scale)
    LOAD_LEVELS = {
        'Low Load': (0, 30),
        'Medium Load': (31, 70),
        'High Load': (71, 100)
    }
    # Detailed explanations for scores
    SCORE_EXPLANATIONS = {
        'task_performance_accuracy_low': 'Low task accuracy suggests difficulty, increasing load.',
        'task_performance_accuracy_high': 'High task accuracy suggests ease, decreasing load.',
        'task_performance_time_slow': 'Slow response times indicate struggle, increasing load.',
        'task_performance_time_fast': 'Fast response times indicate efficiency, decreasing load.',
        'mouse_clicks_high': 'High number of clicks might indicate uncertainty or repeated attempts, increasing load.',
        'mouse_clicks_low': 'Low number of clicks might indicate smooth interaction, decreasing load.',
        'mouse_speed_high': 'High mouse speed could indicate agitation or rushing, potentially increasing load.',
        'mouse_speed_low': 'Low mouse speed could indicate carefulness or hesitation; context dependent.',
        'mouse_path_erratic': 'Erratic mouse path suggests indecision or difficulty, increasing load.',
        'mouse_path_smooth': 'Smooth mouse path suggests confidence, decreasing load.',
        'click_delay_high': 'Long delays between clicks can indicate deep thought or distraction, potentially increasing load.',
        'click_delay_low': 'Short delays between clicks can indicate rapid decision making or impulsiveness.',
        'idle_time_high': 'Significant idle time may point to disengagement or being stuck, increasing load.',
        'idle_time_low': 'Low idle time suggests continuous engagement.',
        'emotion_negative_high_confidence': 'Strong negative emotions (e.g., frustration, anger) detected with high confidence significantly increase load.',
        'emotion_positive_high_confidence': 'Strong positive emotions (e.g., happiness) detected with high confidence may decrease load or indicate engagement.',
        'emotion_neutral_high_confidence': 'Neutral emotion with high confidence suggests focused state, potentially moderate load.',
        'emotion_low_confidence': 'Low confidence in emotion detection makes this factor less reliable for load assessment.'
    }
    
    def __init__(self, config: Dict[str, Any], csv_logger_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the cognitive load calculator
        
        Args:
            config: Application configuration dictionary
            csv_logger_callback: Optional callback function to log data to CSV
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        self.csv_logger_callback = csv_logger_callback
        
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
            'behavior': 0.0, # Contribution from mouse/click metrics
            'facial': 0.0,   # Contribution from emotion metrics
            'performance': 0.0, # Contribution from task performance (time, accuracy)
            # 'eeg': 0.0 # EEG not part of current request
        }
        self.current_ecls = 50  # Start at moderate load (0-100 scale)
        self.smoothed_ecls = 50
        self.last_update_time = None
        self.last_explanation_keys = []
    
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
    
    def update_behavior_metrics(self, metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Update behavior component of cognitive load calculation.
        Normalizes mouse/click metrics and contributes to the cognitive load score.

        Args:
            metrics: Dictionary of behavior metrics (mouse_path_length, mouse_speed, 
                     num_clicks, avg_click_delay, total_idle_time).
        
        Returns:
            Tuple: (behavior_score_contribution, explanation_keys)
                     Behavior score contribution (0-1, where 1 is high load indication)
                     List of keys for SCORE_EXPLANATIONS.
        """
        explanation_keys = []
        
        # Normalize mouse path length (e.g., 0-10000 pixels, more is higher load)
        # These are example normalizations and may need tuning
        mouse_path_length = metrics.get('mouse_path_length', 0.0)
        norm_path = min(1.0, mouse_path_length / 10000.0) 
        if norm_path > 0.7: explanation_keys.append('mouse_path_erratic')
        elif norm_path < 0.3: explanation_keys.append('mouse_path_smooth')

        # Normalize mouse speed (e.g., 0-1000 pixels/sec, very high or very low could be load)
        mouse_speed = metrics.get('mouse_speed', 0.0)
        norm_speed = 0.0
        if mouse_speed > 800: norm_speed = 1.0; explanation_keys.append('mouse_speed_high')
        elif mouse_speed < 100 and mouse_speed > 0: norm_speed = 0.7 # Slow can also be load
        if mouse_speed > 800 : explanation_keys.append('mouse_speed_high')
        elif mouse_speed < 100 and mouse_speed > 10 : explanation_keys.append('mouse_speed_low')

        # Normalize number of clicks (e.g., 0-50 clicks, more is higher load)
        num_clicks = metrics.get('num_clicks', 0)
        norm_clicks = min(1.0, num_clicks / 50.0)
        if norm_clicks > 0.7: explanation_keys.append('mouse_clicks_high')
        elif norm_clicks < 0.2 and num_clicks > 0: explanation_keys.append('mouse_clicks_low')

        # Normalize average click delay (e.g., 0-5 seconds, longer is higher load)
        avg_click_delay = metrics.get('avg_click_delay', 0.0)
        norm_click_delay = min(1.0, avg_click_delay / 5.0)
        if norm_click_delay > 0.7 : explanation_keys.append('click_delay_high')
        elif norm_click_delay < 0.2 and avg_click_delay > 0: explanation_keys.append('click_delay_low')

        # Normalize total idle time (e.g., 0-60 seconds, more is higher load)
        total_idle_time = metrics.get('total_idle_time', 0.0)
        norm_idle_time = min(1.0, total_idle_time / 60.0)
        if norm_idle_time > 0.6: explanation_keys.append('idle_time_high')
        elif norm_idle_time < 0.1 and total_idle_time > 0 : explanation_keys.append('idle_time_low')

        # Combine components into overall behavior score (example weighting)
        # Higher score indicates higher cognitive load contribution from behavior
        behavior_score_contribution = (0.3 * norm_path + 
                                     0.1 * norm_speed + 
                                     0.2 * norm_clicks + 
                                     0.2 * norm_click_delay + 
                                     0.2 * norm_idle_time)
        
        self.component_scores['behavior'] = behavior_score_contribution
        # self._update_cognitive_load() # We will call this once all data is in
        return behavior_score_contribution, explanation_keys
    
    def update_facial_metrics(self, metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Update facial component of cognitive load calculation.
        Normalizes emotion metrics and contributes to the cognitive load score.

        Args:
            metrics: Dictionary of facial analysis metrics (dominant_emotion, emotion_confidence).
        
        Returns:
            Tuple: (facial_score_contribution, explanation_keys)
                     Facial score contribution (0-1, where 1 is high load indication)
                     List of keys for SCORE_EXPLANATIONS.
        """
        explanation_keys = []
        dominant_emotion = metrics.get('dominant_emotion', 'neutral').lower()
        emotion_confidence = metrics.get('emotion_confidence', 0.0)

        facial_score_contribution = 0.5  # Default neutral

        if emotion_confidence < 0.4: # Low confidence, less impact
            explanation_keys.append('emotion_low_confidence')
            facial_score_contribution = 0.5 # Neutral if low confidence
        else:
            if dominant_emotion in ['anger', 'frustration', 'fear', 'sad', 'disgust', 'confusion']:
                facial_score_contribution = 0.3 + (0.7 * emotion_confidence) # Scale up to 1.0 for strong negative
                explanation_keys.append('emotion_negative_high_confidence')
            elif dominant_emotion in ['happy', 'surprise']:
                facial_score_contribution = 0.5 - (0.3 * emotion_confidence) # Scale down for positive
                explanation_keys.append('emotion_positive_high_confidence')
            elif dominant_emotion == 'neutral':
                facial_score_contribution = 0.4 # Neutral is slightly less than mid-point
                explanation_keys.append('emotion_neutral_high_confidence')
        
        self.component_scores['facial'] = facial_score_contribution
        # self._update_cognitive_load() # We will call this once all data is in
        return facial_score_contribution, explanation_keys
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Update performance component of cognitive load calculation.
        Normalizes task performance metrics and contributes to the cognitive load score.

        Args:
            metrics: Dictionary of task performance metrics (accuracy, avg_response_time_ms).
        
        Returns:
            Tuple: (performance_score_contribution, explanation_keys)
                     Performance score contribution (0-1, where 1 is high load indication)
                     List of keys for SCORE_EXPLANATIONS.
        """
        explanation_keys = []
        accuracy = metrics.get('accuracy', 1.0)  # 0.0 to 1.0
        avg_response_time_s = metrics.get('avg_response_time_ms', 5000) / 1000.0 # Convert ms to s, default 5s

        # Normalize accuracy (lower accuracy = higher load contribution)
        # If accuracy is 1.0, contribution is 0. If accuracy is 0.0, contribution is 1.0.
        norm_accuracy_load = 1.0 - accuracy
        if accuracy < 0.5: explanation_keys.append('task_performance_accuracy_low')
        elif accuracy > 0.9: explanation_keys.append('task_performance_accuracy_high')

        # Normalize response time (longer time = higher load contribution)
        # Example: 0-15 seconds. Cap at 15s for normalization.
        norm_time_load = min(1.0, avg_response_time_s / 15.0)
        if avg_response_time_s > 10.0: explanation_keys.append('task_performance_time_slow')
        elif avg_response_time_s < 3.0: explanation_keys.append('task_performance_time_fast')

        # Combine components (example weighting)
        performance_score_contribution = (0.6 * norm_accuracy_load + 
                                          0.4 * norm_time_load)
        
        self.component_scores['performance'] = performance_score_contribution
        # self._update_cognitive_load() # We will call this once all data is in
        return performance_score_contribution, explanation_keys
    
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
        
        # Calculate final ECLS score (0-100)
        # Weights for each component (sum to 1.0)
        # These are example weights and can be configured
        w_performance = self.weights.get('performance', 0.4)
        w_behavior = self.weights.get('behavior', 0.3)
        w_facial = self.weights.get('facial', 0.3)

        # Ensure weights sum to 1 for normalization, excluding EEG for now
        current_total_weight = w_performance + w_behavior + w_facial
        if current_total_weight == 0: # Avoid division by zero if all weights are zero
            self.current_ecls = 50 # Default to medium if no weights
        else:
            norm_w_performance = w_performance / current_total_weight
            norm_w_behavior = w_behavior / current_total_weight
            norm_w_facial = w_facial / current_total_weight

            self.current_ecls = (
                norm_w_performance * self.component_scores.get('performance', 0.5) * 100 +
                norm_w_behavior * self.component_scores.get('behavior', 0.5) * 100 +
                norm_w_facial * self.component_scores.get('facial', 0.5) * 100
            )
        self.current_ecls = min(100, max(0, self.current_ecls)) # Clamp to 0-100

        # Apply exponential smoothing
        if len(self.cognitive_load_history) > 0:
            self.smoothed_ecls = (self.smoothing_factor * self.current_ecls +
                                 (1 - self.smoothing_factor) * self.smoothed_ecls)
        else:
            self.smoothed_ecls = self.current_ecls
        self.smoothed_ecls = min(100, max(0, self.smoothed_ecls)) # Clamp smoothed value

        # Record cognitive load
        history_entry = {
            'timestamp': timestamp,
            'raw_ecls': self.current_ecls,
            'smoothed_ecls': self.smoothed_ecls,
            'components': self.component_scores.copy(),
            'explanation_keys': self.last_explanation_keys # Store keys used for this score
        }
        self.cognitive_load_history.append(history_entry)

        # Prepare detailed explanation
        detailed_explanation = self.get_detailed_explanation(self.last_explanation_keys, self.component_scores, self.smoothed_ecls)

        # Check if threshold is exceeded (threshold is 0-1, ECLS is 0-100)
        if self.smoothed_ecls > (self.threshold * 100):
            self.threshold_exceeded.emit(self.smoothed_ecls, self.threshold * 100)

        # Log data to CSV if callback is provided
        if self.csv_logger_callback:
            log_data = {
                'timestamp': timestamp,
                'raw_ecls': self.current_ecls,
                'smoothed_ecls': self.smoothed_ecls,
                'behavior_score': self.component_scores.get('behavior', 0.0),
                'facial_score': self.component_scores.get('facial', 0.0),
                'performance_score': self.component_scores.get('performance', 0.0),
                # 'eeg_score': self.component_scores.get('eeg', 0.0), # If EEG is used
                'explanation_keys': ", ".join(self.last_explanation_keys) # Join list into a string
            }
            try:
                self.csv_logger_callback(log_data)
            except Exception as e:
                self.logger.error(f"Error calling CSV logger callback: {e}")

        # Emit load updated signal with score (0-100) and explanation
        self.load_updated.emit(self.smoothed_ecls, {'explanation': detailed_explanation, 'components': self.component_scores.copy()})

        # Notify callbacks
        self._notify_data_callbacks({
            'data_type': 'cognitive_load',
            'timestamp': timestamp,
            'raw_ecls': self.current_ecls,
            'smoothed_ecls': self.smoothed_ecls,
            'components': self.component_scores.copy(),
            'load_level': self.get_load_level(self.smoothed_ecls),
            'explanation': detailed_explanation
        })
    
    def calculate_combined_cognitive_load(self, 
                                          task_metrics: Dict[str, Any], 
                                          behavior_metrics: Dict[str, Any], 
                                          facial_metrics: Dict[str, Any]) -> None:
        """
        Calculates the cognitive load based on combined metrics and updates.
        This method should be called when all data for a time window is available.
        """
        all_explanation_keys = []

        perf_score, perf_keys = self.update_performance_metrics(task_metrics)
        all_explanation_keys.extend(perf_keys)

        behav_score, behav_keys = self.update_behavior_metrics(behavior_metrics)
        all_explanation_keys.extend(behav_keys)

        fac_score, fac_keys = self.update_facial_metrics(facial_metrics)
        all_explanation_keys.extend(fac_keys)
        
        self.last_explanation_keys = list(set(all_explanation_keys)) # Remove duplicates

        self._update_cognitive_load() # This will now use the component_scores set by the update_* methods

    def get_detailed_explanation(self, explanation_keys: List[str], component_scores: Dict[str, float], final_score: float) -> str:
        """
        Generates a detailed explanation for the cognitive load score.
        """
        explanation = f"Cognitive Load Score: {final_score:.2f}/100 ({self.get_load_level(final_score)}).\n"
        explanation += "Contributing factors:\n"
        
        unique_keys = sorted(list(set(explanation_keys))) # Ensure unique and sorted for consistent output
        if not unique_keys:
            explanation += "- No specific strong indicators identified; score based on general assessment.\n"
        else:
            for key in unique_keys:
                if key in self.SCORE_EXPLANATIONS:
                    explanation += f"- {self.SCORE_EXPLANATIONS[key]}\n"
        
        explanation += "\nComponent contributions (0-1, higher indicates more load contribution):\n"
        explanation += f"- Task Performance Score: {component_scores.get('performance', 0.0):.2f}\n"
        explanation += f"- Behavioral Metrics Score: {component_scores.get('behavior', 0.0):.2f}\n"
        explanation += f"- Facial Emotion Score: {component_scores.get('facial', 0.0):.2f}\n"
        
        return explanation

    # _calculate_weighted_sum is effectively replaced by logic in _update_cognitive_load now for 0-100 scale.
    
    def get_load_level(self, ecls: float) -> str:
        """
        Get the cognitive load level classification
        
        Args:
            ecls: Estimated Cognitive Load Score
        
        Returns:
            Load level classification
        """
        # ECLS is now 0-100
        for level, (min_val, max_val) in self.LOAD_LEVELS.items():
            # For the last category (High Load), it should be inclusive of max_val
            if level == 'High Load':
                if min_val <= ecls <= max_val:
                    return level
            elif min_val <= ecls <= max_val: # Adjusted to be inclusive for ranges like 31-70
                return level
        
        if ecls > 100 : return 'High Load' # Cap at high if somehow over
        if ecls < 0 : return 'Low Load' # Cap at low if somehow under
        return 'Medium Load' # Default if somehow missed (should not happen with proper ranges)
    
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