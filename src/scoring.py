"""
NEXARIS Cognitive Load Estimator (NCLE)

Scoring Module

Calculates the cognitive load score based on various inputs like task performance,
behavioral data, and facial analysis.
"""

import time
import numpy as np
import os
import csv
from datetime import datetime

class ScoreCalculator:
    """Calculates and logs cognitive load scores."""
    def __init__(self, log_directory="../logs/"):
        self.weights = {
            "task_performance": 0.4,  # e.g., accuracy, response time
            "behavioral_metrics": 0.3, # e.g., mouse hesitation, click rate
            "facial_emotion": 0.3      # e.g., stress, confusion indicators
        }
        self.log_directory = log_directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.score_log_path = os.path.join(self.log_directory, f"cognitive_score_log_{self.session_id}.csv")
        self._setup_logging()

    def _setup_logging(self):
        """Create log directory and initialize CSV log file with headers."""
        os.makedirs(self.log_directory, exist_ok=True)
        with open(self.score_log_path, 'w', newline='') as f_score:
            score_writer = csv.writer(f_score)
            score_writer.writerow([
                "timestamp", "cognitive_load_score", 
                "task_performance_score", "behavioral_score", "facial_emotion_score",
                "task_id"
            ])
        print(f"Logging cognitive scores to {self.log_directory}")

    def _normalize_score(self, value, min_val, max_val, higher_is_better=True):
        """Normalize a value to a 0-100 scale."""
        if max_val == min_val:
            return 50 # Neutral if no range
        
        normalized = (value - min_val) / (max_val - min_val)
        if not higher_is_better:
            normalized = 1 - normalized
        
        return max(0, min(100, normalized * 100))

    def calculate_task_performance_score(self, task_results):
        """Calculate a score based on task performance.
        
        Args:
            task_results (dict): Output from Task.get_results(). 
                                 Example: {'duration': 60, 'answers': [{'question': ..., 'answer': ..., 'correct_answer': ...}]}
                                 This needs to be more structured, e.g. by adding accuracy.
        Returns:
            float: Score from 0 to 100.
        """
        if not task_results or not task_results.get('answers'):
            return 0 # Default score if no data

        # Example: Score based on accuracy and inverse of time (simplified)
        # This needs a proper definition of 'correctness' in task_results
        # For now, let's assume a mock accuracy and response time component.
        num_questions = len(task_results['answers'])
        if num_questions == 0: return 0
        
        # Mocking accuracy - this should come from task_simulation comparing answers
        # For this example, let's assume 75% accuracy if not provided
        correct_answers = task_results.get('correct_answers', num_questions * 0.75)
        accuracy = (correct_answers / num_questions) * 100
        
        # Normalize accuracy (0-100, higher is better)
        accuracy_score = self._normalize_score(accuracy, 0, 100, higher_is_better=True)

        # Consider average response time (lower is better, needs to be calculated in task_simulation)
        # For now, let's use total duration as a proxy. Assume 5s/question is good, 20s/question is bad.
        avg_time_per_question = task_results.get('duration', num_questions * 10) / num_questions
        time_score = self._normalize_score(avg_time_per_question, 5, 20, higher_is_better=False)
        
        # Combine: e.g. 70% accuracy, 30% time
        performance_score = 0.7 * accuracy_score + 0.3 * time_score
        return performance_score

    def calculate_behavioral_score(self, behavior_data):
        """Calculate a score based on behavioral metrics.
        
        Args:
            behavior_data (dict): Aggregated data from BehaviorTracker.
                                  Example: {'mouse_travel_distance': 1500, 'click_count': 20, 'hesitation_time': 5.2}
                                  This needs to be defined and collected by BehaviorTracker.
        Returns:
            float: Score from 0 to 100 (where higher might mean more stressed/loaded).
        """
        # This is highly dependent on what BehaviorTracker logs and how it's aggregated.
        # Example: Higher hesitation time or erratic movements might increase cognitive load score.
        # For now, a placeholder.
        hesitation_time = behavior_data.get('hesitation_time', 0) # Assume this is available
        # Normalize hesitation (0-10s range, higher hesitation = higher load)
        hesitation_score = self._normalize_score(hesitation_time, 0, 10, higher_is_better=True) 
        return hesitation_score # Higher score means more load from behavior

    def calculate_facial_emotion_score(self, emotion_data):
        """Calculate a score based on facial emotion analysis.
        
        Args:
            emotion_data (dict): Aggregated data from FacialAnalyzer.
                                 Example: {'dominant_emotion': 'stressed', 'stress_confidence': 0.8}
                                 This needs to be defined and collected by FacialAnalyzer.
        Returns:
            float: Score from 0 to 100 (where higher might mean more negative/stressed emotion).
        """
        # Example: Map emotions to load scores
        emotion_map = {
            "neutral": 20, "happy": 10, "surprised": 30,
            "sad": 60, "fear": 70, "angry": 80, "disgust": 75,
            "stressed": 85, "confused": 70 # Example derived emotions
        }
        dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 1.0) # Confidence in emotion detection
        
        base_score = emotion_map.get(dominant_emotion.lower(), 50) # Default to 50 if unknown
        return base_score * confidence # Adjust by confidence

    def calculate_cognitive_load(self, task_perf_score, behav_score, facial_emo_score):
        """Calculate the overall cognitive load score.
        
        Note: For task_perf_score, a higher score usually means better performance (lower load).
              For behav_score and facial_emo_score, a higher score might mean higher load indicators.
              Adjust interpretation accordingly.
        """
        # Inverting task_performance_score as high performance implies low load.
        # So, (100 - task_perf_score) represents load from performance issues.
        load_from_performance = (100 - task_perf_score) * self.weights['task_performance']
        load_from_behavior = behav_score * self.weights['behavioral_metrics']
        load_from_emotion = facial_emo_score * self.weights['facial_emotion']
        
        total_load_score = load_from_performance + load_from_behavior + load_from_emotion
        return max(0, min(100, total_load_score)) # Clamp to 0-100

    def log_score(self, cognitive_load_score, task_perf_score, behav_score, facial_emo_score, task_id="general"):
        """Log the calculated scores to a CSV file."""
        timestamp = time.time()
        with open(self.score_log_path, 'a', newline='') as f_score:
            score_writer = csv.writer(f_score)
            score_writer.writerow([
                timestamp, cognitive_load_score,
                task_perf_score, behav_score, facial_emo_score,
                task_id
            ])

if __name__ == '__main__':
    # Example Usage
    scorer = ScoreCalculator(log_directory="../../logs/") # Adjusted path

    # Simulate data from other modules (these would be actual outputs)
    # Mock task results (assuming some way to determine correctness)
    mock_task_results = {
        'duration': 120, 
        'answers': [{'q':1, 'ans': 'A', 'correct': True}, {'q':2, 'ans': 'B', 'correct': False}],
        'correct_answers': 1 # Manually adding for this example
    }
    mock_behavior_data = {'hesitation_time': 2.5, 'mouse_travel_distance': 2000}
    mock_emotion_data = {'dominant_emotion': 'stressed', 'confidence': 0.75}

    # Calculate individual component scores
    tp_score = scorer.calculate_task_performance_score(mock_task_results)
    b_score = scorer.calculate_behavioral_score(mock_behavior_data)
    fe_score = scorer.calculate_facial_emotion_score(mock_emotion_data)

    print(f"Task Performance Score (0-100, higher is better performance): {tp_score:.2f}")
    print(f"Behavioral Score (0-100, higher indicates more load): {b_score:.2f}")
    print(f"Facial Emotion Score (0-100, higher indicates more load): {fe_score:.2f}")

    # Calculate overall cognitive load
    cognitive_load = scorer.calculate_cognitive_load(tp_score, b_score, fe_score)
    print(f"\nCalculated Cognitive Load (0-100): {cognitive_load:.2f}")

    # Log the scores
    scorer.log_score(cognitive_load, tp_score, b_score, fe_score, task_id="test_scoring_task_001")
    print(f"Scores logged to: {os.path.abspath(scorer.score_log_path)}")