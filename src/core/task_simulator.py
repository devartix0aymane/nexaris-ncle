#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task Simulator for NEXARIS Cognitive Load Estimator

This module provides functionality for simulating various cognitive tasks
to measure user performance and cognitive load.
"""

import os
import json
import random
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# PyQt imports for signals
from PyQt5.QtCore import QObject, pyqtSignal

# Import utilities
from ..utils.logging_utils import get_logger


class TaskSimulator(QObject):
    """
    Simulates cognitive tasks for measuring user performance and cognitive load
    """
    
    # Define signals
    task_started = pyqtSignal(dict)
    task_completed = pyqtSignal(dict)
    task_progress = pyqtSignal(int, int)  # current, total
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the task simulator
        
        Args:
            config: Application configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load task configurations
        self.task_config = config.get('task', {})
        self.default_duration = self.task_config.get('default_duration', 300)  # seconds
        self.difficulty_levels = self.task_config.get('difficulty_levels', ["easy", "medium", "hard"])
        self.default_difficulty = self.task_config.get('default_difficulty', "medium")
        self.question_sets = {}
        self._load_question_sets()
        self.current_task = None
        self.task_start_time = None
        self.task_end_time = None
        self.task_callbacks = {}
        
        self.logger.info("Task Simulator initialized")
    
    def _load_question_sets(self) -> None:
        """
        Load question sets from configuration files
        """
        # Define path to question sets
        question_sets_dir = Path(__file__).resolve().parents[2] / 'config' / 'question_sets'
        question_sets_dir.mkdir(parents=True, exist_ok=True)
        
        # Load each question set
        for question_set in self.task_config.get('question_sets', []):
            question_file = question_sets_dir / f"{question_set}.json"
            
            # Create default question set if file doesn't exist
            if not question_file.exists():
                self._create_default_question_set(question_set, question_file)
            
            # Load question set
            try:
                with open(question_file, 'r') as f:
                    questions = json.load(f)
                
                self.question_sets[question_set] = questions
                self.logger.info(f"Loaded question set '{question_set}' with {len(questions)} questions")
            
            except Exception as e:
                self.logger.error(f"Error loading question set '{question_set}': {e}")
                # Create an empty question set as fallback
                self.question_sets[question_set] = []
    
    def _create_default_question_set(self, set_name: str, file_path: Path) -> None:
        """
        Create a default question set file
        
        Args:
            set_name: Name of the question set
            file_path: Path to save the question set
        """
        default_questions = []
        
        # Create different default questions based on set name
        if set_name == "general":
            default_questions = [
                {
                    "id": "g1",
                    "question": "What is the capital of France?",
                    "options": ["Paris", "London", "Berlin", "Madrid"],
                    "correct_answer": "Paris",
                    "difficulty": "easy",
                    "category": "geography",
                    "time_limit": 30
                },
                {
                    "id": "g2",
                    "question": "What is 15 × 17?",
                    "options": ["255", "257", "267", "277"],
                    "correct_answer": "255",
                    "difficulty": "medium",
                    "category": "mathematics",
                    "time_limit": 45
                },
                {
                    "id": "g3",
                    "question": "Who wrote 'War and Peace'?",
                    "options": ["Leo Tolstoy", "Fyodor Dostoevsky", "Anton Chekhov", "Ivan Turgenev"],
                    "correct_answer": "Leo Tolstoy",
                    "difficulty": "medium",
                    "category": "literature",
                    "time_limit": 30
                },
                {
                    "id": "g4",
                    "question": "What is the chemical symbol for gold?",
                    "options": ["Au", "Ag", "Fe", "Cu"],
                    "correct_answer": "Au",
                    "difficulty": "easy",
                    "category": "science",
                    "time_limit": 20
                },
                {
                    "id": "g5",
                    "question": "In which year did World War II end?",
                    "options": ["1945", "1944", "1946", "1947"],
                    "correct_answer": "1945",
                    "difficulty": "easy",
                    "category": "history",
                    "time_limit": 25
                }
            ]
        elif set_name == "cybersecurity":
            default_questions = [
                {
                    "id": "cs1",
                    "question": "Which of the following is NOT a common type of cyber attack?",
                    "options": ["Data Compression", "Phishing", "Man-in-the-Middle", "DDoS"],
                    "correct_answer": "Data Compression",
                    "difficulty": "easy",
                    "category": "attacks",
                    "time_limit": 30
                },
                {
                    "id": "cs2",
                    "question": "What does SSL stand for?",
                    "options": ["Secure Socket Layer", "System Security Layer", "Secure System Login", "Safe Socket Layer"],
                    "correct_answer": "Secure Socket Layer",
                    "difficulty": "medium",
                    "category": "protocols",
                    "time_limit": 25
                },
                {
                    "id": "cs3",
                    "question": "Which encryption algorithm is considered the most secure as of 2023?",
                    "options": ["AES-256", "DES", "MD5", "SHA-1"],
                    "correct_answer": "AES-256",
                    "difficulty": "hard",
                    "category": "encryption",
                    "time_limit": 40
                },
                {
                    "id": "cs4",
                    "question": "What is the primary purpose of a firewall?",
                    "options": ["Monitor network traffic and block unauthorized access", "Encrypt data transmissions", "Scan for viruses", "Backup data"],
                    "correct_answer": "Monitor network traffic and block unauthorized access",
                    "difficulty": "easy",
                    "category": "network security",
                    "time_limit": 35
                },
                {
                    "id": "cs5",
                    "question": "What type of attack attempts to exhaust system resources?",
                    "options": ["Denial of Service", "SQL Injection", "Cross-Site Scripting", "Social Engineering"],
                    "correct_answer": "Denial of Service",
                    "difficulty": "medium",
                    "category": "attacks",
                    "time_limit": 30
                }
            ]
        elif set_name == "alert_triage":
            default_questions = [
                {
                    "id": "at1",
                    "question": "You receive an alert about multiple failed login attempts from different countries. What should you do first?",
                    "options": ["Lock the account and investigate", "Ignore it as a false positive", "Reset the user's password without investigation", "Wait for more failed attempts"],
                    "correct_answer": "Lock the account and investigate",
                    "difficulty": "medium",
                    "category": "account security",
                    "time_limit": 45
                },
                {
                    "id": "at2",
                    "question": "An IDS alert shows unusual outbound traffic to an unknown IP address. What is your first step?",
                    "options": ["Check IP reputation and investigate the source", "Block all outbound traffic", "Shut down the network", "Ignore the alert"],
                    "correct_answer": "Check IP reputation and investigate the source",
                    "difficulty": "hard",
                    "category": "network security",
                    "time_limit": 50
                },
                {
                    "id": "at3",
                    "question": "You receive an alert about a potential data exfiltration. Which of these is NOT a typical indicator?",
                    "options": ["Regular small HTTP requests", "Large email attachments to external domains", "Unusual database queries", "Encrypted traffic to new destinations"],
                    "correct_answer": "Regular small HTTP requests",
                    "difficulty": "hard",
                    "category": "data security",
                    "time_limit": 60
                },
                {
                    "id": "at4",
                    "question": "A system shows signs of high CPU usage with no apparent cause. What should you check first?",
                    "options": ["Running processes and their resource usage", "System logs from last month", "Network configuration", "User account list"],
                    "correct_answer": "Running processes and their resource usage",
                    "difficulty": "medium",
                    "category": "system security",
                    "time_limit": 40
                },
                {
                    "id": "at5",
                    "question": "You receive an alert about a potential phishing email. What is the most important first step?",
                    "options": ["Analyze the email headers and links without clicking", "Forward the email to all security team members", "Delete the email immediately", "Click links to see where they lead"],
                    "correct_answer": "Analyze the email headers and links without clicking",
                    "difficulty": "easy",
                    "category": "email security",
                    "time_limit": 35
                }
            ]
        else:
            # Generic questions for other sets
            default_questions = [
                {
                    "id": "q1",
                    "question": "Sample question 1?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A",
                    "difficulty": "medium",
                    "category": "general",
                    "time_limit": 30
                },
                {
                    "id": "q2",
                    "question": "Sample question 2?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option B",
                    "difficulty": "easy",
                    "category": "general",
                    "time_limit": 20
                },
                {
                    "id": "q3",
                    "question": "Sample question 3?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option C",
                    "difficulty": "hard",
                    "category": "general",
                    "time_limit": 45
                },
                {
                    "id": "q4",
                    "question": "Sample question 4?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option D",
                    "difficulty": "medium",
                    "category": "general",
                    "time_limit": 30
                },
                {
                    "id": "q5",
                    "question": "Sample question 5?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A",
                    "difficulty": "easy",
                    "category": "general",
                    "time_limit": 25
                }
            ]
        
        # Save the default question set
        try:
            with open(file_path, 'w') as f:
                json.dump(default_questions, f, indent=4)
            
            self.logger.info(f"Created default question set '{set_name}' with {len(default_questions)} questions")
        
        except Exception as e:
            self.logger.error(f"Error creating default question set '{set_name}': {e}")
    
    def get_available_question_sets(self) -> List[str]:
        """
        Get a list of available question sets
        
        Returns:
            List of question set names
        """
        return list(self.question_sets.keys())
    
    def get_question_set(self, set_name: str) -> List[Dict[str, Any]]:
        """
        Get a specific question set
        
        Args:
            set_name: Name of the question set
            
        Returns:
            List of questions in the set
        """
        return self.question_sets.get(set_name, [])
    
    def get_questions_by_difficulty(self, set_name: str, difficulty: str) -> List[Dict[str, Any]]:
        """
        Get questions from a set filtered by difficulty
        
        Args:
            set_name: Name of the question set
            difficulty: Difficulty level to filter by
            
        Returns:
            List of questions matching the difficulty
        """
        questions = self.get_question_set(set_name)
        return [q for q in questions if q.get('difficulty', '') == difficulty]
    
    def get_questions_by_category(self, set_name: str, category: str) -> List[Dict[str, Any]]:
        """
        Get questions from a set filtered by category
        
        Args:
            set_name: Name of the question set
            category: Category to filter by
            
        Returns:
            List of questions matching the category
        """
        questions = self.get_question_set(set_name)
        return [q for q in questions if q.get('category', '') == category]
    
    def start_task(self, task_type: str, question_set: str, difficulty: str = None, 
                  duration: int = None, num_questions: int = 10) -> Dict[str, Any]:
        """
        Start a new cognitive task
        
        Args:
            task_type: Type of task (quiz, monitoring, etc.)
            question_set: Name of the question set to use
            difficulty: Difficulty level (easy, medium, hard)
            duration: Task duration in seconds
            num_questions: Number of questions to include
            
        Returns:
            Task configuration dictionary
        """
        # Use default values if not specified
        if difficulty is None:
            difficulty = self.default_difficulty
        
        if duration is None:
            duration = self.default_duration
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Set up task start and end times
        self.task_start_time = datetime.now()
        self.task_end_time = self.task_start_time + timedelta(seconds=duration)
        
        # Select questions based on difficulty and question set
        available_questions = self.get_questions_by_difficulty(question_set, difficulty)
        
        # If not enough questions of the specified difficulty, use questions from all difficulties
        if len(available_questions) < num_questions:
            available_questions = self.get_question_set(question_set)
        
        # Randomly select questions up to the requested number
        selected_questions = []
        if available_questions:
            selected_questions = random.sample(
                available_questions, 
                min(num_questions, len(available_questions))
            )
        
        # Create task configuration
        self.current_task = {
            'id': task_id,
            'type': task_type,
            'question_set': question_set,
            'difficulty': difficulty,
            'duration': duration,
            'start_time': self.task_start_time,
            'end_time': self.task_end_time,
            'questions': selected_questions,
            'current_question_index': 0,
            'answers': [],
            'metrics': {
                'correct_answers': 0,
                'incorrect_answers': 0,
                'skipped_questions': 0,
                'average_response_time': 0,
                'total_response_time': 0
            }
        }
        
        # Emit task started signal
        self.task_started.emit(self.current_task)
        
        self.logger.info(f"Started task '{task_id}' with {len(selected_questions)} questions")
        
        return self.current_task
    
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the current task configuration
        
        Returns:
            Current task configuration or None if no task is active
        """
        return self.current_task
    
    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """
        Get the current question
        
        Returns:
            Current question or None if no task is active or all questions have been answered
        """
        if not self.current_task:
            return None
        
        questions = self.current_task.get('questions', [])
        current_index = self.current_task.get('current_question_index', 0)
        
        if current_index < len(questions):
            return questions[current_index]
        
        return None
    
    def answer_question(self, answer: str, response_time: float) -> Dict[str, Any]:
        """
        Submit an answer for the current question
        
        Args:
            answer: The selected answer
            response_time: Time taken to respond in seconds
            
        Returns:
            Result dictionary with correctness and feedback
        """
        if not self.current_task:
            return {'error': 'No active task'}
        
        current_question = self.get_current_question()
        if not current_question:
            return {'error': 'No current question'}
        
        # Check if the answer is correct
        correct_answer = current_question.get('correct_answer', '')
        is_correct = (answer == correct_answer)
        
        # Update metrics
        metrics = self.current_task['metrics']
        if is_correct:
            metrics['correct_answers'] += 1
        else:
            metrics['incorrect_answers'] += 1
        
        metrics['total_response_time'] += response_time
        metrics['average_response_time'] = metrics['total_response_time'] / (
            metrics['correct_answers'] + metrics['incorrect_answers'] + metrics['skipped_questions']
        )
        
        # Record the answer
        answer_record = {
            'question_id': current_question.get('id', ''),
            'question': current_question.get('question', ''),
            'user_answer': answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_task['answers'].append(answer_record)
        
        # Move to the next question
        self.current_task['current_question_index'] += 1
        
        # Check if all questions have been answered
        if self.current_task['current_question_index'] >= len(self.current_task['questions']):
            self.complete_task()
        else:
            # Emit progress signal
            self.task_progress.emit(
                self.current_task['current_question_index'],
                len(self.current_task['questions'])
            )
        
        # Prepare result
        result = {
            'is_correct': is_correct,
            'correct_answer': correct_answer,
            'feedback': 'Correct!' if is_correct else f'Incorrect. The correct answer is: {correct_answer}',
            'response_time': response_time,
            'next_question': self.get_current_question()
        }
        
        return result
    
    def skip_question(self) -> Dict[str, Any]:
        """
        Skip the current question
        
        Returns:
            Result dictionary with the skipped question and the next question
        """
        if not self.current_task:
            return {'error': 'No active task'}
        
        current_question = self.get_current_question()
        if not current_question:
            return {'error': 'No current question'}
        
        # Update metrics
        self.current_task['metrics']['skipped_questions'] += 1
        
        # Record the skipped question
        skip_record = {
            'question_id': current_question.get('id', ''),
            'question': current_question.get('question', ''),
            'user_answer': 'SKIPPED',
            'correct_answer': current_question.get('correct_answer', ''),
            'is_correct': False,
            'response_time': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_task['answers'].append(skip_record)
        
        # Move to the next question
        self.current_task['current_question_index'] += 1
        
        # Check if all questions have been answered
        if self.current_task['current_question_index'] >= len(self.current_task['questions']):
            self.complete_task()
        else:
            # Emit progress signal
            self.task_progress.emit(
                self.current_task['current_question_index'],
                len(self.current_task['questions'])
            )
        
        # Prepare result
        result = {
            'skipped_question': current_question,
            'correct_answer': current_question.get('correct_answer', ''),
            'next_question': self.get_current_question()
        }
        
        return result
    
    def complete_task(self) -> Dict[str, Any]:
        """
        Complete the current task and calculate final metrics
        
        Returns:
            Task results dictionary
        """
        if not self.current_task:
            return {'error': 'No active task'}
        
        # Calculate final metrics
        metrics = self.current_task['metrics']
        total_questions = len(self.current_task['questions'])
        answered_questions = metrics['correct_answers'] + metrics['incorrect_answers']
        
        # Calculate accuracy
        if answered_questions > 0:
            accuracy = (metrics['correct_answers'] / answered_questions) * 100
        else:
            accuracy = 0
        
        # Calculate completion percentage
        completion_percentage = ((answered_questions + metrics['skipped_questions']) / total_questions) * 100
        
        # Add final metrics
        metrics['accuracy'] = accuracy
        metrics['completion_percentage'] = completion_percentage
        metrics['total_questions'] = total_questions
        metrics['answered_questions'] = answered_questions
        
        # Record end time
        self.current_task['actual_end_time'] = datetime.now()
        
        # Calculate total duration
        start_time = self.current_task['start_time']
        end_time = self.current_task['actual_end_time']
        total_duration = (end_time - start_time).total_seconds()
        metrics['total_duration'] = total_duration
        
        # Emit task completed signal
        self.task_completed.emit(self.current_task)
        
        self.logger.info(f"Completed task '{self.current_task['id']}' with accuracy {accuracy:.2f}%")
        
        # Store the completed task for reference
        completed_task = self.current_task
        
        # Reset current task
        self.current_task = None
        self.task_start_time = None
        self.task_end_time = None
        
        return completed_task
    
    def cancel_task(self) -> Dict[str, Any]:
        """
        Cancel the current task
        
        Returns:
            Cancelled task dictionary
        """
        if not self.current_task:
            return {'error': 'No active task'}
        
        # Record cancellation
        self.current_task['cancelled'] = True
        self.current_task['actual_end_time'] = datetime.now()
        
        # Calculate partial metrics
        metrics = self.current_task['metrics']
        total_questions = len(self.current_task['questions'])
        answered_questions = metrics['correct_answers'] + metrics['incorrect_answers']
        
        # Calculate accuracy for answered questions
        if answered_questions > 0:
            accuracy = (metrics['correct_answers'] / answered_questions) * 100
        else:
            accuracy = 0
        
        # Calculate completion percentage
        completion_percentage = ((answered_questions + metrics['skipped_questions']) / total_questions) * 100
        
        # Add final metrics
        metrics['accuracy'] = accuracy
        metrics['completion_percentage'] = completion_percentage
        metrics['total_questions'] = total_questions
        metrics['answered_questions'] = answered_questions
        
        # Calculate total duration
        start_time = self.current_task['start_time']
        end_time = self.current_task['actual_end_time']
        total_duration = (end_time - start_time).total_seconds()
        metrics['total_duration'] = total_duration
        
        self.logger.info(f"Cancelled task '{self.current_task['id']}' with {completion_percentage:.2f}% completion")
        
        # Store the cancelled task for reference
        cancelled_task = self.current_task
        
        # Reset current task
        self.current_task = None
        self.task_start_time = None
        self.task_end_time = None
        
        return cancelled_task
    
    def is_task_active(self) -> bool:
        """
        Check if a task is currently active
        
        Returns:
            True if a task is active, False otherwise
        """
        return self.current_task is not None
    
    def get_task_progress(self, task_id: str = None) -> Dict[str, Any]:
        """
        Get the current task progress
        
        Args:
            task_id: Optional task ID (not used in current implementation, but kept for API compatibility)
            
        Returns:
            Dictionary with progress information
        """
        if not self.current_task:
            return None
        
        current_index = self.current_task.get('current_question_index', 0)
        total_questions = len(self.current_task.get('questions', []))
        
        # Calculate progress percentage
        progress = current_index / total_questions if total_questions > 0 else 0
        
        # Calculate remaining time
        remaining_time = self.get_task_time_remaining()
        
        return {
            'progress': progress,
            'items_completed': current_index,
            'items_total': total_questions,
            'remaining_time': remaining_time,
            'is_complete': current_index >= total_questions
        }
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """
        Get the current task metrics
        
        Returns:
            Dictionary of task metrics
        """
        if not self.current_task:
            return {}
        
        return self.current_task.get('metrics', {})
    
    def get_task_time_remaining(self) -> int:
        """
        Get the time remaining for the current task in seconds
        
        Returns:
            Time remaining in seconds, or 0 if no task is active
        """
        if not self.current_task or not self.task_end_time:
            return 0
        
        now = datetime.now()
        if now > self.task_end_time:
            return 0
        
        return int((self.task_end_time - now).total_seconds())
    
    def get_task_elapsed_time(self) -> int:
        """
        Get the elapsed time for the current task in seconds
        
        Returns:
            Elapsed time in seconds, or 0 if no task is active
        """
        if not self.current_task or not self.task_start_time:
            return 0
        
        now = datetime.now()
        return int((now - self.task_start_time).total_seconds())
    
    def analyze_task_results(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze task results to extract insights
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            Analysis results dictionary
        """
        if not task_data:
            return {'error': 'No task data provided'}
        
        # Extract basic metrics
        metrics = task_data.get('metrics', {})
        answers = task_data.get('answers', [])
        
        # Initialize results
        results = {
            'task_id': task_data.get('id', ''),
            'question_set': task_data.get('question_set', ''),
            'difficulty': task_data.get('difficulty', ''),
            'basic_metrics': metrics,
            'response_time_analysis': {},
            'question_difficulty_analysis': {},
            'category_performance': {},
            'time_trend_analysis': {}
        }
        
        # Skip further analysis if no answers
        if not answers:
            return results
        
        # Response time analysis
        response_times = [a.get('response_time', 0) for a in answers if a.get('user_answer') != 'SKIPPED']
        if response_times:
            results['response_time_analysis'] = {
                'min': min(response_times),
                'max': max(response_times),
                'mean': sum(response_times) / len(response_times),
                'median': sorted(response_times)[len(response_times) // 2],
                'total': sum(response_times)
            }
        
        # Question difficulty analysis
        questions = task_data.get('questions', [])
        difficulty_performance = {}
        
        for q in questions:
            difficulty = q.get('difficulty', 'unknown')
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'skipped': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            difficulty_performance[difficulty]['total'] += 1
        
        for a in answers:
            q_id = a.get('question_id', '')
            matching_questions = [q for q in questions if q.get('id', '') == q_id]
            
            if not matching_questions:
                continue
            
            q = matching_questions[0]
            difficulty = q.get('difficulty', 'unknown')
            
            if a.get('user_answer') == 'SKIPPED':
                difficulty_performance[difficulty]['skipped'] += 1
            elif a.get('is_correct', False):
                difficulty_performance[difficulty]['correct'] += 1
                difficulty_performance[difficulty]['response_times'].append(a.get('response_time', 0))
            else:
                difficulty_performance[difficulty]['incorrect'] += 1
                difficulty_performance[difficulty]['response_times'].append(a.get('response_time', 0))
        
        # Calculate average response times by difficulty
        for difficulty, data in difficulty_performance.items():
            if data['response_times']:
                data['avg_response_time'] = sum(data['response_times']) / len(data['response_times'])
            del data['response_times']  # Remove the raw data
        
        results['question_difficulty_analysis'] = difficulty_performance
        
        # Category performance analysis
        category_performance = {}
        
        for q in questions:
            category = q.get('category', 'unknown')
            if category not in category_performance:
                category_performance[category] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'skipped': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            category_performance[category]['total'] += 1
        
        for a in answers:
            q_id = a.get('question_id', '')
            matching_questions = [q for q in questions if q.get('id', '') == q_id]
            
            if not matching_questions:
                continue
            
            q = matching_questions[0]
            category = q.get('category', 'unknown')
            
            if a.get('user_answer') == 'SKIPPED':
                category_performance[category]['skipped'] += 1
            elif a.get('is_correct', False):
                category_performance[category]['correct'] += 1
                category_performance[category]['response_times'].append(a.get('response_time', 0))
            else:
                category_performance[category]['incorrect'] += 1
                category_performance[category]['response_times'].append(a.get('response_time', 0))
        
        # Calculate average response times by category
        for category, data in category_performance.items():
            if data['response_times']:
                data['avg_response_time'] = sum(data['response_times']) / len(data['response_times'])
            del data['response_times']  # Remove the raw data
        
        results['category_performance'] = category_performance
        
        # Time trend analysis
        if len(answers) > 1:
            # Sort answers by timestamp
            sorted_answers = sorted(answers, key=lambda a: a.get('timestamp', ''))
            
            # Analyze performance over time
            time_segments = min(5, len(sorted_answers))  # Divide into up to 5 segments
            segment_size = len(sorted_answers) // time_segments
            
            time_trend = []
            for i in range(time_segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < time_segments - 1 else len(sorted_answers)
                segment_answers = sorted_answers[start_idx:end_idx]
                
                correct = sum(1 for a in segment_answers if a.get('is_correct', False))
                total = len(segment_answers)
                avg_time = sum(a.get('response_time', 0) for a in segment_answers) / total if total > 0 else 0
                
                time_trend.append({
                    'segment': i + 1,
                    'correct': correct,
                    'total': total,
                    'accuracy': (correct / total * 100) if total > 0 else 0,
                    'avg_response_time': avg_time
                })
            
            results['time_trend_analysis'] = time_trend
        
        return results
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback function for a specific event type
        
        Args:
            event_type: Type of event to register for
            callback: Callback function to call when the event occurs
        """
        if event_type not in self.task_callbacks:
            self.task_callbacks[event_type] = []
        
        self.task_callbacks[event_type].append(callback)
        
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """Unregister a callback function
        
        Args:
            event_type: Type of event the callback was registered for
            callback: Callback function to unregister
        """
        if event_type in self.task_callbacks:
            if callback in self.task_callbacks[event_type]:
                self.task_callbacks[event_type].remove(callback)
    
    def is_task_active(self) -> bool:
        """
        Check if a task is currently active
        
        Returns:
            True if a task is active, False otherwise
        """
        return self.current_task is not None
    
    def get_task_progress(self, task_id: str = None) -> Dict[str, Any]:
        """
        Get the current task progress
        
        Args:
            task_id: Optional task ID (not used in current implementation, but kept for API compatibility)
            
        Returns:
            Dictionary with progress information
        """
        if not self.current_task:
            return None
        
        current_index = self.current_task.get('current_question_index', 0)
        total_questions = len(self.current_task.get('questions', []))
        
        # Calculate progress percentage
        progress = current_index / total_questions if total_questions > 0 else 0
        
        # Calculate remaining time
        remaining_time = self.get_task_time_remaining()
        
        return {
            'progress': progress,
            'items_completed': current_index,
            'items_total': total_questions,
            'remaining_time': remaining_time,
            'is_complete': current_index >= total_questions
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current or most recent task
        
        Returns:
            Dictionary of performance metrics
        """
        # If there's an active task, return its metrics
        if self.current_task:
            metrics = self.current_task.get('metrics', {})
            
            # Calculate additional metrics if needed
            total_questions = len(self.current_task.get('questions', []))
            answered_questions = metrics.get('correct_answers', 0) + metrics.get('incorrect_answers', 0)
            
            # Calculate accuracy
            if answered_questions > 0:
                accuracy = (metrics.get('correct_answers', 0) / answered_questions) * 100
            else:
                accuracy = 0
            
            # Calculate completion percentage
            completion_rate = ((answered_questions + metrics.get('skipped_questions', 0)) / total_questions) * 100 if total_questions > 0 else 0
            
            return {
                'accuracy': accuracy,
                'completion_rate': completion_rate,
                'avg_response_time': metrics.get('average_response_time', 0),
                'correct_answers': metrics.get('correct_answers', 0),
                'incorrect_answers': metrics.get('incorrect_answers', 0),
                'skipped_questions': metrics.get('skipped_questions', 0),
                'total_questions': total_questions,
                'task_active': True,
                'difficulty': self.current_task.get('difficulty', 'medium')
            }
        
        # If no active task, return empty metrics
        return {
            'accuracy': 0,
            'completion_rate': 0,
            'avg_response_time': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'skipped_questions': 0,
            'total_questions': 0,
            'task_active': False,
            'difficulty': 'none'
        }
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """
        Get the current task metrics
        
        Returns:
            Dictionary of task metrics
        """
        if not self.current_task:
            return {}
        
        return self.current_task.get('metrics', {})
    
    def get_task_time_remaining(self) -> int:
        """
        Get the time remaining for the current task in seconds
        
        Returns:
            Time remaining in seconds, or 0 if no task is active
        """
        if not self.current_task or not self.task_end_time:
            return 0
        
        now = datetime.now()
        if now > self.task_end_time:
            return 0
        
        return int((self.task_end_time - now).total_seconds())
    
    def get_task_elapsed_time(self) -> int:
        """
        Get the elapsed time for the current task in seconds
        
        Returns:
            Elapsed time in seconds, or 0 if no task is active
        """
        if not self.current_task or not self.task_start_time:
            return 0
        
        now = datetime.now()
        return int((now - self.task_start_time).total_seconds())
    
    def analyze_task_results(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze task results to extract insights
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            Analysis results dictionary
        """
        if not task_data:
            return {'error': 'No task data provided'}
        
        # Extract basic metrics
        metrics = task_data.get('metrics', {})
        answers = task_data.get('answers', [])
        
        # Initialize results
        results = {
            'task_id': task_data.get('id', ''),
            'question_set': task_data.get('question_set', ''),
            'difficulty': task_data.get('difficulty', ''),
            'basic_metrics': metrics,
            'response_time_analysis': {},
            'question_difficulty_analysis': {},
            'category_performance': {},
            'time_trend_analysis': {}
        }
        
        # Skip further analysis if no answers
        if not answers:
            return results
        
        # Response time analysis
        response_times = [a.get('response_time', 0) for a in answers if a.get('user_answer') != 'SKIPPED']
        if response_times:
            results['response_time_analysis'] = {
                'min': min(response_times),
                'max': max(response_times),
                'mean': sum(response_times) / len(response_times),
                'median': sorted(response_times)[len(response_times) // 2],
                'total': sum(response_times)
            }
        
        # Question difficulty analysis
        questions = task_data.get('questions', [])
        difficulty_performance = {}
        
        for q in questions:
            difficulty = q.get('difficulty', 'unknown')
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'skipped': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            difficulty_performance[difficulty]['total'] += 1
        
        for a in answers:
            q_id = a.get('question_id', '')
            matching_questions = [q for q in questions if q.get('id', '') == q_id]
            
            if not matching_questions:
                continue
            
            q = matching_questions[0]
            difficulty = q.get('difficulty', 'unknown')
            
            if a.get('user_answer') == 'SKIPPED':
                difficulty_performance[difficulty]['skipped'] += 1
            elif a.get('is_correct', False):
                difficulty_performance[difficulty]['correct'] += 1
                difficulty_performance[difficulty]['response_times'].append(a.get('response_time', 0))
            else:
                difficulty_performance[difficulty]['incorrect'] += 1
                difficulty_performance[difficulty]['response_times'].append(a.get('response_time', 0))
        
        # Calculate average response times by difficulty
        for difficulty, data in difficulty_performance.items():
            if data['response_times']:
                data['avg_response_time'] = sum(data['response_times']) / len(data['response_times'])
            del data['response_times']  # Remove the raw data
        
        results['question_difficulty_analysis'] = difficulty_performance
        
        # Category performance analysis
        category_performance = {}
        
        for q in questions:
            category = q.get('category', 'unknown')
            if category not in category_performance:
                category_performance[category] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'skipped': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            category_performance[category]['total'] += 1
        
        for a in answers:
            q_id = a.get('question_id', '')
            matching_questions = [q for q in questions if q.get('id', '') == q_id]
            
            if not matching_questions:
                continue
            
            q = matching_questions[0]
            category = q.get('category', 'unknown')
            
            if a.get('user_answer') == 'SKIPPED':
                category_performance[category]['skipped'] += 1
            elif a.get('is_correct', False):
                category_performance[category]['correct'] += 1
                category_performance[category]['response_times'].append(a.get('response_time', 0))
            else:
                category_performance[category]['incorrect'] += 1
                category_performance[category]['response_times'].append(a.get('response_time', 0))
        
        # Calculate average response times by category
        for category, data in category_performance.items():
            if data['response_times']:
                data['avg_response_time'] = sum(data['response_times']) / len(data['response_times'])
            del data['response_times']  # Remove the raw data
        
        results['category_performance'] = category_performance
        
        # Time trend analysis
        if len(answers) > 1:
            # Sort answers by timestamp
            sorted_answers = sorted(answers, key=lambda a: a.get('timestamp', ''))
            
            # Analyze performance over time
            time_segments = min(5, len(sorted_answers))  # Divide into up to 5 segments
            segment_size = len(sorted_answers) // time_segments
            
            time_trend = []
            for i in range(time_segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < time_segments - 1 else len(sorted_answers)
                segment_answers = sorted_answers[start_idx:end_idx]
                
                correct = sum(1 for a in segment_answers if a.get('is_correct', False))
                total = len(segment_answers)
                avg_time = sum(a.get('response_time', 0) for a in segment_answers) / total if total > 0 else 0
                
                time_trend.append({
                    'segment': i + 1,
                    'correct': correct,
                    'total': total,
                    'accuracy': (correct / total * 100) if total > 0 else 0,
                    'avg_response_time': avg_time
                })
            
            results['time_trend_analysis'] = time_trend
        
        return results
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback function for a specific event type
        
        Args:
            event_type: Type of event to register for
            callback: Callback function to call when the event occurs
        """
        if event_type not in self.task_callbacks:
            self.task_callbacks[event_type] = []
        
        self.task_callbacks[event_type].append(callback)
        
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """Unregister a callback function
        
        Args:
            event_type: Type of event the callback was registered for
            callback: Callback function to unregister
        """
        if event_type in self.task_callbacks:
            if callback in self.task_callbacks[event_type]:
                self.task_callbacks[event_type].remove(callback)