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

# Import utilities
from ..utils.logging_utils import get_logger


class TaskSimulator:
    """
    Simulates cognitive tasks for measuring user performance and cognitive load
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the task simulator
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load task configurations
        self.task_config = config.get('task', {})
        self.default_duration = self.task_config.get('default_duration', 300)  # seconds
        self.difficulty_levels = self.task_config.get('difficulty_levels', ["easy", "medium", "hard"])
        self.default_difficulty = self.task_config.get('default_difficulty', "medium")
        
        # Load question sets
        self.question_sets = {}
        self._load_question_sets()
        
        # Initialize task state
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
                    "id": "gen_001",
                    "question": "What is 25 × 4?",
                    "options": ["75", "100", "125", "150"],
                    "correct_answer": "100",
                    "difficulty": "easy",
                    "category": "math"
                },
                {
                    "id": "gen_002",
                    "question": "Which of these is NOT a primary color?",
                    "options": ["Red", "Blue", "Green", "Yellow"],
                    "correct_answer": "Green",
                    "difficulty": "easy",
                    "category": "general_knowledge"
                },
                {
                    "id": "gen_003",
                    "question": "What is the capital of France?",
                    "options": ["London", "Berlin", "Paris", "Madrid"],
                    "correct_answer": "Paris",
                    "difficulty": "easy",
                    "category": "geography"
                },
                {
                    "id": "gen_004",
                    "question": "Solve for x: 3x + 7 = 22",
                    "options": ["3", "5", "7", "15"],
                    "correct_answer": "5",
                    "difficulty": "medium",
                    "category": "math"
                },
                {
                    "id": "gen_005",
                    "question": "Which planet is known as the Red Planet?",
                    "options": ["Venus", "Mars", "Jupiter", "Saturn"],
                    "correct_answer": "Mars",
                    "difficulty": "easy",
                    "category": "science"
                }
            ]
        
        elif set_name == "cybersecurity":
            default_questions = [
                {
                    "id": "cyber_001",
                    "question": "Which of the following is NOT a common type of cyber attack?",
                    "options": ["Phishing", "SQL Injection", "Quantum Tunneling", "DDoS"],
                    "correct_answer": "Quantum Tunneling",
                    "difficulty": "easy",
                    "category": "attacks"
                },
                {
                    "id": "cyber_002",
                    "question": "What does SSL stand for?",
                    "options": ["Secure Socket Layer", "System Security Level", "Secure System Login", "Standard Security License"],
                    "correct_answer": "Secure Socket Layer",
                    "difficulty": "easy",
                    "category": "protocols"
                },
                {
                    "id": "cyber_003",
                    "question": "Which port is commonly used for HTTPS traffic?",
                    "options": ["21", "22", "443", "8080"],
                    "correct_answer": "443",
                    "difficulty": "medium",
                    "category": "networking"
                },
                {
                    "id": "cyber_004",
                    "question": "Which encryption algorithm is considered broken and should not be used?",
                    "options": ["AES-256", "RSA-2048", "MD5", "SHA-256"],
                    "correct_answer": "MD5",
                    "difficulty": "medium",
                    "category": "cryptography"
                },
                {
                    "id": "cyber_005",
                    "question": "What is the primary purpose of a firewall?",
                    "options": [
                        "Detect viruses", 
                        "Filter network traffic", 
                        "Encrypt data", 
                        "Backup data"
                    ],
                    "correct_answer": "Filter network traffic",
                    "difficulty": "easy",
                    "category": "network_security"
                }
            ]
        
        elif set_name == "alert_triage":
            default_questions = [
                {
                    "id": "alert_001",
                    "question": "You receive an alert showing multiple failed login attempts from different countries for the same user account. What is your first action?",
                    "options": [
                        "Immediately lock the account", 
                        "Call the user to verify activity", 
                        "Check if the user is traveling", 
                        "Ignore as it's probably a false positive"
                    ],
                    "correct_answer": "Check if the user is traveling",
                    "difficulty": "medium",
                    "category": "authentication"
                },
                {
                    "id": "alert_002",
                    "question": "An IDS alert shows a potential SQL injection attempt. The target system is a test server with no production data. How would you prioritize this alert?",
                    "options": ["Critical", "High", "Medium", "Low"],
                    "correct_answer": "Medium",
                    "difficulty": "medium",
                    "category": "web_security"
                },
                {
                    "id": "alert_003",
                    "question": "You receive an alert for unusual outbound traffic from a workstation to an IP address in a foreign country. What should you check first?",
                    "options": [
                        "Block the IP immediately", 
                        "Check IP reputation in threat intelligence", 
                        "Format the workstation", 
                        "Disconnect the network"
                    ],
                    "correct_answer": "Check IP reputation in threat intelligence",
                    "difficulty": "medium",
                    "category": "network_monitoring"
                },
                {
                    "id": "alert_004",
                    "question": "A SIEM alert shows a user accessing sensitive files at 3 AM local time. The user normally works 9-5. What is your first response?",
                    "options": [
                        "Revoke access immediately", 
                        "Check if overtime was approved", 
                        "Call the user regardless of time", 
                        "Wait until morning to investigate"
                    ],
                    "correct_answer": "Check if overtime was approved",
                    "difficulty": "hard",
                    "category": "data_access"
                },
                {
                    "id": "alert_005",
                    "question": "You receive an alert for a potential malware detection on a server. The antivirus quarantined the file. What should you do next?",
                    "options": [
                        "Restore the file to check if it's a false positive", 
                        "Immediately reimage the server", 
                        "Check for other indicators of compromise", 
                        "Delete the quarantined file"
                    ],
                    "correct_answer": "Check for other indicators of compromise",
                    "difficulty": "hard",
                    "category": "malware"
                }
            ]
        
        # Save default questions to file
        try:
            with open(file_path, 'w') as f:
                json.dump(default_questions, f, indent=4)
            
            self.logger.info(f"Created default question set '{set_name}' at {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error creating default question set '{set_name}': {e}")
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback function for task events
        
        Args:
            event_type: Event type (e.g., 'task_start', 'task_end', 'question_answered')
            callback: Callback function to be called when the event occurs
        """
        if event_type not in self.task_callbacks:
            self.task_callbacks[event_type] = []
        
        self.task_callbacks[event_type].append(callback)
        self.logger.debug(f"Registered callback for event '{event_type}'")
    
    def _trigger_callbacks(self, event_type: str, **kwargs) -> None:
        """
        Trigger registered callbacks for an event
        
        Args:
            event_type: Event type
            **kwargs: Additional arguments to pass to callbacks
        """
        if event_type in self.task_callbacks:
            for callback in self.task_callbacks[event_type]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    self.logger.error(f"Error in callback for event '{event_type}': {e}")
    
    def create_task(self, task_type: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new task configuration
        
        Args:
            task_type: Type of task (e.g., 'question_set', 'alert_simulation')
            **kwargs: Additional task parameters
            
        Returns:
            Task configuration dictionary
        """
        # Generate task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Set default parameters
        duration = kwargs.get('duration', self.default_duration)
        difficulty = kwargs.get('difficulty', self.default_difficulty)
        
        # Create base task configuration
        task_config = {
            'task_id': task_id,
            'task_type': task_type,
            'duration': duration,
            'difficulty': difficulty,
            'created_at': datetime.now().isoformat()
        }
        
        # Add task-specific configuration
        if task_type == 'question_set':
            question_set = kwargs.get('question_set', self.task_config.get('default_question_set', 'general'))
            num_questions = kwargs.get('num_questions', 10)
            
            # Validate question set
            if question_set not in self.question_sets:
                self.logger.warning(f"Question set '{question_set}' not found, using default")
                question_set = self.task_config.get('default_question_set', 'general')
            
            # Select questions based on difficulty
            available_questions = self.question_sets.get(question_set, [])
            if difficulty != 'mixed':
                available_questions = [q for q in available_questions if q.get('difficulty') == difficulty]
            
            # If not enough questions of the specified difficulty, include other difficulties
            if len(available_questions) < num_questions:
                self.logger.warning(f"Not enough questions with difficulty '{difficulty}', including other difficulties")
                available_questions = self.question_sets.get(question_set, [])
            
            # Randomly select questions
            selected_questions = random.sample(
                available_questions, 
                min(num_questions, len(available_questions))
            )
            
            # Add to task configuration
            task_config.update({
                'question_set': question_set,
                'questions': selected_questions,
                'num_questions': len(selected_questions),
                'time_per_question': duration / max(1, len(selected_questions))
            })
        
        elif task_type == 'alert_simulation':
            alert_type = kwargs.get('alert_type', 'security_incident')
            num_alerts = kwargs.get('num_alerts', 5)
            
            # Generate simulated alerts
            alerts = self._generate_simulated_alerts(alert_type, num_alerts, difficulty)
            
            # Add to task configuration
            task_config.update({
                'alert_type': alert_type,
                'alerts': alerts,
                'num_alerts': len(alerts),
                'time_per_alert': duration / max(1, len(alerts))
            })
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in task_config:
                task_config[key] = value
        
        self.logger.info(f"Created {task_type} task with ID {task_id}")
        return task_config
    
    def _generate_simulated_alerts(self, alert_type: str, num_alerts: int, difficulty: str) -> List[Dict[str, Any]]:
        """
        Generate simulated security alerts for alert simulation tasks
        
        Args:
            alert_type: Type of alerts to generate
            num_alerts: Number of alerts to generate
            difficulty: Difficulty level
            
        Returns:
            List of simulated alert dictionaries
        """
        alerts = []
        
        # Alert templates based on type and difficulty
        alert_templates = {
            'security_incident': {
                'easy': [
                    {
                        'title': 'Failed Login Attempts',
                        'description': 'Multiple failed login attempts detected for user {user} from IP {ip}.',
                        'severity': 'Medium',
                        'source': 'Authentication System',
                        'indicators': ['Multiple authentication failures', 'Known suspicious IP']
                    },
                    {
                        'title': 'Malware Detected',
                        'description': 'Antivirus detected {malware_type} on host {hostname}.',
                        'severity': 'High',
                        'source': 'Endpoint Protection',
                        'indicators': ['Malicious file detected', 'Known malware signature']
                    }
                ],
                'medium': [
                    {
                        'title': 'Unusual Network Traffic',
                        'description': 'Unusual outbound traffic detected from {hostname} to {destination_ip} on port {port}.',
                        'severity': 'Medium',
                        'source': 'Network IDS',
                        'indicators': ['Unusual destination', 'High volume of traffic', 'Non-business hours']
                    },
                    {
                        'title': 'Potential Data Exfiltration',
                        'description': 'Large file transfer detected from {hostname} to external domain {domain}.',
                        'severity': 'High',
                        'source': 'DLP System',
                        'indicators': ['Large file transfer', 'Sensitive data pattern match', 'Unusual destination']
                    }
                ],
                'hard': [
                    {
                        'title': 'Potential Lateral Movement',
                        'description': 'User {user} accessed multiple systems ({systems}) within a short time period.',
                        'severity': 'High',
                        'source': 'SIEM Correlation',
                        'indicators': ['Access to multiple systems', 'Privileged account usage', 'After hours activity']
                    },
                    {
                        'title': 'Suspicious PowerShell Execution',
                        'description': 'Encoded PowerShell command executed on {hostname} by user {user}.',
                        'severity': 'Critical',
                        'source': 'EDR',
                        'indicators': ['Encoded command', 'PowerShell execution', 'Privilege escalation attempt']
                    }
                ]
            }
        }
        
        # Select templates based on alert type and difficulty
        if alert_type in alert_templates and difficulty in alert_templates[alert_type]:
            templates = alert_templates[alert_type][difficulty]
        else:
            # Fallback to all templates for the alert type
            templates = []
            for diff in alert_templates.get(alert_type, {}):
                templates.extend(alert_templates[alert_type][diff])
        
        # Generate alerts using templates
        for i in range(num_alerts):
            # Select a random template
            template = random.choice(templates)
            
            # Generate random values for placeholders
            users = ['admin', 'jsmith', 'aharris', 'mwilliams', 'dthompson']
            ips = ['192.168.1.100', '10.0.0.15', '172.16.5.10', '8.8.8.8', '203.0.113.42']
            hostnames = ['DESKTOP-A1B2C3', 'SERVER-X1Y2Z3', 'LAPTOP-USER1', 'WORKSTATION-5', 'DC-PRIMARY']
            domains = ['example.com', 'filestore.net', 'datacloud.org', 'transfer.io', 'share-docs.com']
            malware_types = ['Trojan.Generic', 'Ransomware.Cryptolocker', 'Backdoor.Access', 'Spyware.Logger', 'Worm.Spread']
            ports = [22, 80, 443, 445, 3389, 8080]
            systems = ['FileServer', 'DatabaseServer', 'DomainController', 'WebServer', 'EmailServer']
            
            # Create alert from template
            alert = template.copy()
            
            # Replace placeholders with random values
            for key, value in alert.items():
                if isinstance(value, str):
                    value = value.replace('{user}', random.choice(users))
                    value = value.replace('{ip}', random.choice(ips))
                    value = value.replace('{hostname}', random.choice(hostnames))
                    value = value.replace('{destination_ip}', random.choice(ips))
                    value = value.replace('{domain}', random.choice(domains))
                    value = value.replace('{malware_type}', random.choice(malware_types))
                    value = value.replace('{port}', str(random.choice(ports)))
                    value = value.replace('{systems}', ', '.join(random.sample(systems, random.randint(2, 4))))
                    alert[key] = value
            
            # Add unique ID and timestamp
            alert['id'] = f"alert_{uuid.uuid4().hex[:6]}"
            alert['timestamp'] = (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat()
            
            alerts.append(alert)
        
        return alerts
    
    def start_task(self, task_config: Dict[str, Any]) -> bool:
        """
        Start a task with the given configuration
        
        Args:
            task_config: Task configuration dictionary
            
        Returns:
            True if task started successfully, False otherwise
        """
        if self.current_task is not None:
            self.logger.warning("Cannot start task: Another task is already running")
            return False
        
        # Set current task and start time
        self.current_task = task_config
        self.task_start_time = datetime.now()
        self.task_end_time = self.task_start_time + timedelta(seconds=task_config.get('duration', self.default_duration))
        
        # Trigger task start callbacks
        self._trigger_callbacks('task_start', task_config=task_config)
        
        self.logger.info(f"Task started: {task_config.get('task_id')} (type: {task_config.get('task_type')})")
        return True
    
    def end_task(self) -> Optional[Dict[str, Any]]:
        """
        End the current task
        
        Returns:
            Task results dictionary, or None if no task is running
        """
        if self.current_task is None:
            self.logger.warning("Cannot end task: No task is running")
            return None
        
        # Calculate task duration
        end_time = datetime.now()
        duration = (end_time - self.task_start_time).total_seconds()
        
        # Create task results
        task_results = {
            'task_id': self.current_task.get('task_id'),
            'task_type': self.current_task.get('task_type'),
            'start_time': self.task_start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': duration,
            'completed': True
        }
        
        # Trigger task end callbacks
        self._trigger_callbacks('task_end', task_results=task_results)
        
        # Clear current task
        self.current_task = None
        self.task_start_time = None
        self.task_end_time = None
        
        self.logger.info(f"Task ended: {task_results.get('task_id')} (duration: {duration:.2f}s)")
        return task_results
    
    def get_remaining_time(self) -> Optional[float]:
        """
        Get the remaining time for the current task in seconds
        
        Returns:
            Remaining time in seconds, or None if no task is running
        """
        if self.current_task is None or self.task_end_time is None:
            return None
        
        remaining = (self.task_end_time - datetime.now()).total_seconds()
        return max(0, remaining)
    
    def get_task_progress(self) -> Dict[str, Any]:
        """
        Get the progress of the current task
        
        Returns:
            Dictionary with task progress information
        """
        if self.current_task is None:
            return {'running': False}
        
        # Calculate elapsed and remaining time
        elapsed = (datetime.now() - self.task_start_time).total_seconds()
        remaining = self.get_remaining_time() or 0
        total_duration = self.current_task.get('duration', self.default_duration)
        
        # Calculate progress percentage
        progress_pct = min(100, (elapsed / total_duration) * 100) if total_duration > 0 else 0
        
        return {
            'running': True,
            'task_id': self.current_task.get('task_id'),
            'task_type': self.current_task.get('task_type'),
            'elapsed_time': elapsed,
            'remaining_time': remaining,
            'progress_percent': progress_pct
        }
    
    def record_answer(self, question_id: str, answer: str) -> Dict[str, Any]:
        """
        Record a user's answer to a question
        
        Args:
            question_id: ID of the question being answered
            answer: User's answer
            
        Returns:
            Dictionary with answer results
            
        Raises:
            RuntimeError: If no task is running or task is not a question set
        """
        if self.current_task is None:
            raise RuntimeError("No task is running")
        
        if self.current_task.get('task_type') != 'question_set':
            raise RuntimeError(f"Current task is not a question set: {self.current_task.get('task_type')}")
        
        # Find the question
        question = None
        for q in self.current_task.get('questions', []):
            if q.get('id') == question_id:
                question = q
                break
        
        if question is None:
            self.logger.warning(f"Question not found: {question_id}")
            return {'success': False, 'error': 'Question not found'}
        
        # Check if answer is correct
        correct = answer == question.get('correct_answer')
        
        # Calculate response time
        response_time = None
        if 'displayed_at' in question:
            displayed_at = datetime.fromisoformat(question['displayed_at'])
            response_time = (datetime.now() - displayed_at).total_seconds()
        
        # Create answer result
        result = {
            'question_id': question_id,
            'answer': answer,
            'correct': correct,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to question data
        question['answer'] = answer
        question['correct'] = correct
        question['response_time'] = response_time
        question['answered_at'] = result['timestamp']
        
        # Trigger answer callbacks
        self._trigger_callbacks('question_answered', question=question, result=result)
        
        return result
    
    def record_alert_action(self, alert_id: str, action: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Record a user's action for an alert
        
        Args:
            alert_id: ID of the alert
            action: Action taken (e.g., 'escalate', 'dismiss', 'investigate')
            notes: Optional notes about the action
            
        Returns:
            Dictionary with action results
            
        Raises:
            RuntimeError: If no task is running or task is not an alert simulation
        """
        if self.current_task is None:
            raise RuntimeError("No task is running")
        
        if self.current_task.get('task_type') != 'alert_simulation':
            raise RuntimeError(f"Current task is not an alert simulation: {self.current_task.get('task_type')}")
        
        # Find the alert
        alert = None
        for a in self.current_task.get('alerts', []):
            if a.get('id') == alert_id:
                alert = a
                break
        
        if alert is None:
            self.logger.warning(f"Alert not found: {alert_id}")
            return {'success': False, 'error': 'Alert not found'}
        
        # Calculate response time
        response_time = None
        if 'displayed_at' in alert:
            displayed_at = datetime.fromisoformat(alert['displayed_at'])
            response_time = (datetime.now() - displayed_at).total_seconds()
        
        # Create action result
        result = {
            'alert_id': alert_id,
            'action': action,
            'notes': notes,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to alert data
        alert['action'] = action
        alert['notes'] = notes
        alert['response_time'] = response_time
        alert['actioned_at'] = result['timestamp']
        
        # Trigger alert action callbacks
        self._trigger_callbacks('alert_actioned', alert=alert, result=result)
        
        return result
    
    def mark_item_displayed(self, item_id: str, item_type: str) -> None:
        """
        Mark a question or alert as displayed to the user
        
        Args:
            item_id: ID of the item (question or alert)
            item_type: Type of item ('question' or 'alert')
            
        Raises:
            RuntimeError: If no task is running or item type is invalid
        """
        if self.current_task is None:
            raise RuntimeError("No task is running")
        
        if item_type == 'question':
            if self.current_task.get('task_type') != 'question_set':
                raise RuntimeError("Current task is not a question set")
            
            # Find the question and mark as displayed
            for q in self.current_task.get('questions', []):
                if q.get('id') == item_id:
                    q['displayed_at'] = datetime.now().isoformat()
                    break
        
        elif item_type == 'alert':
            if self.current_task.get('task_type') != 'alert_simulation':
                raise RuntimeError("Current task is not an alert simulation")
            
            # Find the alert and mark as displayed
            for a in self.current_task.get('alerts', []):
                if a.get('id') == item_id:
                    a['displayed_at'] = datetime.now().isoformat()
                    break
        
        else:
            raise RuntimeError(f"Invalid item type: {item_type}")
    
    def get_task_results(self) -> Dict[str, Any]:
        """
        Get results for the current task
        
        Returns:
            Dictionary with task results
            
        Raises:
            RuntimeError: If no task is running
        """
        if self.current_task is None:
            raise RuntimeError("No task is running")
        
        # Create base results
        results = {
            'task_id': self.current_task.get('task_id'),
            'task_type': self.current_task.get('task_type'),
            'start_time': self.task_start_time.isoformat() if self.task_start_time else None,
            'elapsed_time': (datetime.now() - self.task_start_time).total_seconds() if self.task_start_time else None
        }
        
        # Add task-specific results
        if self.current_task.get('task_type') == 'question_set':
            # Calculate question statistics
            questions = self.current_task.get('questions', [])
            answered = [q for q in questions if 'answer' in q]
            correct = [q for q in answered if q.get('correct', False)]
            
            results.update({
                'total_questions': len(questions),
                'answered_questions': len(answered),
                'correct_answers': len(correct),
                'accuracy': len(correct) / len(answered) if answered else 0,
                'average_response_time': sum(q.get('response_time', 0) for q in answered) / len(answered) if answered else 0
            })
        
        elif self.current_task.get('task_type') == 'alert_simulation':
            # Calculate alert statistics
            alerts = self.current_task.get('alerts', [])
            actioned = [a for a in alerts if 'action' in a]
            
            # Count actions by type
            action_counts = {}
            for alert in actioned:
                action = alert.get('action', 'unknown')
                action_counts[action] = action_counts.get(action, 0) + 1
            
            results.update({
                'total_alerts': len(alerts),
                'actioned_alerts': len(actioned),
                'action_counts': action_counts,
                'average_response_time': sum(a.get('response_time', 0) for a in actioned) / len(actioned) if actioned else 0
            })
        
        return results