#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Manager for NEXARIS Cognitive Load Estimator

This module handles data storage, retrieval, and management for the application.
It provides interfaces for storing user session data, behavioral metrics,
and cognitive load scores.
"""

import os
import json
import time
import uuid
import logging
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utilities
from ..utils.logging_utils import get_logger


class DataManager:
    """
    Manages data storage and retrieval for the NCLE application
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data manager
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Set up data storage paths
        self.data_dir = Path(config['data']['storage_path'])
        self.sessions_dir = self.data_dir / 'sessions'
        self.models_dir = self.data_dir / 'models'
        self.exports_dir = self.data_dir / 'exports'
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.sessions_dir, self.models_dir, self.exports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize session data
        self.current_session_id = None
        self.current_session_data = {}
        self.current_task_id = None
        
        # Set up backup timer
        self.backup_interval = config['data']['backup_interval']
        self.backup_timer = None
        if self.backup_interval > 0:
            self._start_backup_timer()
        
        self.logger.info("Data Manager initialized")
    
    def _start_backup_timer(self) -> None:
        """
        Start the automatic backup timer
        """
        def backup_task():
            self.backup_current_session()
            # Schedule next backup
            self.backup_timer = threading.Timer(self.backup_interval, backup_task)
            self.backup_timer.daemon = True
            self.backup_timer.start()
        
        # Start initial timer
        self.backup_timer = threading.Timer(self.backup_interval, backup_task)
        self.backup_timer.daemon = True
        self.backup_timer.start()
        self.logger.debug(f"Backup timer started with interval {self.backup_interval} seconds")
    
    def start_new_session(self, user_id: Optional[str] = None) -> str:
        """
        Start a new data collection session
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID for the new session
        """
        # Save current session if exists
        if self.current_session_id is not None:
            self.save_session(self.current_session_id)
        
        # Generate new session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if user_id:
            session_id = f"{user_id}_{timestamp}"
        else:
            session_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Initialize new session data
        self.current_session_id = session_id
        self.current_session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'tasks': [],
            'metrics': {
                'overall_cognitive_load': [],
                'task_performance': [],
                'behavior_metrics': []
            },
            'settings': {
                'tracking': self.config['tracking'].copy(),
                'scoring': self.config['scoring'].copy()
            }
        }
        
        self.logger.info(f"New session started: {session_id}")
        return session_id
    
    def end_current_session(self) -> Optional[str]:
        """
        End the current session and save data
        
        Returns:
            Session ID of the ended session, or None if no active session
        """
        if self.current_session_id is None:
            self.logger.warning("No active session to end")
            return None
        
        # Update session end time
        self.current_session_data['end_time'] = datetime.now().isoformat()
        
        # Save session data
        session_id = self.current_session_id
        self.save_session(session_id)
        
        # Clear current session
        self.current_session_id = None
        self.current_session_data = {}
        self.current_task_id = None
        
        self.logger.info(f"Session ended: {session_id}")
        return session_id
    
    def end_session(self, session_id: Optional[str] = None) -> Optional[str]:
        """
        End a session and save data
        
        Args:
            session_id: Session ID to end, or current session if None
            
        Returns:
            Session ID of the ended session, or None if no active session
        """
        # For backward compatibility, just call end_current_session
        return self.end_current_session()
    
    def start_new_task(self, task_type: str, task_config: Dict[str, Any]) -> str:
        """
        Start a new task within the current session
        
        Args:
            task_type: Type of task (e.g., 'question', 'alert')
            task_config: Task configuration parameters
            
        Returns:
            Task ID for the new task
            
        Raises:
            RuntimeError: If no active session exists
        """
        if self.current_session_id is None:
            self.logger.error("Cannot start task: No active session")
            raise RuntimeError("No active session. Call start_new_session() first.")
        
        # Generate task ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"task_{timestamp}_{uuid.uuid4().hex[:6]}"
        
        # Create task data structure
        task_data = {
            'task_id': task_id,
            'task_type': task_type,
            'config': task_config,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': None,
            'completed': False,
            'score': None,
            'cognitive_load': [],
            'behavior_data': [],
            'facial_data': [],
            'eeg_data': []
        }
        
        # Add to session data
        self.current_session_data['tasks'].append(task_data)
        self.current_task_id = task_id
        
        self.logger.info(f"New task started: {task_id} (type: {task_type})")
        return task_id
    
    def end_current_task(self, completed: bool = True, score: Optional[float] = None) -> Optional[str]:
        """
        End the current task and update task data
        
        Args:
            completed: Whether the task was completed successfully
            score: Optional performance score for the task
            
        Returns:
            Task ID of the ended task, or None if no active task
        """
        if self.current_task_id is None or self.current_session_id is None:
            self.logger.warning("No active task to end")
            return None
        
        # Find the current task in session data
        for task in self.current_session_data['tasks']:
            if task['task_id'] == self.current_task_id:
                # Update task data
                end_time = datetime.now()
                task['end_time'] = end_time.isoformat()
                
                # Calculate duration in seconds
                start_time = datetime.fromisoformat(task['start_time'])
                duration = (end_time - start_time).total_seconds()
                task['duration'] = duration
                
                task['completed'] = completed
                if score is not None:
                    task['score'] = score
                
                # Calculate average cognitive load for this task
                if task['cognitive_load']:
                    avg_load = sum(cl['value'] for cl in task['cognitive_load']) / len(task['cognitive_load'])
                    task['avg_cognitive_load'] = avg_load
                
                break
        
        task_id = self.current_task_id
        self.current_task_id = None
        
        # Save session data (backup)
        self.backup_current_session()
        
        self.logger.info(f"Task ended: {task_id} (completed: {completed})")
        return task_id
    
    def record_behavior_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """
        Record behavioral data for the current task
        
        Args:
            data_type: Type of behavioral data (e.g., 'mouse', 'keyboard')
            data: Behavioral data dictionary
            
        Raises:
            RuntimeError: If no active task or session exists
        """
        if self.current_task_id is None or self.current_session_id is None:
            self.logger.error("Cannot record behavior data: No active task")
            raise RuntimeError("No active task. Call start_new_task() first.")
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Add data type
        data['data_type'] = data_type
        
        # Find the current task and add behavior data
        for task in self.current_session_data['tasks']:
            if task['task_id'] == self.current_task_id:
                task['behavior_data'].append(data)
                break
    
    def record_cognitive_load(self, value: float, source: str = 'algorithm') -> None:
        """
        Record a cognitive load measurement for the current task
        
        Args:
            value: Cognitive load value (0-100)
            source: Source of the measurement (e.g., 'algorithm', 'ml_model', 'eeg')
            
        Raises:
            RuntimeError: If no active task or session exists
            ValueError: If value is outside the valid range
        """
        if self.current_task_id is None or self.current_session_id is None:
            self.logger.error("Cannot record cognitive load: No active task")
            raise RuntimeError("No active task. Call start_new_task() first.")
        
        # Validate value
        if not 0 <= value <= 100:
            self.logger.warning(f"Cognitive load value {value} outside valid range (0-100), clamping")
            value = max(0, min(100, value))
        
        # Create measurement record
        measurement = {
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'source': source
        }
        
        # Add to task data
        for task in self.current_session_data['tasks']:
            if task['task_id'] == self.current_task_id:
                task['cognitive_load'].append(measurement)
                break
        
        # Add to overall session metrics
        self.current_session_data['metrics']['overall_cognitive_load'].append(measurement)
    
    def record_facial_data(self, emotions: Dict[str, float]) -> None:
        """
        Record facial emotion data for the current task
        
        Args:
            emotions: Dictionary of emotion scores (e.g., {'happy': 0.7, 'frustrated': 0.2})
            
        Raises:
            RuntimeError: If no active task or session exists
        """
        if self.current_task_id is None or self.current_session_id is None:
            self.logger.error("Cannot record facial data: No active task")
            raise RuntimeError("No active task. Call start_new_task() first.")
        
        # Create facial data record
        facial_data = {
            'timestamp': datetime.now().isoformat(),
            'emotions': emotions
        }
        
        # Add to task data
        for task in self.current_session_data['tasks']:
            if task['task_id'] == self.current_task_id:
                task['facial_data'].append(facial_data)
                break
    
    def record_eeg_data(self, channels: Dict[str, float]) -> None:
        """
        Record EEG data for the current task
        
        Args:
            channels: Dictionary of EEG channel values
            
        Raises:
            RuntimeError: If no active task or session exists
        """
        if self.current_task_id is None or self.current_session_id is None:
            self.logger.error("Cannot record EEG data: No active task")
            raise RuntimeError("No active task. Call start_new_task() first.")
        
        # Create EEG data record
        eeg_data = {
            'timestamp': datetime.now().isoformat(),
            'channels': channels
        }
        
        # Add to task data
        for task in self.current_session_data['tasks']:
            if task['task_id'] == self.current_task_id:
                task['eeg_data'].append(eeg_data)
                break
    
    def save_session(self, session_id: Optional[str] = None) -> bool:
        """
        Save session data to disk
        
        Args:
            session_id: Session ID to save, or current session if None
            
        Returns:
            True if successful, False otherwise
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            self.logger.error("Cannot save session: No session ID provided")
            return False
        
        # Determine which data to save
        if session_id == self.current_session_id:
            session_data = self.current_session_data
        else:
            # Load session data from disk
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                self.logger.error(f"Cannot save session: Session file not found: {session_file}")
                return False
            
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading session data: {e}")
                return False
        
        # Save to disk
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Session data saved to {session_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving session data: {e}")
            return False
    
    def backup_current_session(self) -> bool:
        """
        Create a backup of the current session data
        
        Returns:
            True if successful, False otherwise
        """
        if self.current_session_id is None:
            return False
        
        return self.save_session(self.current_session_id)
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from disk
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Session data dictionary, or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            self.logger.error(f"Session file not found: {session_file}")
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self.logger.info(f"Session data loaded from {session_file}")
            return session_data
        
        except Exception as e:
            self.logger.error(f"Error loading session data: {e}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions with basic metadata
        
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob('*.json'):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                # Extract basic metadata
                session_meta = {
                    'session_id': data.get('session_id', session_file.stem),
                    'user_id': data.get('user_id'),
                    'start_time': data.get('start_time'),
                    'end_time': data.get('end_time'),
                    'task_count': len(data.get('tasks', [])),
                    'file_path': str(session_file)
                }
                
                sessions.append(session_meta)
            
            except Exception as e:
                self.logger.error(f"Error reading session file {session_file}: {e}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return sessions
    
    def export_session_to_csv(self, session_id: str) -> Optional[str]:
        """
        Export session data to CSV format
        
        Args:
            session_id: Session ID to export
            
        Returns:
            Path to the exported CSV file, or None if export failed
        """
        # Load session data
        session_data = self.load_session(session_id)
        if session_data is None:
            return None
        
        # Create export directory if it doesn't exist
        export_dir = self.exports_dir / session_id
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export cognitive load data
            cl_data = []
            for task in session_data.get('tasks', []):
                task_id = task.get('task_id', 'unknown')
                for cl in task.get('cognitive_load', []):
                    cl_data.append({
                        'timestamp': cl.get('timestamp'),
                        'task_id': task_id,
                        'value': cl.get('value'),
                        'source': cl.get('source')
                    })
            
            if cl_data:
                cl_df = pd.DataFrame(cl_data)
                cl_file = export_dir / 'cognitive_load.csv'
                cl_df.to_csv(cl_file, index=False)
            
            # Export behavior data
            behavior_data = []
            for task in session_data.get('tasks', []):
                task_id = task.get('task_id', 'unknown')
                for bd in task.get('behavior_data', []):
                    bd_row = {'task_id': task_id, 'timestamp': bd.get('timestamp'), 'data_type': bd.get('data_type')}
                    # Add all other keys from the behavior data
                    for k, v in bd.items():
                        if k not in ['timestamp', 'data_type']:
                            bd_row[k] = v
                    behavior_data.append(bd_row)
            
            if behavior_data:
                bd_df = pd.DataFrame(behavior_data)
                bd_file = export_dir / 'behavior_data.csv'
                bd_df.to_csv(bd_file, index=False)
            
            # Export facial data if available
            facial_data = []
            for task in session_data.get('tasks', []):
                task_id = task.get('task_id', 'unknown')
                for fd in task.get('facial_data', []):
                    fd_row = {'task_id': task_id, 'timestamp': fd.get('timestamp')}
                    # Add emotion scores
                    for emotion, score in fd.get('emotions', {}).items():
                        fd_row[f'emotion_{emotion}'] = score
                    facial_data.append(fd_row)
            
            if facial_data:
                fd_df = pd.DataFrame(facial_data)
                fd_file = export_dir / 'facial_data.csv'
                fd_df.to_csv(fd_file, index=False)
            
            # Export task summary
            task_summary = []
            for task in session_data.get('tasks', []):
                task_row = {
                    'task_id': task.get('task_id'),
                    'task_type': task.get('task_type'),
                    'start_time': task.get('start_time'),
                    'end_time': task.get('end_time'),
                    'duration': task.get('duration'),
                    'completed': task.get('completed'),
                    'score': task.get('score'),
                    'avg_cognitive_load': task.get('avg_cognitive_load')
                }
                task_summary.append(task_row)
            
            if task_summary:
                ts_df = pd.DataFrame(task_summary)
                ts_file = export_dir / 'task_summary.csv'
                ts_df.to_csv(ts_file, index=False)
            
            # Create a README file with export information
            readme_file = export_dir / 'README.txt'
            with open(readme_file, 'w') as f:
                f.write(f"NEXARIS Cognitive Load Estimator - Session Export\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Export Date: {datetime.now().isoformat()}\n\n")
                f.write(f"Files included in this export:\n")
                for file in export_dir.glob('*.csv'):
                    f.write(f"- {file.name}\n")
            
            self.logger.info(f"Session data exported to {export_dir}")
            return str(export_dir)
        
        except Exception as e:
            self.logger.error(f"Error exporting session data: {e}")
            return None
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a summary of session data
        
        Args:
            session_id: Session ID to summarize, or current session if None
            
        Returns:
            Dictionary with session summary statistics
        """
        # Determine which data to summarize
        if session_id is None or (self.current_session_id is not None and session_id == self.current_session_id):
            session_data = self.current_session_data
        else:
            session_data = self.load_session(session_id)
        
        if not session_data:
            return {}
        
        # Calculate summary statistics
        summary = {
            'session_id': session_data.get('session_id'),
            'user_id': session_data.get('user_id'),
            'start_time': session_data.get('start_time'),
            'end_time': session_data.get('end_time'),
            'task_count': len(session_data.get('tasks', [])),
            'completed_tasks': sum(1 for t in session_data.get('tasks', []) if t.get('completed', False)),
            'total_duration': sum(t.get('duration', 0) or 0 for t in session_data.get('tasks', [])),
            'cognitive_load': {
                'min': None,
                'max': None,
                'avg': None,
                'std': None
            }
        }
        
        # Calculate cognitive load statistics
        cl_values = []
        for task in session_data.get('tasks', []):
            cl_values.extend([cl.get('value', 0) for cl in task.get('cognitive_load', [])])
        
        if cl_values:
            summary['cognitive_load']['min'] = min(cl_values)
            summary['cognitive_load']['max'] = max(cl_values)
            summary['cognitive_load']['avg'] = sum(cl_values) / len(cl_values)
            summary['cognitive_load']['std'] = np.std(cl_values) if len(cl_values) > 1 else 0
        
        return summary
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the data manager
        """
        # Stop backup timer if running
        if self.backup_timer is not None:
            self.backup_timer.cancel()
            self.backup_timer = None
        
        # Save current session if exists
        if self.current_session_id is not None:
            self.save_session(self.current_session_id)
        
        self.logger.info("Data Manager cleanup completed")
    
    def is_session_active(self) -> bool:
        """
        Check if there is an active session
        
        Returns:
            True if a session is active, False otherwise
        """
        return self.current_session_id is not None
    
    def get_current_session_id(self) -> Optional[str]:
        """
        Get the current session ID
        
        Returns:
            Current session ID or None if no session is active
        """
        return self.current_session_id
    
    def _check_session_active(self) -> bool:
        """
        Check if there is an active session
        
        Returns:
            True if a session is active, False otherwise
        """
        return self.current_session_id is not None