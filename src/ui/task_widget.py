#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task Widget for NEXARIS Cognitive Load Estimator

This module provides the task simulation widget for presenting cognitive tasks
to the user and collecting responses.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QSplitter, QFrame,
    QMessageBox, QFileDialog, QStackedWidget, QRadioButton,
    QButtonGroup, QTextEdit, QLineEdit, QProgressBar, QSpacerItem,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QIcon

# Import utilities
from ..utils.logging_utils import get_logger


class QuestionWidget(QWidget):
    """
    Widget for displaying and answering questions
    """
    answer_submitted = pyqtSignal(str, str, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.question_id = ""
        self.start_time = 0
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """
        Set up the user interface
        """
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # Create question label
        self.question_label = QLabel("")
        self.question_label.setWordWrap(True)
        self.question_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        self.question_label.setFont(font)
        self.main_layout.addWidget(self.question_label)
        
        # Create answer options layout
        self.options_layout = QVBoxLayout()
        self.options_layout.setContentsMargins(50, 10, 50, 10)
        self.options_layout.setSpacing(10)
        
        # Create option radio buttons
        self.option_group = QButtonGroup(self)
        self.option_buttons = []
        
        for i in range(4):  # Default to 4 options
            option_button = QRadioButton("")
            option_button.setVisible(False)
            font = QFont()
            font.setPointSize(12)
            option_button.setFont(font)
            self.option_group.addButton(option_button, i)
            self.option_buttons.append(option_button)
            self.options_layout.addWidget(option_button)
        
        self.main_layout.addLayout(self.options_layout)
        
        # Create text input for free-form answers
        self.text_answer = QLineEdit()
        self.text_answer.setPlaceholderText("Enter your answer here...")
        self.text_answer.setVisible(False)
        font = QFont()
        font.setPointSize(12)
        self.text_answer.setFont(font)
        self.main_layout.addWidget(self.text_answer)
        
        # Create submit button
        self.submit_button = QPushButton("Submit Answer")
        self.submit_button.setMinimumHeight(40)
        font = QFont()
        font.setPointSize(12)
        self.submit_button.setFont(font)
        self.submit_button.clicked.connect(self.on_submit)
        self.main_layout.addWidget(self.submit_button)
        
        # Add spacer
        self.main_layout.addStretch()
    
    def set_question(self, question_id: str, question_text: str, options: List[str] = None) -> None:
        """
        Set the current question
        
        Args:
            question_id: Question ID
            question_text: Question text
            options: Answer options (None for free-form answer)
        """
        self.question_id = question_id
        self.question_label.setText(question_text)
        self.start_time = time.time()
        
        # Reset previous answers
        self.option_group.setExclusive(True)
        for button in self.option_buttons:
            button.setChecked(False)
            button.setVisible(False)
        
        self.text_answer.clear()
        
        # Set up answer options
        if options:
            # Multiple choice question
            self.text_answer.setVisible(False)
            
            for i, option in enumerate(options):
                if i < len(self.option_buttons):
                    self.option_buttons[i].setText(option)
                    self.option_buttons[i].setVisible(True)
        else:
            # Free-form question
            self.text_answer.setVisible(True)
    
    def get_selected_answer(self) -> str:
        """
        Get the selected answer
        
        Returns:
            Selected answer text or empty string if none selected
        """
        # Check for multiple choice answer
        selected_button = self.option_group.checkedButton()
        if selected_button:
            return selected_button.text()
        
        # Check for text answer
        if self.text_answer.isVisible():
            return self.text_answer.text()
        
        return ""
    
    @pyqtSlot()
    def on_submit(self) -> None:
        """
        Handle submit button click
        """
        answer = self.get_selected_answer()
        if not answer:
            QMessageBox.warning(self, "No Answer", "Please select or enter an answer.")
            return
        
        # Calculate response time
        response_time = time.time() - self.start_time
        
        # Emit answer submitted signal
        self.answer_submitted.emit(self.question_id, answer, response_time)


class AlertWidget(QWidget):
    """
    Widget for displaying and responding to simulated alerts
    """
    alert_action = pyqtSignal(str, str, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.alert_id = ""
        self.start_time = 0
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """
        Set up the user interface
        """
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # Create alert header
        self.header_layout = QHBoxLayout()
        
        # Create alert icon
        self.alert_icon_label = QLabel()
        self.alert_icon_label.setMinimumSize(QSize(48, 48))
        self.alert_icon_label.setMaximumSize(QSize(48, 48))
        self.alert_icon_label.setScaledContents(True)
        self.header_layout.addWidget(self.alert_icon_label)
        
        # Create alert title
        self.alert_title_label = QLabel("")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.alert_title_label.setFont(font)
        self.header_layout.addWidget(self.alert_title_label)
        
        # Create alert severity
        self.alert_severity_label = QLabel("")
        self.alert_severity_label.setMinimumWidth(100)
        self.alert_severity_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        font = QFont()
        font.setPointSize(12)
        self.alert_severity_label.setFont(font)
        self.header_layout.addWidget(self.alert_severity_label)
        
        self.main_layout.addLayout(self.header_layout)
        
        # Create horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)
        
        # Create alert description
        self.alert_description = QTextEdit()
        self.alert_description.setReadOnly(True)
        self.alert_description.setMinimumHeight(100)
        font = QFont()
        font.setPointSize(12)
        self.alert_description.setFont(font)
        self.main_layout.addWidget(self.alert_description)
        
        # Create alert details
        self.details_group = QGroupBox("Alert Details")
        self.details_layout = QFormLayout(self.details_group)
        
        self.source_label = QLabel("")
        self.details_layout.addRow("Source:", self.source_label)
        
        self.timestamp_label = QLabel("")
        self.details_layout.addRow("Timestamp:", self.timestamp_label)
        
        self.category_label = QLabel("")
        self.details_layout.addRow("Category:", self.category_label)
        
        self.main_layout.addWidget(self.details_group)
        
        # Create action buttons
        self.actions_group = QGroupBox("Response Actions")
        self.actions_layout = QVBoxLayout(self.actions_group)
        
        # Create action radio buttons
        self.action_group = QButtonGroup(self)
        
        self.action_buttons = {
            "acknowledge": QRadioButton("Acknowledge - Note the alert but take no immediate action"),
            "escalate": QRadioButton("Escalate - Send to senior analyst for review"),
            "investigate": QRadioButton("Investigate - Gather more information about this alert"),
            "resolve": QRadioButton("Resolve - Mark as resolved (false positive)"),
            "respond": QRadioButton("Respond - Initiate incident response procedure")
        }
        
        for action_id, button in self.action_buttons.items():
            self.action_group.addButton(button)
            self.actions_layout.addWidget(button)
        
        self.main_layout.addWidget(self.actions_group)
        
        # Create notes field
        self.notes_group = QGroupBox("Notes")
        self.notes_layout = QVBoxLayout(self.notes_group)
        
        self.notes_field = QTextEdit()
        self.notes_field.setPlaceholderText("Enter any notes about this alert...")
        self.notes_field.setMaximumHeight(80)
        self.notes_layout.addWidget(self.notes_field)
        
        self.main_layout.addWidget(self.notes_group)
        
        # Create submit button
        self.submit_button = QPushButton("Submit Response")
        self.submit_button.setMinimumHeight(40)
        font = QFont()
        font.setPointSize(12)
        self.submit_button.setFont(font)
        self.submit_button.clicked.connect(self.on_submit)
        self.main_layout.addWidget(self.submit_button)
    
    def set_alert(self, alert_id: str, alert_data: Dict[str, Any]) -> None:
        """
        Set the current alert
        
        Args:
            alert_id: Alert ID
            alert_data: Alert data dictionary
        """
        self.alert_id = alert_id
        self.start_time = time.time()
        
        # Set alert title
        self.alert_title_label.setText(alert_data.get('title', 'Unknown Alert'))
        
        # Set alert severity and color
        severity = alert_data.get('severity', 'medium').lower()
        self.alert_severity_label.setText(severity.upper())
        
        if severity == 'critical':
            self.alert_severity_label.setStyleSheet("color: red; font-weight: bold;")
        elif severity == 'high':
            self.alert_severity_label.setStyleSheet("color: orange; font-weight: bold;")
        elif severity == 'medium':
            self.alert_severity_label.setStyleSheet("color: yellow;")
        elif severity == 'low':
            self.alert_severity_label.setStyleSheet("color: green;")
        else:
            self.alert_severity_label.setStyleSheet("")
        
        # Set alert icon
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icons', f"{severity}_alert.png")
        if os.path.exists(icon_path):
            self.alert_icon_label.setPixmap(QPixmap(icon_path))
        else:
            self.alert_icon_label.clear()
        
        # Set alert description
        self.alert_description.setText(alert_data.get('description', ''))
        
        # Set alert details
        self.source_label.setText(alert_data.get('source', 'Unknown'))
        self.timestamp_label.setText(alert_data.get('timestamp', 'Unknown'))
        self.category_label.setText(alert_data.get('category', 'Unknown'))
        
        # Reset action buttons
        self.action_group.setExclusive(True)
        for button in self.action_buttons.values():
            button.setChecked(False)
        
        # Reset notes field
        self.notes_field.clear()
    
    def get_selected_action(self) -> str:
        """
        Get the selected action
        
        Returns:
            Selected action ID or empty string if none selected
        """
        for action_id, button in self.action_buttons.items():
            if button.isChecked():
                return action_id
        
        return ""
    
    @pyqtSlot()
    def on_submit(self) -> None:
        """
        Handle submit button click
        """
        action = self.get_selected_action()
        if not action:
            QMessageBox.warning(self, "No Action", "Please select a response action.")
            return
        
        # Get notes
        notes = self.notes_field.toPlainText()
        
        # Calculate response time
        response_time = time.time() - self.start_time
        
        # Create action data
        action_data = f"{action}:{notes}"
        
        # Emit alert action signal
        self.alert_action.emit(self.alert_id, action_data, response_time)


class TaskWidget(QWidget):
    """
    Widget for task simulation
    """
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.task_simulator = None
        self.behavior_tracker = None
        
        # Initialize task state
        self.current_task_id = ""
        self.task_running = False
        self.task_start_time = 0
        self.task_end_time = 0
        self.task_items_completed = 0
        self.task_items_total = 0
        
        # Create layout
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """
        Set up the user interface
        """
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create task configuration section
        self.config_group = QGroupBox("Task Configuration")
        self.config_layout = QGridLayout(self.config_group)
        
        # Task type selection
        self.config_layout.addWidget(QLabel("Task Type:"), 0, 0)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Question Set", "Alert Simulation"])
        self.config_layout.addWidget(self.task_type_combo, 0, 1)
        
        # Question set selection
        self.config_layout.addWidget(QLabel("Question Set:"), 1, 0)
        self.question_set_combo = QComboBox()
        self.question_set_combo.addItems(["General Knowledge", "Cybersecurity", "Alert Triage"])
        self.config_layout.addWidget(self.question_set_combo, 1, 1)
        
        # Difficulty selection
        self.config_layout.addWidget(QLabel("Difficulty:"), 2, 0)
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["Easy", "Medium", "Hard"])
        self.config_layout.addWidget(self.difficulty_combo, 2, 1)
        
        # Time limit selection
        self.config_layout.addWidget(QLabel("Time Limit (seconds):"), 3, 0)
        self.time_limit_spin = QSpinBox()
        self.time_limit_spin.setRange(30, 600)
        self.time_limit_spin.setSingleStep(30)
        self.time_limit_spin.setValue(180)
        self.config_layout.addWidget(self.time_limit_spin, 3, 1)
        
        # Number of items selection
        self.config_layout.addWidget(QLabel("Number of Items:"), 4, 0)
        self.item_count_spin = QSpinBox()
        self.item_count_spin.setRange(1, 20)
        self.item_count_spin.setValue(5)
        self.config_layout.addWidget(self.item_count_spin, 4, 1)
        
        # Add configuration group to main layout
        self.main_layout.addWidget(self.config_group)
        
        # Create task control section
        self.control_layout = QHBoxLayout()
        
        # Start button
        self.start_button = QPushButton("Start Task")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_task)
        self.control_layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop Task")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_task)
        self.control_layout.addWidget(self.stop_button)
        
        self.main_layout.addLayout(self.control_layout)
        
        # Create progress section
        self.progress_group = QGroupBox("Task Progress")
        self.progress_layout = QVBoxLayout(self.progress_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_layout.addWidget(self.progress_bar)
        
        # Progress info layout
        self.progress_info_layout = QHBoxLayout()
        
        # Time remaining label
        self.time_label = QLabel("Time Remaining: 00:00")
        self.progress_info_layout.addWidget(self.time_label)
        
        # Items completed label
        self.items_label = QLabel("Items: 0/0")
        self.progress_info_layout.addWidget(self.items_label)
        
        self.progress_layout.addLayout(self.progress_info_layout)
        
        self.main_layout.addWidget(self.progress_group)
        
        # Create task content section
        self.content_stack = QStackedWidget()
        
        # Create question widget
        self.question_widget = QuestionWidget()
        self.question_widget.answer_submitted.connect(self.on_answer_submitted)
        self.content_stack.addWidget(self.question_widget)
        
        # Create alert widget
        self.alert_widget = AlertWidget()
        self.alert_widget.alert_action.connect(self.on_alert_action)
        self.content_stack.addWidget(self.alert_widget)
        
        # Create empty widget for when no task is running
        self.empty_widget = QWidget()
        self.empty_layout = QVBoxLayout(self.empty_widget)
        
        self.empty_label = QLabel("Configure and start a task to begin.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        self.empty_label.setFont(font)
        self.empty_layout.addWidget(self.empty_label)
        
        self.content_stack.addWidget(self.empty_widget)
        
        # Set default widget
        self.content_stack.setCurrentWidget(self.empty_widget)
        
        self.main_layout.addWidget(self.content_stack, 1)  # Give it stretch factor
        
        # Create timer for updating progress
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_progress)
        self.update_timer.setInterval(100)  # 10 Hz
    
    def set_components(self, task_simulator, behavior_tracker) -> None:
        """
        Set core components
        
        Args:
            task_simulator: Task simulator instance
            behavior_tracker: Behavior tracker instance
        """
        self.task_simulator = task_simulator
        self.behavior_tracker = behavior_tracker
        
        # Update question sets
        self.update_question_sets()
    
    def update_question_sets(self) -> None:
        """
        Update available question sets from task simulator
        """
        if not self.task_simulator:
            return
        
        # Get available question sets
        question_sets = self.task_simulator.get_available_question_sets()
        
        # Update combo box
        self.question_set_combo.clear()
        self.question_set_combo.addItems(question_sets)
    
    def update_ui(self) -> None:
        """
        Update UI components
        """
        if not self.task_simulator:
            return
        
        # Update based on task state
        if self.task_running:
            # Update progress
            self.update_progress()
        
    def update_progress(self) -> None:
        """
        Update task progress display
        """
        if not self.task_running or not self.task_simulator:
            return
        
        # Get task progress
        progress = self.task_simulator.get_task_progress(self.current_task_id)
        if not progress:
            return
        
        # Update progress bar
        self.progress_bar.setValue(int(progress['progress'] * 100))
        
        # Update time remaining
        minutes = int(progress['remaining_time'] // 60)
        seconds = int(progress['remaining_time'] % 60)
        self.time_label.setText(f"Time Remaining: {minutes:02d}:{seconds:02d}")
        
        # Update items completed
        self.task_items_completed = progress['items_completed']
        self.task_items_total = progress['items_total']
        self.items_label.setText(f"Items: {self.task_items_completed}/{self.task_items_total}")
        
        # Check if task is complete
        if progress['is_complete']:
            self.on_task_completed()
        
        # Check if time is up
        if progress['remaining_time'] <= 0:
            self.on_time_up()
    
    def start_task(self) -> None:
        """
        Start a new task
        """
        if self.task_running:
            return
        
        if not self.task_simulator:
            QMessageBox.warning(self, "Error", "Task simulator not initialized.")
            return
        
        # Get task configuration
        task_type = self.task_type_combo.currentText()
        question_set = self.question_set_combo.currentText()
        difficulty = self.difficulty_combo.currentText().lower()
        time_limit = self.time_limit_spin.value()
        item_count = self.item_count_spin.value()
        
        # Start task
        try:
            # Pass individual parameters to match the TaskSimulator.start_task method signature
            self.current_task_id = self.task_simulator.start_task(
                task_type=task_type,
                question_set=question_set,
                difficulty=difficulty,
                duration=time_limit,
                num_questions=item_count
            )
            
            self.task_running = True
            self.task_start_time = time.time()
            self.task_items_completed = 0
            self.task_items_total = item_count
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.config_group.setEnabled(False)
            
            # Start update timer
            self.update_timer.start()
            
            # Show appropriate widget based on task type
            if task_type == "Question Set":
                self.content_stack.setCurrentWidget(self.question_widget)
                self.show_next_question()
            elif task_type == "Alert Simulation":
                self.content_stack.setCurrentWidget(self.alert_widget)
                self.show_next_alert()
            
            self.logger.info(f"Task started: {self.current_task_id}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start task: {str(e)}")
            self.logger.error(f"Failed to start task: {str(e)}")
    
    def stop_task(self) -> None:
        """
        Stop the current task
        """
        if not self.task_running:
            return
        
        # Confirm stop
        reply = QMessageBox.question(
            self, "Stop Task",
            "Are you sure you want to stop the current task?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Stop task
        self.end_task()
    
    def end_task(self) -> None:
        """
        End the current task and clean up
        """
        if not self.task_running:
            return
        
        # Stop update timer
        self.update_timer.stop()
        
        # End task in simulator
        if self.task_simulator and self.current_task_id:
            self.task_simulator.cancel_task()
        
        # Update UI
        self.task_running = False
        self.task_end_time = time.time()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.config_group.setEnabled(True)
        self.content_stack.setCurrentWidget(self.empty_widget)
        
        self.logger.info(f"Task ended: {self.current_task_id}")
        self.current_task_id = ""
    
    def show_next_question(self) -> None:
        """
        Show the next question in the task
        """
        if not self.task_running or not self.task_simulator:
            return
        
        # Get next question
        question = self.task_simulator.get_next_item(self.current_task_id)
        if not question:
            # No more questions, end task
            self.on_task_completed()
            return
        
        # Display question
        self.question_widget.set_question(
            question['id'],
            question['text'],
            question.get('options')
        )
    
    def show_next_alert(self) -> None:
        """
        Show the next alert in the task
        """
        if not self.task_running or not self.task_simulator:
            return
        
        # Get next alert
        alert = self.task_simulator.get_next_item(self.current_task_id)
        if not alert:
            # No more alerts, end task
            self.on_task_completed()
            return
        
        # Display alert
        self.alert_widget.set_alert(alert['id'], alert)
    
    @pyqtSlot(str, str, float)
    def on_answer_submitted(self, question_id: str, answer: str, response_time: float) -> None:
        """
        Handle answer submitted signal
        
        Args:
            question_id: Question ID
            answer: Submitted answer
            response_time: Response time in seconds
        """
        if not self.task_running or not self.task_simulator:
            return
        
        # Record answer
        self.task_simulator.record_answer(self.current_task_id, question_id, answer, response_time)
        
        # Show next question
        self.show_next_question()
    
    @pyqtSlot(str, str, float)
    def on_alert_action(self, alert_id: str, action: str, response_time: float) -> None:
        """
        Handle alert action signal
        
        Args:
            alert_id: Alert ID
            action: Selected action
            response_time: Response time in seconds
        """
        if not self.task_running or not self.task_simulator:
            return
        
        # Record action
        self.task_simulator.record_alert_action(self.current_task_id, alert_id, action, response_time)
        
        # Show next alert
        self.show_next_alert()
    
    def on_task_completed(self) -> None:
        """
        Handle task completion
        """
        if not self.task_running:
            return
        
        # Get task results
        results = self.task_simulator.get_task_results(self.current_task_id)
        
        # End task
        self.end_task()
        
        # Show results
        self.show_results(results)
    
    def on_time_up(self) -> None:
        """
        Handle time up event
        """
        if not self.task_running:
            return
        
        # Show message
        QMessageBox.information(self, "Time Up", "The time limit for this task has been reached.")
        
        # Get task results
        results = self.task_simulator.get_task_results(self.current_task_id)
        
        # End task
        self.end_task()
        
        # Show results
        self.show_results(results)
    
    def show_results(self, results: Dict[str, Any]) -> None:
        """
        Show task results
        
        Args:
            results: Task results dictionary
        """
        # Create message
        message = f"<h2>Task Results</h2>"
        message += f"<p><b>Accuracy:</b> {results.get('accuracy', 0.0):.0%}</p>"
        message += f"<p><b>Average Response Time:</b> {results.get('avg_response_time', 0.0):.2f} seconds</p>"
        message += f"<p><b>Completion Rate:</b> {results.get('completion_rate', 0.0):.0%}</p>"
        message += f"<p><b>Items Completed:</b> {self.task_items_completed}/{self.task_items_total}</p>"
        
        # Show dialog
        QMessageBox.information(self, "Task Results", message)