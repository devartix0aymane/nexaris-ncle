"""
NEXARIS Cognitive Load Estimator (NCLE)

Task Simulation Module

Manages the user tasks, including loading, starting, and tracking progress.
"""

import time
import json
import os

class Task:
    """Represents a single task with questions and properties."""
    def __init__(self, task_id, name, description, questions):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.questions = questions
        self.current_question_index = 0
        self.start_time = None
        self.end_time = None
        self.answers = []

    def start(self):
        """Start the task."""
        self.start_time = time.time()
        self.current_question_index = 0
        self.answers = []

    def get_current_question(self):
        """Get the current question, or None if task is finished."""
        if self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    def submit_answer(self, answer):
        """Submit an answer for the current question."""
        if self.get_current_question():
            self.answers.append({
                "question": self.get_current_question(),
                "answer": answer,
                "timestamp": time.time()
            })
            self.current_question_index += 1

    def is_finished(self):
        """Check if the task is finished."""
        return self.current_question_index >= len(self.questions)

    def complete(self):
        """Mark the task as completed."""
        self.end_time = time.time()

    def get_results(self):
        """Get the results of the task."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "answers": self.answers,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else 0
        }

class TaskManager:
    """Manages a collection of tasks and the current active task."""
    def __init__(self, tasks_file_path="../data/tasks.json"):
        self.tasks = {}
        self.active_task = None
        self.tasks_file_path = tasks_file_path
        self._load_tasks()

    def _load_tasks(self):
        """Load tasks from a JSON file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.tasks_file_path), exist_ok=True)
            if os.path.exists(self.tasks_file_path):
                with open(self.tasks_file_path, 'r') as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        self.tasks[task_data['id']] = Task(
                            task_id=task_data['id'],
                            name=task_data['name'],
                            description=task_data['description'],
                            questions=task_data['questions']
                        )
            else:
                print(f"Tasks file not found at {self.tasks_file_path}. Creating a default one.")
                self._create_default_tasks()
        except Exception as e:
            print(f"Error loading tasks: {e}")
            self._create_default_tasks() # Create default if loading fails

    def _create_default_tasks(self):
        """Create a default set of tasks and save them."""
        default_tasks_data = [
            {
                "id": "task1", "name": "Simple Arithmetic", "description": "Solve basic math problems.",
                "questions": [
                    {"text": "What is 2 + 2?", "options": ["3", "4", "5"], "correct_answer": "4"},
                    {"text": "What is 5 * 3?", "options": ["12", "15", "18"], "correct_answer": "15"}
                ]
            }
        ]
        self.tasks = {}
        for task_data in default_tasks_data:
            self.tasks[task_data['id']] = Task(
                task_id=task_data['id'],
                name=task_data['name'],
                description=task_data['description'],
                questions=task_data['questions']
            )
        self.save_tasks() # Save the newly created default tasks

    def save_tasks(self):
        """Save the current tasks to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self.tasks_file_path), exist_ok=True)
            tasks_data_to_save = []
            for task_id, task_obj in self.tasks.items():
                tasks_data_to_save.append({
                    "id": task_obj.task_id,
                    "name": task_obj.name,
                    "description": task_obj.description,
                    "questions": task_obj.questions
                })
            with open(self.tasks_file_path, 'w') as f:
                json.dump(tasks_data_to_save, f, indent=4)
            print(f"Tasks saved to {self.tasks_file_path}")
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def get_task(self, task_id):
        """Get a specific task by its ID."""
        return self.tasks.get(task_id)

    def start_task(self, task_id):
        """Start a specific task."""
        task = self.get_task(task_id)
        if task:
            self.active_task = task
            self.active_task.start()
            print(f"Task '{self.active_task.name}' started.")
            return self.active_task
        print(f"Task with ID '{task_id}' not found.")
        return None

    def get_current_question(self):
        """Get the current question from the active task."""
        if self.active_task:
            return self.active_task.get_current_question()
        return None

    def submit_answer(self, answer):
        """Submit an answer for the current question in the active task."""
        if self.active_task:
            self.active_task.submit_answer(answer)
            if self.active_task.is_finished():
                self.active_task.complete()
                print(f"Task '{self.active_task.name}' completed.")
                # Log results or notify other components
                # print(self.active_task.get_results())

    def get_active_task_results(self):
        """Get results of the currently active task if it's finished."""
        if self.active_task and self.active_task.is_finished():
            return self.active_task.get_results()
        return None

if __name__ == '__main__':
    # Example Usage
    task_manager = TaskManager(tasks_file_path="../../data/tasks.json") # Adjusted path for direct script run

    # List available tasks
    print("Available tasks:")
    for tid, t in task_manager.tasks.items():
        print(f"- {tid}: {t.name}")

    # Start a task
    current_task = task_manager.start_task("task1")

    if current_task:
        while not current_task.is_finished():
            question = current_task.get_current_question()
            if question:
                print(f"\nQuestion: {question['text']}")
                user_ans = input(f"Your answer ({'/'.join(question['options'])}): ")
                task_manager.submit_answer(user_ans)
            else:
                break # Should not happen if is_finished is false

        results = task_manager.get_active_task_results()
        if results:
            print("\nTask Results:")
            print(json.dumps(results, indent=4))