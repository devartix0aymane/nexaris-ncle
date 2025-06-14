"""
NEXARIS Cognitive Load Estimator (NCLE)

Behavior Tracking Module

Logs mouse movements, clicks, and keyboard activity.
"""

import time
import csv
import os
from datetime import datetime

# Placeholder for a GUI library if direct event hooking is needed (e.g., pynput, PyQt)
# For now, we'll simulate or assume data is pushed from a GUI layer.

class BehaviorTracker:
    """Tracks and logs user's mouse and keyboard behavior."""
    def __init__(self, log_directory="../logs/"):
        self.log_directory = log_directory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mouse_log_path = os.path.join(self.log_directory, f"mouse_log_{self.session_id}.csv")
        self.keyboard_log_path = os.path.join(self.log_directory, f"keyboard_log_{self.session_id}.csv")
        self.active = False

        self._setup_logging()

    def _setup_logging(self):
        """Create log directory and initialize CSV log files with headers."""
        os.makedirs(self.log_directory, exist_ok=True)

        # Mouse log
        with open(self.mouse_log_path, 'w', newline='') as f_mouse:
            mouse_writer = csv.writer(f_mouse)
            mouse_writer.writerow(["timestamp", "event_type", "x", "y", "button", "task_id"])

        # Keyboard log
        with open(self.keyboard_log_path, 'w', newline='') as f_keyboard:
            keyboard_writer = csv.writer(f_keyboard)
            keyboard_writer.writerow(["timestamp", "event_type", "key", "task_id"])
        
        print(f"Logging behavior to {self.log_directory}")

    def start_tracking(self, task_id="general"):
        """Start tracking behavior for a given task."""
        self.active = True
        self.current_task_id = task_id
        print(f"Behavior tracking started for task: {task_id}.")
        # In a real scenario, event listeners (e.g., for mouse/keyboard) would be started here.

    def stop_tracking(self):
        """Stop tracking behavior."""
        self.active = False
        print("Behavior tracking stopped.")
        # Event listeners would be stopped here.

    def log_mouse_event(self, event_type, x, y, button=None):
        """Log a mouse event.

        Args:
            event_type (str): 'move', 'click', 'drag_start', 'drag_end'
            x (int): Mouse x-coordinate.
            y (int): Mouse y-coordinate.
            button (str, optional): Mouse button if applicable (e.g., 'left', 'right').
        """
        if not self.active:
            return
        
        timestamp = time.time()
        with open(self.mouse_log_path, 'a', newline='') as f_mouse:
            mouse_writer = csv.writer(f_mouse)
            mouse_writer.writerow([timestamp, event_type, x, y, button, self.current_task_id])

    def log_keyboard_event(self, event_type, key):
        """Log a keyboard event.

        Args:
            event_type (str): 'key_press', 'key_release'.
            key (str): The key pressed/released.
        """
        if not self.active:
            return

        timestamp = time.time()
        with open(self.keyboard_log_path, 'a', newline='') as f_keyboard:
            keyboard_writer = csv.writer(f_keyboard)
            keyboard_writer.writerow([timestamp, event_type, key, self.current_task_id])

    # --- Example methods to be called by a GUI or event system ---
    def on_mouse_move(self, x, y):
        """Handler for mouse movement events."""
        self.log_mouse_event('move', x, y)

    def on_mouse_click(self, x, y, button, pressed):
        """Handler for mouse click events."""
        event_type = 'button_press' if pressed else 'button_release'
        self.log_mouse_event(event_type, x, y, button)

    def on_key_press(self, key):
        """Handler for key press events."""
        self.log_keyboard_event('key_press', str(key))
    
    def on_key_release(self, key):
        """Handler for key release events."""
        self.log_keyboard_event('key_release', str(key))

if __name__ == '__main__':
    # Example Usage
    tracker = BehaviorTracker(log_directory="../../logs/") # Adjusted path for direct script run

    tracker.start_tracking(task_id="test_task_001")

    # Simulate some events
    tracker.on_mouse_move(100, 150)
    time.sleep(0.1)
    tracker.on_mouse_click(101, 152, 'left', True)
    time.sleep(0.05)
    tracker.on_mouse_click(101, 152, 'left', False)
    time.sleep(0.2)
    tracker.on_key_press('A')
    time.sleep(0.1)
    tracker.on_key_release('A')
    time.sleep(0.1)
    tracker.on_mouse_move(200,250)

    tracker.stop_tracking()

    print(f"Mouse logs saved to: {os.path.abspath(tracker.mouse_log_path)}")
    print(f"Keyboard logs saved to: {os.path.abspath(tracker.keyboard_log_path)}")