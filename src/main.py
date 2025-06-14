"""
NEXARIS Cognitive Load Estimator (NCLE)

Main entry point for the NCLE application.
"""

import sys
# import gui
# import task_simulation
# import tracking
# import face_analysis
# import scoring

class Application:
    """Main application class for NCLE."""
    def __init__(self):
        """Initialize the application components."""
        # self.gui = gui.GUI()
        # self.task_manager = task_simulation.TaskManager()
        # self.behavior_tracker = tracking.BehaviorTracker()
        # self.facial_analyzer = face_analysis.FacialAnalyzer()
        # self.score_calculator = scoring.ScoreCalculator()
        print("NEXARIS Cognitive Load Estimator initialized.")

    def run(self):
        """Run the main application loop."""
        print("NEXARIS Cognitive Load Estimator is running...")
        # self.gui.start()


def main():
    """Main function to start the NCLE application."""
    app = Application()
    app.run()

if __name__ == "__main__":
    main()