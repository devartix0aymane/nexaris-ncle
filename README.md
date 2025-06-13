# NEXARIS Cognitive Load Estimator (NCLE)

## Overview
The NEXARIS Cognitive Load Estimator (NCLE) is a prototype system designed to measure and visualize cognitive load in real-time. This tool supports human-centric cybersecurity operations and burnout prevention by providing objective metrics of mental workload during task execution.

## Purpose
Cybersecurity professionals often work under high cognitive load conditions, which can lead to:
- Decreased performance and accuracy
- Increased response time to critical alerts
- Decision fatigue and burnout
- Higher turnover rates in security operations centers

NCLE addresses these challenges by providing real-time feedback on cognitive load, enabling:
- Optimization of workflow and task distribution
- Early detection of potential burnout conditions
- Data-driven approaches to team management
- Improved human-computer interaction design

## Features
- **Task Simulation**: Configurable timed tasks that mimic real-world cybersecurity scenarios
- **Behavior Tracking**: Monitoring of mouse movements, response times, click patterns, and hesitation
- **Facial Analysis**: Optional emotion recognition to detect frustration, focus, and stress indicators
- **Cognitive Load Scoring**: Proprietary algorithm to calculate Estimated Cognitive Load Score (ECLS)
- **Visualization**: Real-time display of cognitive load metrics and trends
- **Extensibility**: Support for ML model integration and EEG data (NEXARIS NeuroPrint™)

## Technology Stack
- **Python**: Core programming language
- **PyQt**: UI framework for professional interface
- **OpenCV**: Computer vision for facial analysis
- **Pandas/NumPy**: Data processing and analysis
- **Matplotlib/Seaborn**: Data visualization

## Project Structure
```
NEXARIS/
├── main.py                     # Application entry point
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── run.bat                     # Windows execution script
├── run.sh                      # Linux/macOS execution script
├── config/                     # Configuration files
├── data/                       # Data storage
├── docs/                       # Additional documentation
├── src/
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── task_simulator.py   # Task generation and management
│   │   ├── behavior_tracker.py # User interaction monitoring
│   │   ├── facial_analyzer.py  # OpenCV-based emotion detection
│   │   ├── load_calculator.py  # ECLS algorithm implementation
│   │   └── data_manager.py     # Data handling and storage
│   ├── ui/                     # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py      # Main application window
│   │   ├── task_panel.py       # Task display and interaction
│   │   ├── visualization.py    # Score visualization components
│   │   └── settings_dialog.py  # Configuration interface
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── user_model.py       # User data representation
│   │   ├── task_model.py       # Task data structures
│   │   └── score_model.py      # Cognitive load score model
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config_utils.py     # Configuration handling
│       ├── logging_utils.py    # Logging functionality
│       └── ml_utils.py         # Machine learning utilities
└── tests/                      # Unit and integration tests
    ├── __init__.py
    ├── test_core.py
    ├── test_ui.py
    └── test_models.py
```

## Installation and Execution

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Windows
1. Clone the repository:
   ```
   git clone https://github.com/devartix0aymane/nexaris-ncle.git
   
   cd nexaris-ncle
   ```
2. Run the application using the provided batch script:
   ```
   run.bat
   ```
   This script will:
   - Create a virtual environment if it doesn't exist
   - Install required dependencies
   - Launch the application

### Linux/macOS
1. Clone the repository:
   ```
   git clone https://github.com/devartix0aymane/nexaris-ncle.git
   cd NEXARIS
   ```
2. Make the shell script executable:
   ```
   chmod +x run.sh
   ```
3. Run the application using the provided shell script:
   ```
   ./run.sh
   ```
   This script will:
   - Create a virtual environment if it doesn't exist
   - Install required dependencies
   - Launch the application

### Manual Installation
1. Clone the repository:
   ```
   git clone https://github.com/devartix0aymane/nexaris-ncle.git
   cd NEXARIS
   ```
2. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python main.py
   ```

## Future Directions
- **Machine Learning Integration**: Advanced pattern recognition for improved cognitive load estimation
- **EEG Support**: Integration with consumer-grade EEG devices for direct neural activity measurement
- **Team Dashboard**: Aggregate view for monitoring team cognitive load in SOC environments
- **Predictive Analytics**: Forecasting potential burnout conditions before they occur
- **API Development**: Enable integration with other security tools and platforms

## Human-Centric Cybersecurity
NCLE represents a shift toward human-centric approaches in cybersecurity operations. By acknowledging and measuring the human factors in security work, organizations can:

1. **Optimize Human Performance**: Design workflows that maximize effectiveness while minimizing cognitive strain
2. **Enhance Decision Quality**: Ensure critical security decisions are made under optimal cognitive conditions
3. **Improve Retention**: Reduce burnout and turnover by proactively managing cognitive workload
4. **Personalize Training**: Develop targeted training programs based on individual cognitive patterns

## NEXARIS NeuroPrint™
The NCLE prototype lays the groundwork for the more advanced NEXARIS NeuroPrint™ technology, which will incorporate direct neurological measurements for unprecedented accuracy in cognitive load estimation.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*NEXARIS Cognitive Load Estimator (NCLE) is a prototype tool designed for research and development purposes.*

## Author

Developed by Aymane Loukhai (devartix)

© 2023 NEXARIS - All Rights Reserved
