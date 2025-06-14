# NEXARIS Cognitive Load Estimator (NCLE)

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/devartix0aymane/nexaris-ncle)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=flat-square)](https://github.com/devartix0aymane/nexaris-ncle/issues)

## 🔹 What the App Does

The NEXARIS Cognitive Load Estimator (NCLE) is a sophisticated desktop application designed to estimate a user's cognitive load in real-time. It achieves this by intelligently integrating and analyzing data from multiple sources:

*   **Task Performance**: Monitors how a user performs on specific tasks.
*   **Behavioral Patterns**: Tracks mouse movements, clicks, and keyboard inputs.
*   **Facial Emotion Analysis**: Utilizes a webcam to detect faces and recognize emotional cues.

This comprehensive approach provides a nuanced understanding of user engagement, stress levels, and overall cognitive state during computer-based activities. The application features a user-friendly Graphical User Interface (GUI) for ease of operation and real-time feedback.

## 📊 Screenshots and Demo

*(Placeholder for screenshots - Please add actual screenshots of the application GUI, e.g., main dashboard, task view, settings panel)*

**Example Screenshot Placeholder:**

`![NCLE Dashboard](https://via.placeholder.com/600x400.png?text=NCLE+Main+Dashboard+Screenshot)`

**Demo Video Placeholder:**

`![NCLE Demo](https://via.placeholder.com/600x400.png?text=Link+to+App+Demo+Video)`

## 🛠️ How to Install

Follow these steps to get NCLE up and running on your system:

1.  **Prerequisites**:
    *   Python 3.7 or newer.
    *   A webcam (required for the facial analysis feature).
    *   Git (for cloning the repository).

2.  **Clone the Repository**:
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/devartix0aymane/nexaris-ncle.git
    cd nexaris-ncle
    ```

3.  **Create and Activate a Virtual Environment (Recommended)**:
    This helps manage project dependencies and avoid conflicts.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    *   **Windows**: `venv\Scripts\activate`
    *   **macOS/Linux**: `source venv/bin/activate`

4.  **Install Dependencies**:
    With the virtual environment activated, install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

Once the installation is complete, you can launch the NCLE application by running the main script from the project's root directory:

```bash
python src/main.py
```

## 🧠 Technologies Used

NCLE leverages a stack of powerful Python libraries and technologies:

*   **Core Language**: Python 3
*   **GUI Framework**: PyQt5
*   **Computer Vision**: OpenCV (`opencv-python`) for webcam access and face detection
*   **Data Analysis**: NumPy, Pandas for data manipulation
*   **System Monitoring**: `psutil` for CPU usage monitoring
*   **Behavioral Tracking**: `pynput` for mouse/keyboard tracking
*   **Emotion Recognition**: DeepFace for facial emotion analysis
*   **Data Logging**: Python CSV module
*   **Configuration**: JSON/YAML files

## 📦 Project Structure

```
nexaris-ncle/
├── .github/                # GitHub specific files
│   └── workflows/
│       └── python-app.yml  # CI workflow
├── assets/                 # Static assets
│   ├── icons/             # Application icons
│   └── models/            # ML models and weights
├── docs/                   # Documentation
│   ├── api/               # API documentation
│   └── user_guide/        # User manual
├── logs/                   # Application logs
├── src/                    # Source code
│   ├── components/        # Core functional modules
│   │   ├── behavior_tracker.py
│   │   ├── cognitive_load_calculator.py
│   │   ├── facial_analyzer.py
│   │   └── task_simulator.py
│   ├── core/             # Core application logic
│   │   ├── config_manager.py
│   │   └── data_manager.py
│   ├── gui/              # GUI components
│   │   ├── main_window.py
│   │   ├── dashboard_widget.py
│   │   └── settings_widget.py
│   └── utils/            # Utility functions
│       ├── config_utils.py
│       ├── logging_utils.py
│       └── ml_utils.py
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## 🔄 Related Projects

NCLE is part of the NEXARIS ecosystem:

* [NEXARIS Adaptive Scenario Engine (NASE)](https://github.com/devartix0aymane/nexaris-nase) - Adaptive micro-scenario engine for personalized training
* [NEXARIS Mission Genesis (NMG)](https://github.com/devartix0aymane/nexaris-nmg) - Collaborative cyber threat response simulation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Developed by [Aymane Loukhai](https://github.com/devartix0aymane)

© 2024 NEXARIS - All Rights Reserved