#!/bin/bash

echo "Starting NEXARIS Cognitive Load Estimator..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Python 3.8 or higher is required. Found Python $PYTHON_VERSION."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please ensure you have venv module installed."
        exit 1
    fi
fi

# Activate virtual environment and install dependencies if needed
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! pip freeze | grep -q "PyQt5"; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies. Please check requirements.txt and your internet connection."
        deactivate
        exit 1
    fi
fi

# Run the application
echo "Starting application..."
python main.py

# Deactivate virtual environment on exit
deactivate

echo "Application closed."