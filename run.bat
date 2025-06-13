@echo off
echo Starting NEXARIS Cognitive Load Estimator...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please ensure you have venv module installed.
        pause
        exit /b 1
    )
)

:: Activate virtual environment and install dependencies if needed
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if requirements are installed
pip freeze | findstr "PyQt5" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install dependencies. Please check requirements.txt and your internet connection.
        pause
        exit /b 1
    )
)

:: Run the application
echo Starting application...
python main.py

:: Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat

echo Application closed.
pause