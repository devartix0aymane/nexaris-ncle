@echo off
cd /d "%~dp0"
echo Starting NEXARIS Cognitive Load Estimator from %cd%...

:: Check if Python is installed
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please ensure you have venv module installed.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment 'venv' already exists.
)

:: Activate virtual environment
echo Activating virtual environment (venv\Scripts\activate.bat)...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.

echo --- Python and Pip Information --- 
pip --version
where python
where pip
echo ----------------------------------

:: Install dependencies
echo Installing/Verifying dependencies from requirements.txt...
echo --- PIP INSTALL OUTPUT START --- 
pip install -r requirements.txt
echo --- PIP INSTALL OUTPUT END --- 
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo Dependencies installation/verification complete.

echo --- INSTALLED PACKAGES START --- 
pip list
echo --- INSTALLED PACKAGES END --- 

:: Run the application
echo Running application (main.py) and logging to application.log...
python main.py > application.log 2>&1

if %errorlevel% neq 0 (

:: Deactivate virtual environment on exit
echo Application finished. Check application.log for details.
pause
deactivate
echo Virtual environment deactivated.
echo Exiting.
exit /b 0