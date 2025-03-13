@echo off
echo Starting Voice2DuneWeaver application...

:: Check if virtual environment exists
if not exist .venv (
    echo Virtual environment not found!
    echo Please run setup.bat first to set up the environment.
    exit /b 1
)

:: Check for .env file
if not exist .env (
    echo .env file not found!
    echo Please run setup.bat first to create an .env file and add your API keys.
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Failed to activate virtual environment.
    echo Try running setup.bat again or manually activate with '.venv\Scripts\activate.bat'
    exit /b 1
)

:: Check for app.py
if not exist app.py (
    echo app.py not found! Make sure you're in the correct directory.
    exit /b 1
)

:: Run the application
echo Launching Voice2DuneWeaver... Press Ctrl+C to exit.
python app.py
if %ERRORLEVEL% neq 0 (
    echo An error occurred while running the application.
    exit /b 1
)

:: Deactivate virtual environment when done
call deactivate
exit /b 0 