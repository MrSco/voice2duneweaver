@echo off
echo Setting up Voice2DuneWeaver environment for Windows...

:: Check for Python installation
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.7 or later and try again.
    echo You can download Python from https://www.python.org/downloads/
    exit /b 1
)
echo Found Python installation.

:: Check for pip
python -m pip --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo pip not found! Please ensure pip is installed with your Python installation.
    exit /b 1
)
echo Found pip installation.

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Install requirements
echo Installing dependencies...
python -m pip install --upgrade pip
if exist requirements.txt (
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install dependencies.
        exit /b 1
    )
) else (
    echo requirements.txt not found!
    exit /b 1
)

:: Check for .env file
if not exist .env (
    if exist .env.example (
        echo Creating .env file from template...
        copy .env.example .env
        echo Please edit the .env file to add your API keys.
    ) else (
        echo Creating basic .env file...
        (
            echo # DuneWeaver settings
            echo DW_URL=http://localhost:8080
            echo.
            echo # GOOGLE GEMINI API KEY
            echo GEMINI_API_KEY=
        ) > .env
        echo Please edit the .env file to add your API keys.
    )
) else (
    echo .env file already exists.
)

echo Setup complete! You can now run the application with run.bat 