@echo off
cd /d "%~dp0"

echo Starting YouTube Transcript Application...
echo.

REM Try to activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python...
)

REM Run the main application
echo Running application...
python src\main.py

REM Keep window open to see any errors
echo.
echo Application finished. Press any key to close...
pause >nul 