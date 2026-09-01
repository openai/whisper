@echo off
title Voice to Text Converter - Terminal Mode
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please run setup.bat first to install dependencies
    echo.
    pause
    exit /b 1
)

REM Check if voice_to_text.py exists
if not exist "voice_to_text.py" (
    echo.
    echo [ERROR] voice_to_text.py not found
    echo Make sure you're running this from the correct directory
    echo.
    pause
    exit /b 1
)

REM Run the application in terminal mode
echo Starting Voice to Text Converter (Terminal Mode)...
echo.
python voice_to_text.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error
    echo.
    pause
)