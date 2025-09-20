@echo off
title Voice to Text Converter - Setup
cd /d "%~dp0"

echo ========================================
echo Voice to Text Converter - Setup
echo ========================================
echo.

REM Check if Python is available
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

python --version
echo [OK] Python found!
echo.

REM Check if requirements.txt exists
echo [2/4] Checking requirements file...
if not exist "requirements.txt" (
    echo.
    echo [ERROR] requirements.txt not found
    echo Make sure you're running this from the correct directory
    echo.
    pause
    exit /b 1
)
echo [OK] Requirements file found!
echo.

REM Install dependencies
echo [3/4] Installing dependencies...
echo This may take a few minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo Please check your internet connection and try again
    echo.
    pause
    exit /b 1
)
echo.
echo [OK] Dependencies installed successfully!
echo.

REM Test import
echo [4/4] Testing installation...
python -c "import voice_to_text; print('[OK] Voice-to-text module loads successfully!')" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] Installation completed but module test failed
    echo This might still work, but check for any error messages above
    echo.
) else (
    echo [OK] Installation test passed!
    echo.
)

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now use:
echo   - voice_to_text_terminal.bat (for terminal mode)
echo   - voice_to_text_gui.bat (for GUI mode)
echo.
echo Note: First run will download Whisper models (~150MB)
echo.
pause