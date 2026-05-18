@echo off
title AquaIQ v3 Control Panel Launcher
color 0B

echo =================================================═════════════════
echo        AquaIQ v3 - Unified Fuzzy Logic Control Panel Launcher
echo =================================================═════════════════
echo.

:: 1. Verify Python Installation
echo [1/3] Verifying Python Environment...
where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo --------------------------------------------------------------
    echo ERROR: Python is not installed or not in your system PATH!
    echo Please install Python 3.8+ and check "Add Python to PATH".
    echo --------------------------------------------------------------
    pause
    exit /b
)
echo Python environment verified.
echo.

:: 2. Check and Install Missing Packages
echo [2/3] Verifying Python Packages (numpy, scipy, scikit-fuzzy, matplotlib, flask, reportlab)...
python -c "import numpy, scipy, skfuzzy, matplotlib, flask, reportlab" >nul 2>nul
if %errorlevel% neq 0 (
    echo Some dependencies are missing. Installing required packages...
    python -m pip install numpy scipy scikit-fuzzy matplotlib flask reportlab
    if %errorlevel% neq 0 (
        color 0C
        echo --------------------------------------------------------------
        echo ERROR: Package installation failed! 
        echo Please check your internet connection and permissions.
        echo --------------------------------------------------------------
        pause
        exit /b
    )
    echo Dependencies successfully installed!
) else (
    echo All dependencies are already installed and satisfied.
)
echo.

:: 3. Open Browser and Launch App
echo [3/3] Launching AquaIQ Web App...
echo Opening browser to http://127.0.0.1:5000/ ...
start "" "http://127.0.0.1:5000/"

echo.
echo =================================================═════════════════
echo  AquaIQ Server is online! Keep this window open while using the app.
echo  Press CTRL + C inside this window to terminate the server.
echo =================================================═════════════════
echo.

python app.py
pause
