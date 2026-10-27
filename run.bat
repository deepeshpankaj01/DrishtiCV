@echo off
setlocal enabledelayedexpansion

title DrishtiCV — Real-Time Object Detection ^& Tracking

echo ===============================================================================
echo  DrishtiCV — Real-Time Object Detection, Tracking ^& Scene Analytics
echo  Developed by Deepesh Kumar Pankaj
echo  "See Objects. Track Movement. Understand the Scene."
echo ===============================================================================
echo.

:: Check for project virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
    set "PYTHON_CMD=python"
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
    set "PYTHON_CMD=python"
) else (
    echo [!] No local virtual environment found.
    echo [*] Attempting to use system Python...
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_CMD=py -3.11"
    ) else (
        where python >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            set "PYTHON_CMD=python"
        ) else (
            echo [ERROR] Python is not installed or not found on PATH.
            echo Please install Python 3.11 from https://www.python.org/
            pause
            exit /b 1
        )
    )
)

echo [*] Launching DrishtiCV desktop application...
echo.

%PYTHON_CMD% app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    echo Please review the error above or inspect the active session log in outputs/
    pause
)

endlocal
