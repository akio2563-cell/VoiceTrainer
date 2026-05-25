@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo Starting Voice Trainer...
.venv\Scripts\python.exe main.py
