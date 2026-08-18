@echo off
title Sipeed Tang Nano 9K -- BDH Monochrome Chatbot
cd /d "%~dp0"
echo [1/3] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)
echo [2/3] Installing/verifying dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo [3/3] Starting BDH Server and Opening Web UI...
start "" "http://localhost:8000"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
