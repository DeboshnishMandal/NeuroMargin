@echo off
setlocal
cd /d "%~dp0"
title NeuroMargin - Fast Launcher
echo ============================================================
echo       NEUROMARGIN - INCREMENTAL ONE-COMMAND LAUNCHER
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [FIRST RUN] Creating the Python 3.13 environment...
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python Launcher not found. Install 64-bit Python 3.13 with the py launcher.
    pause
    exit /b 1
  )
  py -3.13 -m venv .venv
  if errorlevel 1 (
    echo Could not create Python 3.13 environment. Check: py --list
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" scripts\bootstrap.py
if errorlevel 1 (
  echo.
  echo NeuroMargin stopped because a setup, data, or training step failed.
  pause
  exit /b 1
)
pause
