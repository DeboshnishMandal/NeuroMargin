@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_NEUROMARGIN.bat once first.
  pause
  exit /b 1
)
set NEUROMARGIN_FORCE_RETRAIN=1
".venv\Scripts\python.exe" scripts\bootstrap.py
if errorlevel 1 pause
