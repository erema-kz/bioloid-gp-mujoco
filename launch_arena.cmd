@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  echo Virtual environment not found: %CD%\.venv
  echo Install the project first, then run this shortcut again.
  pause
  exit /b 1
)

start "Bioloid GP" ".venv\Scripts\pythonw.exe" -m bioloid_sim.launcher
