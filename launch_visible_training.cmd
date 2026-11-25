@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found: %CD%\.venv
  echo Install the project first, then run this shortcut again.
  pause
  exit /b 1
)

set "RESUME_MODEL=runs\bioloid_exact_walk_0p5m_v2.zip"
if exist "runs\bioloid_visible_recovery.zip" set "RESUME_MODEL=runs\bioloid_visible_recovery.zip"

echo Starting visible AI training from %RESUME_MODEL%
echo Life: 7 seconds, fallen starts: 35 percent, steps: 3000000
echo Do not open this shortcut twice at the same time.

".venv\Scripts\python.exe" -m bioloid_sim.train ^
  --model exact ^
  --task walk ^
  --render ^
  --steps 3000000 ^
  --episode-seconds 7 ^
  --fallen-reset-prob 0.35 ^
  --resume "%RESUME_MODEL%" ^
  --output runs\bioloid_visible_recovery

pause
