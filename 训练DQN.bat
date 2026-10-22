@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Training locally with periodic checkpoints.
".venv\Scripts\python.exe" -u -m rl.training.local_dqn --output runs/dqn4_manual --episodes 30000 --device cuda --resume
echo Exit code: %errorlevel%
pause
