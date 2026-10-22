@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Training locally with periodic checkpoints.
".venv\Scripts\python.exe" -u -m rl.training.local_alphazero --output runs/az_wide_manual --episodes 2000 --sims 400 --warm-start backups/upstream-2711cca/models/alphazero.pt --device cpu --resume
echo Exit code: %errorlevel%
pause
