@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Do not start while another suite is running.
".venv\Scripts\python.exe" -u -m rl.training.local_suite --device cuda --az-device cpu
echo Exit code: %errorlevel%
pause
