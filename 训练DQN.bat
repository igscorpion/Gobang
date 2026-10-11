@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==============================================
echo   Train DQN Gomoku AI (GPU, auto-resume)
echo   Runs forever until you close this window
echo   Saves checkpoint every 100 games to rl\models\dqn.pt
echo ==============================================
py -3.11 -m rl.training.train_dqn
echo.
echo Stopped. Checkpoint saved.
pause
