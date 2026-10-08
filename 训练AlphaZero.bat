@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==============================================
echo   Train AlphaZero Gomoku AI (GPU, auto-resume)
echo   Runs forever until you close this window
echo   Saves checkpoint every 100 games to rl\models\alphazero.pt
echo   (sims/step = 100 by default)
echo ==============================================
py -3.11 -m rl.training.train_alphazero
echo.
echo Stopped. Checkpoint saved.
pause
