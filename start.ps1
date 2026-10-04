# RL Gomoku 一键启动（Windows PowerShell）
# 用法：在项目根目录执行  .\start.ps1
# 若提示无法运行脚本，见 README 中的执行策略说明。

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 py 启动器，请安装 Python 3.10+ 并勾选 'py launcher'。"
}

if (-not (Test-Path ".venv")) {
    py -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -r web\backend\requirements.txt

Push-Location web\frontend
npm install
Pop-Location

Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"

Start-Process -FilePath ".\.venv\Scripts\uvicorn.exe" -ArgumentList "web.backend.main:app","--reload","--host","127.0.0.1","--port","8000" -WorkingDirectory $PSScriptRoot

Set-Location web\frontend
npm run dev
