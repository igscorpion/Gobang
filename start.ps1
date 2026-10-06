# RL Gomoku 一键启动（Windows PowerShell）
# 用法：在项目根目录执行  .\start.ps1
# 若提示无法运行脚本，见 README 中的执行策略说明。

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Find-AvailablePort {
    param(
        [int]$StartPort = 8000,
        [int]$MaxRetries = 10
    )
    for ($port = $StartPort; $port -lt $StartPort + $MaxRetries; $port++) {
        $isFree = $false
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
            $listener.Start()
            $listener.Stop()
            $isFree = $true
        }
        catch {
            $isFree = $false
        }
        finally {
            if ($listener) {
                try { $listener.Stop() } catch { }
            }
        }
        if ($isFree) {
            Write-Host "Port $port is available"
            return $port
        }
        Write-Host "Port $port is in use, trying next..."
    }
    throw "No available port found in range $StartPort to $($StartPort + $MaxRetries - 1)"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 uv，请先安装 uv。建议使用: https://docs.astral.sh/uv/"
}

if (-not (Test-Path ".venv")) {
    uv sync --python 3.11
}

Push-Location web\frontend
npm install
Pop-Location

$backendPort = Find-AvailablePort -StartPort 8000 -MaxRetries 10

$envContent = "VITE_API_PORT=$backendPort"
$envContent | Out-File -FilePath "web\frontend\.env" -Encoding utf8

Write-Host "Backend: http://127.0.0.1:$backendPort"

$uvArgs = @(
    "run",
    "uvicorn",
    "web.backend.main:app",
    "--reload",
    "--host",
    "127.0.0.1",
    "--port",
    "$backendPort"
)

Start-Process -FilePath "uv" -ArgumentList $uvArgs -WorkingDirectory $PSScriptRoot

Set-Location web\frontend
npm run dev