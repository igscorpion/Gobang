# Run the prepared local environment; own only this script's server.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Missing .venv. Run uv sync --extra test first.' }
$taskNode = (Get-Command node -ErrorAction Stop).Source
$frontendDirectory = Join-Path $PSScriptRoot 'web\frontend'
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory 'node_modules\vite\bin\vite.js'))) {
    throw 'Missing frontend dependencies. Run npm ci in web/frontend first.'
}
function Find-AvailablePort([int]$StartPort) {
    foreach ($candidate in $StartPort..($StartPort+19)) {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$candidate)
        try { $listener.Start();return $candidate } catch { } finally { $listener.Stop() }
    }
    throw 'No free local port found.'
}
$backendPort = Find-AvailablePort 8000
$frontendPort = Find-AvailablePort 5173
$env:VITE_API_PORT = "$backendPort"
$logDirectory = Join-Path $PSScriptRoot 'runs\web'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$backend = Start-Process -FilePath $taskPython -ArgumentList @('-m','uvicorn','web.backend.main:app','--host','127.0.0.1','--port',"$backendPort") -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'backend.log') -RedirectStandardError (Join-Path $logDirectory 'backend-error.log')
try {
    $ready = $false
    for ($attempt=0; $attempt -lt 100; $attempt++) {
        if ($backend.HasExited) { throw 'Backend exited. Read runs/web/backend-error.log.' }
        try { Invoke-RestMethod "http://127.0.0.1:$backendPort/" | Out-Null;$ready=$true;break } catch { Start-Sleep -Milliseconds 100 }
    }
    if (-not $ready) { throw 'Backend startup timed out.' }
    Write-Host "Backend: http://127.0.0.1:$backendPort"
    Write-Host "Open game: http://127.0.0.1:$frontendPort"
    Push-Location -LiteralPath $frontendDirectory
    try { & $taskNode 'node_modules/vite/bin/vite.js' '--host' '127.0.0.1' '--port' "$frontendPort" '--strictPort' }
    finally { Pop-Location }
} finally {
    if (-not $backend.HasExited) { $backend.Kill();$backend.WaitForExit() }
}
