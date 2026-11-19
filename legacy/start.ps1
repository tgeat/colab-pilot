Set-Location $PSScriptRoot
Write-Host "🚀 正在启动 Colab Pilot 可视化管控台: http://127.0.0.1:8765" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8765"
$uvPy = Join-Path $env:APPDATA "uv\tools\google-colab-cli\Scripts\python.exe"
if (Test-Path $uvPy) {
    & $uvPy "$PSScriptRoot\server.py"
} else {
    python "$PSScriptRoot\server.py"
}
