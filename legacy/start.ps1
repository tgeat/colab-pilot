Set-Location $PSScriptRoot
Write-Host "🚀 正在启动 Colab Pilot 可视化管控台: http://127.0.0.1:8765" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8765"
& "C:\Users\86177\AppData\Roaming\uv\tools\google-colab-cli\Scripts\python.exe" "$PSScriptRoot\server.py"
