@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Colab Pilot Visual Dashboard on http://127.0.0.1:8765 ...
start "" "http://127.0.0.1:8765"
"C:\Users\86177\AppData\Roaming\uv\tools\google-colab-cli\Scripts\python.exe" server.py
pause
