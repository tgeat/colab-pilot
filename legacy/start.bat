@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Colab Pilot Visual Dashboard on http://127.0.0.1:8765 ...
start "" "http://127.0.0.1:8765"
if exist "%APPDATA%\uv\tools\google-colab-cli\Scripts\python.exe" (
  "%APPDATA%\uv\tools\google-colab-cli\Scripts\python.exe" server.py
) else (
  python server.py
)
pause
