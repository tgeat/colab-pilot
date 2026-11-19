@echo off
setlocal
set "_ORIG_CP="
for /f "tokens=2 delims=:." %%a in ('chcp') do set "_ORIG_CP=%%a"
python "%~dp0qwen_cli.py" %*
set "_QWEN_RC=%ERRORLEVEL%"
if defined _ORIG_CP chcp %_ORIG_CP% >nul 2>&1
endlocal & exit /b %_QWEN_RC%
