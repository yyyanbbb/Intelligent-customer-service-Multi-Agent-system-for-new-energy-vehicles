@echo off
echo Starting Xiaopeng Auto CS Agent...
echo.

REM Start Ollama in background if not running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama...
    start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 5 /nobreak >nul
)

REM Set environment
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONPATH=%~dp0

REM Run Gradio UI
echo Launching Gradio UI at http://localhost:7860
python -m cs_agent.main --ui

pause
