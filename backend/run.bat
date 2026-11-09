@echo off
title IBVAP Server
cd /d %~dp0

REM ============================================================
REM Change this to your own secret before a real deployment/demo.
REM This is what the dashboard's login prompt checks against --
REM whoever knows this key can use the API. Keep it out of anything
REM you commit to a public repo.
REM ============================================================
set IBVAP_API_KEY=letmein

if not exist .venv (
    echo.
    echo Virtual environment not found in this folder.
    echo Run setup.bat first ^(one-time setup^), then use this file to start the server.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo === Starting IBVAP server ===
echo Dashboard will open in your browser automatically in a few seconds.
echo Close this window (or press Ctrl+C) to stop the server.
echo.

REM Open the dashboard in the default browser after a short delay, in a
REM separate helper process, so it doesn't block uvicorn from starting.
REM Uvicorn itself stays in the foreground so closing this window is a
REM clean way to stop the server -- no orphaned background process.
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
