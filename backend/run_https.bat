@echo off
title IBVAP Server (HTTPS)
cd /d %~dp0

REM Keep this in sync with run.bat's key -- both must match whatever
REM you type into the dashboard's login prompt.
set IBVAP_API_KEY=letmein

if not exist .venv (
    echo.
    echo Virtual environment not found in this folder.
    echo Run setup.bat first ^(one-time setup^), then use this file to start the server.
    echo.
    pause
    exit /b 1
)

if not exist cert.pem (
    echo.
    echo cert.pem / key.pem not found in this folder.
    echo Run generate_cert.bat first to create a local self-signed certificate.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo === Starting IBVAP server over HTTPS ===
echo Dashboard: https://localhost:8000
echo Your browser will warn that the certificate is self-signed -- this is
echo expected for local testing. Click through (e.g. "Advanced -^> Proceed").
echo Close this window (or press Ctrl+C) to stop the server.
echo.

start "" cmd /c "timeout /t 3 >nul & start https://localhost:8000"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload ^
    --ssl-keyfile=key.pem --ssl-certfile=cert.pem

pause
