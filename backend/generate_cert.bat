@echo off
title IBVAP - Generate Self-Signed Certificate
cd /d %~dp0

echo.
echo === Generating a self-signed TLS certificate for local HTTPS testing ===
echo.
echo IMPORTANT: a self-signed cert is for LOCAL DEMO/TESTING only. Your
echo browser will show a "not secure" / "certificate not trusted" warning
echo the first time you visit https://localhost:8000 -- that is expected;
echo click through it (e.g. "Advanced -> Proceed"). For a real public
echo deployment, use a real certificate (e.g. via Let's Encrypt / certbot,
echo or whatever your hosting provider issues) instead of this file.
echo.

where openssl >nul 2>nul
if %errorlevel% neq 0 (
    echo openssl was not found on your PATH.
    echo.
    echo Options:
    echo   1. Install Git for Windows ^(includes openssl via Git Bash^)
    echo      https://git-scm.com/download/win
    echo   2. Use WSL ^(Windows Subsystem for Linux^) and run this from there
    echo   3. Install openssl directly: https://slproweb.com/products/Win32OpenSSL.html
    echo.
    pause
    exit /b 1
)

openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 ^
    -subj "/C=IN/ST=WB/L=Local/O=IBVAP/CN=localhost"

if %errorlevel% neq 0 (
    echo.
    echo Certificate generation failed -- see the openssl output above.
    pause
    exit /b 1
)

echo.
echo === Done ===
echo Created key.pem and cert.pem in this folder.
echo Now run run_https.bat to start the server with HTTPS enabled.
echo.
pause
