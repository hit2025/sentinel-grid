@echo off
REM IBVAP backend setup -- run this from inside the "backend" folder.
REM Encodes the correct install order so the opencv/ultralytics conflict
REM (cv2.CascadeClassifier missing) can never happen from following steps
REM by hand in the wrong order.

echo === IBVAP backend setup ===
echo.

if exist .venv (
    echo Virtual environment already exists -- skipping creation.
    echo ^(If you want a totally clean install, close this terminal, delete
    echo  the .venv folder manually in File Explorer, then re-run this script.^)
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo FAILED: could not create virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo.
echo Installing base requirements...
pip install -r requirements.txt

echo.
echo Installing the accurate AI detector (YOLOv8)...
pip install ultralytics

echo.
echo Fixing the opencv conflict that ultralytics just introduced...
echo (this step is NOT optional -- skipping it breaks face detection and ANPR)
pip install --force-reinstall opencv-python-headless==4.13.0.92

echo.
echo === Setup complete ===
echo Start the server with:
echo   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo Also make sure Tesseract OCR is installed separately for ANPR:
echo   https://github.com/UB-Mannheim/tesseract/wiki
echo.
pause
