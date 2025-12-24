@echo off
chcp 65001 >nul
title Thai OCR Runner

REM ==================== CONFIG ====================
set "PROJECT_DIR=%~dp0"
set "PY=%PROJECT_DIR%..\venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%ocr_smoke.py"

echo --------------------------------------------
echo [Thai OCR] Starting OCR with PaddleOCR v3.x
echo Script : %SCRIPT%
echo Python : %PY%
echo --------------------------------------------

REM ==================== RUN ====================
if not exist "%PY%" (
    echo [ERROR] Python venv not found at %PY%
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] Script not found: %SCRIPT%
    pause
    exit /b 1
)

"%PY%" -u "%SCRIPT%"

echo.
echo [DONE] OCR Finished.
echo Results are shown above.
pause
