@echo on
chcp 65001 >nul
setlocal ENABLEDELAYEDEXPANSION

REM ===== CONFIG =====
set "PROJECT_DIR=%~dp0"
set "PY=%PROJECT_DIR%..\venv\Scripts\python.exe"
set "DET_DIR=%PROJECT_DIR%inference\ch_PP-OCRv3_det_infer"
set "REC_DIR=%PROJECT_DIR%inference\th_PP-OCRv5_mobile_rec_infer"
set "CLS_DIR=%PROJECT_DIR%inference\ch_ppocr_mobile_v2.0_cls_infer"
set "DICT=%REC_DIR%\ppocr_keys.txt"
set "FALLBACK_DICT=%PROJECT_DIR%ppocr\utils\ppocr_keys_v1.txt"
set "DEFAULT_IMG_DIR=%PROJECT_DIR%test_images"
set "OUT_DIR=%PROJECT_DIR%inference_results"
set "USE_GPU=0"   REM 0=CPU, 1=GPU

echo -------- PATH CHECK --------
echo PROJECT_DIR=%PROJECT_DIR%
echo PY=%PY%
if not exist "%PY%" echo [ERROR] Python venv not found & pause & exit /b 1

if not exist "%PROJECT_DIR%tools\infer\predict_system.py" echo [ERROR] predict_system.py not found & pause & exit /b 1

for %%A in ("%DET_DIR%\inference.pdmodel" "%REC_DIR%\inference.pdmodel") do (
  if not exist "%%~A" echo [ERROR] Missing model file: %%~A & pause & exit /b 1
)
if not exist "%DICT%" (
  if exist "%FALLBACK_DICT%" (
    echo [WARN] dict not found in Thai rec; using fallback
    set "DICT=%FALLBACK_DICT%"
  ) else (
    echo [ERROR] dict not found: %DICT% and fallback missing & pause & exit /b 1
  )
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%" >nul 2>nul

REM ===== PICK INPUT =====
set "SRC=%~1"
if "%SRC%"=="" set "SRC=%DEFAULT_IMG_DIR%"

REM --- robust DIR/FILE detection ---
set "IMG_MODE=FILE"
if exist "%SRC%\" set "IMG_MODE=DIR"

if "%IMG_MODE%"=="DIR" (
  echo Input Mode: DIR "%SRC%"
  set /a COUNT=0
  for %%e in (jpg jpeg png bmp tif tiff webp) do (
    for /f "delims=" %%F in ('dir /b /a-d "%SRC%\*.%%e" 2^>nul') do set /a COUNT+=1
  )
  echo Found images: !COUNT!
  if "!COUNT!"=="0" echo [ERROR] No images in folder & pause & exit /b 1
) else (
  echo Input Mode: FILE "%SRC%"
  if not exist "%SRC%" echo [ERROR] File not found & pause & exit /b 1
)


REM ===== RUN =====
set "GPU_FLAG=--use_gpu False"
if "%USE_GPU%"=="1" set "GPU_FLAG=--use_gpu True"

echo Running OCR...
"%PY%" -u "%PROJECT_DIR%tools\infer\predict_system.py" ^
  --image_dir "%SRC%" ^
  --det_model_dir "%DET_DIR%" ^
  --rec_model_dir "%REC_DIR%" ^
  --rec_char_dict_path "%DICT%" ^
  --use_angle_cls True ^
  %GPU_FLAG% ^
  --draw_img_save_dir "%OUT_DIR%" ^
  --save_log_path "%OUT_DIR%"

echo.
echo [DONE] Saved to: %OUT_DIR%
pause
