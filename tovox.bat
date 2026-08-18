@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Project venv not found: .venv\Scripts\python.exe
  echo Create it first, then rerun tovox.bat.
  exit /b 1
)

if not exist "%~1" (
  echo [ERROR] Image not found: %~1
  exit /b 1
)

set "IN_IMG=%~f1"
if "%~2"=="" (
  set "OUT_VOX=%~dpn1.vox"
) else (
  set "OUT_VOX=%~f2"
)

echo Converting:
echo   image: %IN_IMG%
echo   vox:   %OUT_VOX%
echo.

".venv\Scripts\python.exe" "%~dp0run_house_to_vox.py" "%IN_IMG%" "%OUT_VOX%"
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo [ERROR] conversion failed with code %ERR%
)
exit /b %ERR%

:usage
echo Usage: tovox.bat input.png [output.vox]
echo.
echo Examples:
echo   tovox.bat a.png b.vox
echo   tovox.bat a.png
echo.
echo If output is omitted, writes ^<input^>.vox next to the image.
echo Optional env overrides: SEED, PIPELINE_TYPE, MATERIAL_MODE, OUT_RES,
echo TRELLIS_MODEL, ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE
exit /b 1
