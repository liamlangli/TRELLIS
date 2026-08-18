@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

set "SCRIPT_DIR=%~dp0"
set "CALLER_CWD=%CD%"

pushd "%CALLER_CWD%" >nul
for %%I in ("%~1") do set "IN_IMG=%%~fI"
if "%~2"=="" (
  for %%I in ("%~1") do set "OUT_GLB=%%~dpnI.glb"
) else (
  for %%I in ("%~2") do set "OUT_GLB=%%~fI"
)
popd >nul

if not exist "!IN_IMG!" (
  echo [ERROR] Image not found: %~1
  echo         resolved: !IN_IMG!
  echo         cwd: %CALLER_CWD%
  exit /b 1
)

if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  echo [ERROR] Project venv not found: %SCRIPT_DIR%.venv\Scripts\python.exe
  exit /b 1
)

echo Full TRELLIS.2 image -^> GLB pipeline:
echo   image: !IN_IMG!
echo   glb:   !OUT_GLB!
echo.

if not defined REMESH set "REMESH=0"
if not defined PIPELINE_TYPE set "PIPELINE_TYPE=512"
if not defined SPARSE_CONV_BACKEND set "SPARSE_CONV_BACKEND=flex_gemm"

"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%img_to_glb.py" "!IN_IMG!" "!OUT_GLB!"
set "ERR=!ERRORLEVEL!"
if not "!ERR!"=="0" (
  echo.
  echo [ERROR] GLB export failed with code !ERR!
  echo If CUDA extensions are missing, run: %SCRIPT_DIR%.ext_build\build_ext.bat
)
exit /b !ERR!

:usage
echo Usage: toglb.bat input.png [output.glb]
echo.
echo Runs the official TRELLIS.2 mesh + o_voxel.postprocess.to_glb path.
echo Paths are resolved from the current working directory.
echo.
echo Optional env:
echo   SEED, PIPELINE_TYPE, TRELLIS_MODEL, LOW_VRAM
echo   DECIMATE_TARGET, TEXTURE_SIZE, SIMPLIFY_TARGET
echo   REMESH, REMESH_BAND, REMESH_PROJECT
exit /b 1
