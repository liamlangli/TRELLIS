@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

set "SCRIPT_DIR=%~dp0"
set "CALLER_CWD=%CD%"

pushd "%CALLER_CWD%" >nul
for %%I in ("%~1") do set "IN_GLB=%%~fI"
if "%~2"=="" (
  for %%I in ("%~1") do set "OUT_VOX=%%~dpnI.vox"
) else (
  for %%I in ("%~2") do set "OUT_VOX=%%~fI"
)
popd >nul

if not exist "!IN_GLB!" (
  echo [ERROR] GLB/mesh not found: %~1
  echo         resolved: !IN_GLB!
  echo         cwd: %CALLER_CWD%
  exit /b 1
)

if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  echo [ERROR] Project venv not found: %SCRIPT_DIR%.venv\Scripts\python.exe
  exit /b 1
)

echo CUDA GLB -^> VOX2 voxelizer (CuMesh BVH):
echo   glb: !IN_GLB!
echo   vox: !OUT_VOX!
echo.

if not defined OUT_RES set "OUT_RES=256"
if not defined VOX_FILL set "VOX_FILL=1"
if not defined COLOR_MODE set "COLOR_MODE=texture"
if not defined SPARSE_CONV_BACKEND set "SPARSE_CONV_BACKEND=flex_gemm"

"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%glb_to_vox.py" "!IN_GLB!" "!OUT_VOX!"
set "ERR=!ERRORLEVEL!"
if not "!ERR!"=="0" (
  echo.
  echo [ERROR] glb_to_vox failed with code !ERR!
)
exit /b !ERR!

:usage
echo Usage: glb2vox.bat input.glb [output.vox]
echo.
echo Paths are resolved from the current working directory.
echo Optional env: OUT_RES, VOX_FILL, SURFACE_BAND, MAX_COLORS,
echo   SIMPLIFY_FACES, COLOR_MODE, SDF_MODE, PAD_VOXELS, CHUNK
exit /b 1
