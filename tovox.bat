@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

REM Keep caller CWD for relative paths; only locate repo tools by script dir.
set "SCRIPT_DIR=%~dp0"
set "CALLER_CWD=%CD%"

REM Resolve input against the caller's current directory.
pushd "%CALLER_CWD%" >nul
for %%I in ("%~1") do set "IN_IMG=%%~fI"
if "%~2"=="" (
  for %%I in ("%~1") do set "OUT_VOX=%%~dpnI.vox"
) else (
  for %%I in ("%~2") do set "OUT_VOX=%%~fI"
)
popd >nul

if not exist "!IN_IMG!" (
  echo [ERROR] Image not found: %~1
  echo         resolved: !IN_IMG!
  echo         cwd: %CALLER_CWD%
  exit /b 1
)

if not defined VOX_URL (
  if not defined VOX_HOST set "VOX_HOST=127.0.0.1"
  if not defined VOX_PORT set "VOX_PORT=8080"
  set "VOX_URL=http://!VOX_HOST!:!VOX_PORT!"
)

set "QUERY="
if defined SEED set "QUERY=!QUERY!&seed=!SEED!"
if defined PIPELINE_TYPE set "QUERY=!QUERY!&pipeline_type=!PIPELINE_TYPE!"
if defined MATERIAL_MODE set "QUERY=!QUERY!&material_mode=!MATERIAL_MODE!"
if defined OUT_RES set "QUERY=!QUERY!&out_res=!OUT_RES!"
if defined ALPHA_THR set "QUERY=!QUERY!&alpha_threshold=!ALPHA_THR!"
if defined COLOR_AXIS set "QUERY=!QUERY!&color_axis=!COLOR_AXIS!"
if defined DOWNSAMPLE_DEVICE set "QUERY=!QUERY!&downsample_device=!DOWNSAMPLE_DEVICE!"

if defined QUERY (
  set "CONVERT_URL=!VOX_URL!/convert?!QUERY:~1!"
) else (
  set "CONVERT_URL=!VOX_URL!/convert"
)

echo Converting via HTTP:
echo   server: !VOX_URL!
echo   image:  !IN_IMG!
echo   vox:    !OUT_VOX!
echo.

where curl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] curl.exe not found. Install curl or use Windows 10+ built-in curl.
  exit /b 2
)

curl -sS --fail --max-time 5 "!VOX_URL!/health" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Cannot reach VOX server at !VOX_URL!
  echo Start it first: "%SCRIPT_DIR%serve.bat"
  exit /b 3
)

set "TMP_OUT=!TEMP!\tovox_%RANDOM%_%RANDOM%.vox"
curl -sS --fail --show-error ^
  -X POST ^
  --data-binary "@!IN_IMG!" ^
  -H "Content-Type: application/octet-stream" ^
  -o "!TMP_OUT!" ^
  "!CONVERT_URL!"
set "ERR=!ERRORLEVEL!"
if not "!ERR!"=="0" (
  echo.
  echo [ERROR] HTTP conversion failed with code !ERR!
  if exist "!TMP_OUT!" del /q "!TMP_OUT!" >nul 2>nul
  exit /b !ERR!
)

for %%A in ("!TMP_OUT!") do set "OUT_SIZE=%%~zA"
if "!OUT_SIZE!"=="" set "OUT_SIZE=0"
if "!OUT_SIZE!"=="0" (
  echo [ERROR] Server returned empty response.
  del /q "!TMP_OUT!" >nul 2>nul
  exit /b 4
)

for %%A in ("!OUT_VOX!") do set "OUT_DIR=%%~dpA"
if not exist "!OUT_DIR!" mkdir "!OUT_DIR!" >nul 2>nul
move /y "!TMP_OUT!" "!OUT_VOX!" >nul
if errorlevel 1 (
  echo [ERROR] Failed to write !OUT_VOX!
  del /q "!TMP_OUT!" >nul 2>nul
  exit /b 5
)

for %%A in ("!OUT_VOX!") do set "FINAL_SIZE=%%~zA"
echo OK wrote !OUT_VOX!  bytes=!FINAL_SIZE!
exit /b 0

:usage
echo Usage: tovox.bat input.png [output.vox]
echo.
echo Examples:
echo   tovox.bat a.png b.vox
echo   tovox.bat .\photos\a.png out\b.vox
echo.
echo Paths are resolved from the current working directory.
echo Converts via HTTP against a running server_vox instance.
echo Start the server first:
echo   serve.bat
echo.
echo If output is omitted, writes ^<input^>.vox next to the image.
echo Server URL: set VOX_URL=http://127.0.0.1:8080
echo        or: set VOX_HOST / VOX_PORT
echo Optional convert overrides: SEED, PIPELINE_TYPE, MATERIAL_MODE, OUT_RES,
echo ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE
exit /b 1
