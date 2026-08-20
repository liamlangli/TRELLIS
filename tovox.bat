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

REM Optional 3rd arg: max VOX resolution (longest-axis voxels). Overrides OUT_RES env.
if not "%~3"=="" (
  set "OUT_RES=%~3"
)

REM Optional 4th arg: max palette colors (1..255; index 0 is air). Overrides MAX_COLORS env.
if not "%~4"=="" (
  set "MAX_COLORS=%~4"
)

if not exist "!IN_IMG!" (
  echo [ERROR] Image not found: %~1
  echo         resolved: !IN_IMG!
  echo         cwd: %CALLER_CWD%
  exit /b 1
)

if defined OUT_RES (
  echo !OUT_RES!| findstr /R /C:"^[1-9][0-9]*$" >nul
  if errorlevel 1 (
    echo [ERROR] Invalid max resolution: !OUT_RES!
    echo         Expected a positive integer, e.g. 128 or 256.
    exit /b 1
  )
)

if defined MAX_COLORS (
  echo !MAX_COLORS!| findstr /R /C:"^[1-9][0-9]*$" >nul
  if errorlevel 1 (
    echo [ERROR] Invalid max colors: !MAX_COLORS!
    echo         Expected a positive integer 1..255, e.g. 16 or 64.
    exit /b 1
  )
  if !MAX_COLORS! GTR 255 (
    echo [ERROR] Invalid max colors: !MAX_COLORS!
    echo         Expected a positive integer 1..255 ^(VOX palette index 0 is air^).
    exit /b 1
  )
)

if not defined VOX_URL (
  if not defined VOX_HOST set "VOX_HOST=127.0.0.1"
  if not defined VOX_PORT set "VOX_PORT=8080"
  set "VOX_URL=http://!VOX_HOST!:!VOX_PORT!"
)

set "QUERY="
if defined CONVERT_MODE set "QUERY=!QUERY!&mode=!CONVERT_MODE!"
if defined MODE set "QUERY=!QUERY!&mode=!MODE!"
if defined SEED set "QUERY=!QUERY!&seed=!SEED!"
if defined PIPELINE_TYPE set "QUERY=!QUERY!&pipeline_type=!PIPELINE_TYPE!"
if defined MATERIAL_MODE set "QUERY=!QUERY!&material_mode=!MATERIAL_MODE!"
if defined OUT_RES set "QUERY=!QUERY!&out_res=!OUT_RES!"
if defined MAX_COLORS set "QUERY=!QUERY!&max_colors=!MAX_COLORS!"
if defined ALPHA_THR set "QUERY=!QUERY!&alpha_threshold=!ALPHA_THR!"
if defined COLOR_AXIS set "QUERY=!QUERY!&color_axis=!COLOR_AXIS!"
if defined DOWNSAMPLE_DEVICE set "QUERY=!QUERY!&downsample_device=!DOWNSAMPLE_DEVICE!"
if defined COLOR_MODE set "QUERY=!QUERY!&color_mode=!COLOR_MODE!"
if defined VOX_FILL set "QUERY=!QUERY!&vox_fill=!VOX_FILL!"
if defined SURFACE_BAND set "QUERY=!QUERY!&surface_band=!SURFACE_BAND!"
if defined DECIMATE_TARGET set "QUERY=!QUERY!&decimate_target=!DECIMATE_TARGET!"
if defined TEXTURE_SIZE set "QUERY=!QUERY!&texture_size=!TEXTURE_SIZE!"
if defined KEEP_GLB set "QUERY=!QUERY!&keep_glb=!KEEP_GLB!"

if defined QUERY (
  set "CONVERT_URL=!VOX_URL!/convert?!QUERY:~1!"
) else (
  set "CONVERT_URL=!VOX_URL!/convert"
)

echo Converting via HTTP:
echo   server: !VOX_URL!
echo   image:  !IN_IMG!
echo   vox:    !OUT_VOX!
if defined OUT_RES (
  echo   out_res:!OUT_RES!
) else (
  echo   out_res:^(server default^)
)
if defined MAX_COLORS (
  echo   colors: !MAX_COLORS!
) else (
  echo   colors:^(server default^)
)
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
echo Usage: tovox.bat input.png [output.vox] [max_res] [max_colors]
echo.
echo Examples:
echo   tovox.bat a.png
echo   tovox.bat a.png b.vox
echo   tovox.bat a.png b.vox 128
echo   tovox.bat a.png b.vox 128 32
echo   tovox.bat .\photos\a.png out\b.vox 256 64
echo.
echo Paths are resolved from the current working directory.
echo Converts via HTTP against a running server_vox instance.
echo Start the server first:
echo   serve.bat
echo.
echo If output is omitted, writes ^<input^>.vox next to the image.
echo max_res is the longest-axis voxel resolution (query out_res).
echo It overrides OUT_RES env for this call when provided.
echo max_colors limits palette size to 1..255 (query max_colors; VOX index 0 is air).
echo It overrides MAX_COLORS env for this call when provided.
echo Server URL: set VOX_URL=http://127.0.0.1:8080
echo        or: set VOX_HOST / VOX_PORT
echo Optional convert overrides: CONVERT_MODE/MODE=glb^|direct, SEED, PIPELINE_TYPE,
echo OUT_RES, MAX_COLORS, COLOR_MODE, VOX_FILL, SURFACE_BAND, DECIMATE_TARGET(default 150000), TEXTURE_SIZE(default 1024),
echo KEEP_GLB, MATERIAL_MODE(default color), ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE
echo Default server mode is direct ^(TRELLIS voxels + base_color^); override with CONVERT_MODE=glb.
exit /b 1
