@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Project venv not found: .venv\Scripts\python.exe
  echo Create it first, then rerun serve.bat.
  exit /b 1
)

if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8080"
if not defined CONVERT_MODE set "CONVERT_MODE=direct"
if not defined MATERIAL_MODE set "MATERIAL_MODE=color"
if not defined COLOR_MODE set "COLOR_MODE=texture"
if not defined OUT_RES set "OUT_RES=256"
if not defined PIPELINE_TYPE set "PIPELINE_TYPE=512"
if not defined PHOTO_COLOR set "PHOTO_COLOR=1"
if not defined DECIMATE_TARGET set "DECIMATE_TARGET=150000"
if not defined TEXTURE_SIZE set "TEXTURE_SIZE=1024"
if not defined PRELOAD set "PRELOAD=1"
if not defined TRELLIS_MODEL set "TRELLIS_MODEL=microsoft/TRELLIS.2-4B"
if not defined REMESH set "REMESH=0"
if not defined VOX_FILL set "VOX_FILL=1"
if not defined SURFACE_BAND set "SURFACE_BAND=0.75"
if not defined COLOR_AXIS set "COLOR_AXIS=auto"
if not defined ALPHA_THR set "ALPHA_THR=0.5"

set "MODE_NORM=%CONVERT_MODE%"
if /I "%MODE_NORM%"=="full" set "MODE_NORM=glb"
if /I "%MODE_NORM%"=="img_glb_vox" set "MODE_NORM=glb"
if /I "%MODE_NORM%"=="image_glb_vox" set "MODE_NORM=glb"
if /I "%MODE_NORM%"=="glb_path" set "MODE_NORM=glb"
if /I "%MODE_NORM%"=="fast" set "MODE_NORM=direct"
if /I "%MODE_NORM%"=="mesh" set "MODE_NORM=direct"
if /I "%MODE_NORM%"=="mesh_voxel" set "MODE_NORM=direct"
if /I "%MODE_NORM%"=="direct_mesh" set "MODE_NORM=direct"

if /I not "%MODE_NORM%"=="direct" if /I not "%MODE_NORM%"=="glb" (
  echo [WARN] Unknown CONVERT_MODE=%CONVERT_MODE%, treating as glb.
  set "MODE_NORM=glb"
)

if /I "%MODE_NORM%"=="direct" (
  set "FLOW=image -> TRELLIS MeshWithVoxel -> quantize attrs/photo -> VOX2"
  set "FLOW_DETAIL=direct path (no GLB / no xatlas packing)"
  set "COLOR_DESC=MATERIAL_MODE=%MATERIAL_MODE%  (color=official base_color, image=photo project, auto=base_color then photo)"
) else (
  set "FLOW=image -> TRELLIS mesh -> bake GLB -> voxelize -> VOX2"
  set "FLOW_DETAIL=glb path (CuMesh UV unwrap + texture bake)"
  set "COLOR_DESC=COLOR_MODE=%COLOR_MODE%"
  if /I not "%PHOTO_COLOR%"=="0" if /I not "%MATERIAL_MODE%"=="texture" if /I not "%MATERIAL_MODE%"=="glb" if /I not "%MATERIAL_MODE%"=="baked" (
    set "COLOR_DESC=!COLOR_DESC! + PHOTO_COLOR=on (repaint solids from source image)"
  ) else (
    set "COLOR_DESC=!COLOR_DESC! (baked GLB colors only)"
  )
)

echo.
echo ============================================================
echo  TRELLIS.2 VOX server
echo ============================================================
echo   python : %CD%\.venv\Scripts\python.exe
echo   listen : http://%HOST%:%PORT%
echo   model  : %TRELLIS_MODEL%
echo   preload: %PRELOAD%
echo.
echo   CONVERT_MODE = %CONVERT_MODE%  =^>  %MODE_NORM%
echo   flow         : !FLOW!
echo                  !FLOW_DETAIL!
echo   pipeline     : %PIPELINE_TYPE%
echo   out_res      : %OUT_RES%
echo   color        : !COLOR_DESC!
if /I "%MODE_NORM%"=="glb" (
  echo   glb knobs    : DECIMATE_TARGET=%DECIMATE_TARGET%  TEXTURE_SIZE=%TEXTURE_SIZE%
  echo                  REMESH=%REMESH%  VOX_FILL=%VOX_FILL%  SURFACE_BAND=%SURFACE_BAND%
) else (
  echo   direct knobs : MATERIAL_MODE=%MATERIAL_MODE%  COLOR_AXIS=%COLOR_AXIS%  ALPHA_THR=%ALPHA_THR%
)
echo.
echo   Endpoints:
echo     GET  /health
echo     POST /convert[?mode=glb^|direct^&out_res=256^&...]
echo     POST /convert.json
echo   Client tip:
echo     tovox.bat image.jpg out.vox 256
echo ============================================================
echo.

".venv\Scripts\python.exe" "%~dp0server_vox.py" %*
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo [ERROR] server_vox exited with code %ERR%
)
exit /b %ERR%
