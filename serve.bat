@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Project venv not found: .venv\Scripts\python.exe
  echo Create it first, then rerun serve.bat.
  exit /b 1
)

if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8080"
if not defined CONVERT_MODE set "CONVERT_MODE=glb"

echo Starting server_vox with project venv...
echo   python: %CD%\.venv\Scripts\python.exe
echo   host:   %HOST%
echo   port:   %PORT%
echo   mode:   %CONVERT_MODE%  (glb = full image-^>GLB-^>VOX)
echo.

".venv\Scripts\python.exe" "%~dp0server_vox.py" %*
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo [ERROR] server_vox exited with code %ERR%
)
exit /b %ERR%
