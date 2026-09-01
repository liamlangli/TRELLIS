@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Create it first.
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python img_to_vox.py --input_folder "C:\code\TRELLIS.2\vox" --skip
exit /b %ERRORLEVEL%
