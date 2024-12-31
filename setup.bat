@echo off
setlocal enabledelayedexpansion

:: Check for Python in PATH first
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%i in ('python -V 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo Found %PYTHON_VERSION%
    set "PYTHON=python"
    goto :setup
)

:: If not in PATH, check WindowsApps directory
for /f "tokens=*" %%i in ('dir /b /ad "%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\Python*" 2^>nul') do (
    set "PYTHON=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\%%i\python.exe"
    if exist "!PYTHON!" (
        for /f "tokens=*" %%v in ('"!PYTHON!" -V 2^>^&1') do set "PYTHON_VERSION=%%v"
        echo Found %PYTHON_VERSION%
        goto :setup
    )
)

echo Python not found
exit /b 1

:setup
:: Create and set up the virtual environment
"%PYTHON%" -m venv venv

:: Activate virtual environment and run commands
call venv\Scripts\activate.bat
python -m pip install uv
python -m uv clean
python -m uv pip install --upgrade pip
python -m uv pip install wheel
python -m uv pip install -r requirements.txt
pre-commit install
deactivate
echo Setup complete.