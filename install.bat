@echo off
cd /d "%~dp0"
echo === Voice Trainer Setup ===
echo.

set PYEXE=
if exist "C:\Python313\python.exe" set PYEXE=C:\Python313\python.exe
if exist "C:\Python312\python.exe" set PYEXE=C:\Python312\python.exe
if exist "C:\Python311\python.exe" set PYEXE=C:\Python311\python.exe
if exist "C:\Python310\python.exe" set PYEXE=C:\Python310\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PYEXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set PYEXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe

if "%PYEXE%"=="" (
    echo [ERROR] Python not found.
    echo Please install Python first:
    echo   winget install Python.Python.3.12
    echo   or https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python: %PYEXE%

if not exist ".venv\" (
    echo Creating virtual environment...
    "%PYEXE%" -m venv .venv
    if errorlevel 1 ( echo Failed to create venv & pause & exit /b 1 )
)

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip -q

echo Installing libraries (first time takes a few minutes)...
.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 ( echo Install failed & pause & exit /b 1 )

echo.
echo === Setup complete! ===
echo Run: run.bat
echo.
pause
