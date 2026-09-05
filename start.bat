@echo off
cd /d "%~dp0"

echo Starting RiskPred...
echo Dashboard: http://127.0.0.1:8000/dashboard
echo.

call "%~dp0venv\Scripts\activate.bat"

if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Virtual environment activation failed.
    echo Please ensure venv folder exists at: %~dp0venv
    pause
    exit /b 1
)

python -c "import sys; print('Python:', sys.executable)"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend

pause
