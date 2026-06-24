@echo off
echo ==============================================
echo   History AI API — Setup
echo ==============================================

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (echo ERROR: Python not found. Install Python 3.10+ & quit /b 1)

echo [2/3] Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

echo [3/3] Done!
echo.
echo Run the server with:  start.bat
echo.
pause
