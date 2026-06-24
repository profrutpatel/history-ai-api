@echo off
echo ==============================================
echo   History AI API — Running Quality Checks
echo ==============================================
echo.
call venv\Scripts\activate.bat
cd backend
pytest ..\tests\test_api.py -v --tb=short
echo.
pause
