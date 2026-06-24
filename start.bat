@echo off
echo ==============================================
echo   History AI API — Starting Server
echo ==============================================
echo.
echo API will be live at:  http://localhost:8000
echo Swagger docs at:      http://localhost:8000/docs
echo Chat UI — open:       frontend\index.html
echo.
echo Press Ctrl+C to stop the server.
echo.
cd backend
C:\Users\comed\.gemini\antigravity-ide\scratch\qlora-finetune\venv\Scripts\python.exe main.py
