@echo off
chcp 65001 >nul
echo =======================================================
echo 🍽️  Kin Nai Dee (กินไหนดี) Web Application
echo =======================================================
cd /d "%~dp0"
echo Starting web server at http://127.0.0.1:5000 ...
if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe app.py
) else (
    python app.py
)
pause
