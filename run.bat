@echo off
cd /d "%~dp0"
pip install -q flask numpy requests 2>nul
python app.py
pause
