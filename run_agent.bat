@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -m pip install -r requirements.txt
python agent.py
pause
