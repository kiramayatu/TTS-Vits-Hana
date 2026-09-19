@echo off
cd /d "%~dp0"
python -m uvicorn api.server:app --host 127.0.0.1 --port 7860 --workers 1
