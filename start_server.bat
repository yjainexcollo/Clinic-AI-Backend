@echo off
echo Starting Clinic-AI Backend Server...
echo.

REM Set environment variables
set ENCRYPTION_KEY=mZH9wEs_8IGLAkQawgWw31YCDZ6X2LJIWD-w9ZL-iP8=
set OPENAI_API_KEY=sk-placeholder-key-for-development
set DEBUG=true

echo Environment variables set:
echo - ENCRYPTION_KEY: Set
echo - OPENAI_API_KEY: Set (placeholder)
echo - DEBUG: true
echo.

REM Start the server
echo Starting server...
uvicorn --app-dir src clinicai.app:app --reload --host 0.0.0.0 --port 8000

pause
