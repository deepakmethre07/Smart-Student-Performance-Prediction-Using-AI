@echo off
REM Student Performance Prediction Web App Launcher (Windows)
REM This script launches the Streamlit web application

echo ==========================================
echo Student Performance Prediction Web App
echo ==========================================
echo.
echo Starting Streamlit server...
echo.
echo The app will open in your default browser.
echo If it doesn't open automatically, visit:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ==========================================
echo.

streamlit run app.py
pause
