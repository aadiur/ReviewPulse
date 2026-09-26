@echo off
cd /d "%~dp0"
echo Starting ReviewPulse Professional Dashboard...
py -m streamlit run app_professional.py
pause
