@echo off
rem Spusteni FEA Studia (GUI). Lze pretahnout STEP nebo .fea na tento soubor.
rem PYTHONHOME / PYTHONPATH jineho programu (GIS, CAD, starsi Python) by prostredi .venv nespustily.
set "PYTHONHOME="
set "PYTHONPATH="
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -E -m fea_gui %*
