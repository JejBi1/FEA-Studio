@echo off
rem Installs FEA Studio by modules (everything stays in this folder, no admin rights needed).
rem Arguments: --profile minimal^|standard^|full, module names, --no-shortcuts, --no-check, --force. Details: installer\README.md
setlocal
cd /d "%~dp0"
rem PYTHONHOME / PYTHONPATH of another program would break the bundled Python and .venv
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"

rem Package with a bundled Python 3.11 (zip, setup.exe): the installer runs with it directly - no PowerShell (group policy
rem ExecutionPolicy / Constrained Language mode in companies, antivirus flags "powershell -ExecutionPolicy Bypass")
set "PY="
for /d %%d in ("%~dp0tools\python\cpython-3.11.*-windows-x86_64-none") do if exist "%%~d\python.exe" set "PY=%%~d\python.exe"
if not defined PY goto powershell

set "CMD=install"
for %%c in (install update remove list doctor uninstall shortcuts fusion slim mpi) do if /i "%~1"=="%%c" set "CMD="
"%PY%" -E -s "%~dp0installer\fea_setup.py" %CMD% %*
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%

:powershell
rem without a bundled Python: install.ps1 finds Python 3.9+ in the system, otherwise downloads uv
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install.ps1" %*
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%
