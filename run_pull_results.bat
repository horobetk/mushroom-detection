@echo off
REM Pobranie wynikow ze zdalnego PC na laptop (TCP, bez HTTP).
REM Uzycie:  run_pull_results.bat            (host domyslny 10.44.25.85)
REM          run_pull_results.bat 10.44.25.85
cd /d "%~dp0"

set HOST=%1
if "%HOST%"=="" set HOST=10.44.25.85

echo ============================================================
echo   POBIERANIE WYNIKOW Z %HOST% (port 5500)
echo ============================================================

python scripts\pull_results_tcp.py --host %HOST% --port 5500
pause
