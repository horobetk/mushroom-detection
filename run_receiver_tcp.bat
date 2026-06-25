@echo off
REM Serwer TCP synchronizacji (BEZ HTTP) - dla VPN blokujacego HTTP
cd /d "%~dp0"

echo ============================================================
echo   SERWER TCP SYNCHRONIZACJI (port 5500)
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo [BLAD] Brak .venv
    pause
    exit /b 1
)

echo Protokol: SYNC_PING / SYNC_PUT (bez HTTP)
echo Przy pierwszym uruchomieniu: Zezwol w firewall dla Python (sieci prywatne)
echo.

.venv\Scripts\python.exe scripts\remote_receiver_tcp.py --port 5500
