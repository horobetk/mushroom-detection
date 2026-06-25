@echo off
REM Uruchom serwer synchronizacji na ZDALNYM PC (3060TI-1).
REM Okno musi pozostac otwarte. Zatrzymanie: Ctrl+C

cd /d "%~dp0"

echo ============================================================
echo   SERWER SYNCHRONIZACJI (port 9999)
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo [BLAD] Brak .venv - uruchom najpierw setup_remote_win.ps1
    pause
    exit /b 1
)

echo Nasluch: http://0.0.0.0:8080  (dostepny po IP tej maszyny w VPN)
echo.
echo UWAGA: Przy pierwszym uruchomieniu Windows moze zapytac o firewall.
echo        Zaznacz "Sieci prywatne" i kliknij Zezwol.
echo.

.venv\Scripts\python.exe scripts\remote_receiver.py --port 8080
