@echo off
REM Quick start script dla Windows
REM Uruchamia dataset creator z domyślnymi ustawieniami

echo ========================================
echo   MUSHROOM DATASET CREATOR
echo   Politechnika Warszawska 2026
echo ========================================
echo.

REM Sprawdź czy Python jest zainstalowany
python --version >nul 2>&1
if errorlevel 1 (
    echo BLAD: Python nie jest zainstalowany!
    echo Zainstaluj Python z https://python.org
    pause
    exit /b 1
)

echo [1/3] Sprawdzanie zaleznosci...
pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo Instalacja opencv-python...
    pip install opencv-python numpy tqdm
)

echo.
echo [2/3] Sprawdzanie struktury katalogow...
if not exist "data\raw" (
    echo UWAGA: Folder data\raw nie istnieje!
    echo Utworzenie folderu...
    mkdir data\raw
    echo.
    echo UMIES WIDEO W FOLDERZE: data\raw\
    echo Potem uruchom ten skrypt ponownie.
    pause
    exit /b 1
)

REM Sprawdź czy są pliki wideo
dir /b data\raw\*.mp4 data\raw\*.avi data\raw\*.mov 2>nul | findstr . >nul
if errorlevel 1 (
    echo.
    echo UWAGA: Brak plikow wideo w data\raw\
    echo.
    echo Skopiuj pliki wideo do folderu data\raw\ i uruchom ponownie.
    pause
    exit /b 1
)

echo.
echo [3/3] Uruchamianie dataset creator...
echo.
echo Konfiguracja:
echo - Input: data\raw
echo - Output: data\frames
echo - Interval: co 30-ta klatka
echo - Augmentacja: WLACZONA
echo - Resize: 640x640 (YOLO11 ready)
echo.

python scripts\advanced_dataset_creator.py ^
    --input data\raw ^
    --output data\frames ^
    --interval 30 ^
    --resize 640 640

echo.
echo ========================================
echo   GOTOWE!
echo ========================================
echo.
echo Sprawdz rezultaty:
echo - Oryginalne klatki: data\frames\original\
echo - Augmentowane: data\frames\augmented\
echo - Statystyki: data\frames\statistics.json
echo.

pause
