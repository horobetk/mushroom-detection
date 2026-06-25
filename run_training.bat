@echo off
REM ============================================================
REM   PELNY PIPELINE TRENINGU MVP - ZDALNY KOMPUTER (Windows)
REM   Politechnika Warszawska 2026 - Kiril Horobets
REM ------------------------------------------------------------
REM   Aktywuje LOKALNE srodowisko .venv (Python 3.12 zainstalowany
REM   per-user w AppData, BEZ uprawnien administratora) i uruchamia po kolei:
REM     1) download_model.py        - pobranie YOLO11n
REM     2) download_mvp_dataset.py  - pobranie i przygotowanie datasetu
REM     3) train_mvp.py             - trening (50 epok, GPU)
REM     4) export_tflite.py         - eksport do TFLite (int8 -> fallback fp16)
REM ============================================================

setlocal

REM Przejscie do katalogu, w ktorym lezy ten plik (glowny katalog projektu)
cd /d "%~dp0"

echo ============================================================
echo   PIPELINE TRENINGU MVP - START
echo ============================================================

REM --- Aktywacja lokalnego srodowiska wirtualnego Python 3.12 ---
if not exist ".venv\Scripts\activate.bat" (
    echo [BLAD] Nie znaleziono .venv\Scripts\activate.bat
    echo        Utworz srodowisko bez praw admina, np.:
    echo            py -3.12 -m venv .venv
    echo            .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)
call ".venv\Scripts\activate.bat"

echo.
echo [1/4] Pobieranie bazowego modelu YOLO11n...
python scripts\download_model.py
if errorlevel 1 (
    echo [BLAD] download_model.py zakonczyl sie bledem.
    exit /b 1
)

echo.
echo [2/4] Pobieranie i przygotowanie datasetu MVP...
python scripts\download_mvp_dataset.py -c Amanita Boletus Cantharellus Macrolepiota Russula -n 50
if errorlevel 1 (
    echo [BLAD] download_mvp_dataset.py zakonczyl sie bledem.
    exit /b 1
)

echo.
echo [3/4] Trening modelu (50 epok, GPU)...
python scripts\train_mvp.py --data datasets\mushrooms_mvp\mushrooms_mvp.yaml --model weights\yolo11n.pt --epochs 50 --batch 16 --device 0
if errorlevel 1 (
    echo [BLAD] train_mvp.py zakonczyl sie bledem.
    exit /b 1
)

echo.
echo [4/4] Eksport modelu do TensorFlow Lite (int8 z fallbackiem na fp16)...
python scripts\export_tflite.py --weights runs\detect\train\weights\best.pt --quant int8 --output android\app\models\mushroom_detector.tflite
if errorlevel 1 (
    echo [BLAD] export_tflite.py zakonczyl sie bledem.
    exit /b 1
)

echo.
echo ============================================================
echo   PIPELINE ZAKONCZONY POMYSLNIE
echo   Model TFLite: android\app\models\mushroom_detector.tflite
echo ============================================================

endlocal
