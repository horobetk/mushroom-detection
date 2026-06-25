<#
.SYNOPSIS
    Jednorazowa inicjalizacja zdalnego komputera (Windows) pod trening YOLO11.

.DESCRIPTION
    Przygotowuje czysta maszyne uniwersytecka:
      1. Instaluje Python 3.12 (przez winget, z fallbackiem na oficjalny instalator).
      2. Tworzy wirtualne srodowisko .venv (Python 3.12).
      3. Instaluje PyTorch z obsluga CUDA (GPU) z dedykowanego indeksu.
      4. Instaluje pozostale zaleznosci z requirements.txt (ultralytics, tensorflow, ...).

    Python 3.12 jest wymagany, bo TensorFlow (potrzebny do eksportu TFLite) nie
    posiada jeszcze pakietow dla Python 3.14.

.PARAMETER CudaIndex
    Indeks pakietow PyTorch z CUDA (domyslnie cu124 = CUDA 12.4).
    Mozliwe: cu121, cu124, cu118 lub 'cpu' (gdy brak GPU).

.PARAMETER VenvPath
    Sciezka tworzonego srodowiska wirtualnego (domyslnie: .venv).

.EXAMPLE
    .\scripts\setup_remote_win.ps1

.EXAMPLE
    .\scripts\setup_remote_win.ps1 -CudaIndex cu121

.NOTES
    Uruchom z katalogu glownego projektu. Zalecane uprawnienia administratora
    (instalacja Pythona). Autor: Kiril Horobets, Politechnika Warszawska, 2026.
#>

param(
    [ValidateSet("cu118", "cu121", "cu124", "cpu")]
    [string]$CudaIndex = "cu124",

    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

# Katalog glowny projektu (rodzic folderu scripts)
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  INICJALIZACJA ZDALNEGO SRODOWISKA (Python 3.12 + CUDA)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Projekt:    $ProjectRoot"
Write-Host "  CUDA index: $CudaIndex"
Write-Host "  Venv:       $VenvPath"
Write-Host "============================================================`n"

# --- 1. Instalacja Python 3.12 ---
Write-Host "[1/4] Sprawdzanie / instalacja Python 3.12 ..."

function Test-Py312 {
    # Zwraca $true, jesli launcher 'py -3.12' jest dostepny
    try {
        $v = & py -3.12 --version 2>$null
        return ($LASTEXITCODE -eq 0 -and $v -match "3\.12")
    }
    catch { return $false }
}

if (Test-Py312) {
    Write-Host "       Python 3.12 juz zainstalowany.`n" -ForegroundColor Green
}
else {
    $installed = $false

    # Proba 1: winget (Menedzer pakietow Windows)
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "       Instalacja przez winget..."
        winget install --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        # Odswiezenie PATH w biezacej sesji
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        $installed = Test-Py312
    }

    # Proba 2: pobranie oficjalnego instalatora z python.org
    if (-not $installed) {
        Write-Host "       winget niedostepny lub nieudany - pobieram instalator z python.org..."
        $PyVer = "3.12.8"
        $Url = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-amd64.exe"
        $Installer = Join-Path $env:TEMP "python-$PyVer-amd64.exe"

        Invoke-WebRequest -Uri $Url -OutFile $Installer
        # Cicha instalacja: dodaj do PATH, zainstaluj launcher py
        Start-Process -FilePath $Installer -Wait -ArgumentList @(
            "/quiet",
            "InstallAllUsers=1",
            "PrependPath=1",
            "Include_launcher=1"
        )
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        $installed = Test-Py312
    }

    if (-not $installed) {
        Write-Host "[BLAD] Nie udalo sie zainstalowac Pythona 3.12." -ForegroundColor Red
        Write-Host "       Zainstaluj recznie z https://www.python.org/downloads/ i uruchom ponownie." -ForegroundColor Red
        exit 1
    }
    Write-Host "       Python 3.12 zainstalowany.`n" -ForegroundColor Green
}

# --- 2. Tworzenie srodowiska wirtualnego ---
Write-Host "[2/4] Tworzenie srodowiska wirtualnego: $VenvPath ..."
if (-not (Test-Path $VenvPath)) {
    & py -3.12 -m venv $VenvPath
}
else {
    Write-Host "       Srodowisko juz istnieje - pomijam tworzenie."
}

# Sciezki do plikow wykonywalnych w venv
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[BLAD] Nie znaleziono $VenvPython - tworzenie venv nie powiodlo sie." -ForegroundColor Red
    exit 1
}
Write-Host "       Srodowisko gotowe.`n" -ForegroundColor Green

# --- 3. Aktualizacja pip + instalacja PyTorch (CUDA) ---
Write-Host "[3/4] Instalacja PyTorch ($CudaIndex) ..."
& $VenvPython -m pip install --upgrade pip

if ($CudaIndex -eq "cpu") {
    & $VenvPython -m pip install torch torchvision
}
else {
    # PyTorch z obsluga GPU/CUDA z dedykowanego indeksu Ultralytics/PyTorch
    & $VenvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaIndex"
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BLAD] Instalacja PyTorch nie powiodla sie." -ForegroundColor Red
    exit 1
}
Write-Host "       PyTorch zainstalowany.`n" -ForegroundColor Green

# --- 4. Instalacja pozostalych zaleznosci z requirements.txt ---
Write-Host "[4/4] Instalacja zaleznosci z requirements.txt ..."
# Torch jest juz zainstalowany powyzej (z CUDA), pip pominie ponowna instalacje.
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[UWAGA] Niektore pakiety z requirements.txt nie zainstalowaly sie." -ForegroundColor Yellow
    Write-Host "        Sprawdz log powyzej (np. labelImg bywa problematyczny - mozna pominac)." -ForegroundColor Yellow
}

# Szybka weryfikacja kluczowych bibliotek + dostepnosci GPU
Write-Host "`n--- Weryfikacja srodowiska ---"
& $VenvPython -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA dostepne:', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
& $VenvPython -c "import tensorflow as tf; print('tensorflow', tf.__version__)" 2>$null

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  SRODOWISKO GOTOWE. Uruchom trening: .\run_training.bat" -ForegroundColor Green
Write-Host "============================================================"
