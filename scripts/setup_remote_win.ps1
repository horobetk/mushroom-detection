<#
.SYNOPSIS
    Jednorazowa inicjalizacja srodowiska pod trening YOLO11 - BEZ uprawnien administratora.

.DESCRIPTION
    Przygotowuje maszyne (lokalna lub zdalna uniwersytecka) do treningu i eksportu:
      1. Znajduje juz zainstalowany, zgodny Python (3.12 lub 3.13) - bez instalacji.
         TensorFlow (potrzebny do eksportu TFLite) wspiera Python 3.11-3.13,
         dlatego Python 3.14 nie jest uzywany.
      2. Jesli nie ma zgodnego Pythona -> instaluje Python 3.12 PER-USER
         (InstallAllUsers=0, bez UAC / bez praw admina).
      3. Tworzy wirtualne srodowisko .venv.
      4. Instaluje PyTorch z obsluga CUDA (GPU) oraz zaleznosci z requirements.txt.

.PARAMETER CudaIndex
    Indeks pakietow PyTorch z CUDA (domyslnie cu124). Mozliwe: cu118, cu121, cu124, cpu.

.PARAMETER VenvPath
    Sciezka tworzonego srodowiska wirtualnego (domyslnie: .venv).

.PARAMETER ForceInstall
    Wymus instalacje Python 3.12 nawet, gdy istnieje inny zgodny Python.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_remote_win.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_remote_win.ps1 -CudaIndex cu121

.NOTES
    NIE wymaga uprawnien administratora. Uruchom z katalogu glownego projektu.
    Autor: Kiril Horobets, Politechnika Warszawska, 2026.
#>

param(
    [ValidateSet("cu118", "cu121", "cu124", "cpu")]
    [string]$CudaIndex = "cu124",

    [string]$VenvPath = ".venv",

    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"

# Katalog glowny projektu (rodzic folderu scripts)
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

# Wersje Pythona akceptowane przez TensorFlow (kolejnosc = priorytet)
$CompatibleVersions = @("3.12", "3.13")

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  INICJALIZACJA SRODOWISKA (bez praw administratora)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Projekt:    $ProjectRoot"
Write-Host "  CUDA index: $CudaIndex"
Write-Host "  Venv:       $VenvPath"
Write-Host "============================================================`n"


function Get-PythonExe {
    <#
        Zwraca sciezke do python.exe pierwszej znalezionej, zgodnej wersji.
        Najpierw probuje launchera 'py', a w razie braku - bezposrednich
        sciezek instalacji per-user. Zwraca $null, gdy nie znaleziono.
    #>
    param([string[]]$Versions)

    # 1) Launcher 'py'
    foreach ($v in $Versions) {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) {
                return $exe.Trim()
            }
        }
        catch {
            # launcher 'py' moze nie istniec - ignorujemy i probujemy dalej
        }
    }

    # 2) Bezposrednie sciezki instalacji per-user (gdy 'py' jeszcze nie w PATH)
    foreach ($v in $Versions) {
        $ver = $v -replace '\.', ''  # np. 3.12 -> 312
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$ver\python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}


function Install-Python312PerUser {
    <#
        Instaluje Python 3.12 WYLACZNIE dla biezacego uzytkownika - calkowicie
        bez praw administratora i bez okna UAC.

        UWAGA: nie uzywamy 'winget', bo pakiet Python.Python.3.12 zawsze probuje
        instalacji systemowej (wymusza UAC). Oficjalny instalator z dwoma flagami
        *AllUsers=0 instaluje sie cicho w AppData bez podnoszenia uprawnien.
    #>
    Write-Host "       Pobieranie oficjalnego instalatora python.org (per-user)..."
    $PyVer = "3.12.10"
    $Url = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-amd64.exe"
    $Installer = Join-Path $env:TEMP "python-$PyVer-amd64.exe"

    # Wylaczenie paska postepu przyspiesza pobieranie w skrypcie
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Url -OutFile $Installer

    Write-Host "       Cicha instalacja per-user (bez praw admina, bez UAC)..."
    # KLUCZOWE: oba *AllUsers=0 (instalacja ORAZ launcher tylko dla uzytkownika),
    # co eliminuje monit UAC o haslo administratora.
    Start-Process -FilePath $Installer -Wait -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "InstallLauncherAllUsers=0",
        "PrependPath=1",
        "Include_launcher=1",
        "Include_pip=1"
    )

    # Odswiezenie PATH w biezacej sesji
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")

    return ($null -ne (Get-PythonExe -Versions @("3.12")))
}


# --- 1. Znalezienie / instalacja zgodnego Pythona ---
Write-Host "[1/4] Szukanie zgodnego Pythona (3.12 / 3.13) ..."

$PythonExe = $null
if (-not $ForceInstall) {
    $PythonExe = Get-PythonExe -Versions $CompatibleVersions
}

if ($PythonExe) {
    Write-Host "       Znaleziono: $PythonExe" -ForegroundColor Green
    Write-Host "       (Pomijam instalacje - uzywam istniejacego Pythona.)`n"
}
else {
    Write-Host "       Brak zgodnego Pythona - instaluje Python 3.12 (per-user)..."
    if (-not (Install-Python312PerUser)) {
        Write-Host "[BLAD] Nie udalo sie zainstalowac Pythona 3.12 bez praw admina." -ForegroundColor Red
        Write-Host "       Zainstaluj recznie z https://www.python.org/downloads/release/python-3128/" -ForegroundColor Red
        Write-Host "       (zaznacz 'Install for me only' / 'Add python.exe to PATH') i uruchom ponownie." -ForegroundColor Red
        exit 1
    }
    $PythonExe = Get-PythonExe -Versions @("3.12")
    Write-Host "       Zainstalowano: $PythonExe`n" -ForegroundColor Green
}


# --- 2. Tworzenie srodowiska wirtualnego ---
Write-Host "[2/4] Tworzenie srodowiska wirtualnego: $VenvPath ..."
if (-not (Test-Path $VenvPath)) {
    & $PythonExe -m venv $VenvPath
}
else {
    Write-Host "       Srodowisko juz istnieje - pomijam tworzenie."
}

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
    & $VenvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaIndex"
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BLAD] Instalacja PyTorch nie powiodla sie." -ForegroundColor Red
    exit 1
}
Write-Host "       PyTorch zainstalowany.`n" -ForegroundColor Green


# --- 4. Instalacja pozostalych zaleznosci z requirements.txt ---
Write-Host "[4/4] Instalacja zaleznosci z requirements.txt ..."
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[UWAGA] Niektore pakiety z requirements.txt nie zainstalowaly sie." -ForegroundColor Yellow
    Write-Host "        Sprawdz log powyzej (np. labelImg bywa problematyczny - mozna pominac)." -ForegroundColor Yellow
}

# Szybka weryfikacja kluczowych bibliotek + dostepnosci GPU
Write-Host "`n--- Weryfikacja srodowiska ---"
& $VenvPython -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA dostepne:', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"

# TensorFlow loguje ostrzezenia na stderr -> PowerShell traktuje to jako blad.
# Wyciszamy logi TF i filtrujemy tylko linie z wynikiem weryfikacji.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$tfOut = & $VenvPython -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print('tensorflow', tf.__version__)" 2>&1
$ErrorActionPreference = $prevEAP
$tfLine = $tfOut | Where-Object { $_ -match '^tensorflow ' } | Select-Object -First 1
if ($tfLine) {
    Write-Host $tfLine -ForegroundColor Green
} else {
    Write-Host "[UWAGA] Nie udalo sie potwierdzic importu tensorflow." -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  SRODOWISKO GOTOWE. Uruchom trening: .\run_training.bat" -ForegroundColor Green
Write-Host "============================================================"
