<#
.SYNOPSIS
    Pobieranie wynikow treningu ze zdalnego komputera na lokalny laptop (tryb PULL).

.DESCRIPTION
    Odwrotna synchronizacja wzgledem sync_to_remote.ps1. Laczy sie z tym samym
    udzialem sieciowym (\\REMOTE_IP\MushroomProject) i punktowo pobiera WYLACZNIE
    wyniki treningu:
      1. Wyeksportowany model TFLite -> android/app/models/mushroom_detector.tflite
      2. Zawartosc runs/detect/<RunName>/ (wykresy: results.png, confusion_matrix.png,
         *_curve.png oraz results.csv) - do lokalnej analizy metryk.

    Celowo NIE pobiera surowego datasetu, srodowiska .venv ani (domyslnie) wag .pt,
    aby nie przesylac gigabajtow danych przez siec. Operacja jest addytywna
    (nie kasuje lokalnych plikow - brak /MIR).

.PARAMETER RemoteHost
    Adres IP lub nazwa hosta zdalnego komputera (np. 10.8.0.5).

.PARAMETER ShareName
    Nazwa udostepnionego folderu na zdalnej maszynie (domyslnie: MushroomProject).

.PARAMETER RunName
    Nazwa eksperymentu w runs/detect/ do pobrania (domyslnie: train).

.PARAMETER User
    Opcjonalna nazwa uzytkownika do logowania do udzialu sieciowego.

.PARAMETER IncludeWeights
    Dodatkowo pobierz wagi .pt (runs/detect/<RunName>/weights/best.pt, last.pt).
    Domyslnie pomijane.

.PARAMETER DryRun
    Tryb testowy - tylko podglad (Robocopy /L), bez realnego kopiowania.

.EXAMPLE
    .\scripts\pull_results.ps1 -RemoteHost 10.8.0.5

.EXAMPLE
    .\scripts\pull_results.ps1 -RemoteHost 10.8.0.5 -RunName train2 -IncludeWeights

.NOTES
    Autor: Kiril Horobets
    Politechnika Warszawska, 2026
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [string]$ShareName = "MushroomProject",

    [string]$RunName = "train",

    [string]$User = "",

    [switch]$IncludeWeights,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Katalog docelowy = glowny katalog repozytorium (rodzic folderu scripts)
$LocalRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Sciezka UNC do zdalnego udzialu, np. \\10.8.0.5\MushroomProject
$RemotePath = "\\$RemoteHost\$ShareName"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  POBIERANIE WYNIKOW <- ZDALNY KOMPUTER (PULL)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Zrodlo (zdalne): $RemotePath"
Write-Host "  Cel (lokalny):   $LocalRoot"
Write-Host "  Eksperyment:     runs/detect/$RunName"
Write-Host "  Wagi .pt:        $($IncludeWeights.IsPresent)"
Write-Host "  DryRun:          $($DryRun.IsPresent)"
Write-Host "============================================================`n"

# --- 1. Sprawdzenie dostepnosci hosta (ping) ---
Write-Host "[1/4] Sprawdzanie dostepnosci hosta $RemoteHost ..."
if (-not (Test-Connection -ComputerName $RemoteHost -Count 2 -Quiet)) {
    Write-Host "[BLAD] Host $RemoteHost nie odpowiada na ping." -ForegroundColor Red
    Write-Host "       Sprawdz polaczenie VPN oraz czy zdalny komputer jest wlaczony." -ForegroundColor Red
    exit 1
}
Write-Host "       Host dostepny.`n" -ForegroundColor Green

# --- 2. Opcjonalne logowanie do udzialu sieciowego ---
if ($User -ne "") {
    Write-Host "[2/4] Logowanie do udzialu $RemotePath jako $User ..."
    $SecurePass = Read-Host -AsSecureString "Podaj haslo dla $User"
    $PlainPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePass))
    cmd /c "net use `"$RemotePath`" /delete" 2>$null | Out-Null
    cmd /c "net use `"$RemotePath`" /user:$User $PlainPass" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[BLAD] Nie udalo sie zalogowac do udzialu sieciowego." -ForegroundColor Red
        exit 1
    }
    Write-Host "       Zalogowano.`n" -ForegroundColor Green
}
else {
    Write-Host "[2/4] Pomijam logowanie (uzywam biezacych poswiadczen sesji).`n"
}

if (-not (Test-Path $RemotePath)) {
    Write-Host "[BLAD] Nie mozna uzyskac dostepu do $RemotePath" -ForegroundColor Red
    Write-Host "       Upewnij sie, ze folder jest udostepniony i masz uprawnienia." -ForegroundColor Red
    exit 1
}

# Wspolne flagi Robocopy (kopiowanie addytywne, bez /MIR)
$CommonFlags = @("/Z", "/R:2", "/W:3", "/NP", "/NFL", "/NDL")
if ($DryRun) { $CommonFlags += "/L" }

# --- 3. Pobranie modelu TFLite ---
Write-Host "[3/4] Pobieranie modelu TFLite ..."
$RemoteModelDir = Join-Path $RemotePath "android\app\models"
$LocalModelDir = Join-Path $LocalRoot "android\app\models"
$ModelFile = "mushroom_detector.tflite"

if (Test-Path (Join-Path $RemoteModelDir $ModelFile)) {
    # Robocopy kopiuje pojedynczy plik: <katalog_zrodlowy> <katalog_docelowy> <nazwa_pliku>
    robocopy $RemoteModelDir $LocalModelDir $ModelFile @CommonFlags | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Write-Host "[BLAD] Kopiowanie modelu TFLite nie powiodlo sie (kod $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "       Model -> $LocalModelDir\$ModelFile" -ForegroundColor Green
}
else {
    Write-Host "[UWAGA] Nie znaleziono zdalnego pliku modelu: $RemoteModelDir\$ModelFile" -ForegroundColor Yellow
    Write-Host "        Czy eksport (export_tflite.py) zakonczyl sie sukcesem?" -ForegroundColor Yellow
}

# --- 4. Pobranie wynikow treningu (wykresy + results.csv) ---
Write-Host "`n[4/4] Pobieranie wynikow runs/detect/$RunName ..."
$RemoteRunDir = Join-Path $RemotePath "runs\detect\$RunName"
$LocalRunDir = Join-Path $LocalRoot "runs\detect\$RunName"

if (Test-Path $RemoteRunDir) {
    # /E - kopiuj podkatalogi (rowniez puste). Bez /MIR - nie kasujemy lokalnych plikow.
    $RunArgs = @($RemoteRunDir, $LocalRunDir, "/E") + $CommonFlags

    # Domyslnie pomijamy ciezki podkatalog z wagami .pt
    if (-not $IncludeWeights) {
        $RunArgs += "/XD"
        $RunArgs += (Join-Path $RemoteRunDir "weights")
    }

    robocopy @RunArgs | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Write-Host "[BLAD] Kopiowanie wynikow treningu nie powiodlo sie (kod $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "       Wyniki -> $LocalRunDir" -ForegroundColor Green
}
else {
    Write-Host "[UWAGA] Nie znaleziono zdalnego katalogu: $RemoteRunDir" -ForegroundColor Yellow
    Write-Host "        Sprawdz nazwe eksperymentu (--RunName) - moze byc np. train2." -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  GOTOWE. Wyniki pobrane na lokalny laptop." -ForegroundColor Green
Write-Host "============================================================"
exit 0
