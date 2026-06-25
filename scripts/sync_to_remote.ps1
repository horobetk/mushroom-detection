<#
.SYNOPSIS
    Synchronizacja kodu projektu na zdalny komputer uniwersytecki przez siec lokalna (VPN).

.DESCRIPTION
    Tworzy lustrzana kopie (mirror) biezacego katalogu projektu na zdalnym
    udostepnionym folderze Windows przy uzyciu wbudowanej utility Robocopy.
    Pomija ciezkie katalogi (.git, .venv, runs, datasets, weights), aby przez
    siec przesylany byl tylko kod (skrypty .py, .yaml, .bat oraz pliki
    konfiguracyjne), a nie gigabajty danych i wag.

    Przed kopiowaniem sprawdza dostepnosc hosta (ping). Opcjonalnie loguje sie do
    udzialu sieciowego (net use) podanymi danymi.

.PARAMETER RemoteHost
    Adres IP lub nazwa hosta zdalnego komputera (np. 10.8.0.5).

.PARAMETER ShareName
    Nazwa udostepnionego folderu na zdalnej maszynie (domyslnie: MushroomProject).

.PARAMETER User
    Opcjonalna nazwa uzytkownika do logowania do udzialu sieciowego (np. DESKTOP-PW\student).

.PARAMETER DryRun
    Tryb testowy - pokazuje co zostanie skopiowane/usuniete, ale nie wprowadza zmian (Robocopy /L).

.EXAMPLE
    .\scripts\sync_to_remote.ps1 -RemoteHost 10.8.0.5

.EXAMPLE
    .\scripts\sync_to_remote.ps1 -RemoteHost 10.8.0.5 -User DESKTOP-PW\student -DryRun

.NOTES
    Autor: Kiril Horobets
    Politechnika Warszawska, 2026
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [string]$ShareName = "MushroomProject",

    [string]$User = "",

    [switch]$DryRun
)

# Zatrzymaj skrypt przy bledach krytycznych
$ErrorActionPreference = "Stop"

# Katalog zrodlowy = glowny katalog repozytorium (rodzic folderu scripts)
$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Sciezka UNC do zdalnego udzialu, np. \\10.8.0.5\MushroomProject
$RemotePath = "\\$RemoteHost\$ShareName"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SYNCHRONIZACJA KODU -> ZDALNY KOMPUTER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Zrodlo:  $SourceDir"
Write-Host "  Cel:     $RemotePath"
Write-Host "  DryRun:  $($DryRun.IsPresent)"
Write-Host "============================================================`n"

# --- 1. Sprawdzenie dostepnosci hosta (ping) ---
Write-Host "[1/3] Sprawdzanie dostepnosci hosta $RemoteHost ..."
if (-not (Test-Connection -ComputerName $RemoteHost -Count 2 -Quiet)) {
    Write-Host "[BLAD] Host $RemoteHost nie odpowiada na ping." -ForegroundColor Red
    Write-Host "       Sprawdz polaczenie VPN oraz czy zdalny komputer jest wlaczony." -ForegroundColor Red
    exit 1
}
Write-Host "       Host dostepny.`n" -ForegroundColor Green

# --- 2. Opcjonalne logowanie do udzialu sieciowego ---
if ($User -ne "") {
    Write-Host "[2/3] Logowanie do udzialu $RemotePath jako $User ..."
    # Pobranie hasla w sposob bezpieczny (nie zapisujemy go w skrypcie)
    $SecurePass = Read-Host -AsSecureString "Podaj haslo dla $User"
    $PlainPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePass))
    # Czyscimy ewentualne stare polaczenie i tworzymy nowe
    cmd /c "net use `"$RemotePath`" /delete" 2>$null | Out-Null
    cmd /c "net use `"$RemotePath`" /user:$User $PlainPass" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[BLAD] Nie udalo sie zalogowac do udzialu sieciowego." -ForegroundColor Red
        exit 1
    }
    Write-Host "       Zalogowano.`n" -ForegroundColor Green
}
else {
    Write-Host "[2/3] Pomijam logowanie (uzywam biezacych poswiadczen sesji).`n"
}

# Sprawdzenie czy udzial jest osiagalny
if (-not (Test-Path $RemotePath)) {
    Write-Host "[BLAD] Nie mozna uzyskac dostepu do $RemotePath" -ForegroundColor Red
    Write-Host "       Upewnij sie, ze folder jest udostepniony (Network Sharing) i masz uprawnienia." -ForegroundColor Red
    exit 1
}

# --- 3. Robocopy: lustrzana kopia kodu (z pominieciem ciezkich katalogow) ---
Write-Host "[3/3] Synchronizacja plikow (Robocopy /MIR)...`n"

# Katalogi pomijane - nie przesylamy ich przez siec (dane / wagi / srodowisko / historia git)
$ExcludeDirs = @(
    ".git",
    ".venv",
    "venv",
    "runs",
    "datasets",
    "weights",
    "__pycache__",
    ".idea",
    ".vscode",
    (Join-Path "data" "external"),
    (Join-Path "data" "frames")
)

# Pliki pomijane - duze artefakty modeli i logi
$ExcludeFiles = @(
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.tflite",
    "*.pb",
    "*.zip",
    "*_log.txt",
    "kaggle.json"
)

# Budowa argumentow Robocopy
$RoboArgs = @(
    $SourceDir,
    $RemotePath,
    "/MIR",          # mirror (kopiuj + usuwaj nadmiarowe pliki w celu)
    "/Z",            # tryb wznawialny (odporny na chwilowe przerwy sieci)
    "/R:2",          # liczba ponowien przy bledzie
    "/W:3",          # czas oczekiwania (s) miedzy ponowieniami
    "/NP",           # bez procentow (czytelniejszy log)
    "/NFL",          # bez listy plikow
    "/NDL"           # bez listy katalogow
)

foreach ($d in $ExcludeDirs) { $RoboArgs += "/XD"; $RoboArgs += $d }
foreach ($f in $ExcludeFiles) { $RoboArgs += "/XF"; $RoboArgs += $f }

if ($DryRun) { $RoboArgs += "/L" }  # tylko podglad, bez realnych zmian

robocopy @RoboArgs
$RoboExit = $LASTEXITCODE

# Robocopy: kody 0-7 oznaczaja sukces (8+ to bledy)
if ($RoboExit -ge 8) {
    Write-Host "`n[BLAD] Robocopy zakonczyl sie z kodem $RoboExit (blad kopiowania)." -ForegroundColor Red
    exit $RoboExit
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  GOTOWE. Kod zsynchronizowany (Robocopy kod: $RoboExit)." -ForegroundColor Green
Write-Host "============================================================"
exit 0
