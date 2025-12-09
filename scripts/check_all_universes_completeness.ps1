# Check Completeness of All Universes
# Runs completeness check for all universe files

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$Interval = "1d",
    [string]$ExpectedStart = "2000-01-01",
    [string]$ExpectedEnd = "2025-12-03"
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$CheckScript = Join-Path $ROOT 'scripts\check_data_completeness.py'

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Completeness Check - All Universes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find all universe files
$UniverseFiles = @(
    "config\universe_ai_tech_tickers.txt",
    "config\healthcare_biotech_tickers.txt",
    "config\energy_resources_cyclicals_tickers.txt",
    "config\defense_security_aero_tickers.txt",
    "config\consumer_financial_misc_tickers.txt",
    "config\macro_world_etfs_tickers.txt"
)

$AllStats = @{
    TotalUniverses = 0
    TotalSymbols = 0
    FilesExist = 0
    FilesMissing = 0
    FilesWithErrors = 0
}

foreach ($universeFile in $UniverseFiles) {
    $universePath = Join-Path $ROOT $universeFile
    
    if (-not (Test-Path $universePath)) {
        Write-Host "Universe file not found: $universeFile" -ForegroundColor Yellow
        continue
    }
    
    $universeName = [System.IO.Path]::GetFileNameWithoutExtension($universeFile)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Checking: $universeName" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $args = @(
        $CheckScript,
        "--universe-file", $universePath,
        "--target-root", $TargetRoot,
        "--interval", $Interval,
        "--expected-start", $ExpectedStart,
        "--expected-end", $ExpectedEnd
    )
    
    try {
        & $Python $args
        $AllStats.TotalUniverses++
    } catch {
        Write-Host "Error checking $universeName : $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All Universes Checked" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

