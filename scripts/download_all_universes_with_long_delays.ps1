# Download all universes with VERY long delays to avoid rate limits
# This script uses 60+ second delays between downloads

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [float]$SleepSeconds = 60.0,  # 60 seconds between downloads
    [int]$InitialWait = 300  # Wait 5 minutes before starting
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'

function Write-Info($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [DOWNLOAD] $msg" -ForegroundColor Cyan
}

function Write-Success($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [SUCCESS] $msg" -ForegroundColor Green
}

function Write-Warning($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [WARNING] $msg" -ForegroundColor Yellow
}

$Universes = @(
    @{ Name = "AI Tech"; File = "config\universe_ai_tech_tickers.txt" },
    @{ Name = "Healthcare Biotech"; File = "config\healthcare_biotech_tickers.txt" },
    @{ Name = "Energy Resources Cyclicals"; File = "config\energy_resources_cyclicals_tickers.txt" },
    @{ Name = "Defense Security Aero"; File = "config\defense_security_aero_tickers.txt" },
    @{ Name = "Consumer Financial Misc"; File = "config\consumer_financial_misc_tickers.txt" }
)

Write-Info "========================================"
Write-Info "Download All Universes - Long Delays"
Write-Info "========================================"
Write-Info "Target Root: $TargetRoot"
Write-Info "Date Range: $StartDate to $EndDate"
Write-Info "Interval: $Interval"
Write-Info "Sleep Between Downloads: $SleepSeconds seconds"
Write-Info "Initial Wait: $InitialWait seconds (5 minutes)"
Write-Info "========================================"
Write-Host ""

if ($InitialWait -gt 0) {
    Write-Warning "Waiting $InitialWait seconds before starting (to let rate limit reset)..."
    Start-Sleep -Seconds $InitialWait
}

$Stats = @{
    Completed = 0
    Failed = 0
}

$startTime = Get-Date

foreach ($idx in 0..($Universes.Count - 1)) {
    $universe = $Universes[$idx]
    $universeIndex = $idx + 1
    
    Write-Host ""
    Write-Info "========================================"
    Write-Info "Universe $universeIndex/$($Universes.Count): $($universe.Name)"
    Write-Info "========================================"
    
    $symbolsFile = Join-Path $ROOT $universe.File
    
    if (-not (Test-Path $symbolsFile)) {
        Write-Warning "Symbols file not found: $symbolsFile"
        $Stats.Failed++
        continue
    }
    
    # Count symbols
    $symbols = Get-Content $symbolsFile | Where-Object { $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") }
    $symbolCount = ($symbols | Measure-Object).Count
    Write-Info "Found $symbolCount symbols"
    
    if ($symbolCount -eq 0) {
        Write-Warning "No symbols found, skipping"
        continue
    }
    
    # Download universe
    Write-Info "Starting download with $SleepSeconds second delays..."
    
    $args = @(
        $DownloadScript,
        "--symbols-file", "`"$symbolsFile`"",
        "--start", $StartDate,
        "--end", $EndDate,
        "--interval", $Interval,
        "--target-root", "`"$TargetRoot`"",
        "--sleep-seconds", $SleepSeconds.ToString("F1")
    )
    
    $output = & $Python $args 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Success "Universe $($universe.Name) completed!"
        $Stats.Completed++
    } else {
        Write-Warning "Universe $($universe.Name) had errors (exit code: $exitCode)"
        $Stats.Failed++
        
        # Check for rate limit
        $outputText = $output -join "`n"
        if ($outputText -match "rate limit|too many requests|429") {
            Write-Warning "Rate limit detected! Waiting 5 minutes before continuing..."
            Start-Sleep -Seconds 300
        }
    }
    
    # Pause between universes
    if ($idx -lt ($Universes.Count - 1)) {
        Write-Info "Pausing 2 minutes before next universe..."
        Start-Sleep -Seconds 120
    }
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Info "========================================"
Write-Info "Final Summary"
Write-Info "========================================"
Write-Info "Completed: $($Stats.Completed)/$($Universes.Count)"
Write-Info "Failed: $($Stats.Failed)/$($Universes.Count)"
Write-Info "Duration: $($duration.ToString('hh\:mm\:ss'))"
Write-Info "========================================"

if ($Stats.Failed -eq 0) {
    Write-Success "All universes downloaded successfully!"
} else {
    Write-Warning "Some universes failed. Re-run this script to retry failed downloads."
}

