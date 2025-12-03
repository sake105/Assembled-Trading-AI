# ============================================================================
# Download All Universes - Optimized Commands
# ============================================================================
# 
# WICHTIG: Yahoo Finance hat aggressive Rate-Limits!
# 
# Strategie:
# 1. Verwende 60+ Sekunden Pause zwischen Downloads
# 2. Warte 5 Minuten vor dem Start (Rate-Limit-Reset)
# 3. Überspringe bereits existierende Dateien automatisch
# 4. Bei Rate-Limit: Warte 5 Minuten und versuche erneut
#
# ============================================================================

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
$StartDate = "2000-01-01"
$EndDate = "2025-12-03"
$Interval = "1d"
$SleepSeconds = 60.0  # 60 Sekunden zwischen Downloads

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

Write-Info "========================================"
Write-Info "Download All Universes - Optimized"
Write-Info "========================================"
Write-Info "Target Root: $TargetRoot"
Write-Info "Date Range: $StartDate to $EndDate"
Write-Info "Interval: $Interval"
Write-Info "Sleep Between Downloads: $SleepSeconds seconds"
Write-Info "========================================"
Write-Host ""

# Initial wait to let rate limit reset
Write-Warning "Waiting 5 minutes before starting (to let rate limit reset)..."
Write-Warning "This is CRITICAL to avoid immediate rate limiting!"
Start-Sleep -Seconds 300

$Universes = @(
    @{ Name = "AI Tech"; File = "config\universe_ai_tech_tickers.txt" },
    @{ Name = "Healthcare Biotech"; File = "config\healthcare_biotech_tickers.txt" },
    @{ Name = "Energy Resources Cyclicals"; File = "config\energy_resources_cyclicals_tickers.txt" },
    @{ Name = "Defense Security Aero"; File = "config\defense_security_aero_tickers.txt" },
    @{ Name = "Consumer Financial Misc"; File = "config\consumer_financial_misc_tickers.txt" }
)

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
    
    # Download universe
    Write-Info "Starting download..."
    
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
    $outputText = $output -join "`n"
    
    # Check for rate limit
    $isRateLimit = $outputText -match "rate limit|too many requests|429|Rate limited|All downloads failed"
    
    if ($exitCode -eq 0) {
        Write-Success "Universe $($universe.Name) completed!"
        $Stats.Completed++
    } elseif ($isRateLimit) {
        Write-Warning "Rate limit detected for $($universe.Name)!"
        Write-Warning "Waiting 5 minutes before continuing..."
        Start-Sleep -Seconds 300
        $Stats.Failed++
    } else {
        Write-Warning "Universe $($universe.Name) had errors (exit code: $exitCode)"
        $Stats.Failed++
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

# Validate results
Write-Host ""
Write-Info "Validating downloaded files..."
$ValidationScript = Join-Path $ROOT 'scripts\validate_altdata_snapshot.py'
if (Test-Path $ValidationScript) {
    $valArgs = @(
        $ValidationScript,
        "--root", "`"$TargetRoot`"",
        "--interval", $Interval
    )
    & $Python $valArgs
}

Write-Host ""
if ($Stats.Failed -eq 0) {
    Write-Success "All universes downloaded successfully!"
} else {
    Write-Warning "Some universes failed. Re-run this script to retry failed downloads."
    Write-Info "The script will automatically skip already downloaded symbols."
}

