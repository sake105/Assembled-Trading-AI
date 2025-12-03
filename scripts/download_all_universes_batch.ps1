# Robust batch download for all universe tickers
# This script calls download_historical_snapshot.py for each universe with retry logic

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [float]$SleepSeconds = 3.0,
    [int]$MaxRetries = 3,
    [int]$RetryDelay = 120
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'

function Write-Info($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [DOWNLOAD] $msg" -ForegroundColor Cyan
}

function Write-Error-Info($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [ERROR] $msg" -ForegroundColor Red
}

function Write-Success($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [SUCCESS] $msg" -ForegroundColor Green
}

function Write-Warning($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [WARNING] $msg" -ForegroundColor Yellow
}

# Universe files
$Universes = @(
    @{ Name = "AI Tech"; File = "config\universe_ai_tech_tickers.txt" },
    @{ Name = "Healthcare Biotech"; File = "config\healthcare_biotech_tickers.txt" },
    @{ Name = "Energy Resources Cyclicals"; File = "config\energy_resources_cyclicals_tickers.txt" },
    @{ Name = "Defense Security Aero"; File = "config\defense_security_aero_tickers.txt" },
    @{ Name = "Consumer Financial Misc"; File = "config\consumer_financial_misc_tickers.txt" }
)

Write-Info "========================================"
Write-Info "Download All Universes - Batch Script"
Write-Info "========================================"
Write-Info "Target Root: $TargetRoot"
Write-Info "Date Range: $StartDate to $EndDate"
Write-Info "Interval: $Interval"
Write-Info "Sleep Between Symbols: $SleepSeconds seconds"
Write-Info "Max Retries per Universe: $MaxRetries"
Write-Info "Retry Delay: $RetryDelay seconds"
Write-Info "========================================"
Write-Host ""

$Stats = @{
    TotalUniverses = $Universes.Count
    CompletedUniverses = 0
    FailedUniverses = 0
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
        Write-Error-Info "Symbols file not found: $symbolsFile"
        $Stats.FailedUniverses++
        continue
    }
    
    # Count symbols
    $symbols = Get-Content $symbolsFile | Where-Object { $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") }
    $symbolCount = ($symbols | Measure-Object).Count
    Write-Info "Found $symbolCount symbols to download"
    
    if ($symbolCount -eq 0) {
        Write-Warning "No symbols found, skipping"
        continue
    }
    
    # Retry loop for this universe
    $attempt = 0
    $success = $false
    
    while ($attempt -lt $MaxRetries -and -not $success) {
        $attempt++
        
        if ($attempt -gt 1) {
            Write-Warning "Retry attempt $attempt/$MaxRetries for $($universe.Name)"
            Write-Info "Waiting $RetryDelay seconds before retry..."
            Start-Sleep -Seconds $RetryDelay
        }
        
        Write-Info "Starting download (attempt $attempt/$MaxRetries)..."
        
        # Build command arguments
        $args = @(
            $DownloadScript,
            "--symbols-file", "`"$symbolsFile`"",
            "--start", $StartDate,
            "--end", $EndDate,
            "--interval", $Interval,
            "--target-root", "`"$TargetRoot`"",
            "--sleep-seconds", $SleepSeconds.ToString("F1")
        )
        
        # Execute download
        $output = & $Python $args 2>&1
        $exitCode = $LASTEXITCODE
        
        # Check output for rate limits
        $outputText = $output -join "`n"
        $isRateLimit = $outputText -match "rate limit|too many requests|429|Rate limited"
        
        if ($exitCode -eq 0) {
            Write-Success "Universe $($universe.Name) downloaded successfully!"
            $success = $true
            $Stats.CompletedUniverses++
        } elseif ($isRateLimit) {
            Write-Warning "Rate limit detected for $($universe.Name)"
            if ($attempt -lt $MaxRetries) {
                Write-Info "Will retry after delay..."
                continue
            } else {
                Write-Error-Info "Max retries reached for $($universe.Name) due to rate limits"
            }
        } else {
            Write-Error-Info "Download failed with exit code $exitCode"
            Write-Info "Last output:"
            Write-Host ($output[-10..-1] -join "`n")
            
            if ($attempt -lt $MaxRetries) {
                Write-Info "Will retry..."
                continue
            }
        }
    }
    
    if (-not $success) {
        $Stats.FailedUniverses++
        Write-Error-Info "Failed to download $($universe.Name) after $MaxRetries attempts"
    }
    
    # Pause between universes (except after last)
    if ($idx -lt ($Universes.Count - 1)) {
        Write-Info "Pausing 15 seconds before next universe..."
        Start-Sleep -Seconds 15
    }
}

# Final summary
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Info "========================================"
Write-Info "Download Summary"
Write-Info "========================================"
Write-Info "Total Universes: $($Stats.TotalUniverses)"
Write-Info "Completed: $($Stats.CompletedUniverses)"
Write-Info "Failed: $($Stats.FailedUniverses)"
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
if ($Stats.FailedUniverses -eq 0) {
    Write-Success "All universes downloaded successfully!"
} else {
    Write-Warning "$($Stats.FailedUniverses) universe(s) failed. Check logs above for details."
}

