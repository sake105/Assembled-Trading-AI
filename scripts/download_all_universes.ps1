# Download all universe tickers with robust retry logic
param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [float]$SleepSeconds = 3.0,
    [int]$MaxRetries = 5,
    [int]$RetryDelay = 60,
    [switch]$SkipExisting = $true
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
Write-Info "Download All Universes - Robust Batch Script"
Write-Info "========================================"
Write-Info "Target Root: $TargetRoot"
Write-Info "Date Range: $StartDate to $EndDate"
Write-Info "Interval: $Interval"
Write-Info "Sleep Between Symbols: $SleepSeconds seconds"
Write-Info "Max Retries per Universe: $MaxRetries"
Write-Info "Retry Delay: $RetryDelay seconds"
Write-Info "Skip Existing: $SkipExisting"
Write-Info "========================================"
Write-Host ""

# Statistics
$Stats = @{
    TotalUniverses = $Universes.Count
    CompletedUniverses = 0
    FailedUniverses = 0
    TotalSymbols = 0
    SuccessfulDownloads = 0
    FailedDownloads = 0
    SkippedDownloads = 0
    RateLimitHits = 0
}

# Check if file exists and has data
function Test-FileExists($filePath) {
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length
        return $size -gt 1024  # At least 1KB
    }
    return $false
}

# Download single universe with retry logic
function Download-Universe {
    param(
        [string]$UniverseName,
        [string]$SymbolsFile,
        [int]$UniverseIndex,
        [int]$TotalUniverses
    )
    
    Write-Info "Processing Universe ($UniverseIndex/$TotalUniverses): $UniverseName"
    Write-Info "  Symbols File: $SymbolsFile"
    
    if (-not (Test-Path $SymbolsFile)) {
        Write-Error-Info "  Symbols file not found: $SymbolsFile"
        $Stats.FailedUniverses++
        return $false
    }
    
    # Count symbols
    $symbols = Get-Content $SymbolsFile | Where-Object { $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") }
    $symbolCount = ($symbols | Measure-Object).Count
    Write-Info "  Symbols to download: $symbolCount"
    $Stats.TotalSymbols += $symbolCount
    
    if ($symbolCount -eq 0) {
        Write-Warning "  No symbols found in file, skipping"
        return $false
    }
    
    # Retry loop for entire universe
    $attempt = 0
    $universeSuccess = $false
    
    while ($attempt -lt $MaxRetries -and -not $universeSuccess) {
        $attempt++
        
        if ($attempt -gt 1) {
            Write-Warning "  Retry attempt $attempt/$MaxRetries for universe $UniverseName"
            Write-Info "  Waiting $RetryDelay seconds before retry..."
            Start-Sleep -Seconds $RetryDelay
        }
        
        Write-Info "  Starting download attempt $attempt..."
        
        try {
            # Build command
            $args = @(
                $DownloadScript,
                "--symbols-file", $SymbolsFile,
                "--start", $StartDate,
                "--end", $EndDate,
                "--interval", $Interval,
                "--target-root", "`"$TargetRoot`"",
                "--sleep-seconds", $SleepSeconds.ToString("F1")
            )
            
            # Execute download
            $result = & $Python $args 2>&1
            $exitCode = $LASTEXITCODE
            
            # Check for rate limit indicators
            $output = $result -join "`n"
            $isRateLimit = $output -match "rate limit|too many requests|429|Rate limited"
            
            if ($isRateLimit) {
                $Stats.RateLimitHits++
                Write-Warning "  Rate limit detected for universe $UniverseName"
                
                if ($attempt -lt $MaxRetries) {
                    Write-Info "  Will retry after delay..."
                    continue
                } else {
                    Write-Error-Info "  Max retries reached for universe $UniverseName due to rate limits"
                    break
                }
            }
            
            # Check exit code
            if ($exitCode -eq 0) {
                Write-Success "  Universe $UniverseName downloaded successfully"
                $universeSuccess = $true
                $Stats.CompletedUniverses++
            } else {
                Write-Error-Info "  Download failed with exit code $exitCode"
                Write-Info "  Output: $($output[-200..-1] -join '`n')"  # Last 200 chars
                
                if ($attempt -lt $MaxRetries) {
                    continue
                }
            }
            
        } catch {
            Write-Error-Info "  Exception during download: $_"
            if ($attempt -lt $MaxRetries) {
                continue
            }
        }
    }
    
    if (-not $universeSuccess) {
        $Stats.FailedUniverses++
        Write-Error-Info "  Failed to download universe $UniverseName after $MaxRetries attempts"
        return $false
    }
    
    return $true
}

# Main loop
$startTime = Get-Date

foreach ($idx in 0..($Universes.Count - 1)) {
    $universe = $Universes[$idx]
    $universeIndex = $idx + 1
    
    Write-Host ""
    Write-Info "========================================"
    
    $success = Download-Universe `
        -UniverseName $universe.Name `
        -SymbolsFile $universe.File `
        -UniverseIndex $universeIndex `
        -TotalUniverses $Universes.Count
    
    # Pause between universes (except after last)
    if ($idx -lt ($Universes.Count - 1)) {
        Write-Info "Pausing 10 seconds before next universe..."
        Start-Sleep -Seconds 10
    }
}

# Final statistics
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Info "========================================"
Write-Info "Download Summary"
Write-Info "========================================"
Write-Info "Total Universes: $($Stats.TotalUniverses)"
Write-Info "Completed: $($Stats.CompletedUniverses)"
Write-Info "Failed: $($Stats.FailedUniverses)"
Write-Info "Total Symbols: $($Stats.TotalSymbols)"
Write-Info "Rate Limit Hits: $($Stats.RateLimitHits)"
Write-Info "Duration: $($duration.ToString('hh\:mm\:ss'))"
Write-Info "========================================"

# Validate results
Write-Host ""
Write-Info "Validating downloaded files..."
$validationScript = Join-Path $ROOT 'scripts\validate_altdata_snapshot.py'
if (Test-Path $validationScript) {
    $valArgs = @(
        $validationScript,
        "--root", "`"$TargetRoot`"",
        "--interval", $Interval
    )
    & $Python $valArgs
}

Write-Host ""
Write-Success "Batch download completed!"
Write-Info "Check validation output above for details."

