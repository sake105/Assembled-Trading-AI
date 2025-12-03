# Download symbols one-by-one with very long delays to avoid rate limits
# Strategy: Download one symbol at a time with 60+ second delays

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$DelayBetweenSymbols = 60,  # 60 seconds between downloads
    [int]$InitialWait = 300,  # Wait 5 minutes before starting (let rate limit reset)
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

# Collect all symbols from all universe files
Write-Info "Collecting symbols from all universe files..."

$AllSymbols = @()

$UniverseFiles = @(
    "config\universe_ai_tech_tickers.txt",
    "config\healthcare_biotech_tickers.txt",
    "config\energy_resources_cyclicals_tickers.txt",
    "config\defense_security_aero_tickers.txt",
    "config\consumer_financial_misc_tickers.txt"
)

foreach ($file in $UniverseFiles) {
    $filePath = Join-Path $ROOT $file
    if (Test-Path $filePath) {
        $symbols = Get-Content $filePath | Where-Object { 
            $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
        }
        $AllSymbols += $symbols
        Write-Info "  Found $($symbols.Count) symbols in $file"
    }
}

$AllSymbols = $AllSymbols | Sort-Object -Unique
Write-Info "Total unique symbols to download: $($AllSymbols.Count)"
Write-Host ""

# Check which files already exist
$ExistingFiles = @()
$MissingSymbols = @()

foreach ($symbol in $AllSymbols) {
    $filePath = Join-Path $TargetRoot "$Interval\$symbol.parquet"
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length
        if ($size -gt 1024) {
            $ExistingFiles += $symbol
        } else {
            $MissingSymbols += $symbol
        }
    } else {
        $MissingSymbols += $symbol
    }
}

Write-Info "Files already exist: $($ExistingFiles.Count)"
Write-Info "Symbols to download: $($MissingSymbols.Count)"
Write-Host ""

if ($MissingSymbols.Count -eq 0) {
    Write-Success "All symbols already downloaded!"
    exit 0
}

# Initial wait to let rate limit reset
if ($InitialWait -gt 0) {
    Write-Warning "Waiting $InitialWait seconds before starting downloads to let rate limit reset..."
    Write-Warning "This is important to avoid immediate rate limiting."
    Start-Sleep -Seconds $InitialWait
}

Write-Info "========================================"
Write-Info "Starting One-by-One Downloads"
Write-Info "========================================"
Write-Info "Target Root: $TargetRoot"
Write-Info "Date Range: $StartDate to $EndDate"
Write-Info "Interval: $Interval"
Write-Info "Delay Between Symbols: $DelayBetweenSymbols seconds"
Write-Info "========================================"
Write-Host ""

$Stats = @{
    Successful = 0
    Failed = 0
    Skipped = 0
    RateLimited = 0
}

$startTime = Get-Date
$failedSymbols = @()

foreach ($idx in 0..($MissingSymbols.Count - 1)) {
    $symbol = $MissingSymbols[$idx]
    $symbolIndex = $idx + 1
    
    Write-Host ""
    Write-Info "========================================"
    Write-Info "Symbol $symbolIndex/$($MissingSymbols.Count): $symbol"
    Write-Info "========================================"
    
    # Skip if exists and skip flag is set
    if ($SkipExisting) {
        $filePath = Join-Path $TargetRoot "$Interval\$symbol.parquet"
        if (Test-Path $filePath) {
            $size = (Get-Item $filePath).Length
            if ($size -gt 1024) {
                Write-Info "  File already exists, skipping"
                $Stats.Skipped++
                continue
            }
        }
    }
    
    # Download single symbol
    $args = @(
        $DownloadScript,
        "--symbol", $symbol,
        "--start", $StartDate,
        "--end", $EndDate,
        "--interval", $Interval,
        "--target-root", "`"$TargetRoot`""
    )
    
    Write-Info "  Starting download..."
    $output = & $Python $args 2>&1
    $exitCode = $LASTEXITCODE
    $outputText = $output -join "`n"
    
    # Check for rate limit
    $isRateLimit = $outputText -match "rate limit|too many requests|429|Rate limited"
    
    if ($exitCode -eq 0) {
        Write-Success "  ✓ $symbol downloaded successfully"
        $Stats.Successful++
        
        # Extra delay after successful download
        if ($idx -lt ($MissingSymbols.Count - 1)) {
            Write-Info "  Waiting $DelayBetweenSymbols seconds before next symbol..."
            Start-Sleep -Seconds $DelayBetweenSymbols
        }
    } elseif ($isRateLimit) {
        Write-Error-Info "  ✗ $symbol - Rate limited"
        $Stats.Failed++
        $Stats.RateLimited++
        $failedSymbols += $symbol
        
        # If rate limited, wait much longer before continuing
        $longDelay = $DelayBetweenSymbols * 3  # 3x normal delay
        Write-Warning "  Rate limit detected! Waiting $longDelay seconds before continuing..."
        Start-Sleep -Seconds $longDelay
    } else {
        Write-Error-Info "  ✗ $symbol - Download failed (exit code: $exitCode)"
        $Stats.Failed++
        $failedSymbols += $symbol
        
        # Wait before next attempt
        if ($idx -lt ($MissingSymbols.Count - 1)) {
            Write-Info "  Waiting $DelayBetweenSymbols seconds before next symbol..."
            Start-Sleep -Seconds $DelayBetweenSymbols
        }
    }
}

# Final summary
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Info "========================================"
Write-Info "Download Summary"
Write-Info "========================================"
Write-Info "Total Symbols: $($AllSymbols.Count)"
Write-Info "Already Existed: $($ExistingFiles.Count)"
Write-Info "Successful: $($Stats.Successful)"
Write-Info "Failed: $($Stats.Failed)"
Write-Info "Rate Limited: $($Stats.RateLimited)"
Write-Info "Duration: $($duration.ToString('hh\:mm\:ss'))"
Write-Info "========================================"

if ($failedSymbols.Count -gt 0) {
    Write-Host ""
    Write-Warning "Failed Symbols ($($failedSymbols.Count)):"
    Write-Warning ($failedSymbols -join ", ")
    Write-Host ""
    Write-Info "You can re-run this script to retry failed symbols."
    Write-Info "It will automatically skip symbols that were successfully downloaded."
}

Write-Host ""
Write-Success "Batch download completed!"
Write-Info "Re-run this script to retry any failed downloads."

