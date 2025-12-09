# Download All Universes: One-by-One Safe Mode
# Downloads all symbols from all universe ticker files with conservative delays
# Automatically skips existing files

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$DelayBetweenSymbols = 10,  # Seconds to wait between symbols
    [int]$SleepSeconds = 2.0,  # Sleep seconds passed to Python script
    [int]$InitialWait = 60  # Initial wait before starting (seconds)
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$LogFile = Join-Path $ROOT "logs\download_all_universes_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create logs directory if needed
$LogDir = Split-Path $LogFile -Parent
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Log($msg, $color = "White") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMsg = "[$timestamp] $msg"
    Write-Host $logMsg -ForegroundColor $color
    Add-Content -Path $LogFile -Value $logMsg -ErrorAction SilentlyContinue
}

function Write-Success($msg) {
    Write-Log "✓ $msg" "Green"
}

function Write-Error($msg) {
    Write-Log "✗ $msg" "Red"
}

function Write-Info($msg) {
    Write-Log ">>> $msg" "Cyan"
}

# Universe files to process
$UniverseFiles = @(
    "config\healthcare_biotech_tickers.txt",
    "config\consumer_financial_misc_tickers.txt",
    "config\energy_resources_cyclicals_tickers.txt",
    "config\defense_security_aero_tickers.txt",
    "config\universe_ai_tech_tickers.txt",
    "config\macro_world_etfs_tickers.txt"
)

Write-Log "========================================" "Cyan"
Write-Log "All Universes: One-by-One Safe Mode" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Target Root: $TargetRoot" "Cyan"
Write-Log "Date Range: $StartDate to $EndDate" "Cyan"
Write-Log "Interval: $Interval" "Cyan"
Write-Log "Delay Between Symbols: $DelayBetweenSymbols seconds" "Cyan"
Write-Log "Initial Wait: $InitialWait seconds" "Cyan"
Write-Log "Log File: $LogFile" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log ""

# Initial wait to avoid immediate rate limits
if ($InitialWait -gt 0) {
    Write-Log "Waiting $InitialWait seconds before starting (to avoid rate limits)..." "Yellow"
    Start-Sleep -Seconds $InitialWait
}

# Collect all symbols from all universe files
$AllSymbols = @()
foreach ($universeFile in $UniverseFiles) {
    $universePath = Join-Path $ROOT $universeFile
    if (Test-Path $universePath) {
        $symbols = Get-Content $universePath | Where-Object { 
            $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
        } | ForEach-Object { $_.Trim().ToUpper() }
        $AllSymbols += $symbols
        Write-Log "Loaded $($symbols.Count) symbols from $universeFile" "Cyan"
    } else {
        Write-Log "Universe file not found: $universePath" "Yellow"
    }
}

# Remove duplicates and sort
$AllSymbols = $AllSymbols | Sort-Object -Unique

if ($AllSymbols.Count -eq 0) {
    Write-Error "No symbols found in any universe file"
    exit 1
}

Write-Log ""
Write-Log "Total unique symbols: $($AllSymbols.Count)" "Cyan"
Write-Log ""

# Check existing files
$ExistingCount = 0
$ToDownload = @()

foreach ($sym in $AllSymbols) {
    $filePath = Join-Path $TargetRoot "$Interval\$sym.parquet"
    if (Test-Path $filePath) {
        try {
            $fileInfo = Get-Item $filePath
            if ($fileInfo.Length -gt 1024) {
                $ExistingCount++
            } else {
                $ToDownload += $sym
            }
        } catch {
            $ToDownload += $sym
        }
    } else {
        $ToDownload += $sym
    }
}

Write-Log "Already downloaded: $ExistingCount" "Cyan"
Write-Log "To download: $($ToDownload.Count)" "Cyan"
Write-Log ""

if ($ToDownload.Count -eq 0) {
    Write-Success "All symbols already downloaded!"
    exit 0
}

# Statistics
$Stats = @{
    Successful = 0
    Failed = 0
    Skipped = 0
}

$startTime = Get-Date
$failedSymbols = @()

# Download each symbol one by one
foreach ($idx in 0..($ToDownload.Count - 1)) {
    $symbol = $ToDownload[$idx]
    $symbolIndex = $idx + 1
    
    Write-Log ""
    Write-Info "Symbol $symbolIndex/$($ToDownload.Count): $symbol"
    
    # Check again if file exists (might have been created by another process)
    $filePath = Join-Path $TargetRoot "$Interval\$symbol.parquet"
    if (Test-Path $filePath) {
        try {
            $fileInfo = Get-Item $filePath
            if ($fileInfo.Length -gt 1024) {
                Write-Log "  File already exists, skipping" "Yellow"
                $Stats.Skipped++
                continue
            }
        } catch {
            # Continue with download
        }
    }
    
    # Download single symbol
    $args = @(
        $DownloadScript,
        "--symbol", $symbol,
        "--start", $StartDate,
        "--end", $EndDate,
        "--interval", $Interval,
        "--target-root", $TargetRoot,
        "--sleep-seconds", $SleepSeconds
    )
    
    try {
        $downloadStart = Get-Date
        $output = & $Python $args 2>&1
        $exitCode = $LASTEXITCODE
        $downloadDuration = ((Get-Date) - $downloadStart).TotalSeconds
        $outputText = $output -join "`n"
        
        # Check for rate limit
        $isRateLimit = $outputText -match "rate limit|too many requests|429|Rate limited|YFRateLimitError"
        
        if ($exitCode -eq 0) {
            # Verify file was created
            if (Test-Path $filePath) {
                try {
                    $fileInfo = Get-Item $filePath
                    if ($fileInfo.Length -gt 1024) {
                        Write-Success "$symbol downloaded successfully ($([math]::Round($downloadDuration, 1))s, $([math]::Round($fileInfo.Length/1KB, 1)) KB)"
                        $Stats.Successful++
                    } else {
                        Write-Error "$symbol - File too small ($($fileInfo.Length) bytes)"
                        $Stats.Failed++
                        $failedSymbols += $symbol
                    }
                } catch {
                    Write-Error "$symbol - Cannot verify file"
                    $Stats.Failed++
                    $failedSymbols += $symbol
                }
            } else {
                Write-Error "$symbol - File not created"
                $Stats.Failed++
                $failedSymbols += $symbol
            }
        } elseif ($isRateLimit) {
            Write-Error "$symbol - Rate limited"
            $Stats.Failed++
            $failedSymbols += $symbol
            
            # Wait much longer if rate limited (10 minutes)
            $longDelay = 600
            Write-Log "  Rate limit detected! Waiting $longDelay seconds (10 minutes)..." "Yellow"
            Start-Sleep -Seconds $longDelay
            continue
        } else {
            Write-Error "$symbol - Download failed (exit code: $exitCode)"
            $outputPreview = if ($outputText.Length -gt 150) { $outputText.Substring(0, 150) + "..." } else { $outputText }
            Write-Log "  Error output: $outputPreview" "Red"
            $Stats.Failed++
            $failedSymbols += $symbol
        }
    } catch {
        Write-Error "$symbol - Exception: $_"
        $Stats.Failed++
        $failedSymbols += $symbol
    }
    
    # Wait before next symbol (except for the last one)
    if ($idx -lt ($ToDownload.Count - 1)) {
        Write-Log "  Waiting $DelayBetweenSymbols seconds before next symbol..." "Cyan"
        Start-Sleep -Seconds $DelayBetweenSymbols
    }
}

# Final summary
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Log ""
Write-Log "========================================" "Cyan"
Write-Log "Download Summary" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Total Symbols: $($AllSymbols.Count)" "Cyan"
Write-Log "Already Existed: $ExistingCount" "Cyan"
Write-Log "Successful: $($Stats.Successful)" "Green"
Write-Log "Failed: $($Stats.Failed)" "Red"
Write-Log "Skipped: $($Stats.Skipped)" "Yellow"
Write-Log "Duration: $($duration.ToString('hh\:mm\:ss'))" "Cyan"
Write-Log "========================================" "Cyan"

if ($failedSymbols.Count -gt 0) {
    Write-Log ""
    Write-Log "Failed Symbols ($($failedSymbols.Count)):" "Yellow"
    $failedSymbols | ForEach-Object { Write-Log "  - $_" "Yellow" }
    Write-Log ""
    Write-Log "You can re-run this script to retry failed symbols." "Cyan"
    Write-Log "Failed symbols will be automatically skipped if they exist." "Cyan"
}

Write-Log ""
Write-Success "Download completed!"
Write-Log "Log saved to: $LogFile" "Cyan"

