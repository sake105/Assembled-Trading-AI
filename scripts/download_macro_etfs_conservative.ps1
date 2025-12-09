# Download Macro-ETFs: Conservative Strategy
# Short-term solution: Download only Macro-ETFs with shortened date range
# Strategy: One symbol at a time, very long delays, shortened date range (2010-2025)

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2010-01-01",  # Shortened from 2000 to reduce load
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$DelayBetweenSymbols = 60,  # 60 seconds = very conservative
    [int]$SleepSeconds = 2.0,
    [int]$InitialWait = 120  # 2 minutes initial wait
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$UniverseFile = Join-Path $ROOT 'config\macro_world_etfs_tickers.txt'
$LogFile = Join-Path $ROOT "logs\download_macro_etfs_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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

Write-Log "========================================" "Cyan"
Write-Log "Macro-ETFs: Conservative Download" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Universe: Macro World ETFs (10 symbols)" "Cyan"
Write-Log "Target Root: $TargetRoot" "Cyan"
Write-Log "Date Range: $StartDate to $EndDate (shortened for stability)" "Cyan"
Write-Log "Interval: $Interval" "Cyan"
Write-Log "Delay Between Symbols: $DelayBetweenSymbols seconds (1 minute)" "Cyan"
Write-Log "Initial Wait: $InitialWait seconds (2 minutes)" "Cyan"
Write-Log "Log File: $LogFile" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log ""

# Load symbols
if (-not (Test-Path $UniverseFile)) {
    Write-Error "Universe file not found: $UniverseFile"
    exit 1
}

$Symbols = Get-Content $UniverseFile | Where-Object { 
    $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
} | ForEach-Object { $_.Trim().ToUpper() }

if ($Symbols.Count -eq 0) {
    Write-Error "No symbols found in $UniverseFile"
    exit 1
}

Write-Log "Found $($Symbols.Count) symbols to download" "Cyan"
Write-Log ""

# Initial wait
if ($InitialWait -gt 0) {
    Write-Log "Waiting $InitialWait seconds before starting (to avoid rate limits)..." "Yellow"
    Start-Sleep -Seconds $InitialWait
}

# Check existing files
$ExistingCount = 0
$ToDownload = @()

foreach ($sym in $Symbols) {
    $filePath = Join-Path $TargetRoot "$Interval\$sym.parquet"
    if (Test-Path $filePath) {
        try {
            $fileInfo = Get-Item $filePath
            if ($fileInfo.Length -gt 1024) {
                $ExistingCount++
                Write-Log "  $sym - Already exists ($([math]::Round($fileInfo.Length/1KB, 1)) KB)" "Yellow"
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

Write-Log ""
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
    
    # Check again if file exists
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
            
            # Wait much longer if rate limited (15 minutes)
            $longDelay = 900
            Write-Log "  Rate limit detected! Waiting $longDelay seconds (15 minutes)..." "Yellow"
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
Write-Log "Total Symbols: $($Symbols.Count)" "Cyan"
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
Write-Log ""
Write-Log "Note: This is a conservative strategy with shortened date range (2010-2025)." "Yellow"
Write-Log "For full historical data (2000-2025), consider using an alternative API provider." "Yellow"

