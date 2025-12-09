# Download Alt-Daten: One-by-One Safe Mode
# Strategy: Download one symbol at a time with long delays to avoid rate limits
# Automatically skips existing files

param(
    [string]$UniverseFile = "config\healthcare_biotech_tickers.txt",  # Start with small universe for testing
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$DelayBetweenSymbols = 10,  # Seconds to wait between symbols (increased for rate limit safety)
    [int]$SleepSeconds = 2.0,  # Sleep seconds passed to Python script
    [int]$InitialWait = 30  # Initial wait before starting (seconds)
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$LogFile = Join-Path $ROOT "logs\download_one_by_one_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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
Write-Log "One-by-One Safe Mode Download" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Universe File: $UniverseFile" "Cyan"
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

# Load symbols from file
$UniversePath = Join-Path $ROOT $UniverseFile
if (-not (Test-Path $UniversePath)) {
    Write-Error "Universe file not found: $UniversePath"
    exit 1
}

$Symbols = Get-Content $UniversePath | Where-Object { 
    $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
} | ForEach-Object { $_.Trim().ToUpper() }

if ($Symbols.Count -eq 0) {
    Write-Error "No symbols found in $UniverseFile"
    exit 1
}

Write-Log "Found $($Symbols.Count) symbols to download" "Cyan"
Write-Log ""

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
            
            # Wait much longer if rate limited (5 minutes)
            $longDelay = 300
            Write-Log "  Rate limit detected! Waiting $longDelay seconds (5 minutes)..." "Yellow"
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
}

Write-Log ""
Write-Success "Download completed!"
Write-Log "Log saved to: $LogFile" "Cyan"

