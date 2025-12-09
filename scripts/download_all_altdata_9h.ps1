# Download all Alt-Daten symbols with ultra-conservative timing for 9-hour run
# Strategy: Very long delays between downloads to avoid rate limits
# Target: Complete all 59 symbols over 9 hours

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$BaseDelaySeconds = 540,  # 9 minutes base delay (conservative for 9-hour window)
    [int]$JitterSeconds = 60,  # Random 0-60 seconds jitter
    [int]$InitialWait = 300,  # Wait 5 minutes before starting
    [switch]$SkipExisting = $true
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$LogFile = Join-Path $ROOT "logs\download_altdata_9h_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create logs directory if needed
$LogDir = Split-Path $LogFile -Parent
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Log($msg, $color = "White") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMsg = "[$timestamp] [DOWNLOAD] $msg"
    Write-Host $logMsg -ForegroundColor $color
    Add-Content -Path $LogFile -Value $logMsg -ErrorAction SilentlyContinue
}

function Write-Error-Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMsg = "[$timestamp] [ERROR] $msg"
    Write-Host $logMsg -ForegroundColor Red
    Add-Content -Path $LogFile -Value $logMsg -ErrorAction SilentlyContinue
}

function Write-Success-Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMsg = "[$timestamp] [SUCCESS] $msg"
    Write-Host $logMsg -ForegroundColor Green
    Add-Content -Path $LogFile -Value $logMsg -ErrorAction SilentlyContinue
}

function Write-Warning-Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMsg = "[$timestamp] [WARNING] $msg"
    Write-Host $logMsg -ForegroundColor Yellow
    Add-Content -Path $LogFile -Value $logMsg -ErrorAction SilentlyContinue
}

Write-Log "========================================" "Cyan"
Write-Log "Alt-Daten Download - 9 Hour Run" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Log File: $LogFile" "Cyan"
Write-Log ""

# Collect all symbols from all universe files (including macro ETFs)
Write-Log "Collecting symbols from all universe files..." "Cyan"

$AllSymbols = @()

$UniverseFiles = @(
    "config\universe_ai_tech_tickers.txt",
    "config\healthcare_biotech_tickers.txt",
    "config\energy_resources_cyclicals_tickers.txt",
    "config\defense_security_aero_tickers.txt",
    "config\consumer_financial_misc_tickers.txt",
    "config\macro_world_etfs_tickers.txt"
)

foreach ($file in $UniverseFiles) {
    $filePath = Join-Path $ROOT $file
    if (Test-Path $filePath) {
        $symbols = Get-Content $filePath | Where-Object { 
            $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
        }
        $AllSymbols += $symbols
        Write-Log "  Found $($symbols.Count) symbols in $file" "Cyan"
    } else {
        Write-Warning-Log "  File not found: $filePath"
    }
}

$AllSymbols = $AllSymbols | Sort-Object -Unique
Write-Log "Total unique symbols: $($AllSymbols.Count)" "Cyan"
Write-Log ""

# Check which files already exist
$ExistingFiles = @()
$MissingSymbols = @()

foreach ($symbol in $AllSymbols) {
    $filePath = Join-Path $TargetRoot "$Interval\$symbol.parquet"
    if (Test-Path $filePath) {
        try {
            $fileInfo = Get-Item $filePath
            if ($fileInfo.Length -gt 1024) {
                $ExistingFiles += $symbol
            } else {
                $MissingSymbols += $symbol
                Write-Warning-Log "  Found empty/corrupt file for $symbol, will re-download"
            }
        } catch {
            $MissingSymbols += $symbol
        }
    } else {
        $MissingSymbols += $symbol
    }
}

Write-Log "Files already exist: $($ExistingFiles.Count)" "Cyan"
Write-Log "Symbols to download: $($MissingSymbols.Count)" "Cyan"
Write-Log ""

if ($MissingSymbols.Count -eq 0) {
    Write-Success-Log "All symbols already downloaded!"
    Write-Log "Exiting." "Cyan"
    exit 0
}

# Calculate timing for 9-hour window
$TotalSeconds = 9 * 3600  # 9 hours
$EstimatedDelay = $BaseDelaySeconds + ($JitterSeconds / 2)
$EstimatedTotalTime = ($MissingSymbols.Count * $EstimatedDelay) + $InitialWait
$EstimatedHours = [math]::Round($EstimatedTotalTime / 3600, 1)

Write-Log "Timing Estimates:" "Cyan"
Write-Log "  Total symbols to download: $($MissingSymbols.Count)" "Cyan"
Write-Log "  Base delay per symbol: $BaseDelaySeconds seconds (~$([math]::Round($BaseDelaySeconds/60, 1)) minutes)" "Cyan"
Write-Log "  Jitter range: 0-$JitterSeconds seconds" "Cyan"
Write-Log "  Estimated total time: ~$EstimatedHours hours" "Cyan"
Write-Log "  Target window: 9 hours" "Cyan"
Write-Log ""

# Initial wait to let rate limit reset
if ($InitialWait -gt 0) {
    Write-Warning-Log "Waiting $InitialWait seconds ($([math]::Round($InitialWait/60, 1)) minutes) before starting downloads..."
    Write-Warning-Log "This helps reset any existing rate limits."
    Start-Sleep -Seconds $InitialWait
}

Write-Log "========================================" "Cyan"
Write-Log "Starting Downloads" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Target Root: $TargetRoot" "Cyan"
Write-Log "Date Range: $StartDate to $EndDate" "Cyan"
Write-Log "Interval: $Interval" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log ""

$Stats = @{
    Successful = 0
    Failed = 0
    Skipped = 0
    RateLimited = 0
}

$startTime = Get-Date
$failedSymbols = @()
$random = New-Object System.Random

foreach ($idx in 0..($MissingSymbols.Count - 1)) {
    $symbol = $MissingSymbols[$idx]
    $symbolIndex = $idx + 1
    $elapsed = (Get-Date) - $startTime
    $remaining = $MissingSymbols.Count - $symbolIndex
    
    Write-Log ""
    Write-Log "========================================" "Cyan"
    Write-Log "Symbol $symbolIndex/$($MissingSymbols.Count): $symbol" "Cyan"
    Write-Log "Elapsed: $($elapsed.ToString('hh\:mm\:ss'))" "Cyan"
    Write-Log "Remaining: $remaining symbols" "Cyan"
    Write-Log "========================================" "Cyan"
    
    # Skip if exists and skip flag is set
    if ($SkipExisting) {
        $filePath = Join-Path $TargetRoot "$Interval\$symbol.parquet"
        if (Test-Path $filePath) {
            try {
                $fileInfo = Get-Item $filePath
                if ($fileInfo.Length -gt 1024) {
                    Write-Log "  File already exists (size: $($fileInfo.Length) bytes), skipping" "Yellow"
                    $Stats.Skipped++
                    continue
                }
            } catch {
                # File exists but can't read it, continue with download
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
        "--target-root", $TargetRoot
    )
    
    Write-Log "  Starting download..." "White"
    $downloadStart = Get-Date
    
    try {
        $output = & $Python $args 2>&1
        $exitCode = $LASTEXITCODE
        $outputText = $output -join "`n"
        $downloadDuration = ((Get-Date) - $downloadStart).TotalSeconds
        
        # Check for rate limit
        $isRateLimit = $outputText -match "rate limit|too many requests|429|Rate limited|YFRateLimitError"
        
        if ($exitCode -eq 0) {
            # Verify file was created
            $filePath = Join-Path $TargetRoot "$Interval\$symbol.parquet"
            if (Test-Path $filePath) {
                try {
                    $fileInfo = Get-Item $filePath
                    if ($fileInfo.Length -gt 1024) {
                        Write-Success-Log "  ✓ $symbol downloaded successfully ($([math]::Round($downloadDuration, 1))s, $($fileInfo.Length) bytes)"
                        $Stats.Successful++
                    } else {
                        Write-Error-Log "  ✗ $symbol - File created but too small ($($fileInfo.Length) bytes)"
                        $Stats.Failed++
                        $failedSymbols += $symbol
                    }
                } catch {
                    Write-Error-Log "  ✗ $symbol - Cannot verify file size"
                    $Stats.Failed++
                    $failedSymbols += $symbol
                }
            } else {
                Write-Error-Log "  ✗ $symbol - Download reported success but file not found"
                $Stats.Failed++
                $failedSymbols += $symbol
            }
        } elseif ($isRateLimit) {
            Write-Error-Log "  ✗ $symbol - Rate limited"
            $Stats.Failed++
            $Stats.RateLimited++
            $failedSymbols += $symbol
            
            # If rate limited, wait much longer
            $longDelay = $BaseDelaySeconds * 4  # 4x normal delay = ~36 minutes
            Write-Warning-Log "  Rate limit detected! Waiting $longDelay seconds ($([math]::Round($longDelay/60, 1)) minutes) before continuing..."
            Start-Sleep -Seconds $longDelay
            continue  # Skip normal delay, we already waited
        } else {
            Write-Error-Log "  ✗ $symbol - Download failed (exit code: $exitCode)"
            $outputPreview = if ($outputText.Length -gt 200) { $outputText.Substring(0, 200) + "..." } else { $outputText }
            Write-Error-Log "  Output preview: $outputPreview"
            $Stats.Failed++
            $failedSymbols += $symbol
        }
    } catch {
        Write-Error-Log "  ✗ $symbol - Exception: $_"
        $Stats.Failed++
        $failedSymbols += $symbol
    }
    
    # Calculate delay with jitter
    if ($idx -lt ($MissingSymbols.Count - 1)) {
        $jitter = $random.Next(0, $JitterSeconds + 1)
        $delay = $BaseDelaySeconds + $jitter
        $delayMinutes = [math]::Round($delay / 60, 1)
        
        Write-Log "  Waiting $delay seconds ($delayMinutes minutes) before next symbol..." "Cyan"
        Start-Sleep -Seconds $delay
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
Write-Log "Already Existed: $($ExistingFiles.Count)" "Cyan"
Write-Log "Successful: $($Stats.Successful)" "Green"
Write-Log "Failed: $($Stats.Failed)" "Red"
Write-Log "Rate Limited: $($Stats.RateLimited)" "Yellow"
Write-Log "Skipped: $($Stats.Skipped)" "Yellow"
Write-Log "Duration: $($duration.ToString('hh\:mm\:ss'))" "Cyan"
Write-Log "========================================" "Cyan"

if ($failedSymbols.Count -gt 0) {
    Write-Log ""
    Write-Warning-Log "Failed Symbols ($($failedSymbols.Count)):"
    Write-Warning-Log ($failedSymbols -join ", ")
    Write-Log ""
    Write-Log "You can re-run this script to retry failed symbols." "Cyan"
    Write-Log "It will automatically skip symbols that were successfully downloaded." "Cyan"
}

Write-Log ""
Write-Success-Log "Batch download completed!"
Write-Log "Log saved to: $LogFile" "Cyan"
Write-Log ""
Write-Log "To retry failed downloads, run this script again." "Cyan"
