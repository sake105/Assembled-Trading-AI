# Download Missing Symbols Sequentially
# Downloads all missing symbols one by one with proper rate-limit handling

param(
    [string]$MissingSymbolsFile = "missing_symbols.txt",
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$SleepSeconds = 10,  # 10 seconds = safe margin for 8 Calls/Min limit
    [string]$ApiKey = $env:ASSEMBLED_TWELVE_DATA_API_KEY
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$LogFile = Join-Path $ROOT "logs\download_missing_sequential_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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

if (-not $ApiKey) {
    Write-Error "ASSEMBLED_TWELVE_DATA_API_KEY not set. Please set it as environment variable."
    exit 1
}

# Load missing symbols
if (-not (Test-Path $MissingSymbolsFile)) {
    Write-Error "Missing symbols file not found: $MissingSymbolsFile"
    Write-Info "Run the status check first to generate missing_symbols.txt"
    exit 1
}

$Symbols = Get-Content $MissingSymbolsFile | Where-Object { 
    $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
} | ForEach-Object { $_.Trim().ToUpper() }

if ($Symbols.Count -eq 0) {
    Write-Success "No missing symbols found! All downloads complete."
    exit 0
}

Write-Log "========================================" "Cyan"
Write-Log "Sequential Download - Missing Symbols" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Missing Symbols: $($Symbols.Count)" "Cyan"
Write-Log "Target Root: $TargetRoot" "Cyan"
Write-Log "Date Range: $StartDate to $EndDate" "Cyan"
Write-Log "Interval: $Interval" "Cyan"
Write-Log "Sleep Between Calls: $SleepSeconds seconds" "Cyan"
Write-Log "API Key: $($ApiKey.Substring(0, 8))..." "Cyan"
Write-Log "Log File: $LogFile" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log ""

# Statistics
$Stats = @{
    Successful = 0
    Failed = 0
    Skipped = 0
    RateLimited = 0
}

$startTime = Get-Date
$failedSymbols = @()
$rateLimitedSymbols = @()

# Download each symbol one by one
foreach ($idx in 0..($Symbols.Count - 1)) {
    $symbol = $Symbols[$idx]
    $symbolIndex = $idx + 1
    
    Write-Log ""
    Write-Info "Symbol $symbolIndex/$($Symbols.Count): $symbol"
    
    # Check if file already exists (might have been downloaded in parallel)
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
        "--provider", "twelve_data",
        "--target-root", $TargetRoot
    )
    
    try {
        $downloadStart = Get-Date
        $output = & $Python $args 2>&1
        $exitCode = $LASTEXITCODE
        $downloadDuration = ((Get-Date) - $downloadStart).TotalSeconds
        $outputText = $output -join "`n"
        
        # Check for rate limit
        $isRateLimit = $outputText -match "API credits|rate limit|too many requests|429|Rate limited"
        
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
            $Stats.RateLimited++
            $rateLimitedSymbols += $symbol
            
            # Wait much longer if rate limited (2 minutes)
            $longDelay = 120
            Write-Log "  Rate limit detected! Waiting $longDelay seconds (2 minutes)..." "Yellow"
            Start-Sleep -Seconds $longDelay
            continue
        } else {
            Write-Error "$symbol - Download failed (exit code: $exitCode)"
            $outputPreview = if ($outputText.Length -gt 200) { $outputText.Substring(0, 200) + "..." } else { $outputText }
            $errorLine = ($outputPreview -split "`n" | Where-Object { $_ -match "error|Error|ERROR|invalid|Invalid" } | Select-Object -First 1)
            if ($errorLine) {
                Write-Log "  Error: $errorLine" "Red"
            }
            $Stats.Failed++
            $failedSymbols += $symbol
        }
    } catch {
        Write-Error "$symbol - Exception: $_"
        $Stats.Failed++
        $failedSymbols += $symbol
    }
    
    # Wait before next symbol (except for the last one)
    if ($idx -lt ($Symbols.Count - 1)) {
        Write-Log "  Waiting $SleepSeconds seconds before next symbol..." "Cyan"
        Start-Sleep -Seconds $SleepSeconds
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
Write-Log "Successful: $($Stats.Successful)" "Green"
Write-Log "Failed: $($Stats.Failed)" "Red"
Write-Log "Skipped (already existed): $($Stats.Skipped)" "Yellow"
Write-Log "Rate Limited: $($Stats.RateLimited)" "Yellow"
Write-Log "Duration: $($duration.ToString('hh\:mm\:ss'))" "Cyan"
Write-Log "========================================" "Cyan"

if ($failedSymbols.Count -gt 0) {
    Write-Log ""
    Write-Log "Failed Symbols ($($failedSymbols.Count)):" "Yellow"
    $failedSymbols | ForEach-Object { Write-Log "  - $_" "Yellow" }
    Write-Log ""
    Write-Log "These symbols may not be available in Twelve Data or have invalid ticker formats." "Cyan"
}

if ($rateLimitedSymbols.Count -gt 0) {
    Write-Log ""
    Write-Log "Rate Limited Symbols ($($rateLimitedSymbols.Count)):" "Yellow"
    $rateLimitedSymbols | ForEach-Object { Write-Log "  - $_" "Yellow" }
    Write-Log ""
    Write-Log "You can re-run this script to retry rate-limited symbols." "Cyan"
}

Write-Log ""
Write-Success "Sequential download completed!"
Write-Log "Log saved to: $LogFile" "Cyan"

