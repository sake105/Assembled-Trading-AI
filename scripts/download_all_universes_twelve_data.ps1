# Download All Universes with Twelve Data
# Downloads all symbols from all universe files using Twelve Data API

param(
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d",
    [int]$SleepSeconds = 8,  # 8 Calls/Min = 8 seconds between calls
    [string]$ApiKey = $env:ASSEMBLED_TWELVE_DATA_API_KEY
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'
$LogFile = Join-Path $ROOT "logs\download_all_universes_twelve_data_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

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

Write-Log "========================================" "Cyan"
Write-Log "Download All Universes - Twelve Data" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Target Root: $TargetRoot" "Cyan"
Write-Log "Date Range: $StartDate to $EndDate" "Cyan"
Write-Log "Interval: $Interval" "Cyan"
Write-Log "Sleep Between Calls: $SleepSeconds seconds (8 Calls/Min limit)" "Cyan"
Write-Log "API Key: $($ApiKey.Substring(0, 8))..." "Cyan"
Write-Log "Log File: $LogFile" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log ""

# Find all universe files
$UniverseFiles = @(
    "config\universe_ai_tech_tickers.txt",
    "config\healthcare_biotech_tickers.txt",
    "config\energy_resources_cyclicals_tickers.txt",
    "config\defense_security_aero_tickers.txt",
    "config\consumer_financial_misc_tickers.txt",
    "config\macro_world_etfs_tickers.txt"
)

$AllStats = @{
    TotalUniverses = 0
    TotalSymbols = 0
    Successful = 0
    Failed = 0
    Skipped = 0
}

$startTime = Get-Date

foreach ($universeFile in $UniverseFiles) {
    $universePath = Join-Path $ROOT $universeFile
    
    if (-not (Test-Path $universePath)) {
        Write-Log "Universe file not found: $universeFile" "Yellow"
        continue
    }
    
    $universeName = [System.IO.Path]::GetFileNameWithoutExtension($universeFile)
    Write-Log ""
    Write-Log "========================================" "Cyan"
    Write-Info "Processing Universe: $universeName"
    Write-Log "========================================" "Cyan"
    
    # Load symbols
    $Symbols = Get-Content $universePath | Where-Object { 
        $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") 
    } | ForEach-Object { $_.Trim().ToUpper() }
    
    if ($Symbols.Count -eq 0) {
        Write-Log "  No symbols found in $universeFile" "Yellow"
        continue
    }
    
    Write-Log "  Symbols: $($Symbols.Count)" "Cyan"
    $AllStats.TotalUniverses++
    $AllStats.TotalSymbols += $Symbols.Count
    
    # Download universe
    $args = @(
        $DownloadScript,
        "--symbols-file", $universePath,
        "--start", $StartDate,
        "--end", $EndDate,
        "--interval", $Interval,
        "--provider", "twelve_data",
        "--target-root", $TargetRoot,
        "--sleep-seconds", $SleepSeconds
    )
    
    try {
        $downloadStart = Get-Date
        $output = & $Python $args 2>&1
        $exitCode = $LASTEXITCODE
        $downloadDuration = ((Get-Date) - $downloadStart).TotalSeconds
        
        if ($exitCode -eq 0) {
            # Parse output for success count
            $outputText = $output -join "`n"
            if ($outputText -match "Success: (\d+)/(\d+)") {
                $successCount = [int]$matches[1]
                $totalCount = [int]$matches[2]
                $AllStats.Successful += $successCount
                $AllStats.Failed += ($totalCount - $successCount)
                Write-Success "${universeName}: ${successCount}/${totalCount} symbols downloaded ($([math]::Round($downloadDuration/60, 1)) minutes)"
            } else {
                Write-Success "$universeName: Download completed ($([math]::Round($downloadDuration/60, 1)) minutes)"
            }
        } else {
            Write-Error "$universeName: Download failed (exit code: $exitCode)"
            $AllStats.Failed += $Symbols.Count
        }
    } catch {
        Write-Error "$universeName: Exception: $_"
        $AllStats.Failed += $Symbols.Count
    }
    
    # Wait between universes (except for the last one)
    if ($universeFile -ne $UniverseFiles[-1]) {
        Write-Log "  Waiting 10 seconds before next universe..." "Cyan"
        Start-Sleep -Seconds 10
    }
}

$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Log ""
Write-Log "========================================" "Cyan"
Write-Log "Final Summary" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Total Universes: $($AllStats.TotalUniverses)" "Cyan"
Write-Log "Total Symbols: $($AllStats.TotalSymbols)" "Cyan"
Write-Log "Successful: $($AllStats.Successful)" "Green"
Write-Log "Failed: $($AllStats.Failed)" "Red"
Write-Log "Skipped: $($AllStats.Skipped)" "Yellow"
Write-Log "Total Duration: $($totalDuration.ToString('hh\:mm\:ss'))" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log ""
Write-Success "Download completed!"
Write-Log "Log saved to: $LogFile" "Cyan"

