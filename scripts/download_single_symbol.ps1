# Helper script to download a single symbol with proper delays
# Usage: .\scripts\download_single_symbol.ps1 -Symbol NVDA

param(
    [Parameter(Mandatory=$true)]
    [string]$Symbol,
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Interval = "1d"
)

$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'

Write-Host "[DOWNLOAD] Downloading $Symbol..." -ForegroundColor Cyan

$args = @(
    $DownloadScript,
    "--symbol", $Symbol,
    "--start", $StartDate,
    "--end", $EndDate,
    "--interval", $Interval,
    "--target-root", "`"$TargetRoot`""
)

$output = & $Python $args 2>&1
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "[SUCCESS] $Symbol downloaded successfully!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to download $Symbol (exit code: $exitCode)" -ForegroundColor Red
    Write-Host ($output -join "`n")
}

exit $exitCode

