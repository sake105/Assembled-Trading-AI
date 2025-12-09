# Test Problem Symbols
# Tests individual symbols that failed during bulk download

param(
    [string[]]$Symbols = @("IOS.DE", "SMHN.DE", "BAVA.CO", "EUZ.DE"),
    [string]$StartDate = "2000-01-01",
    [string]$EndDate = "2025-12-03",
    [string]$Provider = "twelve_data",
    [string]$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
)

$ErrorActionPreference = 'Continue'
$ROOT = (Get-Location).Path
$Python = Join-Path $ROOT '.venv\Scripts\python.exe'
$DownloadScript = Join-Path $ROOT 'scripts\download_historical_snapshot.py'

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Problem Symbols" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Symbols: $($Symbols -join ', ')" -ForegroundColor Yellow
Write-Host "Provider: $Provider" -ForegroundColor Yellow
Write-Host ""

foreach ($symbol in $Symbols) {
    Write-Host "Testing $symbol..." -ForegroundColor Cyan
    
    $args = @(
        $DownloadScript,
        "--symbol", $symbol,
        "--start", $StartDate,
        "--end", $EndDate,
        "--interval", "1d",
        "--provider", $Provider,
        "--target-root", $TargetRoot
    )
    
    try {
        $output = & $Python $args 2>&1
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host "  ✓ $symbol - Success" -ForegroundColor Green
        } else {
            $outputText = $output -join "`n"
            Write-Host "  ✗ $symbol - Failed (exit code: $exitCode)" -ForegroundColor Red
            if ($outputText -match "error|Error|ERROR") {
                $errorLines = $outputText -split "`n" | Where-Object { $_ -match "error|Error|ERROR" } | Select-Object -First 3
                $errorLines | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            }
        }
    } catch {
        Write-Host "  ✗ $symbol - Exception: $_" -ForegroundColor Red
    }
    
    Write-Host ""
    Start-Sleep -Seconds 2
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

