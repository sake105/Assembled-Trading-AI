# Setup Pipeline Integration
# Configures environment for using downloaded Alt-Daten in the pipeline

param(
    [string]$LocalDataRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pipeline Integration Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variable
$env:ASSEMBLED_LOCAL_DATA_ROOT = $LocalDataRoot
Write-Host "✓ Set ASSEMBLED_LOCAL_DATA_ROOT = $LocalDataRoot" -ForegroundColor Green

# Verify directory exists
if (Test-Path $LocalDataRoot) {
    Write-Host "✓ Target directory exists" -ForegroundColor Green
    
    # Count downloaded files
    $oneDayDir = Join-Path $LocalDataRoot "1d"
    if (Test-Path $oneDayDir) {
        $parquetFiles = Get-ChildItem $oneDayDir -Filter "*.parquet" | Where-Object { $_.Length -gt 1024 }
        Write-Host "✓ Found $($parquetFiles.Count) downloaded Parquet files" -ForegroundColor Green
    }
} else {
    Write-Host "✗ Target directory does not exist: $LocalDataRoot" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Test Backtest with new data:" -ForegroundColor Yellow
Write-Host "   python scripts/cli.py backtest --freq 1d --symbols-file config/macro_world_etfs_tickers.txt" -ForegroundColor White
Write-Host ""
Write-Host "2. Run Factor Report:" -ForegroundColor Yellow
Write-Host "   python scripts/cli.py factor_report --freq 1d --symbols-file config/macro_world_etfs_tickers.txt --start-date 2010-01-01 --end-date 2025-12-03" -ForegroundColor White
Write-Host ""
Write-Host "3. Verify data loading:" -ForegroundColor Yellow
Write-Host "   python -c `"from src.assembled_core.data.data_source import get_price_data_source; from src.assembled_core.config.settings import Settings; s = Settings(); s.local_data_root = '$LocalDataRoot'; ds = get_price_data_source(s, 'local'); print(ds.get_history(['SPY'], '2010-01-01', '2025-12-03', '1d'))`"" -ForegroundColor White
Write-Host ""
Write-Host "Note: Environment variable is set for this session only." -ForegroundColor Yellow
Write-Host "To make it permanent, add to your PowerShell profile or .env file." -ForegroundColor Yellow
Write-Host ""

