# 2-minute diagnosis when you have Trades but Return/Drawdown stay 0.
# Usage: .\run_zero_return_diagnosis.ps1 -RunDir "output\system_run\benchmark\trend_baseline\1y"
# Or set $env:RUN_DIR and run without args.

param([string]$RunDir = $env:RUN_DIR)

if (-not $RunDir -or -not (Test-Path $RunDir)) {
    Write-Error "Set RunDir: -RunDir '<path>' or `$env:RUN_DIR = 'output\system_run\benchmark\trend_baseline\1y'"
    exit 1
}

Write-Host "== A) Files in run dir =="
Get-ChildItem -Recurse $RunDir -File -ErrorAction SilentlyContinue | Select-Object FullName

Write-Host "`n== B) Equity curve: more than 1 row? min/max? =="
$eq = Get-ChildItem $RunDir -Filter "equity_curve_*.csv" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($eq) {
    Write-Host $eq.FullName
    Import-Csv $eq.FullName | Measure-Object -Property equity -Minimum -Maximum
    Import-Csv $eq.FullName | Select-Object -First 5
} else { Write-Host "no equity_curve_*.csv found" }

Write-Host "`n== C) Trades: status + fill_qty + first 10 rows =="
$tr = Get-ChildItem $RunDir -Filter "trades_*.csv" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($tr) {
    Write-Host $tr.FullName
    $csv = Import-Csv $tr.FullName -ErrorAction SilentlyContinue
    if ($csv | Get-Member -Name status -ErrorAction SilentlyContinue) {
        $csv | Group-Object status | Select-Object Name, Count
    }
    $csv | Select-Object -First 10 symbol, side, qty, fill_qty, price, fill_price, status, total_cost_cash -ErrorAction SilentlyContinue | Format-Table -AutoSize
} else { Write-Host "no trades_*.csv found" }

Write-Host "`n== D) Price movement in slice (total_ret per symbol) =="
$slicePath = Join-Path $RunDir "price_slice.parquet"
if (Test-Path $slicePath) {
    py -3 -c "import pandas as pd; p=r'$($slicePath -replace '\\','/')'; df=pd.read_parquet(p, columns=['symbol','timestamp','close']); df['timestamp']=pd.to_datetime(df['timestamp'], utc=True, errors='coerce'); g=df.sort_values(['symbol','timestamp']).groupby('symbol')['close']; rets=(g.last()/g.first()-1).rename('total_ret'); print(rets.to_string());"
} else { Write-Host "no price_slice.parquet in run dir" }

Write-Host "`nInterpretation: Min==Max equity => constant => Return 0. fill_qty=0 or status=rejected => no fills => Return 0. total_ret ~0 => market flat."
