# OOS-Sweep Debug + Run (PowerShell)
#
# Variante 1 - direkt in Konsole:
#   .\scripts\dev\run_oos_sweep_debug.ps1 -ParquetPath "<DEIN_PARQUET_PFAD>"
#   Dann kopieren: unter "== 2)" die rows/min/max/unique_days/null_timestamps-Zeile;
#   unter "== 4)" die FullName-Liste (oder "none found").
#
# Variante 2 - Output in Datei (Copy/Paste leichter):
#   .\scripts\dev\run_oos_sweep_debug.ps1 -ParquetPath "<DEIN_PARQUET_PFAD>" 2>&1 | Tee-Object -FilePath oos_debug_log.txt
#   notepad oos_debug_log.txt
#
# Bonus - Report irgendwo suchen:
#   Get-ChildItem -Recurse . -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "^oos(_sweep)?_report\.md$" } | Select-Object FullName
# Bonus - wenn nichts gefunden, naechster Check:
#   Get-Content "output\system_run\benchmark\anomalies.json" -TotalCount 80
#
# Prerequisites: parquet with enough history (>= 60 trading days for OOS; > 250 sensible).
# If start_date == end_date in run_inputs.json, OOS is correctly skipped.

param(
    [string]$ParquetPath = $env:PARQUET,
    [string]$OutRoot = "output/system_run"
)

if (-not $ParquetPath) {
    $defaultParquet = Join-Path $OutRoot "benchmark\synthetic\eod.parquet"
    if (Test-Path $defaultParquet) {
        $ParquetPath = $defaultParquet
        Write-Host "Using repo default parquet: $ParquetPath"
    } else {
        Write-Error "Set ParquetPath: -ParquetPath '<path>' or `$env:PARQUET = '<path>'. Or run once with --synthetic-only to create $defaultParquet"
        exit 1
    }
}

Write-Host "== 1) Check parquet exists =="
if (-not (Test-Path $ParquetPath)) {
    Write-Error "Dataset not found: $ParquetPath"
    exit 1
}

Write-Host "== 2) Inspect timestamp history (rows/min/max/unique_days) =="
$quoted = $ParquetPath -replace "'", "''"
py -3 -c "import pandas as pd; path=r'$quoted'; df=pd.read_parquet(path, columns=['timestamp']); ts=pd.to_datetime(df['timestamp'], utc=True, errors='coerce'); print('rows:', len(df)); print('min:', ts.min()); print('max:', ts.max()); print('unique_days:', ts.dt.date.nunique()); print('null_timestamps:', int(ts.isna().sum()))"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Parquet inspect failed; continuing anyway."
}

Write-Host "== 3) Run OOS sweep (no --quick) =="
py -3 scripts/dev/run_strategy_benchmark.py `
  --output-root $OutRoot `
  --dataset $ParquetPath `
  --oos-sweep `
  --max-variants 10

Write-Host "== 4) Locate OOS report(s) =="
$bench = Join-Path $OutRoot "benchmark"
$oosFiles = @(Get-ChildItem -Path $bench -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "^oos(_sweep)?_report\.md$" })
if ($oosFiles.Count -gt 0) {
    $oosFiles | ForEach-Object { $_.FullName }
} else {
    $recursive = @(Get-ChildItem -Recurse $bench -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "^oos(_sweep)?_report\.md$" })
    if ($recursive.Count -gt 0) { $recursive | ForEach-Object { $_.FullName } } else { Write-Host "none found" }
}

Write-Host "== 5) Print oos_sweep_report.md if present =="
$oosSweep = Join-Path $bench "oos_sweep_report.md"
if (Test-Path $oosSweep) {
  Write-Host "--- oos_sweep_report.md ---"
  Get-Content $oosSweep -TotalCount 200
} else {
  Write-Warning "oos_sweep_report.md not found at: $oosSweep"
  Write-Host "Tip: check run_inputs.json for the exact slice used:"
  $ri = Join-Path $bench "trend_baseline/1y/run_inputs.json"
  if (Test-Path $ri) { Get-Content $ri -TotalCount 200 }
}

Write-Host ""
Write-Host "Interpretation:"
Write-Host "  unique_days < 60 => OOS correctly skipped; need dataset with more history."
Write-Host "  unique_days OK but no report => wrong output root or script error (check anomalies.json / stderr)."
Write-Host "  null_timestamps > 0 or min/max = NaT => fix timestamp column or use another parquet."
Write-Host ""
Write-Host "Quick checks if report 'not there':"
Write-Host "  1) Wrong output root:  Get-ChildItem -Recurse . -File -ErrorAction SilentlyContinue | Where-Object { `$_.Name -eq 'oos_sweep_report.md' } | Select-Object FullName"
Write-Host "  2) Skip/exception:     Get-Content (Join-Path $bench 'anomalies.json') -TotalCount 80"
Write-Host "  3) Timestamp broken:   null_timestamps > 0 or min/max = NaT in step 2 output."
Write-Host ""
Write-Host "After run: report + scoreboard + anomalies (copy/paste):"
Write-Host "  Get-Content (Join-Path $bench 'oos_sweep_report.md') -TotalCount 200"
Write-Host "  Import-Csv (Join-Path $bench 'scoreboard.csv') | Sort-Object {[double]`$_.total_return} -Descending | Select-Object -First 15 variant_id,horizon,total_return,sharpe_ratio,max_drawdown_pct,total_trades,turnover"
Write-Host "  Get-Content (Join-Path $bench 'anomalies.json') -TotalCount 80"
Write-Host "If 0 Trades: qc_report  Get-ChildItem -Recurse (Join-Path $bench 'trend_baseline\1y') -Filter 'qc_report.json' | % { Get-Content `$_.FullName -TotalCount 120 }"
Write-Host "  run_inputs  Get-Content (Join-Path $bench 'trend_baseline\1y\run_inputs.json') -TotalCount 200"
