<# DEPRECATED: Use scripts/cli.py batch_run --config-file <yaml> instead.
    
    This script performs parameter sweeps over exposure and commission values.
    The new batch runner provides better reproducibility, parallel execution,
    resume support, and structured outputs.
    
    Example migration:
    
    Old: pwsh -File scripts/sprint10_param_sweep.ps1 -Freq 5min -Exposures @(0.6,0.8,1.0,1.2) -CommBps @(0.0,0.5,1.0)
    
    New: Create a YAML config (see docs/BATCH_RUNNER_P4.md) with multiple runs
    and execute: python scripts/cli.py batch_run --config-file configs/sweep.yaml
    
    See: docs/BATCH_RUNNER_P4.md for examples and documentation.
#>

param(
  [string]$Freq = '5min',
  [double[]]$Exposures = @(0.6, 0.8, 1.0, 1.2),
  [double[]]$CommBps   = @(0.0, 0.5, 1.0)
)
$ErrorActionPreference='Stop'
$ROOT = (Get-Location).Path
$Py   = Join-Path $ROOT '.venv/Scripts/python.exe'
$Run  = Join-Path $ROOT 'scripts/sprint10_portfolio.py'
$OUT  = Join-Path $ROOT 'output'
$CSV  = Join-Path $OUT  'portfolio_sweep.csv'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
"freq,exposure,comm_bps,final_equity,total_ret" | Out-File -Encoding utf8 $CSV

foreach($e in $Exposures){
  foreach($c in $CommBps){
    & $Py $Run --freq $Freq --start-capital 10000 --exposure $e --max-leverage 1 `
               --commission-bps $c --spread-w 1 --impact-w 1 | Out-Null
    $eq = Import-Csv (Join-Path $OUT "portfolio_equity_${Freq}.csv")
    $start=[double]$eq[0].equity; $end=[double]$eq[-1].equity
    $ret = if($start -ne 0){ ($end/$start - 1) } else { 0 }
    "$Freq,$e,$c,$end,$ret" | Out-File -Append -Encoding utf8 $CSV
    Write-Host "[SWEEP] exp=$e comm=$c → final=${end}"
  }
}
Write-Host "[DONE] Sweep → $CSV"

