# scripts/sprint10_portfolio.ps1
[CmdletBinding()]
param(
  [ValidateSet('5min','1d')] [string]$Freq = '5min',
  [double]$StartCapital = 10000,
  [double]$Exposure = 1.0,
  [double]$MaxLeverage = 1.0,
  [double]$CommissionBps = 0.0,
  [double]$SpreadW = 0.25,
  [double]$ImpactW = 0.5
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -LiteralPath $PSScriptRoot -Parent
Write-Host "[2025-11-02T$(Get-Date -UFormat %H:%M:%S)Z] [PF10] Start Portfolio | freq=$Freq cap=$StartCapital exp=$Exposure lev=$MaxLeverage" 

# Python bestimmen (aus .venv)
$py = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Python nicht gefunden: $py" }

$pyScript = @'
import argparse, pathlib as pl, numpy as np, pandas as pd
ROOT = pl.Path(__file__).resolve().parents[2]
OUT  = ROOT / "output"

def read_eq(freq:str)->pd.DataFrame:
    p = OUT / f"equity_curve_{freq}.csv"
    return pd.read_csv(p, parse_dates=["timestamp"])

def main(freq, start_capital, comm_bps, spread_w, impact_w):
    eq = read_eq(freq)
    eq["equity"] = eq["equity"].astype(float)
    rep = {
        "final_pf": float(eq["equity"].iloc[-1]/eq["equity"].iloc[0]),
        "rows": int(len(eq))
    }
    (OUT / f"portfolio_equity_{freq}.csv").write_text(eq.to_csv(index=False))
    with open(OUT / "portfolio_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Portfolio Report ({freq})\n\n")
        f.write(f"- Final PF: {rep['final_pf']:.4f}\n")
        f.write(f"- Points: {rep['rows']}\n")

if __name__ == "__main__":
    import sys
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
'@

$tmp = New-TemporaryFile
Set-Content -LiteralPath $tmp -Value $pyScript -Encoding UTF8

& $py $tmp $Freq $StartCapital $CommissionBps $SpreadW $ImpactW | Out-Null
Remove-Item $tmp -Force -ErrorAction SilentlyContinue

Write-Host "[PF10] [OK] written: $(Join-Path $repo 'output\portfolio_equity_{0}.csv' -f $Freq)"
Write-Host "[PF10] [OK] written: $(Join-Path $repo 'output\portfolio_report.md')"
Write-Host "[PF10] DONE Portfolio"

