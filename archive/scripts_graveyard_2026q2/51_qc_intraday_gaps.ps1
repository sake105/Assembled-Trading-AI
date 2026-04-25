# QC: Lücken (Gaps) in 1-Minuten-Zeitreihen finden
# PS 7.x, UTF-8, keine BOM

param(
  [string]$ProjectRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [ValidateSet('1min','5min','15min','30min','60min')]
  [string]$ExpectedFreq  = '1min'
)

function WInfo([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function WOk  ([string]$m){ Write-Host "[OK]   $m" -ForegroundColor Green }
function WErr ([string]$m){ Write-Host "[ERR]  $m" -ForegroundColor Red }

$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if(-not (Test-Path $py)){ $py = "python" }

$inputParquet = Join-Path $ProjectRoot "output\assembled_intraday\assembled_intraday.parquet"
$outCsv       = Join-Path $ProjectRoot "output\qc\intraday_gaps.csv"
$outJson      = Join-Path $ProjectRoot "output\qc\intraday_gaps_summary.json"
$outDir       = Split-Path $outCsv -Parent
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

WInfo "Starte QC Gaps ..."

$code = @'
import sys, os, json
import pandas as pd
import numpy as np

project_root   = r"{PR}"
input_parquet  = os.path.join(project_root, r"output\assembled_intraday\assembled_intraday.parquet")
out_csv        = os.path.join(project_root, r"output\qc\intraday_gaps.csv")
out_json       = os.path.join(project_root, r"output\qc\intraday_gaps_summary.json")
expected_freq  = sys.argv[1]  # '1min' etc.

if not os.path.exists(input_parquet):
    raise FileNotFoundError(input_parquet)

df = pd.read_parquet(input_parquet)

# robuste Zeitstempelbehandlung:
# - egal ob aware/naiv -> in UTC konvertieren, dann tz entfernen
ts = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
df = df.copy()
df["ts"] = ts

ticker_col = "ticker" if "ticker" in df.columns else None
if ticker_col is None:
    df["ticker"] = "ALL"
    ticker_col = "ticker"

df = df.sort_values([ticker_col, "ts"], kind="stable")

exp = pd.to_timedelta(expected_freq)

rows = []
summary = []
total_gaps = 0
max_gap_minutes_overall = 0.0

for tkr, g in df.groupby(ticker_col, dropna=False, sort=False):
    ts = g["ts"].reset_index(drop=True)
    diffs = ts.diff().dropna()
    mask  = diffs > exp
    gaps  = diffs[mask]

    gaps_count = int(mask.sum())
    max_gap_minutes = float((gaps.max() / pd.Timedelta(minutes=1)) if gaps_count > 0 else 0.0)

    total_gaps += gaps_count
    max_gap_minutes_overall = max(max_gap_minutes_overall, max_gap_minutes)

    # konkrete Gap-Zeilen (Start/Ende)
    for idx in gaps.index:
        gap_start = ts.iloc[idx-1]
        gap_end   = ts.iloc[idx]
        gap_min   = float((gap_end - gap_start) / pd.Timedelta(minutes=1))
        rows.append({"ticker": tkr, "gap_start": gap_start, "gap_end": gap_end, "gap_minutes": gap_min})

    summary.append({"ticker": tkr, "total_gaps": gaps_count, "max_gap_minutes": max_gap_minutes})

# Ausgaben
pd.DataFrame(rows).to_csv(out_csv, index=False)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "expected_freq": expected_freq,
        "total_gaps": int(total_gaps),
        "max_gap_minutes_overall": float(max_gap_minutes_overall),
        "by_ticker": summary
    }, f, ensure_ascii=False, indent=2)

print("[OK]   QC Gaps abgeschlossen")
print("[OK]   CSV: ", out_csv)
print("[OK]   JSON:", out_json)
'@.Replace('{PR}', $ProjectRoot.Replace('\','\\'))

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName               = $py
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute        = $false
$null = $psi.ArgumentList.Add("-c")
$null = $psi.ArgumentList.Add($code)
$null = $psi.ArgumentList.Add($ExpectedFreq)

$proc   = [System.Diagnostics.Process]::Start($psi)
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()

if($stdout){ Write-Host $stdout }
if($stderr){ WErr $stderr }
if($proc.ExitCode -ne 0){ WErr $stderr; throw "Python-QC hat ExitCode $($proc.ExitCode)" }



