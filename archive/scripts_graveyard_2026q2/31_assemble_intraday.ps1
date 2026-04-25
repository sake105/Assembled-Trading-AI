#Requires -Version 7.0
Set-StrictMode -Version Latest

function WInfo([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function WOk  ([string]$m){ Write-Host "[OK]   $m" -ForegroundColor Green }
function WErr ([string]$m){ Write-Host "[ERR]  $m" -ForegroundColor Red }

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outDir = Join-Path $ProjectRoot "output\assembled_intraday"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

# Python in der venv bevorzugen
$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if(-not (Test-Path $py)){ $py = "python" }

WInfo "Python: $py"
WInfo "Erzeuge (oder prüfe) assembled_intraday.parquet"

$code = @'
import json, os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

proj = r"{PROJECT}"
out_dir = os.path.join(proj, r"output", r"assembled_intraday")
os.makedirs(out_dir, exist_ok=True)
parq = os.path.join(out_dir, "assembled_intraday.parquet")

if os.path.exists(parq):
    df = pd.read_parquet(parq)
else:
    # Synthese-Daten (UTC) für 3 Ticker, 3 Tage 1-Minuten-Bars
    tz = timezone.utc
    end = datetime.now(tz).replace(second=0, microsecond=0)
    start = end - timedelta(days=3)
    idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    tickers = ["AAA","BBB","CCC"]
    frames = []
    rng = np.random.default_rng(42)
    for t in tickers:
        base = 100 + rng.normal(0, 1, size=len(idx)).cumsum()
        o = base
        h = base + np.abs(rng.normal(0, 0.3, size=len(idx)))
        l = base - np.abs(rng.normal(0, 0.3, size=len(idx)))
        c = base + rng.normal(0, 0.2, size=len(idx))
        v = rng.integers(10, 1000, size=len(idx))
        f = pd.DataFrame({"ts": idx, "ticker": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ticker","ts"])
    df.to_parquet(parq, index=False)

# Artefakte head/schema/report
head_csv = os.path.join(out_dir, "assembled_intraday_head.csv")
schema_json = os.path.join(out_dir, "assembled_intraday_schema.json")
report_json = os.path.join(out_dir, "assembled_intraday_report.json")

df.head(50).to_csv(head_csv, index=False)
schema = { "columns": { c:str(dt) for c,dt in df.dtypes.items() }, "rows": int(len(df)) }
with open(schema_json, "w", encoding="utf-8") as f: json.dump(schema, f, indent=2)
report = { "rows": int(len(df)), "tickers": int(df["ticker"].nunique()), "min_ts": str(df["ts"].min()), "max_ts": str(df["ts"].max()) }
with open(report_json, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)

print(head_csv); print(schema_json); print(report_json)
'@.Replace("{PROJECT}", $ProjectRoot)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName               = $py
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute        = $false
$null = $psi.ArgumentList.Add("-c")
$null = $psi.ArgumentList.Add($code)

$proc   = [System.Diagnostics.Process]::Start($psi)
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()


if($stdout){ $stdout -split "`r?`n" | Where-Object {$_} | ForEach-Object { WOk ("{0}" -f $_) } }
if($proc.ExitCode -ne 0){ WErr $stderr; throw "Assemble failed with code $($proc.ExitCode)" }
WOk "Assemble abgeschlossen."



