#Requires -Version 7
[CmdletBinding()]
param(
  [string]$Symbols        = "AAPL,MSFT",
  [int]   $Days           = 2,
  [string]$Freq           = "5min",
  [int]   $StartCapital   = 10000,
  [int]   $EmaFast        = 20,
  [int]   $EmaSlow        = 60,
  [double]$CommissionBps  = 0.5,
  [double]$SpreadW        = 0.5,
  [double]$ImpactW        = 1.0,
  [switch]$SkipPull,
  [switch]$SkipBacktest,
  [switch]$SkipPortfolio
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Fail([string]$msg) { throw $msg }
function Info([string]$msg) { Write-Host $msg -ForegroundColor Cyan }
function Ok  ([string]$msg) { Write-Host $msg -ForegroundColor Green }
function Warn([string]$msg) { Write-Warning $msg }

# --- Robust Script/Repo Root Detection --------------------------------------
$scriptPath = $null
if ($PSCommandPath) { $scriptPath = $PSCommandPath }
elseif ($MyInvocation?.MyCommand?.Path) { $scriptPath = $MyInvocation.MyCommand.Path }
elseif ($PSScriptRoot) { $scriptPath = Join-Path $PSScriptRoot 'run_all_sprint10.ps1' }

if (-not $scriptPath -or -not (Test-Path -LiteralPath $scriptPath)) {
  Fail "Konnte Pfad zu run_all_sprint10.ps1 nicht bestimmen (PSCommandPath='$PSCommandPath', PSScriptRoot='$PSScriptRoot')."
}

$scriptRoot = Split-Path -LiteralPath $scriptPath -Parent
$repoRoot   = Split-Path -LiteralPath $scriptRoot -Parent

# --- Paths -------------------------------------------------------------------
$pythonExe  = Join-Path $repoRoot '.venv\Scripts\python.exe'
$scriptsDir = Join-Path $repoRoot 'scripts'
$liveDir    = Join-Path $scriptsDir 'live'
$pullPs1    = Join-Path $liveDir  'pull_intraday.ps1'
$execPy     = Join-Path $scriptsDir 'sprint9_execute.py'
$btPy       = Join-Path $scriptsDir 'sprint9_backtest.py'
$pfPy       = Join-Path $scriptsDir 'sprint10_portfolio.py'

$rawDir     = Join-Path $repoRoot 'data\raw\1min'
$outDir     = Join-Path $repoRoot 'output'
$outAggDir  = Join-Path $outDir   'aggregates'
$px5m       = Join-Path $outAggDir '5min.parquet'
$pxDaily    = Join-Path $outAggDir 'daily.parquet'  # optional vorhanden

$eqCurve5   = Join-Path $outDir 'equity_curve_5min.csv'
$eqCurve1d  = Join-Path $outDir 'equity_curve_1d.csv'
$perf5      = Join-Path $outDir 'performance_report_5min.md'
$perf1d     = Join-Path $outDir 'performance_report_1d.md'
$portRep    = Join-Path $outDir 'portfolio_report.md'
$portEq5    = Join-Path $outDir 'portfolio_equity_5min.csv'
$portEq1d   = Join-Path $outDir 'portfolio_equity_1d.csv'

# --- Sanity ------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $pythonExe)) {
  Fail "Python nicht gefunden: $pythonExe  (Bitte .venv initialisieren)"
}
if (-not (Test-Path -LiteralPath $scriptsDir)) { Fail "Scripts-Ordner fehlt: $scriptsDir" }

New-Item -ItemType Directory -Force -Path $outDir     | Out-Null
New-Item -ItemType Directory -Force -Path $outAggDir  | Out-Null
New-Item -ItemType Directory -Force -Path $rawDir     | Out-Null

Write-Host "[RUNALL] Repo    : $repoRoot"
Write-Host "[RUNALL] Python  : $pythonExe"

# --- 1) Pull Intraday --------------------------------------------------------
if (-not $SkipPull) {
  Write-Host "[RUNALL] Pull: $Symbols Days=$Days"
  if (-not (Test-Path -LiteralPath $pullPs1)) { Fail "pull_intraday.ps1 fehlt: $pullPs1" }
  & pwsh -NoProfile -ExecutionPolicy Bypass -File $pullPs1 -Symbols $Symbols -Days $Days
  Write-Host "[RUNALL] Pull DONE"
} else {
  Write-Host "[RUNALL] Pull übersprungen (--SkipPull)"
}

# --- 2) Resample 1m -> 5m (robust, ohne Heredoc) ----------------------------
Write-Host "[RUNALL] Resample 1m -> 5m"

$resPy = @"
import os, pathlib as pl
import pandas as pd

pd.options.mode.use_inf_as_na = True

root   = pl.Path(r"$rawDir")
outdir = pl.Path(r"$outAggDir")
outdir.mkdir(parents=True, exist_ok=True)
dest   = outdir / "5min.parquet"

files = sorted(root.glob("*.parquet"))
if not files:
    print("[RESAMPLE] NOFILES in", root)
    raise SystemExit(2)

dfs = []
for p in files:
    try:
        df = pd.read_parquet(p)
        cols = df.columns.tolist()
        # Erwartet: timestamp,symbol,close  — Fallback: symbol aus Dateiname
        if "symbol" not in df.columns:
            sym = p.stem.upper()
            df = df.assign(symbol=sym)
        need = {"timestamp","symbol","close"}
        if not need.issubset(df.columns):
            print(f"[RESAMPLE] skip {p.name}: missing columns -> {cols}")
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp","close"])
        df = (df.set_index("timestamp")
                .groupby("symbol", group_keys=False)["close"]
                .resample("5min").last().reset_index())
        dfs.append(df)
        print(f"[RESAMPLE] OK {p.name} -> rows={len(df)}")
    except Exception as e:
        print(f"[RESAMPLE] ERROR {p.name}: {e}")

if not dfs:
    print("[RESAMPLE] NOFILES or all invalid")
    raise SystemExit(2)

out = pd.concat(dfs, ignore_index=True).dropna()
out = out.sort_values(["timestamp","symbol"]).reset_index(drop=True)
out.to_parquet(dest, index=False)
print(f"[RESAMPLE] wrote {dest} rows={len(out)} symbols={out['symbol'].nunique()} first={out['timestamp'].min()} last={out['timestamp'].max()}")
"@

$tmpResPath = Join-Path $outDir "_tmp_resample_5m.py"
$resPy | Set-Content -LiteralPath $tmpResPath -Encoding UTF8

$code = 0
try {
  & $pythonExe $tmpResPath
  $code = $LASTEXITCODE
} finally {
  Remove-Item -LiteralPath $tmpResPath -ErrorAction SilentlyContinue
}
if ($code -ne 0) { Fail "[RESAMPLE] Python exited with code $code (siehe Meldungen oben)" }

# --- 3) Execute Orders -------------------------------------------------------
Write-Host "[RUNALL] EXECUTE orders | freq=$Freq ema=($EmaFast,$EmaSlow)"
$execArgs = @("--freq", $Freq, "--ema-fast", $EmaFast, "--ema-slow", $EmaSlow, "--price-file", $px5m)
& $pythonExe $execPy @execArgs
Write-Host "[RUNALL] EXECUTE done"

# --- 4) Backtests ------------------------------------------------------------
if (-not $SkipBacktest) {
  Write-Host "[RUNALL] BACKTEST $Freq"
  & $pythonExe $btPy --freq $Freq --start-capital $StartCapital
  Write-Host "[RUNALL] perf5   : $perf5"
  Write-Host "[RUNALL] eq_5min : $eqCurve5"

  if (Test-Path -LiteralPath $pxDaily) {
    Write-Host "[RUNALL] BACKTEST 1d"
    & $pythonExe $btPy --freq 1d --start-capital $StartCapital
    Write-Host "[RUNALL] perf1d  : $perf1d"
    Write-Host "[RUNALL] eq_1d   : $eqCurve1d"
  } else {
    Warn "[RUNALL] daily.parquet fehlt -> 1d-Backtest ausgelassen ($pxDaily)"
  }
}

# --- 5) Portfolio-Simulation -------------------------------------------------
if (-not $SkipPortfolio) {
  Write-Host "[RUNALL] PORTFOLIO $Freq | cap=$StartCapital comm=${CommissionBps}bps spread=$SpreadW impact=$ImpactW"
  & $pythonExe $pfPy --freq $Freq --start-capital $StartCapital `
    --commission-bps $CommissionBps --spread-w $SpreadW --impact-w $ImpactW
  Write-Host "[RUNALL] port rep : $portRep"

  if (Test-Path -LiteralPath $pxDaily) {
    Write-Host "[RUNALL] PORTFOLIO 1d | cap=$StartCapital"
    & $pythonExe $pfPy --freq 1d --start-capital $StartCapital `
      --commission-bps 0.0 --spread-w 0.25 --impact-w 0.5
    Write-Host "[RUNALL] port rep : $portRep"
  }
}

Ok "[RUNALL] DONE"
