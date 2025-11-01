<# 
.SYNOPSIS
  PS7 wrapper to pull live 1m intraday data via yfinance and write Parquet compatible with Sprint8+.

.PARAMETER Symbols
  Comma separated list (default: AAPL,MSFT)

.PARAMETER Days
  Days of 1m history to fetch (default: 5; Yahoo supports up to 7 for 1m)

.EXAMPLE
  pwsh -File .\scripts\live\pull_intraday.ps1 -Symbols "AAPL,MSFT" -Days 5
#>

[CmdletBinding()]
param(
  [string]$Symbols = "AAPL,MSFT",
  [int]$Days = 5
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent -Path $PSCommandPath | Split-Path -Parent  # repo root from scripts/live
Write-Host "[LIVE] Repo: $repo"

# Ensure venv + packages
$activate = Join-Path $repo "scripts\tools\activate_python.ps1"
if(-not (Test-Path $activate -PathType Leaf)){
  throw "activate_python.ps1 nicht gefunden: $activate"
}

# Activate (ensures pip + requirements)
& $activate | Write-Host

# Ensure yfinance is available (lightweight, no pin forced)
$python = Join-Path $repo ".venv\Scripts\python.exe"
$null = & $python -c "import importlib, sys; sys.exit(0 if importlib.util.find_spec('yfinance') else 1)"
if($LASTEXITCODE -ne 0){
  Write-Host "[LIVE] Installing yfinance…"
  & $python -m pip install --upgrade "yfinance==0.2.54"
  if($LASTEXITCODE -ne 0){ throw "pip install yfinance failed" }
}

# Run puller
$pull = Join-Path $repo "scripts\live\pull_intraday.py"
if(-not (Test-Path $pull -PathType Leaf)){
  throw "pull_intraday.py nicht gefunden: $pull"
}

Write-Host "[LIVE] Running puller…"
& $python $pull --symbols $Symbols --days $Days --repo-root $repo
if($LASTEXITCODE -ne 0){ throw "Live-Pull fehlgeschlagen" }

Write-Host "[LIVE] DONE"
