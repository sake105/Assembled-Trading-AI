<# Orchestriert Sprint 8–10 Pipeline. #>
[CmdletBinding()]
param(
  [switch]$Seed,
  [switch]$Rehydrate,
  [switch]$Features,
  [switch]$Cost,
  [switch]$Backtest,
  [switch]$CostGrid,
  [switch]$Portfolio,
  [switch]$Sync,

  [ValidateSet('1min','5min')]
  [string]$Freq = '5min',

  [double]$StartCapital = 10000,
  [double]$Exposure     = 1,
  [double]$MaxLeverage  = 1,

  [double]$CommissionBps = 0.5,
  [double]$SpreadW       = 1.0,
  [double]$ImpactW       = 1.0,

  # Quick-Window für Sprint8 (Tage)
  [int]$QuickDays = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Step {
  param([string]$Label, [ScriptBlock]$Action)
  Write-Host "[$Label] START" -ForegroundColor Cyan
  try {
    & $Action
    Write-Host "[$Label] DONE" -ForegroundColor Green
  } catch {
    Write-Host "[$Label] FEHLER: $($_.Exception.Message)" -ForegroundColor Red
    throw
  }
}

# --- Pfade
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptRoot
$VenvPath   = Join-Path $RepoRoot ".venv"
$ReqPath    = Join-Path $RepoRoot "requirements.txt"

# --- Python / Venv sicherstellen
$ActivatePath = Join-Path $ScriptRoot "tools\activate_python.ps1"
if (-not (Test-Path $ActivatePath -PathType Leaf)) {
  throw "tools\activate_python.ps1 nicht gefunden."
}
. $ActivatePath   # dot-source: bringt Ensure-Venv in den aktuellen Scope

$Global:VenvPython = Ensure-Venv -RepoRoot $RepoRoot -VenvPath $VenvPath -Requirements $ReqPath -Quiet:$false

Write-Host "[RUNALL] Using Python: $Global:VenvPython" -ForegroundColor DarkCyan
$ts = Get-Date -AsUTC -Format "yyyy-MM-ddTHH:mm:ssZ"
Write-Host "[$ts] [RUNALL Start | freq=$Freq cap=$StartCapital exp=$Exposure lev=$MaxLeverage comm=${CommissionBps}bps spread=$SpreadW impact=$ImpactW]" -ForegroundColor DarkCyan

# --- Helpers zu Skriptpfaden
$PS8_Rehydrate = Join-Path $ScriptRoot "run_sprint8_rehydrate.ps1"
$PS8_Cost      = Join-Path $ScriptRoot "sprint8_cost_model.ps1"
$PS9_Backtest  = Join-Path $ScriptRoot "sprint9_backtest.ps1"
$PS9_Grid      = Join-Path $ScriptRoot "sprint9_cost_grid.ps1"
$PS10_Port     = Join-Path $ScriptRoot "sprint10_portfolio.ps1"
$PS_GitSync    = Join-Path $ScriptRoot "tools\git_sync.ps1"
$SeedPy        = Join-Path $ScriptRoot "tools\seed_demo_data.py"

# --- SEED
if ($Seed) {
  Invoke-Step -Label "SEED" -Action {
    if (Test-Path $SeedPy -PathType Leaf) {
      & $Global:VenvPython "$SeedPy" --freq "$Freq"
      if ($LASTEXITCODE -ne 0) { throw "seed_demo_data.py exit $LASTEXITCODE" }
    } else {
      Write-Warning "seed_demo_data.py nicht gefunden – überspringe Seed."
    }
  }
}

# --- REHYDRATE (benannte Parameter, kein Positionsbinding)
if ($Rehydrate) {
  Invoke-Step -Label "REHYDRATE" -Action {
    if (-not (Test-Path $PS8_Rehydrate -PathType Leaf)) { throw "run_sprint8_rehydrate.ps1 nicht gefunden." }
    & $PS8_Rehydrate -Freq $Freq -Quick -QuickDays $QuickDays
    if ($LASTEXITCODE -ne 0) { throw "run_sprint8_rehydrate.ps1 exit $LASTEXITCODE" }
  }
}

# --- FEATURES (idempotent – Sprint8 schreibt Dateien; optional erneut)
if ($Features) {
  Invoke-Step -Label "FEATURES" -Action {
    if (-not (Test-Path $PS8_Rehydrate -PathType Leaf)) { throw "run_sprint8_rehydrate.ps1 nicht gefunden." }
    & $PS8_Rehydrate -Freq $Freq -Quick -QuickDays $QuickDays
    if ($LASTEXITCODE -ne 0) { throw "Features (via Rehydrate) exit $LASTEXITCODE" }
  }
}

# --- COST
if ($Cost) {
  Invoke-Step -Label "COST" -Action {
    if (-not (Test-Path $PS8_Cost)) { throw "sprint8_cost_model.ps1 nicht gefunden." }
    & $PS8_Cost -Freq $Freq -Notional $StartCapital -CommissionBps $CommissionBps
    if ($LASTEXITCODE -ne 0) { throw "sprint8_cost_model exit $LASTEXITCODE" }
  }
}

# --- BACKTEST
if ($Backtest) {
  Invoke-Step -Label "BACKTEST" -Action {
    if (-not (Test-Path $PS9_Backtest)) { throw "sprint9_backtest.ps1 nicht gefunden." }
    & $PS9_Backtest -Freq $Freq
    if ($LASTEXITCODE -ne 0) { throw "sprint9_backtest exit $LASTEXITCODE" }
  }
}

# --- COST GRID
if ($CostGrid) {
  Invoke-Step -Label "COSTGRID" -Action {
    if (-not (Test-Path $PS9_Grid)) { throw "sprint9_cost_grid.ps1 nicht gefunden." }
    & $PS9_Grid -Freq $Freq -Notional $StartCapital `
# NEU (echte Float-Arrays)
	& "$PSScriptRoot\sprint9_cost_grid.ps1" `
  -Freq $Freq `
  -CommissionBps @(0.0, 0.5, 1.0) `
  -SpreadW @(0.5, 1.0, 2.0) `
  -ImpactW @(0.5, 1.0, 2.0) `
  -Notional $StartCapital
if($LASTEXITCODE -ne 0){ throw "COSTGRID fehlgeschlagen" }
  }
}

# --- PORTFOLIO
if ($Portfolio) {
  Invoke-Step -Label "PF10" -Action {
    if (-not (Test-Path $PS10_Port)) { throw "sprint10_portfolio.ps1 nicht gefunden." }
    & $PS10_Port -Freq $Freq `
      -StartCapital $StartCapital -Exposure $Exposure -MaxLeverage $MaxLeverage `
      -CommissionBps $CommissionBps -SpreadW $SpreadW -ImpactW $ImpactW
    if ($LASTEXITCODE -ne 0) { throw "sprint10_portfolio exit $LASTEXITCODE" }
  }
}

# --- GIT SYNC
if ($Sync) {
  Invoke-Step -Label "GIT" -Action {
    if (-not (Test-Path $PS_GitSync)) { throw "tools\git_sync.ps1 nicht gefunden." }
    $msg = "RunAll: $Freq, cap=$StartCapital, exp=$Exposure, comm=$CommissionBps" +
           "bps, spread=$SpreadW, impact=$ImpactW"
    & $PS_GitSync -Message $msg
    if ($LASTEXITCODE -ne 0) { throw "git_sync exit $LASTEXITCODE" }
  }
}

$tsDone = Get-Date -AsUTC -Format "yyyy-MM-ddTHH:mm:ssZ"
Write-Host "[$tsDone] [RUNALL] DONE" -ForegroundColor Green
