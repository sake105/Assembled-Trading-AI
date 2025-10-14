param(
  [string]$Freq = '5min',
  [switch]$Seed,           # optional: Demo-Daten seed
  [switch]$Rehydrate,      # optional: Rehydrate-Flow
  [switch]$Features,       # Sprint 8 Features
  [switch]$Exec,           # Sprint 8 Execution & Costs
  [switch]$Backtest,       # Sprint 9 Backtest
  [switch]$Grid,           # Sprint 9 Cost Grid
  [switch]$Portfolio,      # Sprint 10 Portfolio
  [switch]$Sync,           # Git Sync am Ende
  [double]$StartCapital = 10000,
  [double]$Exposure = 1.0,
  [double]$MaxLeverage = 1.0,
  [double]$CommissionBps = 0.5,
  [double]$SpreadW = 1.0,
  [double]$ImpactW = 1.0
)

$ErrorActionPreference = 'Stop'

function TS(){ (Get-Date).ToUniversalTime().ToString('s') + 'Z' }
function Info($tag,$m){ Write-Host "[$(TS)] [$tag] $m" }

# --- Projektpfade & Python venv ------------------------------------------------
$ROOT = (Get-Location).Path
$Py = Join-Path $ROOT '.venv/Scripts/python.exe'
if(-not (Test-Path $Py)){ throw "Python venv nicht gefunden: $Py. Bitte .\.venv\Scripts\Activate.ps1 ausführen oder venv erstellen." }

# --- Hilfsaufrufer -------------------------------------------------------------
function RunPS([string]$scriptPath, [string[]]$args){
  if(-not (Test-Path $scriptPath)){ throw "Script nicht gefunden: $scriptPath" }
  Info 'RUN' "$([System.IO.Path]::GetFileName($scriptPath)) $($args -join ' ')"
  & pwsh -File $scriptPath @args
  if($LASTEXITCODE -ne 0){ throw "Subprozess fehlgeschlagen: $scriptPath" }
}

# --- Defaults: wenn keine Flags gesetzt -> alles laufen lassen -----------------
$anySwitch = $PSBoundParameters.Keys | Where-Object { $_ -in @('Seed','Rehydrate','Features','Exec','Backtest','Grid','Portfolio','Sync') }
if(-not $anySwitch){ $Seed=$true; $Rehydrate=$true; $Features=$true; $Exec=$true; $Backtest=$true; $Grid=$true; $Portfolio=$true; $Sync=$true }

Info 'RUNALL' "Start | freq=$Freq cap=$StartCapital exp=$Exposure lev=$MaxLeverage comm=${CommissionBps}bps spread=$SpreadW impact=$ImpactW"

# --- 0) Seed Demo-Daten (optional) --------------------------------------------
if($Seed){
  $SeedPy = Join-Path $ROOT 'scripts/00_seed_demo_data.py'
  if(Test-Path $SeedPy){
    Info 'SEED' 'Seede Demo-Daten…'
    & $Py $SeedPy
    if($LASTEXITCODE -ne 0){ throw "Seed fehlgeschlagen" }
  } else {
    Info 'SEED' 'Überspringe Seed (Datei fehlt).'
  }
}

# --- 1) Rehydrate / Orchestrator (optional) -----------------------------------
if($Rehydrate){
  $Rehy = Join-Path $ROOT 'scripts/run_sprint8_rehydrate.ps1'
  if(Test-Path $Rehy){
    RunPS $Rehy @()
  } else {
    Info 'REHYDRATE' 'Überspringe Rehydrate (Datei fehlt).'
  }
}

# --- 2) Sprint 8: Feature Engineering -----------------------------------------
if($Features){
  $FeatPy = Join-Path $ROOT 'scripts/sprint8_feature_engineering.py'
  if(Test-Path $FeatPy){
    Info 'SPRINT8' "Features bauen… ($Freq)"
    & $Py $FeatPy --freq $Freq --quick
    if($LASTEXITCODE -ne 0){ throw "Sprint8 Features fehlgeschlagen" }
  } else {
    Info 'SPRINT8' 'Überspringe Features (Datei fehlt).'
  }
}

# --- 3) Sprint 8: Execution & Kosten ------------------------------------------
if($Exec){
  $CostPS = Join-Path $ROOT 'scripts/sprint8_cost_model.ps1'
  if(Test-Path $CostPS){
    RunPS $CostPS @()
  } else {
    # Fallback: direkter Python-Call, falls PS-Orchestrator fehlt
    $ExecPy = Join-Path $ROOT 'scripts/sprint8_execution.py'
    if(Test-Path $ExecPy){
      Info 'EXEC' "Direkt: execution.py --freq $Freq --commission-bps $CommissionBps"
      & $Py $ExecPy --freq $Freq --commission-bps $CommissionBps
      if($LASTEXITCODE -ne 0){ throw "Execution fehlgeschlagen" }
    } else {
      Info 'EXEC' 'Überspringe Execution (Dateien fehlen).'
    }
  }
}

# --- 4) Sprint 9: Backtest ----------------------------------------------------
if($Backtest){
  $BT = Join-Path $ROOT 'scripts/sprint9_backtest.ps1'
  if(Test-Path $BT){
    RunPS $BT @('-Freq', $Freq)
  } else {
    Info 'BT9' 'Überspringe Backtest (Datei fehlt).'
  }
}

# --- 5) Sprint 9: Cost Grid ---------------------------------------------------
if($Grid){
  $GridPS = Join-Path $ROOT 'scripts/sprint9_cost_grid.ps1'
  if(Test-Path $GridPS){
    RunPS $GridPS @('-Freq', $Freq)
  } else {
    Info 'GRID' 'Überspringe Cost Grid (Datei fehlt).'
  }
}

# --- 6) Sprint 10: Portfolio --------------------------------------------------
if($Portfolio){
  $PF = Join-Path $ROOT 'scripts/sprint10_portfolio.ps1'
  if(Test-Path $PF){
    RunPS $PF @('-Freq',$Freq,
                '-StartCapital',"$StartCapital",
                '-Exposure',"$Exposure",
                '-MaxLeverage',"$MaxLeverage",
                '-CommissionBps',"$CommissionBps",
                '-SpreadW',"$SpreadW",
                '-ImpactW',"$ImpactW")
  } else {
    Info 'PF10' 'Überspringe Portfolio (Datei fehlt).'
  }
}

# --- 7) Git Sync (optional) ---------------------------------------------------
if($Sync){
  $SyncPS = Join-Path $ROOT 'scripts/tools/git_sync.ps1'
  if(Test-Path $SyncPS){
    $msg = "RunAll: freq=$Freq cap=$StartCapital exp=$Exposure comm=${CommissionBps}bps spread=$SpreadW impact=$ImpactW ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
    RunPS $SyncPS @('-Message', $msg)
  } else {
    Info 'GIT' 'Überspringe Git Sync (Datei fehlt).'
  }
}

Info 'RUNALL' 'DONE'
