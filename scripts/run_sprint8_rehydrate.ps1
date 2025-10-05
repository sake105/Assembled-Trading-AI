param(
  [switch]$UseDemo = $false,           # Falls keine Daten vorhanden: Demo-Daten generieren
  [string]$Symbols = "AAPL,MSFT",     # Standard-Symbole
  [int]$QuickDays = 180                # Quick-Window
)

$ErrorActionPreference = 'Stop'

# --- Paths --------------------------------------------------------------
$ROOT   = (Get-Location).Path
$Config = Join-Path $ROOT 'config'
$Data   = Join-Path $ROOT 'data'
$Raw1m  = Join-Path $Data 'raw/1min'
$Out    = Join-Path $ROOT 'output'
$Agg    = Join-Path $Out 'aggregates'
$AsmOut = Join-Path $Out 'assembled_intraday'
$Feat   = Join-Path $Out 'features'
$Logs   = Join-Path $ROOT 'logs'
$Scripts= Join-Path $ROOT 'scripts'

$Py    = Join-Path $ROOT '.venv/Scripts/python.exe'
$FEpy  = Join-Path $Scripts 'sprint8_feature_engineering.py'
$AsmPS = Join-Path $Scripts '31_assemble_intraday.ps1'
$ResPS = Join-Path $Scripts '50_resample_intraday.ps1'

# --- Helpers ------------------------------------------------------------
function Write-Info($msg){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [REHYDRATE] $msg" }
function Ensure-Dir($p){ if(-not (Test-Path $p)){ New-Item -ItemType Directory -Force -Path $p | Out-Null } }
function Count-Files($path,$pattern){ if(-not (Test-Path $path)){ return 0 }; (Get-ChildItem -Path $path -Recurse -File -Include $pattern -ErrorAction SilentlyContinue).Count }

# --- Ensure structure ---------------------------------------------------
$null = ($Config,$Data,$Raw1m,$Out,$Agg,$AsmOut,$Feat,$Logs,$Scripts) | ForEach-Object{ Ensure-Dir $_ }

# --- Sanity: venv / python --------------------------------------------
if(-not (Test-Path $Py)){ throw "Python venv nicht gefunden: $Py. Bitte .\\.venv\\Scripts\\Activate.ps1 ausführen und Dependencies prüfen." }

Write-Info "Python: $Py"
Write-Info "Checking core packages..."
& $Py -c @"
import importlib, sys
for pkg in ("pandas","numpy","pyarrow"):
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except Exception as e:
        print(f"[FAIL] {pkg}: {e}"); sys.exit(1)
print("[OK] Packages ready")
"@
if($LASTEXITCODE -ne 0){
    throw "Python-Pakete fehlen. Bitte 'pip install pandas numpy pyarrow' in der venv ausführen."
}

# --- Step 1: (Re)Assemble / Resample ----------------------------------
$needAssemble = ($null -eq (Get-ChildItem $AsmOut -Recurse -File -ErrorAction SilentlyContinue))
$needResample = ($null -eq (Get-ChildItem $Agg -Recurse -File -Include '*5min*.parquet','*5min*.csv' -ErrorAction SilentlyContinue))

if($needAssemble){ Write-Info "Assemble-Outputs fehlen → starte 31_assemble_intraday.ps1"; & pwsh -File $AsmPS }
else { Write-Info "Assemble-Outputs gefunden (überspringe Assemble)" }

if($needResample){ Write-Info "5min-Resample fehlt → starte 50_resample_intraday.ps1"; & pwsh -File $ResPS }
else { Write-Info "5min-Resample vorhanden (überspringe Resample)" }

# --- Step 2: Datenlage prüfen -----------------------------------------
$raw1mCnt = Count-Files $Raw1m @('*.parquet','*.csv')
$asm1mCnt = Count-Files (Join-Path $AsmOut '1min') @('*.parquet','*.csv')
$agg5mCnt = Count-Files $Agg @('*5min*.parquet','*5min*.csv')

Write-Info "Files: raw/1min=$raw1mCnt, assembled/1min=$asm1mCnt, aggregates(*5min*)=$agg5mCnt"

# --- Step 3: Feature Build Strategy -----------------------------------
$freq = $null
if($agg5mCnt -gt 0){ $freq = '5min' }
elseif($raw1mCnt -gt 0 -or $asm1mCnt -gt 0){ $freq = '1min' }
elseif($UseDemo){ $freq = '1min' }
else {
  Write-Info "Keine Daten gefunden. Starte mit -UseDemo oder fülle data/raw/1min bzw. output/aggregates mit Dateien."
  exit 2
}

# --- Step 4: Features bauen -------------------------------------------
Write-Info "Baue Features | freq=$freq | symbols=$Symbols | quickDays=$QuickDays | demo=$UseDemo"
$args = @('scripts/sprint8_feature_engineering.py','--freq', $freq,'--symbols', $Symbols,'--quick','--quick-days', "$QuickDays")
if($UseDemo){ $args += @('--demo','--demo-days','30') }

& $Py @args
if($LASTEXITCODE -ne 0){ throw "Feature-Build gescheitert (siehe Log)." }

# --- Step 5: Ergebnisübersicht ----------------------------------------
$base = Join-Path $Feat "base_${freq}.parquet"
$micro = Join-Path $Feat "micro_${freq}.parquet"
$reg = Join-Path $Feat "regime_${freq}.parquet"

$written = @()
if (Test-Path $base)  { $written += $base }
if (Test-Path $micro) { $written += $micro }
if (Test-Path $reg)   { $written += $reg }

if ($written.Count -gt 0) {
  Write-Info "[OK] Features geschrieben:"
  $written | ForEach-Object { Write-Host " - $_" }
} else {
  Write-Info "[WARN] Keine Feature-Dateien gefunden – bitte Log prüfen: $($Logs)\sprint8_feature_engineering.log"
}

Write-Info "Rehydrate + Orchestrator DONE"


