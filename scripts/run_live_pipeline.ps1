<# 
  Startet eine einfache Live-Pipeline (Download → Assemble → Features → Execution).

  Beispiel:
    pwsh -File .\scripts\run_live_pipeline.ps1 `
      -LiveSymbols "AAPL,MSFT" -LiveDays 3 -Freq 5min
#>

param(
  [string]$LiveSymbols = 'AAPL,MSFT',
  [int]$LiveDays = 3,
  [ValidateSet('1min','5min','15min','60min')][string]$Freq = '5min'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptRoot = Split-Path -Parent $PSCommandPath
$RepoRoot   = Split-Path -Parent $ScriptRoot
$ToolsDir   = Join-Path $ScriptRoot 'tools'
$ActivatePS = Join-Path $ToolsDir   'activate_python.ps1'

function Stamp([string]$t){ $ts=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); Write-Host "[$ts] $t" -ForegroundColor Cyan }

# --- Venv sicherstellen ------------------------------------------------------
if (-not (Test-Path $ActivatePS -PathType Leaf)) {
  throw "tools\activate_python.ps1 nicht gefunden: $ActivatePS"
}

. $ActivatePS
$venvPath = Join-Path $RepoRoot '.venv'
$reqTxt   = Join-Path $RepoRoot 'requirements.txt'
$Global:VenvPython = Ensure-Venv -RepoRoot $RepoRoot -VenvPath $venvPath -Requirements $reqTxt -Quiet:$false

if (-not $Global:VenvPython -or -not (Test-Path $Global:VenvPython -PathType Leaf)) {
  throw "Konnte Venv-Python nicht bestimmen. Erhalten: '$Global:VenvPython'"
}

Stamp "LIVE Using Python: $Global:VenvPython"

function PyRun([string]$script,[string[]]$args=@()){
  & $Global:VenvPython $script @args
  if ($LASTEXITCODE -ne 0) { throw "Python fehlgeschlagen: $script" }
}

# --- (1) Live-Download (falls vorhanden) ------------------------------------
$dlPy = Join-Path $ScriptRoot 'live_download.py'
if (Test-Path $dlPy -PathType Leaf) {
  Stamp "LIVE Download…"
  PyRun $dlPy @('--symbols',$LiveSymbols,'--days',$LiveDays.ToString())
} else {
  Write-Warning "live_download.py nicht gefunden – überspringe Download."
}

# --- (2) Assemble / Rehydrate (falls vorhanden) ------------------------------
$rehPS = Join-Path $ScriptRoot 'run_sprint8_rehydrate.ps1'
if (Test-Path $rehPS -PathType Leaf) {
  Stamp "LIVE Rehydrate…"
  & $rehPS -Freq $Freq -Symbols $LiveSymbols -Quick -QuickDays 180
  if ($LASTEXITCODE -ne 0) { throw "Rehydrate fehlgeschlagen" }
} else {
  Write-Warning "run_sprint8_rehydrate.ps1 nicht gefunden – überspringe Rehydrate."
}

# --- (3) Execution v1 (optional) --------------------------------------------
$execPy = Join-Path $ScriptRoot 'sprint8_execution.py'
if (Test-Path $execPy -PathType Leaf) {
  Stamp "LIVE Execution…"
  PyRun $execPy @('--freq',$Freq,'--commission-bps','0.5')
} else {
  Write-Warning "sprint8_execution.py nicht gefunden – überspringe Execution."
}

Stamp "LIVE DONE"

