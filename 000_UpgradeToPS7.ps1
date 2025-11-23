#Requires -Version 7.0
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$OldRoot,
  [Parameter(Mandatory=$true)][string]$NewRoot,
  [switch]$RunSprint7,
  [string]$ZipOut
)

function Write-Info([string]$m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Ok  ([string]$m){ Write-Host "[OK]   $m" -ForegroundColor Green }
function Write-Err ([string]$m){ Write-Host "[ERR]  $m" -ForegroundColor Red }

function Ensure-Dir([string]$p){
  if(-not (Test-Path -LiteralPath $p)){ New-Item -ItemType Directory -Path $p | Out-Null }
  return (Resolve-Path -LiteralPath $p).Path
}

# --- 1) PS-Version absichern ---------------------------------------------------
if($PSVersionTable.PSEdition -ne 'Core' -or $PSVersionTable.PSVersion.Major -lt 7){
  throw "Dieses Skript muss in PowerShell 7+ (pwsh) laufen. Aktuell: $($PSVersionTable.PSVersion)"
}

# --- 2) Pfade auflösen ---------------------------------------------------------
$OldRoot = (Resolve-Path -LiteralPath $OldRoot).Path
$NewRoot = $NewRoot.TrimEnd('\')
Ensure-Dir $NewRoot | Out-Null

Write-Info "Quelle: $OldRoot"
Write-Info "Ziel:   $NewRoot"

# --- 3) Projekt kopieren (ohne .venv, Caches, alte QC/Aggregate) ---------------
$excludeDirs = @(
  '.venv','__pycache__','.git','.idea','.vscode','tmp','.mypy_cache',
  'output\qc','output\aggregates'
)

Get-ChildItem -LiteralPath $OldRoot -Force | ForEach-Object {
  $src = $_.FullName
  $rel = $src.Substring($OldRoot.Length).TrimStart('\')

  $skip = $false
  foreach($ex in $excludeDirs){
    if($rel -like "$ex*"){ $skip = $true; break }
  }
  if($skip){ return }

  $dst = Join-Path $NewRoot $rel
  if($_.PSIsContainer){
    Ensure-Dir $dst | Out-Null
  } else {
    Ensure-Dir (Split-Path $dst -Parent) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
  }
}

# --- 4) Skripte für PS7 "säubern" ---------------------------------------------
function Clean-For-PS7([string]$text){
  if($text -notmatch '#Requires\s+-Version\s+7'){
    $text = "#Requires -Version 7.0`r`n$text"
  }

  # Sonderzeichen über Unicode-Codes ersetzen (kein Parser-Ärger)
  $map = [ordered]@{}
  $map[[string][char]0x2013] = '-'   # -  en dash
  $map[[string][char]0x2014] = '-'   # -  em dash
  $map[[string][char]0x2192] = '->'  # ->  right arrow
  $map[[string][char]0x201C] = '"'   # "
  $map[[string][char]0x201D] = '"'   # "
  $map[[string][char]0x201E] = '"'   # "
  $map[[string][char]0x2019] = "'"   # '
  $map[[string][char]0x00B4] = "'"   # '

  foreach($k in $map.Keys){
    $text = $text -replace [regex]::Escape($k), $map[$k]
  }

  # Encoding-Parameter vereinheitlichen (PS7 versteht utf8NoBOM)
  $text = $text -replace '-Encoding\s+utf8(NoBOM)?','-Encoding utf8NoBOM'
  return $text
}

$psScripts = Get-ChildItem -LiteralPath $NewRoot -Recurse -Include *.ps1,*.psm1 -File -ErrorAction SilentlyContinue
foreach($f in $psScripts){
  try{
    $raw = Get-Content -LiteralPath $f.FullName -Raw -Encoding utf8NoBOM
  } catch {
    $raw = Get-Content -LiteralPath $f.FullName -Raw
  }
  $new = Clean-For-PS7 $raw
  if($new -ne $raw){
    $new | Set-Content -LiteralPath $f.FullName -Encoding utf8NoBOM
    Write-Ok "bereinigt: $($f.FullName)"
  }
}

# --- 5) (Optional) Python-Venv anlegen ----------------------------------------
$py = Get-Command python -ErrorAction SilentlyContinue
if($py){
  $venv = Join-Path $NewRoot '.venv'
  if(-not (Test-Path -LiteralPath $venv)){
    Write-Info "Erzeuge Python venv …"
    & python -m venv $venv
  }
} else {
  Write-Err "Python nicht gefunden (python.exe). Überspringe venv-Erstellung."
}

# --- 6) (Optional) Sprint 7 laufen lassen -------------------------------------
if($RunSprint7){
  $env:Path = (Join-Path $NewRoot '.venv\Scripts') + ';' + $env:Path
  Push-Location (Join-Path $NewRoot 'scripts')
  try{
    if(Test-Path .\50_resample_intraday.ps1){
      Write-Info "Sprint 7: Resample"
      .\50_resample_intraday.ps1
    }
    if(Test-Path .\51_qc_intraday_gaps.ps1){
      Write-Info "Sprint 7: QC Gaps"
      .\51_qc_intraday_gaps.ps1
    }
    if(Test-Path .\52_make_acceptance_intraday_sprint7.ps1){
      Write-Info "Sprint 7: Acceptance"
      .\52_make_acceptance_intraday_sprint7.ps1
    }
  } finally {
    Pop-Location
  }
}

# --- 7) ZIP erzeugen ----------------------------------------------------------
if($ZipOut){
  $zip = $ZipOut
  if(-not $zip.ToLower().EndsWith('.zip')){ $zip = "$zip.zip" }
  if(Test-Path -LiteralPath $zip){ Remove-Item -LiteralPath $zip -Force }
  Compress-Archive -Path (Join-Path $NewRoot '*') -DestinationPath $zip
  Write-Ok "ZIP geschrieben: $zip"
}

Write-Ok "Upgrade fertig."


