<#
  tools/activate_python.ps1
  - Erstellt/aktiviert .venv
  - Prüft Python-Version (min. 3.10)
  - Upgraded pip
  - Installiert optional requirements.txt
  - Mit -ReturnPath gibt das Script NUR den Pfad zur venv-Python auf STDOUT aus
#>

[CmdletBinding()]
param(
  [string]$RepoRoot = (Resolve-Path -Path "$PSScriptRoot\..\..").Path,
  [string]$VenvDir  = ".venv",
  [string]$MinPython = "3.10",
  [switch]$CreateIfMissing = $true,
  [switch]$InstallRequirements = $true,
  [switch]$ReturnPath = $false
)

$ErrorActionPreference = "Stop"

function Info([string]$msg){ Write-Host "[PYENV] $msg" -ForegroundColor Cyan }
function Ok([string]$msg){ Write-Host "[OK] $msg" -ForegroundColor Green }
function Fail([string]$msg){ Write-Error $msg; exit 1 }

# 1) Pfade
$repo = Resolve-Path -Path $RepoRoot
Set-Location $repo
$venvPath = Join-Path $repo $VenvDir
$pyWin   = Join-Path $venvPath "Scripts\python.exe"
$pyNix   = Join-Path $venvPath "bin/python"
$pythonInVenv = $null

Info "Repo : $repo"
Info "Venv : $venvPath"

# 2) .venv erstellen (falls fehlt)
if (-not (Test-Path -Path $venvPath -PathType Container)) {
  if ($CreateIfMissing) {
    Info ".venv nicht gefunden → erstelle…"
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) { Fail "Kein 'python' im PATH gefunden. Bitte Python $MinPython+ installieren." }

    & $pythonCmd.Source -m venv "$venvPath" | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "python -m venv fehlgeschlagen." }
  }
  else {
    Fail ".venv fehlt. Starte mit -CreateIfMissing oder lege das venv manuell an."
  }
}

# 3) Python im venv finden
if (Test-Path -Path $pyWin -PathType Leaf)      { $pythonInVenv = $pyWin }
elseif (Test-Path -Path $pyNix -PathType Leaf)   { $pythonInVenv = $pyNix }
else { Fail "Konnte python in .venv nicht finden (weder $pyWin noch $pyNix)." }

# 4) Version prüfen
$verOut = & $pythonInVenv --version
if ($LASTEXITCODE -ne 0) { Fail "Konnte Python-Version nicht abrufen." }
Info "Python: $verOut"

$pyVersionCode = @"
import sys
print(".".join(map(str, sys.version_info[:3])))
"@
$versionStr = (& $pythonInVenv -c $pyVersionCode)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($versionStr)) {
  Fail "Konnte Python-Versionsstring nicht ermitteln."
}
$versionStr = $versionStr.Trim()

if ([version]$versionStr -lt [version]$MinPython) {
  Fail "Python $MinPython+ benötigt (gefunden: $versionStr)."
}

# 5) pip upgraden (Ausgabe nicht in STDOUT zurückgeben)
& $pythonInVenv -m pip install --upgrade pip | Out-Host
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade fehlgeschlagen." }
Ok "pip upgrade ok"

# 6) requirements.txt installieren (wenn vorhanden)
$req = Join-Path $repo "requirements.txt"
if ($InstallRequirements -and (Test-Path -Path $req -PathType Leaf)) {
  Info "requirements.txt gefunden → installiere…"
  & $pythonInVenv -m pip install -r "$req" | Out-Host
  if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements.txt fehlgeschlagen." }
  Ok "requirements installiert"
}
else {
  Info "requirements.txt nicht vorhanden oder Installation übersprungen."
}

if ($ReturnPath) {
  # Nur den Pfad als "saubere" Ausgabe zurückgeben
  Write-Output $pythonInVenv
} else {
  Ok "Venv bereit: $pythonInVenv"
}
exit 0
