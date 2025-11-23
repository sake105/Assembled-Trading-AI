<#
  Stellt sicher, dass eine funktionsfähige venv existiert.
  Definiert die Funktion Ensure-Venv (liefert Pfad zur python.exe).

  Nutzung in Skripten:
    . $PSScriptRoot\activate_python.ps1          # dot-sourcen
    $py = Ensure-Venv -RepoRoot $repo -VenvPath "$repo\.venv" -Requirements "$repo\requirements.txt"

  Nutzung als Modul (PSM1):
    Import-Module $PSScriptRoot\activate_python.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info([string]$msg){ Write-Host $msg -ForegroundColor DarkCyan }
function Write-Ok  ([string]$msg){ Write-Host $msg -ForegroundColor Green }
function Write-Warn([string]$msg){ Write-Warning $msg }

function Ensure-Venv {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$VenvPath,
    [Parameter()][string]$Requirements,
    [switch]$Quiet
  )

  $pyExe = Join-Path $VenvPath 'Scripts\python.exe'

  if (-not $Quiet) {
    Write-Info "[PYENV] Repo : $RepoRoot"
    Write-Info "[PYENV] Venv : $VenvPath"
  }

  # 1) System-Python finden
  $sysPython = $null
  foreach ($cand in @('python','py')) {
    try {
      $c = Get-Command $cand -ErrorAction Stop
      if ($c -and $c.CommandType -in 'Application','ExternalScript') {
        $sysPython = $c.Source
        break
      }
    } catch {}
  }
  if (-not $sysPython) { throw "Kein System-Python gefunden (python/py)." }

  if (-not $Quiet) {
    try {
      $ver = & $sysPython -c "import sys;print('.'.join(map(str,sys.version_info[:3])))"
      Write-Info "[PYENV] Python: Python $ver"
    } catch {
      Write-Info "[PYENV] Python: $sysPython"
    }
  }

  # 2) Venv erstellen falls nötig
  if (-not (Test-Path $pyExe -PathType Leaf)) {
    if (-not (Test-Path $VenvPath)) { New-Item -ItemType Directory -Path $VenvPath | Out-Null }
    & $sysPython -m venv "$VenvPath"
    if ($LASTEXITCODE -ne 0) { throw "Venv-Erstellung fehlgeschlagen: $VenvPath" }
  }

  # 3) pip upgraden
  & $pyExe -m pip install --upgrade pip
  if ($LASTEXITCODE -ne 0) { throw "pip upgrade fehlgeschlagen" }
  if (-not $Quiet) { Write-Ok "[OK] pip upgrade ok" }

  # 4) requirements installieren (wenn Datei existiert)
  if ($Requirements -and (Test-Path $Requirements -PathType Leaf)) {
    if (-not $Quiet) { Write-Info "[PYENV] requirements.txt gefunden → installiere…" }
    & $pyExe -m pip install -r "$Requirements"
    if ($LASTEXITCODE -ne 0) { throw "requirements install fehlgeschlagen" }
    if (-not $Quiet) { Write-Ok "[OK] requirements installiert" }
  } else {
    if (-not $Quiet) { Write-Warn "requirements.txt fehlt oder Pfad ungültig – überspringe Installation." }
  }

  if (-not $Quiet) { Write-Ok "[OK] Venv bereit: $pyExe" }
  return $pyExe
}

# Nur exportieren, wenn das Skript als Modul geladen wurde
if ($ExecutionContext.SessionState.Module) {
  Export-ModuleMember -Function Ensure-Venv
}

