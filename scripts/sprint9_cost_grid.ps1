<#  sprint9_cost_grid.ps1
    Robust wrapper für sprint9_cost_grid.py
    - akzeptiert Werte als String mit Leerzeichen/Komma/Semikolon
    - Dezimaltrennzeichen , oder .
    - Aliase für alte Parameternamen (CommissionBps / SpreadW / ImpactW)
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [string]$Freq = "5min",

  # Beispiele: "0 0.5 1", "0,0.5,1", "0;0,5;1"
  [Parameter(Mandatory = $false)]
  [Alias('Commission','CommissionBps')]
  [string]$CommissionValues = "0 0.5 1",

  [Parameter(Mandatory = $false)]
  [Alias('Spread','SpreadW')]
  [string]$SpreadValues = "0.5 1 2",

  [Parameter(Mandatory = $false)]
  [Alias('Impact','ImpactW')]
  [string]$ImpactValues = "0.5 1 2"
)

# --- Utils -------------------------------------------------------------

function Parse-Floats {
  param([string]$s)
  # Trenner normalisieren (Leerzeichen, Komma, Semikolon -> Leerzeichen)
  $norm = ($s -replace '[;,\s]+',' ').Trim()
  if ([string]::IsNullOrWhiteSpace($norm)) { return @() }

  $vals = @()
  foreach ($tok in ($norm -split ' +')) {
    if ($tok -eq '') { continue }
    # Dezimal-Komma -> Punkt
    $tok2 = $tok -replace ',', '.'
    try {
      $vals += [single]$tok2
    } catch {
      throw "Ungültiger Float-Wert: '$tok' (aus '$s')"
    }
  }
  return ,$vals
}

function Get-Python {
  # 1) bevorzugt global gesetztes Venv-Python
  if ($Global:VenvPython -and (Test-Path $Global:VenvPython -PathType Leaf)) {
    return $Global:VenvPython
  }
  # 2) versuche .venv relativ zum Repo (zwei Ebenen über diesem Skript)
  try {
    $venvCandidate = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path '.venv\Scripts\python.exe'
    if (Test-Path $venvCandidate -PathType Leaf) { return $venvCandidate }
  } catch { }
  # 3) Fallback: python im PATH
  return 'python'
}

# --- Parse Inputs ------------------------------------------------------

$commissions = Parse-Floats -s $CommissionValues
$spreadWs    = Parse-Floats -s $SpreadValues
$impactWs    = Parse-Floats -s $ImpactValues

Write-Host ("[GRID] START Grid | freq={0} | commission=[{1}] | spread_w=[{2}] | impact_w=[{3}]" -f `
  $Freq, ($commissions -join ', '), ($spreadWs -join ', '), ($impactWs -join ', '))

# --- Call Python -------------------------------------------------------

$py = Get-Python
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
  throw "Python nicht gefunden: $py"
}

# Pfad zum Python-Skript relativ zu diesem Skript
$pyScript = Join-Path $PSScriptRoot 'sprint9_cost_grid.py'
if (-not (Test-Path $pyScript -PathType Leaf)) {
  throw "Python-Skript nicht gefunden: $pyScript"
}

# Arg-Liste für argparse (nargs='+')
$pyArgs = @(
  $pyScript,
  '--freq', $Freq,
  '--commission-bps'
) + $commissions + @(
  '--spread-w'
) + $spreadWs + @(
  '--impact-w'
) + $impactWs

# Ausführen
& $py @pyArgs
$exit = $LASTEXITCODE
if ($exit -ne 0) {
  throw "Grid fehlgeschlagen (ExitCode $exit)."
}

Write-Host "[GRID] DONE"

