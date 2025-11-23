# scripts/live/pull_intraday.ps1
[CmdletBinding()]
param(
  [string] $Symbols = "AAPL,MSFT",
  [int]    $Days    = 3,
  [string] $RepoRoot,
  [switch] $NoThrow
)

$ErrorActionPreference = "Stop"

# --- robuste Pfaderkennung ---
$scriptPath = $PSCommandPath
if (-not $scriptPath) { $scriptPath = $MyInvocation.MyCommand.Definition }
if (-not $scriptPath) { $scriptPath = [System.IO.Path]::Combine((Get-Location).Path, "scripts","live","pull_intraday.ps1") }

$scriptDir = [System.IO.Path]::GetDirectoryName($scriptPath)
if ([string]::IsNullOrWhiteSpace($scriptDir)) { $scriptDir = (Get-Location).Path }

# RepoRoot korrekt: zwei Ebenen hoch (…\scripts\live -> …\scripts -> …\<RepoRoot>)
if (-not $RepoRoot -or $RepoRoot -eq "") {
  $p1 = [System.IO.Directory]::GetParent($scriptDir)       # ...\scripts
  if ($null -eq $p1) { throw "Konnte scripts-Ordner nicht bestimmen (scriptDir='$scriptDir')." }
  $p2 = [System.IO.Directory]::GetParent($p1.FullName)     # ...\<RepoRoot>
  if ($null -eq $p2) { throw "Konnte RepoRoot nicht bestimmen (scripts='$($p1.FullName)')." }
  $RepoRoot = $p2.FullName
}

# --- Python aus .venv, sonst global ---
$py = [System.IO.Path]::Combine($RepoRoot, ".venv", "Scripts", "python.exe")
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }

# --- Python-Skript ---
$pyScript = [System.IO.Path]::Combine($RepoRoot, "scripts", "live", "pull_intraday.py")
if (-not (Test-Path -LiteralPath $pyScript)) {
  throw "pull_intraday.py nicht gefunden: $pyScript"
}

Write-Host "[PYENV] Python:" $py
Write-Host "[LIVE]  Repo:  " $RepoRoot
Write-Host "[LIVE]  Symbols:" $Symbols
Write-Host "[LIVE]  Days:   " $Days

# --- Call ---
& $py $pyScript `
  --symbols $Symbols `
  --days $Days `
  --repo-root $RepoRoot
$exit = $LASTEXITCODE

if ($exit -ne 0) {
  $msg = "pull_intraday.py fehlgeschlagen (ExitCode $exit)."
  if ($NoThrow) { Write-Warning $msg } else { throw $msg }
} else {
  Write-Host "[LIVE] DONE"
}

