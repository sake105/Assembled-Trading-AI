# scripts/sprint10_portfolio_simple.ps1  (robuste Version mit RepoRoot-Override)
param(
  [string]$Freq,
  [double]$StartCapital = 10000,
  [double]$CommissionBps = 0.0,
  [double]$SpreadW = 0.25,
  [double]$ImpactW = 0.5,
  # Optional: gib dein Repo explizit an, um alle Path-Detections zu umgehen
  [string]$RepoRoot
)

$ErrorActionPreference = 'Stop'

# --- RepoRoot ermitteln (mit sicherem Fallback) ---
function Get-SafeRepoRoot {
  param([string]$RepoRootIn)

  if ($RepoRootIn -and (Test-Path -LiteralPath $RepoRootIn)) {
    return (Resolve-Path -LiteralPath $RepoRootIn).Path
  }

  $sr = $null
  try {
    if ($PSScriptRoot) {
      $sr = $PSScriptRoot
    } elseif ($PSCommandPath) {
      $sr = Split-Path -LiteralPath $PSCommandPath -Parent
    } elseif ($MyInvocation -and $MyInvocation.MyCommand -and $MyInvocation.MyCommand.Path) {
      $sr = Split-Path -LiteralPath $MyInvocation.MyCommand.Path -Parent
    }
  } catch {}

  if (-not $sr) {
    # Fallback: versuche ./scripts als Skript-Ordner relativ zum CWD
    $cwd = (Get-Location).Path
    $tryScripts = Join-Path $cwd 'scripts'
    if (Test-Path -LiteralPath $tryScripts) {
      $sr = $tryScripts
    } else {
      # Letzter Fallback: nimm CWD selbst als Repo
      return $cwd
    }
  }

  # Repo ist der Parent von scripts
  $rr = $sr
  try {
    # Wenn $sr ...\scripts ist, gehe eins hoch
    if (Split-Path -Leaf $sr -ErrorAction SilentlyContinue -ne '') {
      $leaf = Split-Path -Leaf $sr
      if ($leaf -ieq 'scripts') {
        $rr = Split-Path -Path $sr -Parent
      }
    }
  } catch {
    $rr = (Get-Location).Path
  }

  return $rr
}

$repo = Get-SafeRepoRoot -RepoRootIn $RepoRoot

# --- Python & Skriptpfade ---
$pyVenv   = Join-Path $repo '.venv\Scripts\python.exe'
$py       = if (Test-Path -LiteralPath $pyVenv) { $pyVenv } else { 'python' }
$pyScript = Join-Path $repo 'scripts\sprint10_portfolio.py'
if (-not (Test-Path -LiteralPath $pyScript)) {
  throw "Portfolio-Skript nicht gefunden: $pyScript"
}

# --- Zahlen invariant formatieren (Punkt statt Komma) ---
$nf = [System.Globalization.CultureInfo]::InvariantCulture.NumberFormat
$cap = [string]([double]::Parse(($StartCapital.ToString($nf)),$nf))
$cbp = [string]([double]::Parse(($CommissionBps.ToString($nf)),$nf))
$spw = [string]([double]::Parse(($SpreadW.ToString($nf)),$nf))
$ipw = [string]([double]::Parse(($ImpactW.ToString($nf)),$nf))

# --- Argumente ---
$argList = @(
  $pyScript,
  '--freq',          $Freq,
  '--start-capital', $cap,
  '--commission-bps',$cbp,
  '--spread-w',      $spw,
  '--impact-w',      $ipw
)

Write-Host "[PF10] Repo:   $repo"
Write-Host "[PF10] Python: $py"
Write-Host "[PF10] Script: $pyScript"
Write-Host "[PF10] Args:   freq=$Freq cap=$cap comm=$cbp spread=$spw impact=$ipw"

# --- Ausführen ---
$proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru -Wait
if ($proc.ExitCode -ne 0) {
  throw "sprint10_portfolio.py fehlgeschlagen (ExitCode $($proc.ExitCode))."
}

# --- Outputs (falls vorhanden) ---
$outs = @(
  'output\portfolio_equity_1d.csv',
  'output\portfolio_equity_5min.csv',
  'output\portfolio_report.md',
  'output\portfolio_trades.csv'
) | ForEach-Object { Join-Path $repo $_ }

foreach ($p in $outs) {
  if (Test-Path -LiteralPath $p) { Write-Host "[PF10] [OK] $p" }
}
Write-Host "[PF10] DONE"

