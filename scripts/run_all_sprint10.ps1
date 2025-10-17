<# 
  scripts/run_all_sprint10.ps1
  Orchestriert den kompletten Sprint-Flow (Seed → Rehydrate → Features → Cost → Backtest → CostGrid → Portfolio → Git Sync)
  - Nutzt tools/activate_python.ps1 -ReturnPath, um robust die .venv-Python zu bekommen (ohne pip-Output als "Kommando" zu interpretieren)
  - Optionales Discord-Notify, wenn DISCORD_WEBHOOK_URL (oder -DiscordWebhook) gesetzt ist
#>

[CmdletBinding()]
param(
  # Welche Schritte?
  [switch]$Seed,
  [switch]$Rehydrate,
  [switch]$Features,
  [switch]$Cost,
  [switch]$Backtest,
  [switch]$CostGrid,
  [switch]$Portfolio,
  [switch]$Sync,

  # Häufig genutzte Parameter
  [ValidateSet("1min","5min","15min","30min","60min","day")]
  [string]$Freq = "5min",
  [double]$StartCapital = 10000,
  [double]$Exposure = 1.0,
  [double]$MaxLeverage = 1.0,
  [double]$CommissionBps = 0.5,
  [double]$SpreadW = 1.0,
  [double]$ImpactW = 1.0,

  # Sprint8 Feature-Build quick-Modus
  [switch]$Quick = $true,
  [int]$QuickDays = 180,
  [string[]]$Symbols = @("AAPL","MSFT"),

  # Discord
  [string]$DiscordWebhook = $null
)

$ErrorActionPreference = "Stop"

function Info([string]$m){ Write-Host $m -ForegroundColor Cyan }
function Ok([string]$m){ Write-Host $m -ForegroundColor Green }
function Warn([string]$m){ Write-Warning $m }
function Fail([string]$m){ Write-Error $m; exit 1 }

# ------------------------------------------------------------
# 0) Projektpfade
# ------------------------------------------------------------
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = (Resolve-Path -Path (Join-Path $ScriptRoot "..")).Path

Set-Location $RepoRoot

# ------------------------------------------------------------
# 1) .venv aktivieren und Python-Pfad sauber beziehen
# ------------------------------------------------------------
$activate = Join-Path $ScriptRoot "tools\activate_python.ps1"
if (-not (Test-Path -Path $activate -PathType Leaf)) {
  Fail "activate_python.ps1 nicht gefunden: $activate"
}

# Nur der Python-Pfad kommt als Output zurück, alle anderen Ausgaben gehen direkt zur Konsole
$VenvPython = & $activate -RepoRoot $RepoRoot -VenvDir ".venv" -CreateIfMissing -InstallRequirements -ReturnPath
if (-not (Test-Path -Path $VenvPython -PathType Leaf)) {
  Fail "Venv-Python nicht gefunden: $VenvPython"
}
Info "[RUNALL] Using Python: $VenvPython"

# ------------------------------------------------------------
# 2) Discord Webhook ermitteln (ENV oder Param)
# ------------------------------------------------------------
if (-not $DiscordWebhook -and $env:DISCORD_WEBHOOK_URL) {
  $DiscordWebhook = $env:DISCORD_WEBHOOK_URL
}
$CanNotifyDiscord = -not [string]::IsNullOrWhiteSpace($DiscordWebhook)

$notifyScript = Join-Path $ScriptRoot "tools\notify_discord.ps1"
function NotifyDiscord([string]$title, [string]$content){
  if (-not $CanNotifyDiscord) { return }
  if (-not (Test-Path -Path $notifyScript -PathType Leaf)) { return }
  try {
    & $notifyScript -Title $title -Content $content -WebhookUrl $DiscordWebhook
  } catch {
    Warn "Discord-Notify fehlgeschlagen: $($_.Exception.Message)"
  }
}

# ------------------------------------------------------------
# 3) Hilfsfunktion zum Run eines .ps1 mit Fehlerbehandlung
# ------------------------------------------------------------
function Run-Step([string]$label, [scriptblock]$action) {
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"), $label) -ForegroundColor Yellow
  & $action
  if($LASTEXITCODE -ne 0){ throw "$label fehlgeschlagen" }
}

# ------------------------------------------------------------
# 4) Anzeige Start
# ------------------------------------------------------------
Write-Host ("[{0}] [RUNALL] Start | freq={1} cap={2} exp={3} lev={4} comm={5}bps spread={6} impact={7}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"), $Freq, $StartCapital, $Exposure, $MaxLeverage, $CommissionBps, $SpreadW, $ImpactW) -ForegroundColor Magenta
Write-Host "[RUNALL] START" -ForegroundColor Magenta
NotifyDiscord "RunAll gestartet" "freq=$Freq | cap=$StartCapital | exp=$Exposure | lev=$MaxLeverage | comm=${CommissionBps}bps | spread=$SpreadW | impact=$ImpactW"

# ------------------------------------------------------------
# 5) Seed (optional – wenn Script vorhanden)
# ------------------------------------------------------------
if ($Seed) {
  Run-Step "Seede Demo-Daten…" {
    $seedPy = Join-Path $RepoRoot "scripts\seed_demo_data.py"
    if (Test-Path -Path $seedPy -PathType Leaf) {
      & $VenvPython $seedPy --freq $Freq
    } else {
      Warn "seed_demo_data.py nicht gefunden – überspringe Seed."
    }
  }
}

# ------------------------------------------------------------
# 6) Rehydrate (Sprint 8)
# ------------------------------------------------------------
if ($Rehydrate) {
  Run-Step "run_sprint8_rehydrate.ps1" {
    $rehydrate = Join-Path $RepoRoot "scripts\run_sprint8_rehydrate.ps1"
    if (-not (Test-Path -Path $rehydrate -PathType Leaf)) { throw "Script fehlt: $rehydrate" }

    # Primär mit Quick + QuickDays
    try {
      & pwsh -File $rehydrate -Freq $Freq -Symbols ($Symbols -join ",") -Quick:$Quick -QuickDays $QuickDays
    } catch {
      Warn "Rehydrate mit -Quick/-QuickDays fehlgeschlagen → Fallback ohne Quick-Flags."
      & pwsh -File $rehydrate -Freq $Freq -Symbols ($Symbols -join ",")
    }
  }
}

# ------------------------------------------------------------
# 7) Features explizit (optional – meist redundant zu Rehydrate)
# ------------------------------------------------------------
if ($Features) {
  Run-Step "SPRINT8 Features bauen… ($Freq)" {
    $featPy = Join-Path $RepoRoot "scripts\sprint8_feature_engineering.py"
    if (-not (Test-Path -Path $featPy -PathType Leaf)) { throw "Script fehlt: $featPy" }

    $args = @("--freq", $Freq)
    if ($Quick)    { $args += @("--quick", "--qdays", "$QuickDays") }
    if ($Symbols -and $Symbols.Count -gt 0) { $args += @("--symbols", ($Symbols -join ",")) }

    & $VenvPython $featPy @args
  }
}

# ------------------------------------------------------------
# 8) Execution & Costs
# ------------------------------------------------------------
if ($Cost) {
  Run-Step "sprint8_cost_model.ps1" {
    $costPS1 = Join-Path $RepoRoot "scripts\sprint8_cost_model.ps1"
    if (-not (Test-Path -Path $costPS1 -PathType Leaf)) { throw "Script fehlt: $costPS1" }
    & pwsh -File $costPS1 -Freq $Freq -Notional $StartCapital -CommissionBps $CommissionBps
  }
}

# ------------------------------------------------------------
# 9) Backtest (Sprint 9)
# ------------------------------------------------------------
if ($Backtest) {
  Run-Step "sprint9_backtest.ps1" {
    $btPS1 = Join-Path $RepoRoot "scripts\sprint9_backtest.ps1"
    if (-not (Test-Path -Path $btPS1 -PathType Leaf)) { throw "Script fehlt: $btPS1" }
    & pwsh -File $btPS1 -Freq $Freq
  }
}

# ------------------------------------------------------------
# 10) Cost Grid (Sprint 9)
# ------------------------------------------------------------
$bestGridLine = $null
if ($CostGrid) {
  Run-Step "sprint9_cost_grid.ps1" {
    $gridPS1 = Join-Path $RepoRoot "scripts\sprint9_cost_grid.ps1"
    if (-not (Test-Path -Path $gridPS1 -PathType Leaf)) { throw "Script fehlt: $gridPS1" }
    & pwsh -File $gridPS1 -Freq $Freq
  }

  # Versuch, "Bestes Grid" aus Report zu lesen (optional)
  $gridReport = Join-Path $RepoRoot "output\cost_grid_report.md"
  if (Test-Path -Path $gridReport -PathType Leaf) {
    $bestGridLine = (Get-Content $gridReport | Select-String -Pattern "Bestes Grid").Line
    if ($bestGridLine) { Write-Host "[RUNALL] $bestGridLine" -ForegroundColor DarkCyan }
  }
}

# ------------------------------------------------------------
# 11) Portfolio (Sprint 10)
# ------------------------------------------------------------
if ($Portfolio) {
  Run-Step "sprint10_portfolio.ps1" {
    $pfPS1 = Join-Path $RepoRoot "scripts\sprint10_portfolio.ps1"
    if (-not (Test-Path -Path $pfPS1 -PathType Leaf)) { throw "Script fehlt: $pfPS1" }
    & pwsh -File $pfPS1 -Freq $Freq -StartCapital $StartCapital -Exposure $Exposure -MaxLeverage $MaxLeverage `
      -CommissionBps $CommissionBps -SpreadW $SpreadW -ImpactW $ImpactW
  }
}

# ------------------------------------------------------------
# 12) Git Sync
# ------------------------------------------------------------
if ($Sync) {
  Run-Step "git_sync.ps1" {
    $syncPS1 = Join-Path $RepoRoot "scripts\tools\git_sync.ps1"
    if (-not (Test-Path -Path $syncPS1 -PathType Leaf)) { throw "Script fehlt: $syncPS1" }
    $msg = "RunAll: $Freq, cap=$StartCapital, exp=$Exposure, comm=${CommissionBps}bps, spread=$SpreadW, impact=$ImpactW"
    & pwsh -File $syncPS1 -Message $msg
  }
}

# ------------------------------------------------------------
# 13) Fertig / Notify
# ------------------------------------------------------------
if ($bestGridLine) {
  NotifyDiscord "RunAll abgeschlossen" "$bestGridLine"
} else {
  NotifyDiscord "RunAll abgeschlossen" "freq=$Freq | cap=$StartCapital | exp=$Exposure | lev=$MaxLeverage | comm=${CommissionBps}bps | spread=$SpreadW | impact=$ImpactW"
}

Write-Host ("[{0}] [RUNALL] DONE" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")) -ForegroundColor Magenta
exit 0
