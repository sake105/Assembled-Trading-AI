param(
  # Core
  [string]$Freq = '5min',
  [double]$StartCapital = 10000,
  [double]$Exposure = 1,
  [double]$MaxLeverage = 1,
  [double]$CommissionBps = 0.5,
  [double]$SpreadW = 1,
  [double]$ImpactW = 1,

  # Sprint 8 Feature Build
  [string]$Symbols = 'AAPL,MSFT',   # 'ALL' = alle
  [int]$QuickDays = 180,
  [switch]$Demo,

  # Stages
  [switch]$Seed,
  [switch]$Rehydrate,
  [switch]$Features,
  [switch]$Costs,
  [switch]$Backtest,
  [switch]$Grid,
  [switch]$Portfolio,
  [switch]$Sync,

  # Discord
  [switch]$NotifyDiscord,
  [string]$DiscordWebhook = $null,   # optional; wenn leer → $env:DISCORD_WEBHOOK oder .env

  # Convenience: alles in einem Rutsch
  [switch]$All
)

$ErrorActionPreference = 'Stop'

function Info($m){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [RUNALL] $m" }

# --- Pfade/Umgebung ---
$ROOT = (Get-Location).Path
$Py = Join-Path $ROOT '.venv/Scripts/python.exe'
if(-not (Test-Path $Py)){ throw "Python venv nicht gefunden: $Py" }

$S8_Feature = Join-Path $ROOT 'scripts/sprint8_feature_engineering.py'
$S8_CostPS  = Join-Path $ROOT 'scripts/sprint8_cost_model.ps1'
$S9_BackPS  = Join-Path $ROOT 'scripts/sprint9_backtest.ps1'
$S9_GridPS  = Join-Path $ROOT 'scripts/sprint9_cost_grid.ps1'
$S10_PF_PS  = Join-Path $ROOT 'scripts/sprint10_portfolio.ps1'
$SeedPy     = Join-Path $ROOT 'scripts/00_seed_demo_data.py'
$RehydPS    = Join-Path $ROOT 'scripts/run_sprint8_rehydrate.ps1'
$SyncPS     = Join-Path $ROOT 'scripts/tools/git_sync.ps1'
$DiscordPS  = Join-Path $ROOT 'scripts/tools/notify_discord.ps1'

$OUT = Join-Path $ROOT 'output'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# --- Auswahl-Logik ---
if(-not ($Seed -or $Rehydrate -or $Features -or $Costs -or $Backtest -or $Grid -or $Portfolio -or $Sync -or $NotifyDiscord)){
  # Nichts explizit gewählt → All
  $All = $true
}
if($All){
  $Seed = $true
  $Rehydrate = $true
  $Features = $true
  $Costs = $true
  $Backtest = $true
  $Grid = $true
  $Portfolio = $true
  # $Sync lassen wir bewusst separat, weil manche erst prüfen wollen
}

Info "Start | freq=$Freq cap=$StartCapital exp=$Exposure lev=$MaxLeverage comm=${CommissionBps}bps spread=$SpreadW impact=$ImpactW"

# --- (1) Optional Seed ---
if($Seed -and (Test-Path $SeedPy)){
  Info "Seede Demo-Daten…"
  & $Py $SeedPy
  if($LASTEXITCODE -ne 0){ throw "Seeding fehlgeschlagen" }
}

# --- (2) Optional Rehydrate (orchestriert Assemble/Resample/Feature) ---
if($Rehydrate -and (Test-Path $RehydPS)){
  Info "run_sprint8_rehydrate.ps1"
  $rehydOk = $false

  try {
    # Versuch 1: neue Schnittstelle (mit -Quick / -QuickDays)
    pwsh -File $RehydPS -Freq $Freq -Symbols $Symbols -Quick -QuickDays $QuickDays -Demo:$Demo
    if($LASTEXITCODE -eq 0){ $rehydOk = $true }
  } catch { }

  if(-not $rehydOk){
    Info "Rehydrate (Fallback) ohne -Quick/-QuickDays…"
    try {
      # Versuch 2: ältere Schnittstelle
      pwsh -File $RehydPS -Freq $Freq -Symbols $Symbols
      if($LASTEXITCODE -eq 0){ $rehydOk = $true }
    } catch { }
  }

  if(-not $rehydOk){ throw "Rehydrate fehlgeschlagen" }
}

# --- (3) Optional: nur Features (direkter Call) ---
if($Features){
  Info "SPRINT8 Features bauen… ($Freq)"
  # Kompatible Flags (wir haben --quick + --qdays in sprint8_feature_engineering.py)
  $symArg = $Symbols
  if([string]::IsNullOrWhiteSpace($symArg)){ $symArg = 'ALL' }
  & $Py $S8_Feature --freq $Freq --symbols $symArg --quick --qdays $QuickDays
  if($LASTEXITCODE -ne 0){ throw "Feature-Build gescheitert" }
}

# --- (4) Execution+Kosten ---
if($Costs){
  Info "sprint8_cost_model.ps1"
  pwsh -File $S8_CostPS -Freq $Freq -Notional $StartCapital -CommissionBps $CommissionBps
  if($LASTEXITCODE -ne 0){ throw "Execution fehlgeschlagen" }
}

# --- (5) Backtest ---
if($Backtest){
  Info "sprint9_backtest.ps1"
  pwsh -File $S9_BackPS -Freq $Freq
  if($LASTEXITCODE -ne 0){ throw "Backtest fehlgeschlagen" }
}

# --- (6) Kosten-Grid ---
if($Grid){
  Info "sprint9_cost_grid.ps1"
  pwsh -File $S9_GridPS -Freq $Freq
  if($LASTEXITCODE -ne 0){ throw "Cost-Grid fehlgeschlagen" }
}

# --- (7) Portfolio ---
if($Portfolio){
  Info "sprint10_portfolio.ps1"
  pwsh -File $S10_PF_PS -Freq $Freq -StartCapital $StartCapital -Exposure $Exposure -MaxLeverage $MaxLeverage -CommissionBps $CommissionBps -SpreadW $SpreadW -ImpactW $ImpactW
  if($LASTEXITCODE -ne 0){ throw "Portfolio run fehlgeschlagen" }
}

# --- (8) Zusammenfassung erzeugen (JSON), ggf. Discord senden ---
$perfMd     = Join-Path $OUT 'performance_report.md'
$pfReportMd = Join-Path $OUT 'portfolio_report.md'
$eqCsv      = Join-Path $OUT ("portfolio_equity_{0}.csv" -f $Freq)
$ordersCsv  = Join-Path $OUT 'orders.csv'
$gridMd     = Join-Path $OUT 'cost_grid_report.md'

# Parser-Helfer: einfache Werte aus MD ziehen
function Parse-Metric {
  param([string]$Path, [string]$Key)
  if(-not (Test-Path $Path)){ return $null }
  $line = Select-String -Path $Path -Pattern $Key -SimpleMatch | Select-Object -First 1
  if(-not $line){ return $null }
  # Zahlen aus der Zeile ziehen (erste Zahl / signed float)
  $m = [regex]::Match($line.Line, '(-?\d+(\.\d+)?)')
  if($m.Success){ return $m.Value } else { return $null }
}

$summary = [ordered]@{
  Freq      = $Freq
  StartCap  = $StartCapital
  Exposure  = $Exposure
  Leverage  = $MaxLeverage
  CommissionBps = $CommissionBps
  SpreadW   = $SpreadW
  ImpactW   = $ImpactW
  Performance = @{
    Report = (Test-Path $perfMd) ? $perfMd : $null
    PF     = (Parse-Metric -Path $perfMd -Key 'PF')
    Sharpe = (Parse-Metric -Path $perfMd -Key 'Sharpe')
    Trades = (Parse-Metric -Path $perfMd -Key 'Trades')
  }
  Portfolio = @{
    Report = (Test-Path $pfReportMd) ? $pfReportMd : $null
    EquityCsv = (Test-Path $eqCsv) ? $eqCsv : $null
  }
  Costs = @{
    OrdersCsv = (Test-Path $ordersCsv) ? $ordersCsv : $null
    GridReport = (Test-Path $gridMd) ? $gridMd : $null
  }
  Timestamp = (Get-Date).ToString('s')
}

$summaryPath = Join-Path $OUT 'run_summary.json'
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summaryPath

if($NotifyDiscord){
  Info "Discord-Notify…"
  if(-not (Test-Path $DiscordPS)){ throw "notify_discord.ps1 nicht gefunden: $DiscordPS" }

  $title = "RunAll abgeschlossen"
  $content = "Kompletter Sprint-10-Durchlauf erfolgreich ✅`nFreq=$Freq | Cap=$StartCapital | Exp=$Exposure | Lev=$MaxLeverage | Comm=${CommissionBps}bps"

  if([string]::IsNullOrWhiteSpace($DiscordWebhook)){
    # verlasse dich auf $env:DISCORD_WEBHOOK oder .env
    pwsh -File $DiscordPS -Title $title -Content $content -DataJsonPath $summaryPath
  }else{
    pwsh -File $DiscordPS -Title $title -Content $content -DataJsonPath $summaryPath -Webhook $DiscordWebhook
  }
  if($LASTEXITCODE -ne 0){ throw "Discord-Notify fehlgeschlagen" }
}

# --- (9) Git Sync ---
if($Sync){
  Info "git_sync.ps1"
  if(-not (Test-Path $SyncPS)){ throw "git_sync.ps1 nicht gefunden: $SyncPS" }
  $msg = "Auto-sync $(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
  pwsh -File $SyncPS -Message $msg
  if($LASTEXITCODE -ne 0){ throw "Git-Sync fehlgeschlagen" }
}

Info "DONE"
