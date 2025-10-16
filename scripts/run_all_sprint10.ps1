param(
  [string]$Freq = '5min',
  [double]$StartCapital = 10000,
  [double]$Exposure = 1,
  [double]$MaxLeverage = 1,
  [double]$CommissionBps = 0.5,
  [double]$SpreadW = 1,
  [double]$ImpactW = 1,

  # Rehydrate/Features
  [switch]$Quick = $true,
  [int]$QuickDays = 180,
  [string]$Symbols = 'AAPL,MSFT',

  # Steps
  [switch]$Seed = $true,
  [switch]$Rehydrate = $true,
  [switch]$Features = $true,
  [switch]$Costs = $true,
  [switch]$Backtest = $true,
  [switch]$Grid = $true,
  [switch]$Portfolio = $true,
  [switch]$Sync = $true,
  [switch]$Notify = $false
)

$ErrorActionPreference = 'Stop'
function Info($m){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [RUNALL] $m" }

$ROOT    = (Get-Location).Path
$Py      = Join-Path $ROOT '.venv/Scripts/python.exe'
$Scripts = Join-Path $ROOT 'scripts'
$Out     = Join-Path $ROOT 'output'

function Run([string]$label, [ScriptBlock]$block){
  Info $label
  & $block
  if($LASTEXITCODE -ne 0){ throw "$label fehlgeschlagen" }
}

function Get-CostGridBest {
  $md  = Join-Path $Out 'cost_grid_report.md'
  $csv = Join-Path $Out 'cost_grid_results.csv'

  if(Test-Path $csv){
    try{
      $rows = Import-Csv $csv
      $best = $rows | ForEach-Object {
        [pscustomobject]@{
          commission_bps = [double]$_.commission_bps
          spread_w       = [double]$_.spread_w
          impact_w       = [double]$_.impact_w
          PF             = [double]$_.PF
          Sharpe         = [double]$_.Sharpe
          Trades         = [int]$_.Trades
          FinalEquity    = [double]$_.FinalEquity
        }
      } | Sort-Object PF -Descending | Select-Object -First 1
      if($best){ return $best }
    } catch { }
  }

  if(-not (Test-Path $md)){ return $null }

  $text = Get-Content $md -Raw
  $pf = $null
  $m = [regex]::Match($text, '\*\*Best PF:\*\*\s*(?<pf>[0-9]+(?:\.[0-9]+)?)', 'IgnoreCase')
  if($m.Success){ $pf = [double]$m.Groups['pf'].Value }

  $mc = [regex]::Match($text, 'Params:\s*commission\s*(?<c>[0-9]+(?:\.[0-9]+)?)\s*bps,\s*spread_w\s*(?<sw>[0-9]+(?:\.[0-9]+)?),\s*impact_w\s*(?<iw>[0-9]+(?:\.[0-9]+)?)', 'IgnoreCase')
  $m2 = [regex]::Match($text, 'Trades:\s*(?<t>[0-9]+)\s*•\s*Final\s*Equity:\s*(?<eq>[0-9\.\s]+)\s*•\s*Sharpe:\s*(?<sh>[0-9\.\-]+)', 'IgnoreCase')

  if($pf -ne $null -and $mc.Success){
    $obj = [pscustomobject]@{
      commission_bps = [double]$mc.Groups['c'].Value
      spread_w       = [double]$mc.Groups['sw'].Value
      impact_w       = [double]$mc.Groups['iw'].Value
      PF             = [double]$pf
      Sharpe         = $null
      Trades         = $null
      FinalEquity    = $null
    }
    if($m2.Success){
      $obj.Trades = [int]$m2.Groups['t'].Value
      $eq = ($m2.Groups['eq'].Value -replace '\s','')
      $obj.FinalEquity = [double]$eq
      $obj.Sharpe = [double]$m2.Groups['sh'].Value
    }
    return $obj
  }
  return $null
}

$commStr   = ([string]$CommissionBps).Replace(',','.')
$spreadStr = ([string]$SpreadW).Replace(',','.')
$impactStr = ([string]$ImpactW).Replace(',','.')
Info ("Start | freq={0} cap={1} exp={2} lev={3} comm={4}bps spread={5} impact={6}" -f $Freq,$StartCapital,$Exposure,$MaxLeverage,$commStr,$spreadStr,$impactStr)

if($Seed){
  Run "Seede Demo-Daten…" {
    & $Py (Join-Path $Scripts '00_seed_demo_data.py') --freq $Freq
  }
}

if($Rehydrate){
  Run "run_sprint8_rehydrate.ps1" {
    $rehydrateArgs = @('-File', (Join-Path $Scripts 'run_sprint8_rehydrate.ps1'), '-Freq', $Freq)
    if($Quick){ $rehydrateArgs += '-Quick' }
    if(-not [string]::IsNullOrWhiteSpace($Symbols)){ $rehydrateArgs += @('-Symbols', $Symbols) }

    try { & pwsh @rehydrateArgs }
    catch {
      Info "Rehydrate (Fallback) ohne Quick/Symbols…"
      & pwsh -File (Join-Path $Scripts 'run_sprint8_rehydrate.ps1') -Freq $Freq
    }
  }
}

if($Features){
  Run "SPRINT8 Features bauen… ($Freq)" {
    $args = @('--freq', $Freq)
    if($Quick){
      $args += '--quick'
      if($PSBoundParameters.ContainsKey('QuickDays')){ $args += @('--qdays', "$QuickDays") }
    }
    if(-not [string]::IsNullOrWhiteSpace($Symbols)){ $args += @('--symbols', $Symbols) }
    & $Py (Join-Path $Scripts 'sprint8_feature_engineering.py') @args
  }
}

if($Costs){
  Run "sprint8_cost_model.ps1" {
    & pwsh -File (Join-Path $Scripts 'sprint8_cost_model.ps1') -Freq $Freq -Notional $StartCapital -CommissionBps $CommissionBps
  }
}

if($Backtest){
  Run "sprint9_backtest.ps1" {
    & pwsh -File (Join-Path $Scripts 'sprint9_backtest.ps1') -Freq $Freq
  }
}

$Best = $null
if($Grid){
  Run "sprint9_cost_grid.ps1" {
    & pwsh -File (Join-Path $Scripts 'sprint9_cost_grid.ps1') -Freq $Freq
  }
  $Best = Get-CostGridBest
  if($Best){
    Info ("Bestes Grid → PF={0} | comm={1}bps | spread={2} | impact={3} | trades={4} | equity={5} | sharpe={6}" -f `
      $Best.PF, $Best.commission_bps, $Best.spread_w, $Best.impact_w, $Best.Trades, $Best.FinalEquity, $Best.Sharpe)
  } else {
    Info "Hinweis: Konnte keinen Best-Block aus cost_grid_report extrahieren."
  }
}

if($Portfolio){
  Run "sprint10_portfolio.ps1" {
    & pwsh -File (Join-Path $Scripts 'sprint10_portfolio.ps1') `
      -Freq $Freq -StartCapital $StartCapital -Exposure $Exposure -MaxLeverage $MaxLeverage `
      -CommissionBps $CommissionBps -SpreadW $SpreadW -ImpactW $ImpactW
  }
}

if($Sync){
  Run "git_sync.ps1" {
    $msg = "RunAll: $($Freq), cap=$($StartCapital), exp=$($Exposure), comm=$($CommissionBps)bps, spread=$($SpreadW), impact=$($ImpactW)"
    & pwsh -File (Join-Path $Scripts 'tools/git_sync.ps1') -Message $msg
  }
}

if($Notify){
  $title = "RunAll abgeschlossen"
  $report = @(
    "Freq: $Freq",
    "Portfolio: cap=$StartCapital | exp=$Exposure | lev=$MaxLeverage | comm=$CommissionBps bps | spread=$SpreadW | impact=$ImpactW"
  )

  if($Best){
    $report += @(
      "",
      "**Cost-Grid (Best):**",
      "PF=$($Best.PF) | Sharpe=$($Best.Sharpe) | Trades=$($Best.Trades) | Final=$($Best.FinalEquity)",
      "Params: commission=$($Best.commission_bps) bps | spread_w=$($Best.spread_w) | impact_w=$($Best.impact_w)"
    )
  } else {
    $report += @("", "_Hinweis: Konnte Best-Block aus Cost-Grid nicht extrahieren._")
  }

  $content = ($report -join "`n")
  Run "Discord-Notify…" {
    & pwsh -File (Join-Path $Scripts 'tools/notify_discord.ps1') -Title $title -Content $content
  }
}

Info "DONE"
