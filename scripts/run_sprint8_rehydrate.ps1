param(
  [string]$Freq = '5min',
  [switch]$Quick = $true,
  [int]$QuickDays = 180,
  [string]$Symbols = 'AAPL,MSFT'
)

$ErrorActionPreference = 'Stop'
function Info($m){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [REHYDRATE] $m" }

$ROOT    = (Get-Location).Path
$Py      = Join-Path $ROOT '.venv/Scripts/python.exe'
$Out     = Join-Path $ROOT 'output'
$Data    = Join-Path $ROOT 'data'

Info "Python: $Py"
Info "Checking core packages..."
& $Py -c "import pandas, numpy, pyarrow; print('[OK] pandas'); print('[OK] numpy'); print('[OK] pyarrow'); print('[OK] Packages ready')"

# Assemble/Resample werden übersprungen, wenn Artefakte existieren
$agg5  = Join-Path $Out 'aggregates'
$asm   = Join-Path $Out 'assembled_intraday'
if(Test-Path (Join-Path $agg5 "assembled_intraday_${Freq}.parquet")){
  Info "Assemble-Outputs gefunden (überspringe Assemble)"
}
if(Test-Path (Join-Path $agg5 "assembled_intraday_${Freq}.parquet")){
  Info "$Freq-Resample vorhanden (überspringe Resample)"
}

$raw1m = Join-Path $Data 'raw\1min'
$filesRaw   = (Test-Path $raw1m) ? (Get-ChildItem $raw1m -Filter *.csv -File | Measure-Object).Count : 0
$filesAsm   = (Test-Path $asm) ? (Get-ChildItem $asm -Filter *.parquet -File | Measure-Object).Count : 0
$filesAgg   = (Test-Path $agg5) ? (Get-ChildItem $agg5 -Filter *$Freq*.parquet -File | Measure-Object).Count : 0
Info ("Files: raw/1min={0}, assembled/1min={1}, aggregates(*{2}*)={3}" -f $filesRaw,$filesAsm,$Freq,$filesAgg)

# Feature-Build
$demo = $false
$pyArgs = @('--freq', $Freq)
if($Quick){ $pyArgs += @('--quick', '--qdays', "$QuickDays") }
if(-not [string]::IsNullOrWhiteSpace($Symbols)){ $pyArgs += @('--symbols', $Symbols) }

Info ("Baue Features | freq={0} | symbols={1} | quickDays={2} | demo={3}" -f $Freq, $Symbols, $QuickDays, $demo)
& $Py (Join-Path $ROOT 'scripts\sprint8_feature_engineering.py') @pyArgs
if($LASTEXITCODE -ne 0){ throw "Feature-Build gescheitert (siehe Log)." }

# Ausgabe
$feat = Join-Path $Out 'features'
if(Test-Path $feat){
  Info "[OK] Features geschrieben:"
  Get-ChildItem $feat -Filter *.parquet | ForEach-Object {
    Write-Host " - $($_.FullName)"
  }
  $manifest = Join-Path $feat 'feature_manifest.json'
  if(Test-Path $manifest){ Info "[OK] manifest updated: $manifest" }
}
Info "Rehydrate + Orchestrator DONE"

