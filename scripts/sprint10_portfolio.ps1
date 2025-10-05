param(
[string]$Freq = '5min',
[float]$StartCapital = 10000,
[float]$Exposure = 1.0,
[float]$MaxLeverage = 1.0,
[float]$CommissionBps = 0.5,
[float]$SpreadW = 1.0,
[float]$ImpactW = 1.0
)
$ErrorActionPreference = 'Stop'
$ROOT = (Get-Location).Path
$Py = Join-Path $ROOT '.venv/Scripts/python.exe'
$Script = Join-Path $ROOT 'scripts/sprint10_portfolio.py'
function Info($m){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [PF10] $m" }
if(-not (Test-Path $Py)){ throw "Python venv nicht gefunden: $Py" }
if(-not (Test-Path $Script)){ throw "Nicht gefunden: $Script" }
Info "Start Portfolio | freq=$Freq cap=$StartCapital exp=$Exposure lev=$MaxLeverage"
& $Py $Script --freq $Freq --start-capital $StartCapital --exposure $Exposure --max-leverage $MaxLeverage --commission-bps $CommissionBps --spread-w $SpreadW --impact-w $ImpactW
if($LASTEXITCODE -ne 0){ throw "Portfolio run fehlgeschlagen" }
$OUT = Join-Path $ROOT 'output'
$eq = Join-Path $OUT "portfolio_equity_${Freq}.csv"
$rep = Join-Path $OUT 'portfolio_report.md'
$trd = Join-Path $OUT 'portfolio_trades.csv'
if(Test-Path $eq){ Info "[OK] Equity: $eq" }
if(Test-Path $rep){ Info "[OK] Report: $rep" }
if(Test-Path $trd){ Info "[OK] Trades: $trd" }
Info "DONE"

