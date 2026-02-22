# Quick verification that equity_curve = MTM and cash_curve exists (after benchmark run).
# Usage: .\scripts\dev\verify_equity_cash_system_run.ps1
#        Or pass base dir: .\scripts\dev\verify_equity_cash_system_run.ps1 -BaseDir "output\system_run\benchmark\trend_baseline\1y"

param([string]$BaseDir = "output\system_run\benchmark\trend_baseline\1y")

$EQ = Join-Path $BaseDir "equity_curve_1d.csv"
$C  = Join-Path $BaseDir "cash_curve_1d.csv"
$TR = Join-Path $BaseDir "trades_1d.csv"

Write-Host "=== Equity vs Cash verification (MTM fix) ===" -ForegroundColor Cyan
if (-not (Test-Path $EQ)) { Write-Host "Missing: $EQ"; exit 1 }
$eq = Import-Csv $EQ
Write-Host "equity_curve rows: $($eq.Count)"
$eqMin = ($eq | Measure-Object -Property equity -Minimum).Minimum
$eqMax = ($eq | Measure-Object -Property equity -Maximum).Maximum
Write-Host "equity min: $eqMin  max: $eqMax"

if (Test-Path $C) {
    $ca = Import-Csv $C
    Write-Host "cash_curve rows: $($ca.Count)"
    $caMin = ($ca | Measure-Object -Property cash -Minimum).Minimum
    $caMax = ($ca | Measure-Object -Property cash -Maximum).Maximum
    Write-Host "cash min: $caMin  max: $caMax"
    $diff = 0
    for ($i = 0; $i -lt [Math]::Min($eq.Count, $ca.Count); $i++) {
        if ([Math]::Abs([double]$eq[$i].equity - [double]$ca[$i].cash) -gt 1e-6) { $diff++ }
    }
    Write-Host "Rows where equity != cash: $diff of $($eq.Count)"
    if ($diff -gt 0) { Write-Host "OK: Equity and cash are different (MTM fix active)" -ForegroundColor Green }
} else {
    Write-Host "cash_curve_1d.csv not found (old run or writer not writing cash?)"
}

if (Test-Path $TR) {
    Write-Host ""
    Write-Host "=== Trades ===" -ForegroundColor Cyan
    $tr = Import-Csv $TR
    $byStatus = $tr | Group-Object status | Select-Object Name, Count
    $byStatus | Format-Table -AutoSize
    $filled = ($tr | Where-Object { [double]$_.fill_qty -gt 0 }).Count
    Write-Host "Filled (fill_qty>0): $filled"
    $rej = $tr | Where-Object { $_.status -eq "rejected" }
    if ($rej) {
        Write-Host "Rejected sample reject_reason: $(($rej | Select-Object -First 5).reject_reason -join ', ')"
    }
}
