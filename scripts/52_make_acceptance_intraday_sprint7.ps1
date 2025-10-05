# requires -Version 7.0
param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$OutPath,
  [int]$MinRowsPerInterval = 50,
  [int]$MaxGapMinutes      = 480,
  [int]$MaxTotalGaps       = 100000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if(-not $OutPath){
  $OutPath = Join-Path $ProjectRoot 'docs\SPRINT_7_ACCEPTANCE.md'
}
$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutPath)

# Pfade zu Artefakten
$resampleReport = Join-Path $ProjectRoot 'output\aggregates\resample_report.json'
$qcCsv          = Join-Path $ProjectRoot 'output\qc\intraday_gaps.csv'
$qcSummary      = Join-Path $ProjectRoot 'output\qc\intraday_gaps_summary.json'

function Read-JsonOrNull([string]$path){
  if(Test-Path -LiteralPath $path){
    try   { Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
    catch { Write-Warning ('JSON lesen fehlgeschlagen: {0} -> {1}' -f $path, $_.Exception.Message); $null }
  } else { $null }
}

$rep   = Read-JsonOrNull $resampleReport
$qcsum = Read-JsonOrNull $qcSummary

# Gates
$passResample = $false
if($rep -and $rep.summary -and $rep.summary.Count -gt 0){
  $passResample = $true
  foreach($row in $rep.summary){
    if([int]$row.rows -lt $MinRowsPerInterval){ $passResample = $false; break }
  }
}

$passQc = $false
if($qcsum){
  $tg = [int]$qcsum.total_gaps
  $mg = [int][double]$qcsum.max_gap_minutes_overall
  $passQc = ($tg -le $MaxTotalGaps -and $mg -le $MaxGapMinutes)
}

$status = if($passResample -and $passQc) { 'PASS' } else { 'FAIL' }

# Datum Berlin
try {
  $tz = [System.TimeZoneInfo]::FindSystemTimeZoneById('W. Europe Standard Time')
  $nowBerlin = [System.TimeZoneInfo]::ConvertTime([DateTimeOffset]::Now, $tz)
  $dateStr = $nowBerlin.ToString('yyyy-MM-dd HH:mm:ss zzz')
} catch {
  $dateStr = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
}

# Markdown zusammenbauen – NUR Stringverkettungen, KEINE Backticks im Inhalt
$lines = @()
$lines += '# SPRINT 7 - ACCEPTANCE (Resample + QC)'
$lines += ''
$lines += '**Datum (Berlin):** ' + $dateStr
$lines += ''
$lines += '## Ergebnis'
$lines += '- **Status:** ' + $status
$lines += ''
$lines += '## Gate-Parameter'
$lines += '- MinRowsPerInterval: **' + $MinRowsPerInterval + '**'
$lines += '- MaxGapMinutes: **'      + $MaxGapMinutes      + '**'
$lines += '- MaxTotalGaps: **'       + $MaxTotalGaps       + '**'
$lines += ''
$lines += '## Resample-Report'

if($rep -and $rep.summary -and $rep.summary.Count -gt 0){
  foreach($it in $rep.summary){
    $iv   = [string]$it.interval
    $rows = [int]$it.rows
    $file = [string]$it.file
    if($file){
      $prefix = "$ProjectRoot\"
      if($file.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)){
        $file = $file.Substring($prefix.Length)
      }
    }
    $lines += '- Intervall **' + $iv + '**: Zeilen **' + $rows + '** -> Datei: ' + $file
  }
} else {
  $lines += '- (Keine Resample-Zusammenfassung gefunden.)'
}

$lines += ''
$lines += '## QC Gaps (Kurzfassung)'
if($qcsum){
  $lines += '- total_gaps: **'                + $qcsum.total_gaps                + '**'
  $lines += '- max_gap_minutes_overall: **'   + $qcsum.max_gap_minutes_overall   + '**'
} else {
  $lines += '- (Keine QC-Zusammenfassung gefunden.)'
}

$lines += ''
$lines += '## Artefakte'
$lines += 'output/aggregates/resample_report.json'
$lines += 'output/qc/intraday_gaps.csv'
$lines += 'output/qc/intraday_gaps_summary.json'
$lines += ''

$md = $lines -join "`r`n"
$md | Set-Content -LiteralPath $OutPath -Encoding utf8

Write-Host ('[OK]   written: {0}' -f $OutPath)
if($status -eq 'PASS'){ Write-Host '[OK]   Acceptance PASS (Sprint 7)' } else { Write-Warning 'Acceptance FAIL (Sprint 7)' }


