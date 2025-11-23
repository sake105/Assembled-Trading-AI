#Requires -Version 7.0
param([switch]$Quick)  # -Quick überspringt Backfill

$ROOT = "D:\PROJEKT_AKTIE\Projekt_1\Grundsachen"
Set-Location $ROOT
.\.venv\Scripts\Activate.ps1

# PYTHONPATH für alle Module
$env:PYTHONPATH = "$PWD\Sprint1_Update_v24;$PWD\Sprint2_Update_v24;$PWD\OneClick_Autopilot_Addon_v24;$PWD\OneClick_Autopilot_Addon_v24\tools;$PWD\News_AllIn_TrustedFeeds_v24;$PWD\News_TrustedPack_v24"

# 1) Daten-Check (mind. 1 Datei)
$raw = Join-Path $ROOT "data\raw"
if (-not (Test-Path $raw) -or ((Get-ChildItem $raw -File).Count -eq 0)) {
  Write-Host "[STOP] Keine Rohdaten in data\raw - zuerst Backfill laufen lassen." -ForegroundColor Yellow
  if (-not $Quick) { python ".\Sprint1_Update_v24\scripts\backfill.py" }
}

# 2) Sprint-2 Verzeichnis finden
$s2 = Get-ChildItem -Directory -Path $ROOT -Filter "Sprint2*" | Select-Object -First 1
if (-not $s2) {
  Write-Host "[INFO] Kein 'Sprint2*' Ordner gefunden. Bitte prüfen/auspacken." -ForegroundColor Yellow
  exit 1
}

# 3) Entry-Script autodiscovery
$candidates = @("run_sprint2.py","run_all.py","train.py","pipeline.py","main.py","sprint2.py")
$entry = $null
foreach ($pat in $candidates) {
  $hit = Get-ChildItem -Recurse -Path $s2.FullName -Include $pat -File | Select-Object -First 1
  if ($hit) { $entry = $hit.FullName; break }
}
if (-not $entry) {
  # generischer Fallback: irgendein run_*.py
  $hit = Get-ChildItem -Recurse -Path $s2.FullName -Include "run_*.py" -File | Select-Object -First 1
  if ($hit) { $entry = $hit.FullName }
}

if (-not $entry) {
  Write-Host "[INFO] Kein Entry-Script in $($s2.Name) gefunden. Suche nach 'run_*.py', 'train.py', 'pipeline.py', 'main.py'." -ForegroundColor Yellow
  Get-ChildItem -Recurse -Path $s2.FullName -Include "*.py" -File | Select-Object FullName | Format-Table -AutoSize
  exit 1
}

Write-Host "[OK] Sprint-2 Entry: $entry" -ForegroundColor Green

# 4) (Optional) Quick-Backfill
if (-not $Quick) {
  python ".\Sprint1_Update_v24\scripts\backfill.py"
}

# 5) Start Sprint-2
python $entry


