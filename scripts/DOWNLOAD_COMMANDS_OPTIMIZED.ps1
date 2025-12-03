# ============================================================================
# OPTIMIERTE DOWNLOAD-BEFEHLE FÜR ALLE UNIVERSEN
# ============================================================================
#
# WICHTIG: Yahoo Finance Rate-Limits sind sehr aggressiv!
#
# Diese Befehle verwenden:
# - 60 Sekunden Pause zwischen Downloads (statt 2.0)
# - Automatisches Überspringen bereits existierender Dateien
# - Robuste Retry-Logik bei Rate-Limits
#
# ============================================================================
# VOR DEM START:
# ============================================================================
# 1. Warte mindestens 3-4 Stunden nach dem letzten Rate-Limit-Fehler
# 2. Oder starte morgen früh (Rate-Limits werden über Nacht zurückgesetzt)
# 3. Stelle sicher, dass deine Internetverbindung stabil ist
#
# ============================================================================

$ErrorActionPreference = 'Continue'
$Python = ".\.venv\Scripts\python.exe"
$TargetRoot = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
$StartDate = "2000-01-01"
$EndDate = "2025-12-03"
$Interval = "1d"
$SleepSeconds = 60.0  # 60 Sekunden zwischen Downloads (KRITISCH!)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "OPTIMIERTE DOWNLOAD-BEFEHLE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WICHTIG: Diese Befehle verwenden 60 Sekunden Pause zwischen Downloads!" -ForegroundColor Yellow
Write-Host "Das ist notwendig, um Rate-Limits zu vermeiden." -ForegroundColor Yellow
Write-Host ""
Write-Host "Erwartete Dauer pro Universe:" -ForegroundColor Cyan
Write-Host "  - AI Tech (24 Symbole): ca. 24 Minuten" -ForegroundColor Cyan
Write-Host "  - Healthcare (4 Symbole): ca. 4 Minuten" -ForegroundColor Cyan
Write-Host "  - Energy (7 Symbole): ca. 7 Minuten" -ForegroundColor Cyan
Write-Host "  - Defense (11 Symbole): ca. 11 Minuten" -ForegroundColor Cyan
Write-Host "  - Consumer (3 Symbole): ca. 3 Minuten" -ForegroundColor Cyan
Write-Host "  - GESAMT: ca. 49 Minuten + Pausen zwischen Universen" -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. AI / Tech
# ============================================================================
Write-Host "[1/5] AI / Tech Universe" -ForegroundColor Green
& $Python scripts\download_historical_snapshot.py `
  --symbols-file config\universe_ai_tech_tickers.txt `
  --start $StartDate `
  --end $EndDate `
  --interval $Interval `
  --target-root "`"$TargetRoot`"" `
  --sleep-seconds $SleepSeconds

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] AI Tech download had errors. Check logs above." -ForegroundColor Yellow
    Write-Host "Waiting 5 minutes before continuing..." -ForegroundColor Yellow
    Start-Sleep -Seconds 300
}

Write-Host ""
Write-Host "Pausing 2 minutes before next universe..." -ForegroundColor Cyan
Start-Sleep -Seconds 120

# ============================================================================
# 2. Healthcare / Biotech
# ============================================================================
Write-Host "[2/5] Healthcare / Biotech Universe" -ForegroundColor Green
& $Python scripts\download_historical_snapshot.py `
  --symbols-file config\healthcare_biotech_tickers.txt `
  --start $StartDate `
  --end $EndDate `
  --interval $Interval `
  --target-root "`"$TargetRoot`"" `
  --sleep-seconds $SleepSeconds

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Healthcare download had errors. Check logs above." -ForegroundColor Yellow
    Write-Host "Waiting 5 minutes before continuing..." -ForegroundColor Yellow
    Start-Sleep -Seconds 300
}

Write-Host ""
Write-Host "Pausing 2 minutes before next universe..." -ForegroundColor Cyan
Start-Sleep -Seconds 120

# ============================================================================
# 3. Energy / Resources / Cyclicals
# ============================================================================
Write-Host "[3/5] Energy / Resources / Cyclicals Universe" -ForegroundColor Green
& $Python scripts\download_historical_snapshot.py `
  --symbols-file config\energy_resources_cyclicals_tickers.txt `
  --start $StartDate `
  --end $EndDate `
  --interval $Interval `
  --target-root "`"$TargetRoot`"" `
  --sleep-seconds $SleepSeconds

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Energy download had errors. Check logs above." -ForegroundColor Yellow
    Write-Host "Waiting 5 minutes before continuing..." -ForegroundColor Yellow
    Start-Sleep -Seconds 300
}

Write-Host ""
Write-Host "Pausing 2 minutes before next universe..." -ForegroundColor Cyan
Start-Sleep -Seconds 120

# ============================================================================
# 4. Defense / Security / Aero
# ============================================================================
Write-Host "[4/5] Defense / Security / Aero Universe" -ForegroundColor Green
& $Python scripts\download_historical_snapshot.py `
  --symbols-file config\defense_security_aero_tickers.txt `
  --start $StartDate `
  --end $EndDate `
  --interval $Interval `
  --target-root "`"$TargetRoot`"" `
  --sleep-seconds $SleepSeconds

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Defense download had errors. Check logs above." -ForegroundColor Yellow
    Write-Host "Waiting 5 minutes before continuing..." -ForegroundColor Yellow
    Start-Sleep -Seconds 300
}

Write-Host ""
Write-Host "Pausing 2 minutes before next universe..." -ForegroundColor Cyan
Start-Sleep -Seconds 120

# ============================================================================
# 5. Consumer / Financial / Misc
# ============================================================================
Write-Host "[5/5] Consumer / Financial / Misc Universe" -ForegroundColor Green
& $Python scripts\download_historical_snapshot.py `
  --symbols-file config\consumer_financial_misc_tickers.txt `
  --start $StartDate `
  --end $EndDate `
  --interval $Interval `
  --target-root "`"$TargetRoot`"" `
  --sleep-seconds $SleepSeconds

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Consumer download had errors. Check logs above." -ForegroundColor Yellow
}

# ============================================================================
# Validation
# ============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Validating downloaded files..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
& $Python scripts\validate_altdata_snapshot.py `
  --root "`"$TargetRoot`"" `
  --interval $Interval

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Download-Prozess abgeschlossen!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Tipp: Falls einige Downloads fehlgeschlagen sind," -ForegroundColor Yellow
Write-Host "      führe dieses Skript erneut aus." -ForegroundColor Yellow
Write-Host "      Bereits existierende Dateien werden automatisch übersprungen." -ForegroundColor Yellow

