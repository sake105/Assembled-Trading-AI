#Requires -Version 7.0
param([switch]$Quick)  # mit -Quick überspringst du den Backfill
Set-Location "D:\PROJEKT_AKTIE\Projekt_1\Grundsachen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\Sprint1_Update_v24;$PWD\OneClick_Autopilot_Addon_v24;$PWD\OneClick_Autopilot_Addon_v24\tools;$PWD\News_AllIn_TrustedFeeds_v24;$PWD\News_TrustedPack_v24"

if (-not $Quick) { python ".\Sprint1_Update_v24\scripts\backfill.py" }  # Daten aktualisieren
python ".\Sprint1_Update_v24\scripts\screener_local.py"                 # Kennzahlen erzeugen
# optional - nur wenn whitelist/blacklist konfiguriert ist:
# python ".\News_AllIn_TrustedFeeds_v24\scripts\run_news.py"

