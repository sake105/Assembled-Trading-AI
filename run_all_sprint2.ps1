#Requires -Version 7.0
Set-Location "$PSScriptRoot"
.\.venv\Scripts\Activate.ps1 | Out-Null

# PYTHONPATH for Sprint 2 + optional modules
$env:PYTHONPATH = "$PWD\Sprint2_Update_v24;$PWD\Sprint1_Update_v24;$PWD\OneClick_Autopilot_Addon_v24;$PWD\OneClick_Autopilot_Addon_v24\tools;$PWD\News_AllIn_TrustedFeeds_v24;$PWD\News_TrustedPack_v24"

# Optional: Data migration (convert flat files to yearly parquet). Safe to re-run.
python ".\Sprint2_Update_v24\scripts\migrate_raw_layout.py"

# Run Sprint-2 pipelines
if (Test-Path ".\Sprint2_Update_v24\scripts\run_screener.py") { python ".\Sprint2_Update_v24\scripts\run_screener.py" }
if (Test-Path ".\Sprint2_Update_v24\scripts\run_events.py")   { python ".\Sprint2_Update_v24\scripts\run_events.py" }
if (Test-Path ".\Sprint2_Update_v24\scripts\run_news.py")     { python ".\Sprint2_Update_v24\scripts\run_news.py" }

