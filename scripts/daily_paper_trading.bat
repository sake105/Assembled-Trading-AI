@echo off
REM ============================================
REM  Daily Paper Trading Runner
REM  Run this once per trading day (after US market open)
REM  Recommended: 16:30 UTC / 18:30 CEST (30 min before close)
REM ============================================

cd /d "F:\Python_Projekt\Aktiengerüst"

echo [%date% %time%] Starting daily paper trading cycle...

REM Step 1: Update price cache (Polygon, ~10 min for free tier)
echo [%date% %time%] Updating price cache...
.venv\Scripts\python.exe scripts\update_prices.py --days 10
if errorlevel 1 (
    echo [WARN] Price update failed - using cached data
)

REM Step 2: Run paper trading cycle
echo [%date% %time%] Running paper trading cycle...
.venv\Scripts\python.exe scripts\run_live_paper.py once

echo [%date% %time%] Done. Exit code: %errorlevel%
pause
