@echo off
REM ============================================
REM  Daily Paper Trading Runner
REM  Triggered by Windows Task Scheduler (see scripts/ops/register_paper_pilot_task.ps1)
REM  Recommended schedule: 21:30 CEST (15:30 ET, 30 min before NYSE close)
REM  Runs Mon-Fri only (handled by task trigger)
REM ============================================

cd /d "F:\Python_Projekt\Aktiengerüst"

REM Per-day log file for audit trail
if not exist logs\scheduler mkdir logs\scheduler
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="') do set datetime=%%i
set LOGFILE=logs\scheduler\daily_paper_trading_%datetime:~0,8%.log

echo [%date% %time%] === Starting daily paper trading cycle === >> "%LOGFILE%" 2>&1

REM Step 0: Refresh daily.parquet from master_universe_panel.parquet (offline, no network).
REM This bridges the freshness gap that broke the pilot 2026-05-15..20: when the
REM cache aged past 3 days, run_live_paper.py fell back to a sequential yfinance
REM fetch (197 syms x ~10s with rate-limit retries) that exceeded the 15-min Task
REM Scheduler ExecutionTimeLimit and got hard-terminated. The master panel is
REM produced earlier in the daily cycle by the build pipeline and typically has
REM fresher OHLCV than the EOD cache. No-op when panel is not newer than cache.
echo [%date% %time%] Refreshing daily.parquet from master panel... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\ops\refresh_daily_cache_from_panel.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo [WARN] daily.parquet refresh failed — proceeding with existing cache >> "%LOGFILE%" 2>&1
)

REM Step 1: Refresh price cache for the watchlist (catch gaps from new symbols)
echo [%date% %time%] Pre-warming price cache... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\ops\prewarm_price_cache.py --years 2 >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo [WARN] Price cache prewarm failed — proceeding with existing cache >> "%LOGFILE%" 2>&1
)

REM Step 2: Run one pilot day via the pilot script (NOT run_live_paper.py directly).
REM This preserves the pilot manifest day-count + GO/NO-GO tracking.
echo [%date% %time%] Running paper pilot --run-day... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\run_paper_pilot.py --run-day >> "%LOGFILE%" 2>&1
set PILOT_RC=%errorlevel%

echo [%date% %time%] === Done. Pilot exit code: %PILOT_RC% === >> "%LOGFILE%" 2>&1

REM NO `pause` — this runs unattended via Task Scheduler.
exit /b %PILOT_RC%
