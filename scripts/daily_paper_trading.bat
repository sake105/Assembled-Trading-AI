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

REM Step 0 ENTFERNT (Audit 2026-08-16): refresh_daily_cache_from_panel.py lief
REM seit 2026-05-21 als taeglicher No-op — data/sample/master_universe_panel.parquet
REM wird von keinem Prozess mehr neu gebaut (der Builder haengt an EODHD, tot seit
REM 2026-08-05). Die Freshness-Bridge, fuer die Step 0 gebaut wurde, uebernimmt
REM Step 1 (prewarm via yfinance). Wiedereinbau nur mit lebendem Panel-Builder.

REM Step 1: Refresh price cache for the watchlist (catch gaps from new symbols)
echo [%date% %time%] Pre-warming price cache... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\ops\prewarm_price_cache.py --years 2 >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo [WARN] Price cache prewarm failed — proceeding with existing cache >> "%LOGFILE%" 2>&1
)

REM Step 1b: Refresh sector-ETF + SPY closes in daily.parquet (yfinance, ~9 syms).
REM The sector ETFs (XLK/XLF/XLE/XLV/XLI/XLU/XLP/XLY) are NOT in configs/watchlist.txt
REM and not reliably in the master panel, so Steps 0/1 never refresh them. They go
REM stale and the live multifactor_v2 sector_rotation_bias factor neutralises to 0.0
REM via its 7-day staleness guard. This step keeps them fresh so the factor computes.
REM Today's (partial) bar is excluded for PIT safety. No-op / WARN on yfinance outage.
echo [%date% %time%] Refreshing sector-ETF + SPY cache... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\ops\refresh_sector_etf_cache.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo [WARN] sector-ETF cache refresh failed — sector_rotation_bias may neutralise >> "%LOGFILE%" 2>&1
)

REM Step 2: Run one pilot day via the pilot script (NOT run_live_paper.py directly).
REM This preserves the pilot manifest day-count + GO/NO-GO tracking.
echo [%date% %time%] Running paper pilot --run-day... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\run_paper_pilot.py --run-day >> "%LOGFILE%" 2>&1
set PILOT_RC=%errorlevel%

REM Step 3: Attribution- + Signal-Score-Report (Audit-Plan 5.3, 2026-08-16).
REM Producer fuer output/attribution/* und /monitoring/signals. Non-fatal:
REM ein Report-Fehler darf den Pilot-Exit-Code nicht ueberschreiben.
echo [%date% %time%] Generating attribution report... >> "%LOGFILE%" 2>&1
.venv\Scripts\python.exe scripts\generate_attribution_report.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo [WARN] attribution report failed — pilot result unaffected >> "%LOGFILE%" 2>&1
)

echo [%date% %time%] === Done. Pilot exit code: %PILOT_RC% === >> "%LOGFILE%" 2>&1

REM NO `pause` — this runs unattended via Task Scheduler.
exit /b %PILOT_RC%
