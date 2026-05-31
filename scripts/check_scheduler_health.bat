@echo off
REM ============================================
REM  Scheduler Health Check (OPS-03)
REM  Independent watchdog: runs check_scheduler_health.py against the REAL
REM  deployed heartbeat (output\state\heartbeat.json, written by the daily
REM  pilot via _tc_execution) and alerts if it is stale.
REM
REM  Registered as a SEPARATE Windows Task from the pilot
REM  (see scripts/ops/register_health_check_task.ps1) so that if the pilot
REM  task itself stops firing — the exact 2026-04-10 silent-stall mode — this
REM  watchdog still runs and detects the missing/stale heartbeat.
REM
REM  Schedule: 22:30 local (Mon-Fri), ~1h after the 21:30 pilot window.
REM  Threshold 1080 min (18h): a healthy day's heartbeat is minutes old; a
REM  stall (no run today) leaves yesterday's heartbeat >= ~24h old.
REM ============================================

cd /d "F:\Python_Projekt\Aktiengerüst"

if not exist logs\scheduler mkdir logs\scheduler
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="') do set datetime=%%i
set LOGFILE=logs\scheduler\health_check_%datetime:~0,8%.log

echo [%date% %time%] === Scheduler health check === >> "%LOGFILE%" 2>&1

.venv\Scripts\python.exe scripts\check_scheduler_health.py ^
    --heartbeat-path output\state\heartbeat.json ^
    --ignore-market-hours ^
    --stale-minutes 1080 ^
    --notify >> "%LOGFILE%" 2>&1
set HEALTH_RC=%errorlevel%

echo [%date% %time%] === Done. Health exit code: %HEALTH_RC% === >> "%LOGFILE%" 2>&1

REM NO `pause` — this runs unattended via Task Scheduler.
exit /b %HEALTH_RC%
