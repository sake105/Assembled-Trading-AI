@echo off
REM ============================================
REM  Dead-Man's Switch daemon launcher (OPS-01)
REM  Launches the continuous passive heartbeat monitor scripts/dms_daemon.py,
REM  which reads configs/policy.yaml (dead_man_switch block).
REM
REM  Registered as the AssembledTradingAI-DMS Windows Task (at-logon, continuous)
REM  via scripts/ops/register_dms_task.ps1. The daemon runs forever until the
REM  session ends or it receives SIGTERM; this wrapper exits with the daemon's
REM  exit code at that point.
REM
REM  MODE: SHADOW (observe-only) per configs/policy.yaml — it LOGS what it would
REM  flatten to output/ops/dms_audit.jsonl but does NOT touch the kill switch.
REM  Arming to "market" is blocked on OPS-02 (see policy.yaml + runbook 12 §4).
REM ============================================

cd /d "F:\Python_Projekt\Aktiengerüst"

if not exist logs\scheduler mkdir logs\scheduler
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="') do set datetime=%%i
set LOGFILE=logs\scheduler\dms_daemon_%datetime:~0,8%.log

echo [%date% %time%] === DMS daemon starting (shadow mode) === >> "%LOGFILE%" 2>&1

.venv\Scripts\python.exe scripts\dms_daemon.py --policy configs\policy.yaml >> "%LOGFILE%" 2>&1
set DMS_RC=%errorlevel%

echo [%date% %time%] === DMS daemon exited. Exit code: %DMS_RC% === >> "%LOGFILE%" 2>&1

REM NO `pause` — this runs unattended via Task Scheduler.
exit /b %DMS_RC%
