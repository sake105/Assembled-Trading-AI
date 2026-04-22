@echo off
REM Launcher for paper-trading scheduler daemon — used by Windows Task Scheduler.
REM Redirects stdout + stderr to a dated log file so failures are not lost.

cd /d "F:\Python_Projekt\Aktiengerüst"
if not exist logs\scheduler mkdir logs\scheduler

for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="') do set datetime=%%i
set LOGFILE=logs\scheduler\scheduler_%datetime:~0,8%.log

".venv\Scripts\python.exe" -u scripts\paper_trading_scheduler.py >> "%LOGFILE%" 2>&1
