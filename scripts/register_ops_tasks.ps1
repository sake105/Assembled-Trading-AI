# scripts/register_ops_tasks.ps1 — register the ops watchdog (every 20 min) + DMS daemon (on logon).
# Run ONCE in an elevated PowerShell. Idempotent: deletes+recreates by name.
$ErrorActionPreference = "Stop"
$repo = "F:\Python_Projekt\Aktiengerüst"
$py   = "python"

# 1) Watchdog — every 20 minutes, indefinitely
$wdName = "AssembledTradingAI-OpsWatchdog"
$wdAct  = New-ScheduledTaskAction -Execute $py -Argument "scripts\ops_watchdog.py" -WorkingDirectory $repo
$wdTrig = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 20)
schtasks /delete /tn $wdName /f 2>$null
Register-ScheduledTask -TaskName $wdName -Action $wdAct -Trigger $wdTrig -Description "Paper-pilot ops watchdog: halt/heartbeat/run-quality/drawdown alerts"

# 2) DMS daemon — at logon, long-running heartbeat-stale flatten guard (was never registered)
$dmsName = "AssembledTradingAI-DMSDaemon"
$dmsAct  = New-ScheduledTaskAction -Execute $py -Argument "scripts\dms_daemon.py" -WorkingDirectory $repo
$dmsTrig = New-ScheduledTaskTrigger -AtLogOn
schtasks /delete /tn $dmsName /f 2>$null
Register-ScheduledTask -TaskName $dmsName -Action $dmsAct -Trigger $dmsTrig -Description "Dead-Man's Switch: heartbeat-stale auto-flatten guard"

Write-Host "Registered: $wdName (every 20m), $dmsName (at logon)."
