# Register Windows Scheduled Task for the OPS-03 scheduler-health watchdog.
#
# This is an INDEPENDENT task from the paper-pilot task on purpose: if the pilot
# task stops firing (the 2026-04-10 silent-stall mode), this watchdog still runs
# and detects the stale/missing heartbeat. Do NOT chain it off the pilot.
#
# Run ONCE in an elevated PowerShell (Run as Administrator):
#   .\scripts\ops\register_health_check_task.ps1
#
# To unregister:
#   .\scripts\ops\register_health_check_task.ps1 -Unregister
#
# What this creates:
# - A weekday Windows Task that runs scripts/check_scheduler_health.bat
# - Trigger: 22:30 local time, Mon-Fri (~1h after the 21:30 pilot window)
# - The detector reads output\state\heartbeat.json and, on stale/missing,
#   posts to DISCORD_WEBHOOK (with SMTP email fallback).
#
# Verification after registration:
#   Get-ScheduledTask -TaskName 'AssembledTradingAI-HealthCheck'

[CmdletBinding()]
param(
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'

$TaskName = 'AssembledTradingAI-HealthCheck'
$RepoRoot = 'F:\Python_Projekt\Aktiengerüst'
$BatchPath = Join-Path $RepoRoot 'scripts\check_scheduler_health.bat'

# Pre-check: must run elevated for system tasks
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator (right-click PowerShell -> 'Run as Administrator')."
    exit 1
}

if ($Unregister) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[OK] Unregistered task '$TaskName'."
    } else {
        Write-Host "[INFO] Task '$TaskName' was not registered — nothing to do."
    }
    exit 0
}

if (-not (Test-Path $BatchPath)) {
    Write-Error "Batch file not found: $BatchPath"
    exit 1
}

# Remove existing task if present (idempotent re-registration)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[INFO] Existing task '$TaskName' found — replacing."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Action: invoke the batch via powershell.exe (handles UTF-16 args; cmd.exe garbles the
# umlaut in 'Aktiengerüst' under Task Scheduler's non-interactive codepage and exits rc=3).
$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"& '$BatchPath'`"" `
    -WorkingDirectory $RepoRoot

# Trigger: weekdays at 22:30 local (~1h after the 21:30 pilot window)
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 22:30

# Settings: robust to missed runs, can wake computer, short time limit
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 2)

# Principal: run as current user
$principalConfig = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description 'OPS-03 watchdog: checks the daily-pilot heartbeat (output\state\heartbeat.json) for staleness Mon-Fri 22:30 local and alerts on Discord/email. Independent of the pilot task.' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principalConfig | Out-Null

Write-Host "[OK] Registered task '$TaskName'."
Write-Host ""
Write-Host "Verify with:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host ""
Write-Host "Test-run immediately (writes logs\scheduler\health_check_<YYYYMMDD>.log):"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "NOTE: set DISCORD_WEBHOOK (and optionally SMTP_* / ALERT_EMAIL_TO) in the"
Write-Host "task's environment for alerts to actually deliver."
