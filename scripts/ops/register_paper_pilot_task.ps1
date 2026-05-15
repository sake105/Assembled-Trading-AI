# Register Windows Scheduled Task for autonomous paper-pilot daily cycle.
#
# Run ONCE in an elevated PowerShell (Run as Administrator):
#   .\scripts\ops\register_paper_pilot_task.ps1
#
# To unregister:
#   .\scripts\ops\register_paper_pilot_task.ps1 -Unregister
#
# What this creates:
# - A daily-recurring Windows Task that runs scripts/daily_paper_trading.bat
# - Trigger: 21:30 local time, Mon-Fri (= 15:30 ET, 30min before NYSE close)
# - Settings: wake computer, run regardless of login, auto-retry on missed
#
# Verification after registration:
#   Get-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot'
#
# Run task immediately (test):
#   Start-ScheduledTask -TaskName 'AssembledTradingAI-PaperPilot'

[CmdletBinding()]
param(
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'

$TaskName = 'AssembledTradingAI-PaperPilot'
$RepoRoot = 'F:\Python_Projekt\Aktiengerüst'
$BatchPath = Join-Path $RepoRoot 'scripts\daily_paper_trading.bat'

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

# Action: invoke the batch via cmd.exe
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$BatchPath`""

# Trigger: daily at 21:30 local time, weekdays only
# (Mon-Fri encoded via DaysOfWeek property after creation)
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 21:30

# Settings: robust to missed runs, can wake computer, no time-limit
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Principal: run as current user, only when logged on by default.
# For unattended ops (run even when logged off), pass -LogonType S4U
# and ensure the user has 'Log on as a batch job' rights.
$principalConfig = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description 'Daily paper-trading cycle for Assembled-Trading-AI (Mon-Fri 21:30 local = 15:30 ET, 30 min before NYSE close).' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principalConfig | Out-Null

Write-Host "[OK] Registered task '$TaskName'."
Write-Host ""
Write-Host "Verify with:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host ""
Write-Host "Test-run immediately:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "View last-run status:"
Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host ""
Write-Host "Logs land in: logs\scheduler\daily_paper_trading_<YYYYMMDD>.log"
Write-Host "Pilot manifest: output\pilot\pilot_manifest.json"
