# Register the Windows Scheduled Task for the OPS-01 Dead-Man's Switch daemon.
#
# This launches scripts/dms_daemon.py as a CONTINUOUS at-logon background process
# (not a timed check like the OPS-03 health watchdog). The daemon reads
# configs/policy.yaml and runs in SHADOW mode: it logs what it would flatten to
# output/ops/dms_audit.jsonl but does NOT activate the kill switch.
#
# Why shadow (operator must read before arming to market): the deployed pilot writes
# output/state/heartbeat.json only ONCE per weekday. A continuous fixed-timeout DMS
# cannot tell a healthy weekend (Fri->Mon = ~72h, no trading days) from a real stall,
# so market mode would activate the kill switch every weekend and block the pilot.
# Arming to market is blocked on OPS-02 (heartbeat-topology unification). See
# configs/policy.yaml (dead_man_switch block) and docs/runbooks/12_paper_entry_point.md.
#
# Run ONCE in an elevated PowerShell (Run as Administrator):
#   .\scripts\ops\register_dms_task.ps1
#
# To unregister:
#   .\scripts\ops\register_dms_task.ps1 -Unregister
#
# Verification after registration:
#   Get-ScheduledTask -TaskName 'AssembledTradingAI-DMS' | Format-List
#   Start-ScheduledTask -TaskName 'AssembledTradingAI-DMS'   # launches it now
#   Get-Content logs\scheduler\dms_daemon_*.log -Tail 20     # confirm it started

[CmdletBinding()]
param(
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'

$TaskName = 'AssembledTradingAI-DMS'
$RepoRoot = 'F:\Python_Projekt\Aktiengerüst'
$BatchPath = Join-Path $RepoRoot 'scripts\dms_daemon.bat'

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
# umlaut in 'Aktiengerüst' under Task Scheduler's non-interactive codepage and exits rc=3,
# the 2026-05-15 incident).
$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"& '$BatchPath'`"" `
    -WorkingDirectory $RepoRoot

# Trigger: at logon — the daemon is a long-running continuous monitor.
$trigger = New-ScheduledTaskTrigger -AtLogOn

# Settings: keep it alive. No execution time limit (continuous daemon); restart if it dies.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Principal: run as current user (the daemon needs the same env/paths as the pilot).
$principalConfig = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description 'OPS-01 Dead-Man''s Switch daemon: continuous passive heartbeat monitor (output\state\heartbeat.json). Runs in SHADOW mode per configs/policy.yaml — logs to output\ops\dms_audit.jsonl, does NOT touch the kill switch. Do NOT arm to market without OPS-02. Independent of the pilot task.' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principalConfig | Out-Null

Write-Host "[OK] Registered task '$TaskName' (at-logon, continuous, SHADOW mode)."
Write-Host ""
Write-Host "Verify with:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host ""
Write-Host "Launch it now (otherwise it starts at next logon):"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Get-Content logs\scheduler\dms_daemon_*.log -Tail 20"
Write-Host ""
Write-Host "NOTE: this is OBSERVE-ONLY (shadow). It will log to output\ops\dms_audit.jsonl"
Write-Host "what it WOULD flatten. Review that trail across real calendar weeks (incl."
Write-Host "weekends) before considering OPS-02 + a market-mode promotion."
