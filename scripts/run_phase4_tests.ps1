# scripts/run_phase4_tests.ps1
<#
.SYNOPSIS
    Thin wrapper around the central Python CLI to run Phase-4 test suite.

.DESCRIPTION
    This PowerShell script is a thin wrapper that calls:
        python scripts/cli.py run_phase4_tests [--verbose] [--durations N]

    The actual test execution logic lives in scripts/cli.py.

.PARAMETER Verbose
    Show detailed test output (maps to --verbose flag).

.PARAMETER Durations
    Show slowest tests (maps to --durations 10).

.EXAMPLE
    .\scripts\run_phase4_tests.ps1

.EXAMPLE
    .\scripts\run_phase4_tests.ps1 -Verbose -Durations
#>
param(
    [switch]$Verbose,
    [switch]$Durations
)

$ErrorActionPreference = "Stop"

# Get repo root
$RepoRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $RepoRoot

# Activate venv
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv"
    exit 1
}

# Build CLI command arguments
$CliArgs = @("scripts/cli.py", "run_phase4_tests")

if ($Verbose) {
    $CliArgs += "--verbose"
}

if ($Durations) {
    $CliArgs += "--durations"
    $CliArgs += "10"
}

# Run CLI
Write-Host "Running Phase-4 test suite via Python CLI..." -ForegroundColor Cyan
Write-Host "Command: python $($CliArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

& $VenvPython @CliArgs

$ExitCode = $LASTEXITCODE
if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "✅ Phase-4 tests passed!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Phase-4 tests failed (exit code: $ExitCode)" -ForegroundColor Red
}

exit $ExitCode

