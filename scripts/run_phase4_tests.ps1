# scripts/run_phase4_tests.ps1
"""Run Phase-4 test suite (TA, QA-Metrics, Gates, Backtest, Reports, Pipelines).

This script runs the fast, reliable Phase-4 regression tests (~17s, 110 tests).

Usage:
    .\scripts\run_phase4_tests.ps1

Options:
    -Verbose    Show detailed test output
    -Durations  Show slowest tests
"""
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

# Build pytest command
$PytestArgs = @(
    "-m", "phase4"
    "-q"
    "--maxfail=1"
    "--tb=short"
)

if ($Verbose) {
    $PytestArgs = $PytestArgs | Where-Object { $_ -ne "-q" }
    $PytestArgs += "-v"
}

if ($Durations) {
    $PytestArgs += "--durations=10"
}

# Run tests
Write-Host "Running Phase-4 test suite..." -ForegroundColor Cyan
Write-Host "Command: pytest $($PytestArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

& $VenvPython -m pytest @PytestArgs

$ExitCode = $LASTEXITCODE
if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "✅ Phase-4 tests passed!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Phase-4 tests failed (exit code: $ExitCode)" -ForegroundColor Red
}

exit $ExitCode

