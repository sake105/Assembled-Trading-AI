# start_pilot_v2.ps1 — Pilot v2 pre-flight startup script (Backlog Item 132)
#
# Checks performed before starting Assembled-Trading-AI Pilot v2:
#   1. Required ENV vars present (ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
#   2. Python ENV validator (env_validator.validate_env)
#   3. configs/policy.yaml exists
#   4. Smoke tests pass (pytest -m smoke)
#   5. Status summary + user confirmation before pilot starts
#
# feature_store.py wiring status (checked 2026-05-08):
#   File: src/assembled_core/data/feature_store.py — EXISTS
#   Callers in src/assembled_core/pipeline/: NONE (0 files import feature_store)
#   Callers in src/assembled_core/strategies/: NONE (0 files import feature_store)
#   Callers project-wide: src/assembled_core/intel/conviction_engine.py,
#                          src/assembled_core/signals/base.py
#   Verdict: feature_store is NOT wired into the main pipeline or strategy layer.
#            It is referenced only from conviction_engine (EDCL path) and signals/base.py.
#            The EDCL path is policy-gated (edcl_conviction_overlay.enabled=true in policy.yaml
#            but allow_in_backtest=false). Feature store data is read during EDCL enrichment
#            only — it is NOT a hard dependency for daily pilot cycles to function.
#
# Usage:
#   .\scripts\start_pilot_v2.ps1 [-SkipSmoke] [-Force]
#
# Exit codes:
#   0 = all checks passed, user confirmed, ready to start
#   1 = one or more checks failed (see [FAIL] lines)
#   2 = user declined to start

param(
    [switch]$SkipSmoke,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve project root (scripts/ is one level below root)
$ROOT = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$PASS = "[PASS]"
$FAIL = "[FAIL]"
$WARN = "[WARN]"
$SKIP = "[SKIP]"

$failures = 0

function Write-Check {
    param([string]$Status, [string]$Message)
    $color = switch ($Status) {
        "[PASS]" { "Green"  }
        "[FAIL]" { "Red"    }
        "[WARN]" { "Yellow" }
        "[SKIP]" { "Cyan"   }
        default  { "White"  }
    }
    Write-Host "$Status $Message" -ForegroundColor $color
}

function Load-DotEnv {
    # Load .env into current process environment (non-destructive — skips already-set vars)
    $envFile = Join-Path $ROOT ".env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match "^\s*#" -or $line -notmatch "=") { continue }
            $parts = $line -split "=", 2
            $k = $parts[0].Trim()
            $v = $parts[1].Trim().Trim('"').Trim("'")
            if (-not [System.Environment]::GetEnvironmentVariable($k)) {
                [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
            }
        }
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Assembled-Trading-AI — Pilot v2 Start  " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Load .env so checks below can find values even if shell didn't export them
Load-DotEnv

# ─── Check 1: Required ENV vars ───────────────────────────────────────────────
Write-Host "--- Check 1: Required ENV vars ---"
$required = @("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL")
$missing = @()
foreach ($key in $required) {
    $val = [System.Environment]::GetEnvironmentVariable($key)
    if (-not $val) { $missing += $key }
}

if ($missing.Count -gt 0) {
    Write-Check $FAIL "Missing required ENV vars: $($missing -join ', ')"
    Write-Host "  Fix: add these to your .env file or export them in the shell." -ForegroundColor Red
    $failures++
} else {
    Write-Check $PASS "ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL all present"
}

# ─── Check 2: Python ENV validator ────────────────────────────────────────────
Write-Host ""
Write-Host "--- Check 2: ENV validator (src.assembled_core.config.env_validator) ---"
$validatorScript = @"
import sys, os
from pathlib import Path

# Load .env before importing validator so it sees the vars
env_file = Path(r'$ROOT') / '.env'
if env_file.exists():
    for line in env_file.read_text(encoding='utf-8').splitlines():
        if line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(Path(r'$ROOT') / 'src'))
try:
    from assembled_core.config.env_validator import validate_env
    validate_env(warn_missing_optional=False)
    print('OK: all required ENV vars validated')
    sys.exit(0)
except RuntimeError as e:
    print(f'FAIL: {e}')
    sys.exit(1)
except ImportError as e:
    print(f'SKIP: env_validator not importable — {e}')
    sys.exit(2)
"@

$validatorResult = & $Python -c $validatorScript 2>&1
$validatorCode = $LASTEXITCODE
if ($validatorCode -eq 0) {
    Write-Check $PASS "env_validator: $validatorResult"
} elseif ($validatorCode -eq 2) {
    Write-Check $WARN "env_validator: $validatorResult"
} else {
    Write-Check $FAIL "env_validator: $validatorResult"
    $failures++
}

# ─── Check 3: configs/policy.yaml exists ─────────────────────────────────────
Write-Host ""
Write-Host "--- Check 3: configs/policy.yaml ---"
$policyPath = Join-Path $ROOT "configs\policy.yaml"
if (Test-Path $policyPath) {
    $policyContent = Get-Content $policyPath -Raw
    # Quick sanity: must have risk_limits and execution_policy sections
    if ($policyContent -match "risk_limits:" -and $policyContent -match "execution_policy:") {
        Write-Check $PASS "configs/policy.yaml exists and has risk_limits + execution_policy sections"
    } else {
        Write-Check $WARN "configs/policy.yaml exists but may be incomplete (missing risk_limits or execution_policy)"
    }
} else {
    Write-Check $FAIL "configs/policy.yaml not found at: $policyPath"
    $failures++
}

# ─── Check 4: Smoke tests ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "--- Check 4: Smoke tests (pytest -m smoke) ---"
if ($SkipSmoke) {
    Write-Check $SKIP "Smoke tests skipped (-SkipSmoke flag)"
} else {
    $smokeOut = & $Python -m pytest (Join-Path $ROOT "tests") -m "smoke" -q --tb=short 2>&1
    $smokeCode = $LASTEXITCODE
    # Extract summary line (e.g. "36 passed in 4.2s")
    $summaryLine = ($smokeOut | Select-String -Pattern "passed|failed|error" | Select-Object -Last 1)
    if ($smokeCode -eq 0) {
        Write-Check $PASS "Smoke tests: $summaryLine"
    } else {
        Write-Check $FAIL "Smoke tests failed (exit $smokeCode) — $summaryLine"
        Write-Host "--- smoke test output ---" -ForegroundColor DarkGray
        $smokeOut | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
        Write-Host "--- end smoke test output ---" -ForegroundColor DarkGray
        $failures++
    }
}

# ─── Status Summary ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Pre-flight Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Show key policy values for operator awareness
if (Test-Path $policyPath) {
    $leverageMatch = [regex]::Match($policyContent, "leverage_allowed:\s*(\S+)")
    $modeMatch     = [regex]::Match($policyContent, "mode_default:\s*""?(\S+?)""?")
    $grossMatch    = [regex]::Match($policyContent, "max_gross_exposure:\s*([\d.]+)")
    $leverage = if ($leverageMatch.Success) { $leverageMatch.Groups[1].Value } else { "?" }
    $mode     = if ($modeMatch.Success)     { $modeMatch.Groups[1].Value }     else { "?" }
    $gross    = if ($grossMatch.Success)    { $grossMatch.Groups[1].Value }     else { "?" }
    Write-Host "  Policy: mode=$mode  leverage=$leverage  max_gross_exposure=$gross" -ForegroundColor White
}

if ($failures -eq 0) {
    Write-Host "  Status: ALL CHECKS PASSED" -ForegroundColor Green
} else {
    Write-Host "  Status: $failures CHECK(S) FAILED" -ForegroundColor Red
    if (-not $Force) {
        Write-Host ""
        Write-Host "  Resolve [FAIL] items above, then re-run this script." -ForegroundColor Red
        Write-Host "  Use -Force to bypass (not recommended)." -ForegroundColor Yellow
        Write-Host "==========================================" -ForegroundColor Cyan
        exit 1
    } else {
        Write-Host "  -Force set: proceeding despite failures. USE WITH CAUTION." -ForegroundColor Yellow
    }
}

# ─── User Confirmation ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Ready to start Pilot v2." -ForegroundColor Green
Write-Host "  This will run: python scripts/run_paper_pilot.py --run-day"
Write-Host ""
$confirm = Read-Host "  Confirm start? [y/N]"
if ($confirm -notin @("y", "Y", "yes", "YES")) {
    Write-Host "  Aborted by user." -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Cyan
    exit 2
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Starting Pilot v2 cycle..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

& $Python (Join-Path $ROOT "scripts\run_paper_pilot.py") --run-day
exit $LASTEXITCODE
