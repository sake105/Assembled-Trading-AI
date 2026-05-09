<#
  Nightly smoke-test: verifies core pipeline can be imported and CLI responds.
  Replaces the original Phase-10 full-run script (deprecated after Phase 11+).

  Parameters kept for backward compatibility with nightly-runall.yml:
    -Freq, -StartCapital, -CommissionBps, -SpreadW, -ImpactW
#>
param(
    [string]$Freq          = '5min',
    [int]$StartCapital     = 10000,
    [double]$CommissionBps = 0.5,
    [double]$SpreadW       = 1,
    [double]$ImpactW       = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Stamp([string]$msg) {
    $ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Write-Host "[$ts] $msg"
}

Stamp "Nightly smoke-test starting (Freq=$Freq, Capital=$StartCapital)"

# 1 — Core import check
Stamp "Checking core package imports..."
python -c "
import sys
modules = [
    'assembled_core.signals.meta_model',
    'assembled_core.accounting.ledger',
    'assembled_core.config.env_validator',
    'assembled_core.features.seasonal_features',
]
failed = []
for m in modules:
    try:
        __import__(m)
        print(f'  OK  {m}')
    except Exception as e:
        print(f'  ERR {m}: {e}')
        failed.append(m)
if failed:
    sys.exit(1)
print('All core imports OK')
"
if ($LASTEXITCODE -ne 0) { throw "Core import check failed" }

# 2 — CLI help (no env vars required)
Stamp "Checking CLI help..."
python scripts/cli.py --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "CLI --help failed" }
Stamp "CLI help OK"

Stamp "Nightly smoke-test PASSED"
