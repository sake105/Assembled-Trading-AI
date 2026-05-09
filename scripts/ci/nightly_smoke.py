"""Nightly smoke-test — verifies core pipeline imports and CLI."""

import subprocess
import sys
from pathlib import Path

MODULES = [
    "assembled_core.signals.meta_model",
    "assembled_core.accounting.ledger",
    "assembled_core.config.env_validator",
    "assembled_core.features.seasonal_features",
]

failed = []
for m in MODULES:
    try:
        __import__(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  ERR {m}: {e}")
        failed.append(m)

if failed:
    sys.exit(1)

print("All core imports OK")

cli = Path("scripts/cli.py")
if cli.exists():
    result = subprocess.run(
        [sys.executable, str(cli), "--help"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("ERR: cli.py --help failed")
        sys.exit(1)
    print("CLI --help OK")

print("Nightly smoke-test PASSED")
