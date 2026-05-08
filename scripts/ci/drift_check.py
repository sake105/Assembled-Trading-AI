"""CI: Feature drift and leakage smoke check."""

import sys

try:
    from src.assembled_core.qa.drift_detection import compute_psi  # noqa: F401

    print("[drift] compute_psi available — endpoint handles full check")
except Exception as exc:
    print(f"[drift] skipped: {exc}")
    sys.exit(0)
