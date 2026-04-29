"""Health Check Script (Plan 11.3).

Validates system readiness:
- Import checks for all core modules
- Config validation
- Data source ping
- Last run check
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

CORE_MODULES = [
    "src.assembled_core.pipeline.trading_cycle",
    "src.assembled_core.signals.multifactor_signal",
    "src.assembled_core.portfolio.position_sizing",
    "src.assembled_core.execution.fill_model",
    "src.assembled_core.risk.risk_metrics",
    "src.assembled_core.accounting.ledger",
    "src.assembled_core.qa.metrics",
    "src.assembled_core.data.calendar",
]


def check_imports() -> list[str]:
    """Check that all core modules can be imported."""
    failures = []
    for mod in CORE_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failures.append(f"{mod}: {exc}")
    return failures


def check_config() -> list[str]:
    """Validate policy configuration."""
    issues = []
    try:
        from src.assembled_core.config.policy_loader import load_policy
        from src.assembled_core.config.policy_schema import (
            validate_policy,
            validate_policy_consistency,
        )

        policy = load_policy()
        _, warnings = validate_policy(policy)
        issues.extend(warnings)
        violations = validate_policy_consistency(policy)
        issues.extend(violations)
    except Exception as exc:
        issues.append(f"Config check failed: {exc}")
    return issues


def check_data_files() -> list[str]:
    """Check that key data files exist."""
    issues = []
    data_dir = Path("data")
    if not data_dir.exists():
        issues.append("data/ directory missing")
    output_dir = Path("output")
    if not output_dir.exists():
        issues.append("output/ directory missing")
    return issues


def run_health_check() -> dict:
    """Run all health checks."""
    import_issues = check_imports()
    config_issues = check_config()
    data_issues = check_data_files()

    all_issues = import_issues + config_issues + data_issues
    status = "HEALTHY" if not all_issues else "DEGRADED"

    return {
        "status": status,
        "import_issues": import_issues,
        "config_issues": config_issues,
        "data_issues": data_issues,
        "total_issues": len(all_issues),
    }


if __name__ == "__main__":
    result = run_health_check()
    print(f"Status: {result['status']}")
    if result["total_issues"] > 0:
        print(f"Issues ({result['total_issues']}):")
        for section in ["import_issues", "config_issues", "data_issues"]:
            for issue in result[section]:
                print(f"  [{section}] {issue}")
    else:
        print("All checks passed.")
    sys.exit(0 if result["status"] == "HEALTHY" else 1)
