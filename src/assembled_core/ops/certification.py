"""Certification & Sign-Off Checklist for Go-Live.

M32: Integration Testing + Certification module.
Provides structured sign-off checks for system readiness:
- Data pipeline health
- Feature computation validity
- Signal generation consistency
- Portfolio optimization convergence
- Risk controls active
- E2E pipeline smoke test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Single certification check result."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CertificationReport:
    """Full certification report."""
    timestamp: str
    checks: list[CheckResult]
    all_passed: bool
    total_checks: int
    passed_count: int
    failed_count: int
    summary: str

    @property
    def pass_rate(self) -> float:
        return self.passed_count / max(self.total_checks, 1)


class CertificationRunner:
    """Runs a suite of certification checks.

    Usage:
        runner = CertificationRunner()
        runner.add_check("data_pipeline", lambda: check_data())
        runner.add_check("risk_controls", lambda: check_risk())
        report = runner.run()
    """

    def __init__(self) -> None:
        self._checks: list[tuple[str, Any]] = []

    def add_check(self, name: str, check_fn: Any) -> None:
        """Register a certification check.

        Args:
            name: Check identifier.
            check_fn: Callable returning (bool, str) or CheckResult.
        """
        self._checks.append((name, check_fn))

    def run(self) -> CertificationReport:
        """Execute all certification checks.

        Returns:
            CertificationReport with results.
        """
        results = []
        for name, fn in self._checks:
            try:
                result = fn()
                if isinstance(result, CheckResult):
                    results.append(result)
                elif isinstance(result, tuple) and len(result) >= 2:
                    results.append(CheckResult(
                        name=name,
                        passed=bool(result[0]),
                        message=str(result[1]),
                    ))
                elif isinstance(result, bool):
                    results.append(CheckResult(
                        name=name,
                        passed=result,
                        message="OK" if result else "FAILED",
                    ))
                else:
                    results.append(CheckResult(
                        name=name,
                        passed=False,
                        message=f"Unexpected return type: {type(result).__name__}",
                    ))
            except Exception as e:
                results.append(CheckResult(
                    name=name,
                    passed=False,
                    message=f"Exception: {e}",
                ))

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        report = CertificationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=results,
            all_passed=(failed == 0),
            total_checks=len(results),
            passed_count=passed,
            failed_count=failed,
            summary=f"{passed}/{len(results)} checks passed.",
        )

        logger.info("[Certification] %s", report.summary)
        return report


# ---------------------------------------------------------------------------
# Built-in certification checks
# ---------------------------------------------------------------------------

def check_imports_ok() -> CheckResult:
    """Verify core imports succeed."""
    errors = []
    modules = [
        "src.assembled_core.pipeline",
        "src.assembled_core.portfolio",
        "src.assembled_core.risk",
        "src.assembled_core.execution",
    ]
    for mod in modules:
        try:
            __import__(mod)
        except ImportError as e:
            errors.append(f"{mod}: {e}")

    return CheckResult(
        name="core_imports",
        passed=len(errors) == 0,
        message="All core imports OK" if not errors else f"Failed: {', '.join(errors)}",
    )


def check_numpy_scipy() -> CheckResult:
    """Verify numpy and scipy available."""
    try:
        import numpy as np
        from scipy.optimize import minimize
        return CheckResult(name="numpy_scipy", passed=True, message="OK")
    except ImportError as e:
        return CheckResult(name="numpy_scipy", passed=False, message=str(e))


def build_default_runner() -> CertificationRunner:
    """Create a CertificationRunner with built-in checks."""
    runner = CertificationRunner()
    runner.add_check("core_imports", check_imports_ok)
    runner.add_check("numpy_scipy", check_numpy_scipy)
    return runner


__all__ = [
    "CheckResult",
    "CertificationReport",
    "CertificationRunner",
    "build_default_runner",
    "check_imports_ok",
    "check_numpy_scipy",
]
