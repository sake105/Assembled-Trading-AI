"""Pre-Live Gate — 8 Mandatory Checks Before Going Live (M24 Task 24.3).

No strategy goes live without passing all 8 checks:
1. 30-day paper trading without critical failures
2. Reconciliation diff < 1% on 30/30 days
3. Max DD in paper < policy.max_dd * 1.5
4. CPCV P(Sharpe>0) > 0.60
5. Feature drift score < 0.30
6. Kill switch test passed
7. Pre-trade checks all active and configured
8. TCA model calibrated (estimated vs real < 2x)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GateCheckResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    value: float | str
    threshold: float | str
    detail: str = ""


@dataclass
class PreLiveGateResult:
    """Result of the full pre-live gate evaluation."""
    all_passed: bool
    checks: list[GateCheckResult]
    passed_count: int
    total_count: int
    blockers: list[str]

    @property
    def pass_rate(self) -> float:
        return self.passed_count / max(self.total_count, 1)


class PreLiveGate:
    """Pre-live gate evaluator with 8 mandatory checks."""

    def __init__(self, policy_max_dd: float = 0.20):
        """Initialize gate.

        Args:
            policy_max_dd: Policy maximum drawdown (fraction, e.g. 0.20 = 20%).
        """
        self.policy_max_dd = policy_max_dd

    def evaluate(
        self,
        paper_days: int = 0,
        paper_critical_failures: int = 0,
        reconciliation_pass_days: int = 0,
        reconciliation_total_days: int = 30,
        paper_max_dd: float = 0.0,
        cpcv_prob_positive_sharpe: float = 0.0,
        feature_drift_score: float = 1.0,
        kill_switch_tested: bool = False,
        pre_trade_checks_active: bool = False,
        tca_ratio: float = 999.0,
    ) -> PreLiveGateResult:
        """Run all 8 pre-live checks.

        Args:
            paper_days: Days of paper trading completed.
            paper_critical_failures: Number of critical failures in paper.
            reconciliation_pass_days: Days where recon diff < 1%.
            reconciliation_total_days: Total recon days.
            paper_max_dd: Maximum drawdown observed in paper (fraction).
            cpcv_prob_positive_sharpe: P(OOS Sharpe > 0) from CPCV.
            feature_drift_score: Feature drift score (0=no drift, 1=total drift).
            kill_switch_tested: Whether kill switch was manually tested.
            pre_trade_checks_active: Whether all pre-trade checks are configured.
            tca_ratio: Ratio of estimated to real transaction costs.

        Returns:
            PreLiveGateResult with all check results.
        """
        checks = []

        # Check 1: 30 days paper trading without critical failures
        c1 = GateCheckResult(
            name="paper_trading_duration",
            passed=paper_days >= 30 and paper_critical_failures == 0,
            value=f"{paper_days} days, {paper_critical_failures} failures",
            threshold=">=30 days, 0 critical failures",
            detail=f"Paper ran {paper_days} days with {paper_critical_failures} critical failures",
        )
        checks.append(c1)

        # Check 2: Reconciliation accuracy
        recon_rate = reconciliation_pass_days / max(reconciliation_total_days, 1)
        c2 = GateCheckResult(
            name="reconciliation_accuracy",
            passed=reconciliation_pass_days >= reconciliation_total_days and recon_rate >= 1.0,
            value=f"{reconciliation_pass_days}/{reconciliation_total_days}",
            threshold="30/30 days < 1% diff",
            detail=f"Recon passed {reconciliation_pass_days} of {reconciliation_total_days} days",
        )
        checks.append(c2)

        # Check 3: Paper max DD < policy * 1.5
        dd_limit = self.policy_max_dd * 1.5
        c3 = GateCheckResult(
            name="paper_max_drawdown",
            passed=abs(paper_max_dd) < dd_limit,
            value=f"{abs(paper_max_dd)*100:.1f}%",
            threshold=f"<{dd_limit*100:.1f}%",
            detail=f"Paper max DD {abs(paper_max_dd)*100:.1f}% vs limit {dd_limit*100:.1f}%",
        )
        checks.append(c3)

        # Check 4: CPCV P(Sharpe>0) > 0.60
        c4 = GateCheckResult(
            name="cpcv_confidence",
            passed=cpcv_prob_positive_sharpe > 0.60,
            value=f"{cpcv_prob_positive_sharpe:.2f}",
            threshold=">0.60",
            detail=f"CPCV P(Sharpe>0) = {cpcv_prob_positive_sharpe:.2f}",
        )
        checks.append(c4)

        # Check 5: Feature drift < 0.30
        c5 = GateCheckResult(
            name="feature_drift",
            passed=feature_drift_score < 0.30,
            value=f"{feature_drift_score:.2f}",
            threshold="<0.30",
            detail=f"Feature drift score = {feature_drift_score:.2f}",
        )
        checks.append(c5)

        # Check 6: Kill switch tested
        c6 = GateCheckResult(
            name="kill_switch_test",
            passed=kill_switch_tested,
            value="tested" if kill_switch_tested else "NOT tested",
            threshold="manual test passed",
        )
        checks.append(c6)

        # Check 7: Pre-trade checks active
        c7 = GateCheckResult(
            name="pre_trade_checks",
            passed=pre_trade_checks_active,
            value="active" if pre_trade_checks_active else "NOT active",
            threshold="all checks configured",
        )
        checks.append(c7)

        # Check 8: TCA calibrated
        c8 = GateCheckResult(
            name="tca_calibration",
            passed=tca_ratio < 2.0,
            value=f"{tca_ratio:.2f}x",
            threshold="<2.0x",
            detail=f"Estimated/Real cost ratio = {tca_ratio:.2f}",
        )
        checks.append(c8)

        # Aggregate
        passed_count = sum(1 for c in checks if c.passed)
        blockers = [c.name for c in checks if not c.passed]
        all_passed = len(blockers) == 0

        if all_passed:
            logger.info("[PreLiveGate] ALL 8 CHECKS PASSED — ready for live")
        else:
            logger.warning("[PreLiveGate] %d/%d passed — BLOCKED by: %s",
                           passed_count, len(checks), ", ".join(blockers))

        return PreLiveGateResult(
            all_passed=all_passed,
            checks=checks,
            passed_count=passed_count,
            total_count=len(checks),
            blockers=blockers,
        )


__all__ = [
    "GateCheckResult",
    "PreLiveGateResult",
    "PreLiveGate",
]
