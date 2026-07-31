"""QA gates for performance metrics evaluation.

This module provides structured QA gates that evaluate performance metrics
and determine if a backtest/portfolio passes quality checks.

QA Gates:
- Sharpe ratio threshold (out-of-sample performance)
- Maximum drawdown limit
- Turnover threshold
- CAGR threshold
- Volatility limit
- Hit rate threshold (if trades available)
- Profit factor threshold (if trades available)
- Leakage detection (only checks when the caller supplies a feature frame;
  otherwise reported as ``details["skipped"]`` — see ``check_leakage``)

Each gate returns a structured result (OK, WARNING, BLOCK) with reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
from src.assembled_core.qa.metrics import PerformanceMetrics


class QAResult(str, Enum):
    """QA gate result status."""

    OK = "ok"  # Gate passed, no issues
    WARNING = "warning"  # Gate passed but with concerns
    BLOCK = "block"  # Gate failed, should block deployment/production


@dataclass
class QAGateResult:
    """Result of a single QA gate evaluation.

    Attributes:
        gate_name: Name of the gate (e.g., "sharpe_ratio", "max_drawdown")
        result: QAResult (OK, WARNING, BLOCK)
        reason: Human-readable reason for the result
        details: Additional details (e.g., actual value, threshold, metric)
    """

    gate_name: str
    result: QAResult
    reason: str
    details: dict[str, float | str | None] | None = None


@dataclass
class QAGatesSummary:
    """Summary of all QA gate evaluations.

    Attributes:
        overall_result: Overall result (worst case of all gates)
        passed_gates: Number of gates that actually passed (OK and CHECKED).
            Gates that returned OK only because they were skipped (no input
            supplied, ``details["skipped"] is True``) are EXCLUDED — a
            not-checked gate must never show up as green aggregate evidence
            (E-066). Those are counted in ``skipped_gates`` instead.
        warning_gates: Number of gates with warnings
        blocked_gates: Number of gates that blocked
        gate_results: List of individual gate results
        skipped_gates: Number of gates that were NOT checked (OK by
            fail-open, not by evidence). Trailing default so existing
            constructions stay valid.
    """

    overall_result: QAResult
    passed_gates: int
    warning_gates: int
    blocked_gates: int
    gate_results: list[QAGateResult]
    skipped_gates: int = 0


def check_sharpe_ratio(
    metrics: PerformanceMetrics, min_sharpe: float = 1.0, warning_sharpe: float = 0.5
) -> QAGateResult:
    """Check if Sharpe ratio meets quality threshold.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        min_sharpe: Minimum Sharpe ratio to pass (default: 1.0)
        warning_sharpe: Sharpe ratio below which to issue warning (default: 0.5)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK
    """
    gate_name = "sharpe_ratio"

    if metrics.sharpe_ratio is None:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="Sharpe ratio cannot be computed (insufficient data or zero volatility)",
            details={
                "sharpe_ratio": None,
                "min_sharpe": min_sharpe,
                "warning_sharpe": warning_sharpe,
            },
        )

    sharpe = metrics.sharpe_ratio

    if sharpe < warning_sharpe:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Sharpe ratio {sharpe:.4f} is below warning threshold {warning_sharpe:.2f}",
            details={
                "sharpe_ratio": sharpe,
                "min_sharpe": min_sharpe,
                "warning_sharpe": warning_sharpe,
            },
        )
    elif sharpe < min_sharpe:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Sharpe ratio {sharpe:.4f} is below minimum threshold {min_sharpe:.2f}",
            details={
                "sharpe_ratio": sharpe,
                "min_sharpe": min_sharpe,
                "warning_sharpe": warning_sharpe,
            },
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Sharpe ratio {sharpe:.4f} meets minimum threshold {min_sharpe:.2f}",
            details={
                "sharpe_ratio": sharpe,
                "min_sharpe": min_sharpe,
                "warning_sharpe": warning_sharpe,
            },
        )


def check_max_drawdown(
    metrics: PerformanceMetrics,
    max_dd_pct_limit: float = -20.0,
    warning_dd_pct: float = -15.0,
) -> QAGateResult:
    """Check if maximum drawdown is within acceptable limits.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        max_dd_pct_limit: Maximum drawdown percentage to block (default: -20.0%)
        warning_dd_pct: Drawdown percentage to issue warning (default: -15.0%)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK
    """
    gate_name = "max_drawdown"

    # max_drawdown_pct is negative, so we compare with negative limits
    max_dd = metrics.max_drawdown_pct

    if max_dd is None:
        # Mirror the sharpe gate: an incomputable drawdown must DEGRADE, not crash
        # evaluate_all_gates with `None < float` TypeError (Diagnostik A3).
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="Maximum drawdown cannot be computed (insufficient data)",
            details={
                "max_drawdown_pct": None,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct,
            },
        )

    if max_dd < max_dd_pct_limit:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Maximum drawdown {max_dd:.2f}% exceeds limit {max_dd_pct_limit:.2f}%",
            details={
                "max_drawdown_pct": max_dd,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct,
            },
        )
    elif max_dd < warning_dd_pct:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Maximum drawdown {max_dd:.2f}% exceeds warning threshold {warning_dd_pct:.2f}%",
            details={
                "max_drawdown_pct": max_dd,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct,
            },
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Maximum drawdown {max_dd:.2f}% is within acceptable limits",
            details={
                "max_drawdown_pct": max_dd,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct,
            },
        )


def check_turnover(
    metrics: PerformanceMetrics,
    max_turnover: float = 50.0,
    warning_turnover: float = 30.0,
) -> QAGateResult:
    """Check if portfolio turnover is within acceptable limits.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        max_turnover: Maximum annualized turnover to allow (default: 50.0x)
        warning_turnover: Turnover above which to issue warning (default: 30.0x)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK (or WARNING if no trades)
    """
    gate_name = "turnover"

    if metrics.turnover is None:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="Turnover cannot be computed (no trades provided)",
            details={
                "turnover": None,
                "max_turnover": max_turnover,
                "warning_turnover": warning_turnover,
            },
        )

    turnover = metrics.turnover

    if turnover > max_turnover:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Turnover {turnover:.2f}x exceeds maximum limit {max_turnover:.2f}x",
            details={
                "turnover": turnover,
                "max_turnover": max_turnover,
                "warning_turnover": warning_turnover,
            },
        )
    elif turnover > warning_turnover:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Turnover {turnover:.2f}x exceeds warning threshold {warning_turnover:.2f}x",
            details={
                "turnover": turnover,
                "max_turnover": max_turnover,
                "warning_turnover": warning_turnover,
            },
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Turnover {turnover:.2f}x is within acceptable limits",
            details={
                "turnover": turnover,
                "max_turnover": max_turnover,
                "warning_turnover": warning_turnover,
            },
        )


def check_cagr(
    metrics: PerformanceMetrics, min_cagr: float = 0.05, warning_cagr: float = 0.0
) -> QAGateResult:
    """Check if CAGR meets minimum threshold.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        min_cagr: Minimum CAGR to pass (default: 0.05 = 5%)
        warning_cagr: CAGR below which to issue warning (default: 0.0 = 0%)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK
    """
    gate_name = "cagr"

    if metrics.cagr is None:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="CAGR cannot be computed (less than 1 year of data)",
            details={"cagr": None, "min_cagr": min_cagr, "warning_cagr": warning_cagr},
        )

    cagr = metrics.cagr

    if cagr < warning_cagr:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"CAGR {cagr:.2%} is below warning threshold {warning_cagr:.2%}",
            details={"cagr": cagr, "min_cagr": min_cagr, "warning_cagr": warning_cagr},
        )
    elif cagr < min_cagr:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"CAGR {cagr:.2%} is below minimum threshold {min_cagr:.2%}",
            details={"cagr": cagr, "min_cagr": min_cagr, "warning_cagr": warning_cagr},
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"CAGR {cagr:.2%} meets minimum threshold {min_cagr:.2%}",
            details={"cagr": cagr, "min_cagr": min_cagr, "warning_cagr": warning_cagr},
        )


def check_volatility(
    metrics: PerformanceMetrics,
    max_volatility: float = 0.30,
    warning_volatility: float = 0.25,
) -> QAGateResult:
    """Check if volatility is within acceptable limits.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        max_volatility: Maximum annualized volatility to allow (default: 0.30 = 30%)
        warning_volatility: Volatility above which to issue warning (default: 0.25 = 25%)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK
    """
    gate_name = "volatility"

    if metrics.volatility is None:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="Volatility cannot be computed (insufficient data)",
            details={
                "volatility": None,
                "max_volatility": max_volatility,
                "warning_volatility": warning_volatility,
            },
        )

    volatility = metrics.volatility

    if volatility > max_volatility:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Volatility {volatility:.2%} exceeds maximum limit {max_volatility:.2%}",
            details={
                "volatility": volatility,
                "max_volatility": max_volatility,
                "warning_volatility": warning_volatility,
            },
        )
    elif volatility > warning_volatility:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Volatility {volatility:.2%} exceeds warning threshold {warning_volatility:.2%}",
            details={
                "volatility": volatility,
                "max_volatility": max_volatility,
                "warning_volatility": warning_volatility,
            },
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Volatility {volatility:.2%} is within acceptable limits",
            details={
                "volatility": volatility,
                "max_volatility": max_volatility,
                "warning_volatility": warning_volatility,
            },
        )


def check_hit_rate(
    metrics: PerformanceMetrics,
    min_hit_rate: float = 0.50,
    warning_hit_rate: float = 0.40,
) -> QAGateResult:
    """Check if hit rate (win rate) meets minimum threshold.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        min_hit_rate: Minimum hit rate to pass (default: 0.50 = 50%)
        warning_hit_rate: Hit rate below which to issue warning (default: 0.40 = 40%)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK (or WARNING if no trades)
    """
    gate_name = "hit_rate"

    if metrics.hit_rate is None:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="Hit rate cannot be computed (no trades or position tracking not available)",
            details={
                "hit_rate": None,
                "min_hit_rate": min_hit_rate,
                "warning_hit_rate": warning_hit_rate,
            },
        )

    hit_rate = metrics.hit_rate

    if hit_rate < warning_hit_rate:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Hit rate {hit_rate:.2%} is below warning threshold {warning_hit_rate:.2%}",
            details={
                "hit_rate": hit_rate,
                "min_hit_rate": min_hit_rate,
                "warning_hit_rate": warning_hit_rate,
            },
        )
    elif hit_rate < min_hit_rate:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Hit rate {hit_rate:.2%} is below minimum threshold {min_hit_rate:.2%}",
            details={
                "hit_rate": hit_rate,
                "min_hit_rate": min_hit_rate,
                "warning_hit_rate": warning_hit_rate,
            },
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Hit rate {hit_rate:.2%} meets minimum threshold {min_hit_rate:.2%}",
            details={
                "hit_rate": hit_rate,
                "min_hit_rate": min_hit_rate,
                "warning_hit_rate": warning_hit_rate,
            },
        )


def check_profit_factor(
    metrics: PerformanceMetrics,
    min_profit_factor: float = 1.5,
    warning_profit_factor: float = 1.2,
) -> QAGateResult:
    """Check if profit factor meets minimum threshold.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        min_profit_factor: Minimum profit factor to pass (default: 1.5)
        warning_profit_factor: Profit factor below which to issue warning (default: 1.2)

    Returns:
        QAGateResult with OK, WARNING, or BLOCK (or WARNING if no trades)
    """
    gate_name = "profit_factor"

    if metrics.profit_factor is None:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason="Profit factor cannot be computed (no trades or position tracking not available)",
            details={
                "profit_factor": None,
                "min_profit_factor": min_profit_factor,
                "warning_profit_factor": warning_profit_factor,
            },
        )

    profit_factor = metrics.profit_factor

    if profit_factor < warning_profit_factor:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Profit factor {profit_factor:.2f} is below warning threshold {warning_profit_factor:.2f}",
            details={
                "profit_factor": profit_factor,
                "min_profit_factor": min_profit_factor,
                "warning_profit_factor": warning_profit_factor,
            },
        )
    elif profit_factor < min_profit_factor:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Profit factor {profit_factor:.2f} is below minimum threshold {min_profit_factor:.2f}",
            details={
                "profit_factor": profit_factor,
                "min_profit_factor": min_profit_factor,
                "warning_profit_factor": warning_profit_factor,
            },
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Profit factor {profit_factor:.2f} meets minimum threshold {min_profit_factor:.2f}",
            details={
                "profit_factor": profit_factor,
                "min_profit_factor": min_profit_factor,
                "warning_profit_factor": warning_profit_factor,
            },
        )


def check_leakage(
    feature_df: pd.DataFrame | None = None,
    feature_col: str = "feature",
    disclosure_col: str = "disclosure_date",
    timestamp_col: str = "timestamp",
) -> QAGateResult:
    """Check for look-ahead bias / data leakage.

    HONESTY NOTE (W4/P5, 2026-07-22 GESAMTBEWERTUNG; updated 2026-08-01):
    earlier docs called this a "mandatory gate" — it is NOT, and it still is
    not. Since 2026-08-01 it IS part of ``evaluate_all_gates`` (E-059
    follow-up), so its state is at least VISIBLE in every gate summary
    instead of silently absent. But visibility is not enforcement: the gate
    only checks anything when the caller passes ``feature_df``; with
    ``feature_df=None`` it returns OK (fail-open by design, because "no
    altdata features" is a legitimate state — but that also means forgetting
    to pass the frame still passes). No production caller supplies a frame
    today; consumers MUST treat ``details["skipped"] is True`` as "NOT
    checked", never as "no leakage".

    If feature_df is provided, validates row-wise that feature values are
    zero (or NaN, treated as zero) before their disclosure date: any row
    where ``timestamp_col`` < ``disclosure_col`` must not carry a non-zero
    value in ``feature_col``.  Rows with a missing (NaT) disclosure date and
    a non-zero feature value are treated as violations (fail-closed: a value
    without a known disclosure date cannot be proven PIT-safe).  If
    feature_df is not provided, returns OK with ``details["skipped"]`` —
    that means "no frame was handed in", NOT "this backtest has no altdata
    features".

    Note: this check is intentionally implemented inline.  The helper
    ``qa.leakage_tests.assert_feature_zero_before_disclosure`` is a
    re-computation harness (prices + events + feature_fn at two as_of
    points) and cannot validate a precomputed flat feature frame, which is
    all this gate receives.

    Args:
        feature_df: DataFrame with feature values and disclosure dates.
        feature_col: Column containing feature values.
        disclosure_col: Column containing disclosure dates.
        timestamp_col: Column containing observation timestamps.

    Returns:
        QAGateResult: BLOCK if leakage is detected or the frame cannot be
        validated (missing columns, unparseable dates), OK otherwise.
    """
    gate_name = "leakage_detection"

    if feature_df is None or feature_df.empty:
        # The reason string is the ONLY honesty anchor that reaches the
        # markdown report and the backtest log — both render `reason` but
        # not `details`. It must therefore say "not checked", never imply
        # "clean" (Stage-1 MAJOR-1, 2026-08-01). skip_kind distinguishes
        # "caller passed nothing" from "caller passed an empty frame"
        # (e.g. a failed altdata load), which are NOT the same finding.
        skip_kind = "no_frame" if feature_df is None else "empty_frame"
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=(
                "leakage NOT CHECKED — "
                + (
                    "no feature_df supplied by caller"
                    if skip_kind == "no_frame"
                    else "caller supplied an EMPTY feature_df"
                )
                + " (absence of evidence, not 'clean')"
            ),
            details={"skipped": True, "skip_kind": skip_kind},
        )

    missing = [
        col
        for col in (feature_col, disclosure_col, timestamp_col)
        if col not in feature_df.columns
    ]
    if missing:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=(
                f"Leakage check impossible: feature_df is missing required "
                f"columns {missing}"
            ),
            details={
                "missing_columns": ", ".join(missing),
                "available_columns": ", ".join(str(c) for c in feature_df.columns),
            },
        )

    try:
        timestamps = pd.to_datetime(feature_df[timestamp_col], utc=True)
        disclosures = pd.to_datetime(feature_df[disclosure_col], utc=True)
    except (ValueError, TypeError) as exc:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Leakage check impossible: unparseable date column(s): {exc}",
            details={"error": str(exc)},
        )

    # NaN/NA feature values count as zero (consistent with the
    # assert_feature_zero_before_disclosure helper semantics).
    non_zero = feature_df[feature_col].fillna(0) != 0
    # Fail-closed on BOTH unknown-time axes (F-senior-6): a non-zero value
    # with an unknown disclosure time OR an unknown observation time cannot
    # be proven point-in-time-safe and counts as a violation.
    pre_disclosure = (timestamps < disclosures) | disclosures.isna() | timestamps.isna()
    violation_mask = non_zero & pre_disclosure
    n_violations = int(violation_mask.sum())

    if n_violations > 0:
        # details values are typed float|str|None -> serialize sample as JSON
        sample = json.dumps(
            feature_df.loc[violation_mask, [timestamp_col, disclosure_col, feature_col]]
            .head(5)
            .astype(str)
            .to_dict(orient="records")
        )
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=(
                f"LEAKAGE DETECTED: {n_violations} row(s) have non-zero "
                f"'{feature_col}' before their disclosure date"
            ),
            details={
                "violations": n_violations,
                "rows_checked": len(feature_df),
                "sample_violations": sample,
            },
        )

    return QAGateResult(
        gate_name=gate_name,
        result=QAResult.OK,
        reason="No look-ahead leakage detected in altdata features",
        details={"rows_checked": len(feature_df)},
    )


def evaluate_all_gates(
    metrics: PerformanceMetrics,
    gate_config: dict[str, dict[str, float]] | None = None,
    *,
    feature_df: pd.DataFrame | None = None,
    leakage_feature_col: str = "feature",
    leakage_disclosure_col: str = "disclosure_date",
    leakage_timestamp_col: str = "timestamp",
) -> QAGatesSummary:
    """Evaluate all QA gates and return summary.

    Leakage gate (E-059 follow-up, 2026-08-01): ``check_leakage`` is part of
    the summary so that its state is visible in QA artifacts. Without
    ``feature_df`` it contributes an OK result carrying
    ``details["skipped"] = True`` — that means "NOT checked", not "clean".
    Passing a frame turns it into a real fail-closed PIT check of EXACTLY
    ONE feature column (a multi-factor store needs one call per column).

    COUNT HONESTY (E-066): a skipped gate is NOT counted in ``passed_gates``
    — it lands in ``skipped_gates``. So ``passed_gates`` stays at 7 for the
    metric gates, exactly as before this gate was added; the aggregate never
    turns green for a check that did not run. Residual gap: the counts in
    ``pipeline/orchestrator.py:326-328`` and its BLOCK log line
    (``:1002-1011``) are RE-derived from ``gate_results`` and still count the
    skipped gate as OK. That file is a protected path — fixing it is a
    separate, explicitly scoped step.

    HALT FOOTGUN for the first real caller (Stage-1 MAJOR-3): the default
    column names are generic placeholders. A column-name MISMATCH makes
    ``check_leakage`` return BLOCK (fail-closed — correct, but it is a schema
    problem, not a leak). Via ``orchestrator`` -> ``write_qa_block_flag`` that
    BLOCK writes ``output/ops/qa_block.json``, which makes the live-pilot
    preflight refuse to trade until an operator runs ``ack_qa_block.py``. Dry-
    run any new production ``feature_df`` through ``check_leakage`` first.

    Args:
        metrics: PerformanceMetrics from qa.metrics
        gate_config: Optional configuration dict with custom thresholds:
            {
                "sharpe": {"min": 1.0, "warning": 0.5},
                "max_drawdown": {"max": -20.0, "warning": -15.0},
                "turnover": {"max": 50.0, "warning": 30.0},
                "cagr": {"min": 0.05, "warning": 0.0},
                "volatility": {"max": 0.30, "warning": 0.25},
                "hit_rate": {"min": 0.50, "warning": 0.40},
                "profit_factor": {"min": 1.5, "warning": 1.2}
            }
        feature_df: Optional altdata feature frame for the leakage gate.
        leakage_feature_col: Feature value column in ``feature_df``.
        leakage_disclosure_col: Disclosure-date column in ``feature_df``.
        leakage_timestamp_col: Observation-timestamp column in ``feature_df``.

    Returns:
        QAGatesSummary with overall result and individual gate results
    """
    if gate_config is None:
        gate_config = {}

    # Get thresholds from config or use defaults
    sharpe_config = gate_config.get("sharpe", {})
    max_dd_config = gate_config.get("max_drawdown", {})
    turnover_config = gate_config.get("turnover", {})
    cagr_config = gate_config.get("cagr", {})
    volatility_config = gate_config.get("volatility", {})
    hit_rate_config = gate_config.get("hit_rate", {})
    profit_factor_config = gate_config.get("profit_factor", {})

    # Evaluate all gates
    gate_results = [
        check_sharpe_ratio(
            metrics,
            min_sharpe=sharpe_config.get("min", 1.0),
            warning_sharpe=sharpe_config.get("warning", 0.5),
        ),
        check_max_drawdown(
            metrics,
            max_dd_pct_limit=max_dd_config.get("max", -20.0),
            warning_dd_pct=max_dd_config.get("warning", -15.0),
        ),
        check_turnover(
            metrics,
            max_turnover=turnover_config.get("max", 50.0),
            warning_turnover=turnover_config.get("warning", 30.0),
        ),
        check_cagr(
            metrics,
            min_cagr=cagr_config.get("min", 0.05),
            warning_cagr=cagr_config.get("warning", 0.0),
        ),
        check_volatility(
            metrics,
            max_volatility=volatility_config.get("max", 0.30),
            warning_volatility=volatility_config.get("warning", 0.25),
        ),
        check_hit_rate(
            metrics,
            min_hit_rate=hit_rate_config.get("min", 0.50),
            warning_hit_rate=hit_rate_config.get("warning", 0.40),
        ),
        check_profit_factor(
            metrics,
            min_profit_factor=profit_factor_config.get("min", 1.5),
            warning_profit_factor=profit_factor_config.get("warning", 1.2),
        ),
        check_leakage(
            feature_df=feature_df,
            feature_col=leakage_feature_col,
            disclosure_col=leakage_disclosure_col,
            timestamp_col=leakage_timestamp_col,
        ),
    ]

    # Count results. A gate that returned OK only because it was SKIPPED
    # (no input supplied) is not a passed check — counting it as one would
    # manufacture green aggregate evidence for a no-op (E-066). Aggregate
    # consumers (API gate_counts, "**Passed:** N" in the daily QA report,
    # the backtest log line) read passed_gates and never see the reason.
    skipped_gates = sum(
        1
        for r in gate_results
        if r.result == QAResult.OK and (r.details or {}).get("skipped") is True
    )
    passed_gates = (
        sum(1 for r in gate_results if r.result == QAResult.OK) - skipped_gates
    )
    warning_gates = sum(1 for r in gate_results if r.result == QAResult.WARNING)
    blocked_gates = sum(1 for r in gate_results if r.result == QAResult.BLOCK)

    # Determine overall result (worst case wins)
    if blocked_gates > 0:
        overall_result = QAResult.BLOCK
    elif warning_gates > 0:
        overall_result = QAResult.WARNING
    else:
        overall_result = QAResult.OK

    return QAGatesSummary(
        overall_result=overall_result,
        passed_gates=passed_gates,
        warning_gates=warning_gates,
        blocked_gates=blocked_gates,
        gate_results=gate_results,
        skipped_gates=skipped_gates,
    )


# ---------------------------------------------------------------------------
# W4 (2026-07-24, GESAMTBEWERTUNG Schritt 8): QA-BLOCK flag bridge to the
# live pilot.
#
# Design honesty: the QA gates evaluate BACKTEST quality; the daily pilot
# cycle (paper_runner) never computes them, so a freshness-gated artifact
# would be permanently stale — a dead gate (the E-054 lesson: a gate whose
# path no writer serves). Chosen semantics instead (ack_halt pattern):
#   - a BLOCK verdict from an orchestrator run over the ROOT output dir
#     writes a persistent flag file,
#   - run_live_paper's preflight refuses to trade WHILE the flag exists
#     (positive block evidence -> fail-closed),
#   - ABSENCE of the flag means "no known QA block", NOT "QA passed" —
#     the pilot is not dead-locked by the orchestrator simply never running,
#   - clearing is an explicit operator act (delete after review; the flag
#     carries the reasons), audit-logged by the preflight when honored.
# ---------------------------------------------------------------------------

# Repo-root anchored (not CWD-relative — the costs.py CWD-trap class):
# qa_gates.py lives at src/assembled_core/qa/, three parents up = repo root.
QA_BLOCK_FLAG_PATH = (
    Path(__file__).resolve().parents[3] / "output" / "ops" / "qa_block.json"
)


def write_qa_block_flag(
    summary: QAGatesSummary,
    *,
    source: str,
    flag_path: Path | str | None = None,
) -> Path | None:
    """Persist a QA-BLOCK verdict so the live pilot preflight can refuse to trade.

    Only writes when ``summary.overall_result == BLOCK`` (returns None
    otherwise). Atomic tmp+replace write. Never raises — a flag-write
    failure must not break the calling pipeline; it logs ERROR instead
    (the pipeline's own BLOCK logging remains the primary signal).

    Timing note (Stage-1 B3): the pilot preflight reads this flag once per
    cycle start — a BLOCK written while a pilot cycle is already running
    takes effect at the NEXT cycle.
    """
    import json
    import logging
    from datetime import datetime, timezone

    log = logging.getLogger(__name__)
    if summary.overall_result != QAResult.BLOCK:
        return None
    path = Path(flag_path) if flag_path is not None else QA_BLOCK_FLAG_PATH
    payload = {
        "schema": "qa_block.v1",
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "overall": summary.overall_result.value,
        "blocked_gates": [
            {"gate": r.gate_name, "reason": r.reason}
            for r in summary.gate_results
            if r.result == QAResult.BLOCK
        ],
        "clear_instructions": (
            "Operator: review the blocked gates, then clear via "
            'scripts/ops/ack_qa_block.py --reason "..." (reason-gated, '
            "ledger-appended, flag archived). Do NOT bare-delete this file."
        ),
        "note": (
            "Distinct from ctx.qa_block_trading (in-cycle data-QC gate): "
            "this flag carries a cross-process BACKTEST QA verdict."
        ),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        log.error(
            "[QA-BLOCK] flag written to %s (%d blocked gate(s)) — live pilot "
            "preflight will refuse to trade until an operator clears it",
            path,
            len(payload["blocked_gates"]),
        )
        return path
    except Exception as exc:  # noqa: BLE001 — Stage-1 B6: docstring promises
        # "never raises"; the calling pipeline must not break on flag I/O.
        log.error("[QA-BLOCK] flag write FAILED (%s): %s", path, exc)
        return None


def read_qa_block_flag(
    flag_path: Path | str | None = None,
) -> dict | None:
    """Return the parsed QA-block flag, or None when absent.

    Fail-closed on unreadable/corrupt content: returns a minimal dict with
    ``{"schema": "unreadable"}`` so callers treat a corrupt flag as a block
    (a safety flag that cannot be read must not be silently ignored).
    """
    import json

    path = Path(flag_path) if flag_path is not None else QA_BLOCK_FLAG_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"schema": "unreadable"}
    except (OSError, json.JSONDecodeError):
        return {"schema": "unreadable"}
