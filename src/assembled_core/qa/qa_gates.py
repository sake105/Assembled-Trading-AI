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

Each gate returns a structured result (OK, WARNING, BLOCK) with reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
        passed_gates: Number of gates that passed (OK)
        warning_gates: Number of gates with warnings
        blocked_gates: Number of gates that blocked
        gate_results: List of individual gate results
    """
    overall_result: QAResult
    passed_gates: int
    warning_gates: int
    blocked_gates: int
    gate_results: list[QAGateResult]


def check_sharpe_ratio(
    metrics: PerformanceMetrics,
    min_sharpe: float = 1.0,
    warning_sharpe: float = 0.5
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
            details={"sharpe_ratio": None, "min_sharpe": min_sharpe, "warning_sharpe": warning_sharpe}
        )
    
    sharpe = metrics.sharpe_ratio
    
    if sharpe < warning_sharpe:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Sharpe ratio {sharpe:.4f} is below warning threshold {warning_sharpe:.2f}",
            details={"sharpe_ratio": sharpe, "min_sharpe": min_sharpe, "warning_sharpe": warning_sharpe}
        )
    elif sharpe < min_sharpe:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Sharpe ratio {sharpe:.4f} is below minimum threshold {min_sharpe:.2f}",
            details={"sharpe_ratio": sharpe, "min_sharpe": min_sharpe, "warning_sharpe": warning_sharpe}
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Sharpe ratio {sharpe:.4f} meets minimum threshold {min_sharpe:.2f}",
            details={"sharpe_ratio": sharpe, "min_sharpe": min_sharpe, "warning_sharpe": warning_sharpe}
        )


def check_max_drawdown(
    metrics: PerformanceMetrics,
    max_dd_pct_limit: float = -20.0,
    warning_dd_pct: float = -15.0
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
    
    if max_dd < max_dd_pct_limit:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Maximum drawdown {max_dd:.2f}% exceeds limit {max_dd_pct_limit:.2f}%",
            details={
                "max_drawdown_pct": max_dd,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct
            }
        )
    elif max_dd < warning_dd_pct:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Maximum drawdown {max_dd:.2f}% exceeds warning threshold {warning_dd_pct:.2f}%",
            details={
                "max_drawdown_pct": max_dd,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct
            }
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Maximum drawdown {max_dd:.2f}% is within acceptable limits",
            details={
                "max_drawdown_pct": max_dd,
                "max_dd_limit": max_dd_pct_limit,
                "warning_dd": warning_dd_pct
            }
        )


def check_turnover(
    metrics: PerformanceMetrics,
    max_turnover: float = 50.0,
    warning_turnover: float = 30.0
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
            details={"turnover": None, "max_turnover": max_turnover, "warning_turnover": warning_turnover}
        )
    
    turnover = metrics.turnover
    
    if turnover > max_turnover:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Turnover {turnover:.2f}x exceeds maximum limit {max_turnover:.2f}x",
            details={"turnover": turnover, "max_turnover": max_turnover, "warning_turnover": warning_turnover}
        )
    elif turnover > warning_turnover:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Turnover {turnover:.2f}x exceeds warning threshold {warning_turnover:.2f}x",
            details={"turnover": turnover, "max_turnover": max_turnover, "warning_turnover": warning_turnover}
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Turnover {turnover:.2f}x is within acceptable limits",
            details={"turnover": turnover, "max_turnover": max_turnover, "warning_turnover": warning_turnover}
        )


def check_cagr(
    metrics: PerformanceMetrics,
    min_cagr: float = 0.05,
    warning_cagr: float = 0.0
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
            details={"cagr": None, "min_cagr": min_cagr, "warning_cagr": warning_cagr}
        )
    
    cagr = metrics.cagr
    
    if cagr < warning_cagr:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"CAGR {cagr:.2%} is below warning threshold {warning_cagr:.2%}",
            details={"cagr": cagr, "min_cagr": min_cagr, "warning_cagr": warning_cagr}
        )
    elif cagr < min_cagr:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"CAGR {cagr:.2%} is below minimum threshold {min_cagr:.2%}",
            details={"cagr": cagr, "min_cagr": min_cagr, "warning_cagr": warning_cagr}
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"CAGR {cagr:.2%} meets minimum threshold {min_cagr:.2%}",
            details={"cagr": cagr, "min_cagr": min_cagr, "warning_cagr": warning_cagr}
        )


def check_volatility(
    metrics: PerformanceMetrics,
    max_volatility: float = 0.30,
    warning_volatility: float = 0.25
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
            details={"volatility": None, "max_volatility": max_volatility, "warning_volatility": warning_volatility}
        )
    
    volatility = metrics.volatility
    
    if volatility > max_volatility:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Volatility {volatility:.2%} exceeds maximum limit {max_volatility:.2%}",
            details={"volatility": volatility, "max_volatility": max_volatility, "warning_volatility": warning_volatility}
        )
    elif volatility > warning_volatility:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Volatility {volatility:.2%} exceeds warning threshold {warning_volatility:.2%}",
            details={"volatility": volatility, "max_volatility": max_volatility, "warning_volatility": warning_volatility}
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Volatility {volatility:.2%} is within acceptable limits",
            details={"volatility": volatility, "max_volatility": max_volatility, "warning_volatility": warning_volatility}
        )


def check_hit_rate(
    metrics: PerformanceMetrics,
    min_hit_rate: float = 0.50,
    warning_hit_rate: float = 0.40
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
            details={"hit_rate": None, "min_hit_rate": min_hit_rate, "warning_hit_rate": warning_hit_rate}
        )
    
    hit_rate = metrics.hit_rate
    
    if hit_rate < warning_hit_rate:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Hit rate {hit_rate:.2%} is below warning threshold {warning_hit_rate:.2%}",
            details={"hit_rate": hit_rate, "min_hit_rate": min_hit_rate, "warning_hit_rate": warning_hit_rate}
        )
    elif hit_rate < min_hit_rate:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Hit rate {hit_rate:.2%} is below minimum threshold {min_hit_rate:.2%}",
            details={"hit_rate": hit_rate, "min_hit_rate": min_hit_rate, "warning_hit_rate": warning_hit_rate}
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Hit rate {hit_rate:.2%} meets minimum threshold {min_hit_rate:.2%}",
            details={"hit_rate": hit_rate, "min_hit_rate": min_hit_rate, "warning_hit_rate": warning_hit_rate}
        )


def check_profit_factor(
    metrics: PerformanceMetrics,
    min_profit_factor: float = 1.5,
    warning_profit_factor: float = 1.2
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
            details={"profit_factor": None, "min_profit_factor": min_profit_factor, "warning_profit_factor": warning_profit_factor}
        )
    
    profit_factor = metrics.profit_factor
    
    if profit_factor < warning_profit_factor:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.BLOCK,
            reason=f"Profit factor {profit_factor:.2f} is below warning threshold {warning_profit_factor:.2f}",
            details={"profit_factor": profit_factor, "min_profit_factor": min_profit_factor, "warning_profit_factor": warning_profit_factor}
        )
    elif profit_factor < min_profit_factor:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.WARNING,
            reason=f"Profit factor {profit_factor:.2f} is below minimum threshold {min_profit_factor:.2f}",
            details={"profit_factor": profit_factor, "min_profit_factor": min_profit_factor, "warning_profit_factor": warning_profit_factor}
        )
    else:
        return QAGateResult(
            gate_name=gate_name,
            result=QAResult.OK,
            reason=f"Profit factor {profit_factor:.2f} meets minimum threshold {min_profit_factor:.2f}",
            details={"profit_factor": profit_factor, "min_profit_factor": min_profit_factor, "warning_profit_factor": warning_profit_factor}
        )


def evaluate_all_gates(
    metrics: PerformanceMetrics,
    gate_config: dict[str, dict[str, float]] | None = None
) -> QAGatesSummary:
    """Evaluate all QA gates and return summary.
    
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
            warning_sharpe=sharpe_config.get("warning", 0.5)
        ),
        check_max_drawdown(
            metrics,
            max_dd_pct_limit=max_dd_config.get("max", -20.0),
            warning_dd_pct=max_dd_config.get("warning", -15.0)
        ),
        check_turnover(
            metrics,
            max_turnover=turnover_config.get("max", 50.0),
            warning_turnover=turnover_config.get("warning", 30.0)
        ),
        check_cagr(
            metrics,
            min_cagr=cagr_config.get("min", 0.05),
            warning_cagr=cagr_config.get("warning", 0.0)
        ),
        check_volatility(
            metrics,
            max_volatility=volatility_config.get("max", 0.30),
            warning_volatility=volatility_config.get("warning", 0.25)
        ),
        check_hit_rate(
            metrics,
            min_hit_rate=hit_rate_config.get("min", 0.50),
            warning_hit_rate=hit_rate_config.get("warning", 0.40)
        ),
        check_profit_factor(
            metrics,
            min_profit_factor=profit_factor_config.get("min", 1.5),
            warning_profit_factor=profit_factor_config.get("warning", 1.2)
        ),
    ]
    
    # Count results
    passed_gates = sum(1 for r in gate_results if r.result == QAResult.OK)
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
        gate_results=gate_results
    )

