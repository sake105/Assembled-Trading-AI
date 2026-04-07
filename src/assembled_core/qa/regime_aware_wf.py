"""Regime-aware walk-forward validation (V8).

Extends standard walk-forward analysis with HMM regime conditioning:
- Tags each test window with its dominant regime.
- Computes per-regime metrics (Sharpe, DD, hit_rate).
- Detects regime-conditional overfitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class RegimeWalkForwardResult:
    """Results of regime-aware walk-forward analysis."""

    overall_sharpe: float | None
    overall_max_dd: float | None
    per_regime_metrics: dict[str, dict[str, float]]
    # e.g. {"bull": {"sharpe": 1.2, "max_dd": -0.05, "n_days": 120}, ...}
    regime_stability_score: float  # 0-1: how consistent is performance across regimes
    worst_regime: str | None
    best_regime: str | None
    n_windows: int
    window_results: list[dict]


def tag_regime_for_window(
    regime_state_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    timestamp_col: str = "timestamp",
) -> str:
    """Determine the dominant regime for a date window.

    Args:
        regime_state_df: DataFrame with timestamp and regime_label columns.
        start_date: Start of window.
        end_date: End of window.
        timestamp_col: Column name for timestamps.

    Returns:
        Dominant regime label string (most frequent in window).
    """
    if regime_state_df is None or regime_state_df.empty:
        return "unknown"

    mask = (regime_state_df[timestamp_col] >= start_date) & (
        regime_state_df[timestamp_col] <= end_date
    )
    window_regimes = regime_state_df.loc[mask, "regime_label"]

    if window_regimes.empty:
        return "unknown"

    return str(window_regimes.value_counts().index[0])


def run_regime_aware_walk_forward(
    equity_curves: list[pd.DataFrame],
    window_dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    regime_state_df: pd.DataFrame | None = None,
    regime_series: pd.Series | None = None,
) -> RegimeWalkForwardResult:
    """Analyze walk-forward results conditioned on market regime.

    Args:
        equity_curves: List of equity DataFrames (one per WF window),
            each with 'timestamp' and 'equity' columns.
        window_dates: List of (start_date, end_date) tuples for each window.
        regime_state_df: Optional DataFrame with timestamp and regime_label columns.
        regime_series: Optional Series indexed by date with regime labels.

    Returns:
        RegimeWalkForwardResult with per-regime analysis.
    """
    if not equity_curves or not window_dates:
        return RegimeWalkForwardResult(
            overall_sharpe=None, overall_max_dd=None,
            per_regime_metrics={}, regime_stability_score=0.0,
            worst_regime=None, best_regime=None, n_windows=0, window_results=[],
        )

    # Build regime_state_df from regime_series if needed
    if regime_state_df is None and regime_series is not None:
        regime_state_df = pd.DataFrame({
            "timestamp": regime_series.index,
            "regime_label": regime_series.values,
        })

    # Analyze each window
    window_results = []
    regime_returns: dict[str, list[float]] = {}

    for i, (eq, (start, end)) in enumerate(zip(equity_curves, window_dates)):
        if eq.empty or "equity" not in eq.columns:
            continue

        eq = eq.sort_values("timestamp") if "timestamp" in eq.columns else eq
        equity_values = eq["equity"].values
        if len(equity_values) < 2:
            continue

        returns = np.diff(equity_values) / equity_values[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            continue

        # Compute window metrics
        sharpe = float(np.mean(returns) / max(np.std(returns), 1e-10) * np.sqrt(252))
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = float(dd.min())
        total_return = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0

        # Tag regime
        regime = "unknown"
        if regime_state_df is not None:
            regime = tag_regime_for_window(regime_state_df, start, end)

        window_result = {
            "window_idx": i,
            "start": start,
            "end": end,
            "regime": regime,
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4),
            "total_return": round(total_return, 4),
            "n_days": len(returns),
        }
        window_results.append(window_result)

        if regime not in regime_returns:
            regime_returns[regime] = []
        regime_returns[regime].extend(returns.tolist())

    # Compute per-regime metrics
    per_regime: dict[str, dict[str, float]] = {}
    for regime, rets in regime_returns.items():
        r = np.array(rets)
        if len(r) < 2:
            continue
        regime_sharpe = float(np.mean(r) / max(np.std(r), 1e-10) * np.sqrt(252))
        cum = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        per_regime[regime] = {
            "sharpe": round(regime_sharpe, 4),
            "max_dd": round(float(dd.min()), 4),
            "n_days": len(r),
            "avg_daily_return": round(float(np.mean(r)), 6),
            "volatility": round(float(np.std(r) * np.sqrt(252)), 4),
        }

    # Overall metrics
    all_returns = []
    for rets in regime_returns.values():
        all_returns.extend(rets)
    all_r = np.array(all_returns)

    overall_sharpe = None
    overall_max_dd = None
    if len(all_r) > 2:
        overall_sharpe = round(float(np.mean(all_r) / max(np.std(all_r), 1e-10) * np.sqrt(252)), 4)
        cum = np.cumprod(1 + all_r)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        overall_max_dd = round(float(dd.min()), 4)

    # Regime stability: std of per-regime Sharpes (lower = more stable)
    regime_sharpes = [v["sharpe"] for v in per_regime.values() if "sharpe" in v]
    if len(regime_sharpes) >= 2:
        sharpe_std = float(np.std(regime_sharpes))
        sharpe_mean = float(np.mean(np.abs(regime_sharpes)))
        stability = max(0.0, 1.0 - sharpe_std / max(sharpe_mean, 0.01))
    else:
        stability = 1.0

    # Best/worst regime
    best_regime = max(per_regime, key=lambda k: per_regime[k]["sharpe"]) if per_regime else None
    worst_regime = min(per_regime, key=lambda k: per_regime[k]["sharpe"]) if per_regime else None

    result = RegimeWalkForwardResult(
        overall_sharpe=overall_sharpe,
        overall_max_dd=overall_max_dd,
        per_regime_metrics=per_regime,
        regime_stability_score=round(stability, 4),
        worst_regime=worst_regime,
        best_regime=best_regime,
        n_windows=len(window_results),
        window_results=window_results,
    )

    _log.info(
        "Regime-Aware WF: %d windows, stability=%.2f, best=%s (%.2f), worst=%s (%.2f)",
        len(window_results), stability,
        best_regime, per_regime.get(best_regime, {}).get("sharpe", 0) if best_regime else 0,
        worst_regime, per_regime.get(worst_regime, {}).get("sharpe", 0) if worst_regime else 0,
    )

    return result


__all__ = [
    "RegimeWalkForwardResult",
    "tag_regime_for_window",
    "run_regime_aware_walk_forward",
]
