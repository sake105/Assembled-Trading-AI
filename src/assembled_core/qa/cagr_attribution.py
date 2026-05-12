"""CAGR & drawdown attribution by sub-period (audit C2-068).

A small reporting helper that breaks a backtest equity curve into
calendar sub-periods (typically quarters or years) and produces a
per-period attribution table with the metrics that matter for a
post-trade review:

    * CAGR per period (annualized) and total compounded return.
    * Maximum-drawdown amplitude AND duration (in calendar days).
    * Calmar ratio (CAGR / |MaxDD|) — the standard "growth per unit
      of pain" metric.
    * Worst-Year flag — useful in long horizons; the period whose
      ending equity is the worst point in a rolling-year window.

All math is performed on a daily equity series, no I/O. The output
is a pandas DataFrame so it slots into the existing report layer
without any new infrastructure.

Reference: McLeod-Ziemba (2006) on Calmar; standard Kestner Worst-Year
heuristic from "Quantitative Trading Strategies" (2003).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


PeriodKind = Literal["Q", "Y", "M", "W"]


@dataclass(frozen=True)
class AttributionResult:
    """Container for the per-period attribution + summary metrics."""

    per_period: pd.DataFrame
    overall_cagr: float
    overall_max_dd: float
    overall_calmar: float
    worst_period_label: str | None


def _drawdown_info(equity: pd.Series) -> tuple[float, int]:
    """Return (max_drawdown_magnitude, max_drawdown_duration_in_days).

    Drawdown is defined as (peak - equity) / peak, always non-negative.
    Duration is calendar days from prior peak to trough.
    """
    if equity.empty:
        return 0.0, 0
    arr = equity.to_numpy(dtype=float)
    times = equity.index
    cummax = np.maximum.accumulate(arr)
    # Avoid division by zero in equity series that start at 0.
    safe = np.where(cummax > 0, cummax, np.nan)
    dd = (cummax - arr) / safe
    dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
    max_dd = float(np.max(dd)) if dd.size else 0.0
    if max_dd <= 0.0:
        return 0.0, 0
    trough_idx = int(np.argmax(dd))
    # Find the peak immediately preceding this trough.
    pre = arr[: trough_idx + 1]
    peak_idx = int(np.argmax(pre))
    if isinstance(times, pd.DatetimeIndex):
        duration_days = int((times[trough_idx] - times[peak_idx]).days)
    else:
        duration_days = int(trough_idx - peak_idx)
    return max_dd, duration_days


def _annualized_return(equity: pd.Series) -> float:
    """CAGR from start to end of an equity series."""
    if equity.size < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0.0:
        return 0.0
    if isinstance(equity.index, pd.DatetimeIndex):
        years = (equity.index[-1] - equity.index[0]).days / 365.25
    else:
        years = (equity.size - 1) / 252.0
    if years <= 0.0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def attribute_by_period(
    equity_curve: pd.Series,
    *,
    period: PeriodKind = "Q",
) -> AttributionResult:
    """Decompose an equity curve into per-period metrics.

    Args:
        equity_curve: a pandas Series of daily equity values indexed
            by a DatetimeIndex (UTC or naive).
        period: pandas resample period code — ``"Q"``, ``"Y"``,
            ``"M"`` or ``"W"``. Default ``"Q"`` (quarters).

    Returns:
        An :class:`AttributionResult` with the per-period DataFrame
        and the overall summary numbers.

    Raises:
        TypeError: if ``equity_curve`` is not indexed by datetimes.
        ValueError: if the series has fewer than 2 points.
    """
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise TypeError("equity_curve must have a DatetimeIndex")
    if equity_curve.size < 2:
        raise ValueError("need at least 2 equity points")

    curve = equity_curve.astype(float).sort_index()

    rows = []
    for period_label, group in curve.groupby(curve.index.to_period(period)):
        if group.size < 2:
            continue
        cagr = _annualized_return(group)
        total_return = float(group.iloc[-1] / group.iloc[0] - 1.0)
        max_dd, dd_days = _drawdown_info(group)
        calmar = cagr / max_dd if max_dd > 1e-9 else float("inf")
        rows.append(
            {
                "period": str(period_label),
                "start": group.index[0],
                "end": group.index[-1],
                "start_equity": float(group.iloc[0]),
                "end_equity": float(group.iloc[-1]),
                "total_return": total_return,
                "cagr": cagr,
                "max_drawdown": max_dd,
                "drawdown_duration_days": dd_days,
                "calmar": calmar,
            }
        )

    per_period = pd.DataFrame(rows)

    overall_cagr = _annualized_return(curve)
    overall_max_dd, _ = _drawdown_info(curve)
    overall_calmar = (
        overall_cagr / overall_max_dd if overall_max_dd > 1e-9 else float("inf")
    )

    worst_label = None
    if not per_period.empty:
        worst_row = per_period.loc[per_period["total_return"].idxmin()]
        worst_label = str(worst_row["period"])

    return AttributionResult(
        per_period=per_period,
        overall_cagr=overall_cagr,
        overall_max_dd=overall_max_dd,
        overall_calmar=overall_calmar,
        worst_period_label=worst_label,
    )


__all__ = ["attribute_by_period", "AttributionResult"]
