"""Data quality checks for price and OHLCV data.

Detects common data issues: null prices, extreme jumps, duplicate timestamps,
multi-day gaps, and stale data. Returns structured check results that can
be used for gating, logging, or degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a data quality check run."""

    symbol: str
    passed: bool
    issues: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_checked: int = 0


def check_price_quality(
    df: pd.DataFrame,
    symbol: str,
    *,
    max_return_pct: float = 50.0,
    max_gap_days: int = 5,
    min_rows: int = 10,
) -> QualityCheckResult:
    """Run quality checks on a single symbol's price data.

    Args:
        df: DataFrame with at least 'close' column and a datetime index or 'date' column.
        symbol: Ticker for logging.
        max_return_pct: Flag absolute daily returns exceeding this threshold.
        max_gap_days: Flag gaps (trading day holes) longer than this many calendar days.
        min_rows: Minimum data points required for meaningful checks.

    Returns:
        QualityCheckResult with issues list.
    """
    result = QualityCheckResult(symbol=symbol, passed=True, rows_checked=len(df))

    if len(df) < min_rows:
        result.passed = False
        result.issues.append({
            "type": "insufficient_data",
            "detail": f"Only {len(df)} rows (min={min_rows})",
        })
        return result

    # Ensure we have a datetime index
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        result.warnings.append("No datetime index or 'date' column — skipping gap check")
        dates = None

    close = df["close"] if "close" in df.columns else df.get("Close")
    if close is None:
        result.passed = False
        result.issues.append({"type": "missing_column", "detail": "No 'close' or 'Close' column"})
        return result

    close = close.astype(float)

    # 1. Null/NaN check
    null_count = int(close.isna().sum())
    if null_count > 0:
        pct = null_count / len(close) * 100
        result.issues.append({
            "type": "null_prices",
            "count": null_count,
            "pct": round(pct, 2),
        })
        if pct > 10:
            result.passed = False

    # 2. Zero/negative price check
    bad_prices = int((close <= 0).sum())
    if bad_prices > 0:
        result.passed = False
        result.issues.append({
            "type": "invalid_prices",
            "count": bad_prices,
            "detail": "Zero or negative close prices",
        })

    # 3. Extreme return check
    valid_close = close.dropna()
    if len(valid_close) >= 2:
        returns = valid_close.pct_change().dropna()
        extreme_mask = returns.abs() > (max_return_pct / 100.0)
        extreme_count = int(extreme_mask.sum())
        if extreme_count > 0:
            result.issues.append({
                "type": "extreme_returns",
                "count": extreme_count,
                "threshold_pct": max_return_pct,
                "max_return_pct": round(float(returns.abs().max()) * 100, 2),
            })
            result.warnings.append(
                f"{extreme_count} daily return(s) exceed {max_return_pct}%"
            )

    # 4. Duplicate timestamp check
    if dates is not None:
        dup_count = int(dates.duplicated().sum())
        if dup_count > 0:
            result.passed = False
            result.issues.append({
                "type": "duplicate_timestamps",
                "count": dup_count,
            })

    # 5. Gap check
    if dates is not None and len(dates) >= 2:
        sorted_dates = dates.sort_values()
        gaps = sorted_dates.diff().dropna()
        max_gap = gaps.max()
        if isinstance(max_gap, timedelta) and max_gap.days > max_gap_days:
            result.issues.append({
                "type": "data_gap",
                "max_gap_days": max_gap.days,
                "threshold_days": max_gap_days,
            })
            result.warnings.append(
                f"Max gap of {max_gap.days} calendar days (threshold={max_gap_days})"
            )

    # 6. Stale data check (constant price)
    if len(valid_close) >= 5:
        tail = valid_close.tail(5)
        if tail.nunique() == 1:
            result.issues.append({
                "type": "stale_data",
                "detail": "Last 5 prices are identical",
            })
            result.warnings.append("Possible stale/frozen data — last 5 closes identical")

    if result.issues:
        logger.info(
            "[DataQuality] %s: %d issue(s)%s",
            symbol, len(result.issues),
            " — FAILED" if not result.passed else "",
        )

    return result


def check_panel_quality(
    panel_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    max_return_pct: float = 50.0,
    max_gap_days: int = 5,
    min_rows: int = 10,
) -> list[QualityCheckResult]:
    """Run quality checks across all symbols in a panel DataFrame.

    Returns list of QualityCheckResult, one per symbol.
    """
    results = []
    if symbol_col not in panel_df.columns:
        logger.warning("[DataQuality] No '%s' column in panel — cannot check", symbol_col)
        return results

    for symbol, group in panel_df.groupby(symbol_col):
        result = check_price_quality(
            group,
            str(symbol),
            max_return_pct=max_return_pct,
            max_gap_days=max_gap_days,
            min_rows=min_rows,
        )
        results.append(result)

    failed = sum(1 for r in results if not r.passed)
    if failed:
        logger.warning("[DataQuality] Panel check: %d/%d symbols FAILED", failed, len(results))

    return results
