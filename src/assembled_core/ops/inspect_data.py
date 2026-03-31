"""OPS-8: EOD price coverage inspector — min/max timestamps and recommended experiment windows."""

from __future__ import annotations

from typing import Any

import pandas as pd

SCHEMA_VERSION = "ops.eod_coverage.v1"


def inspect_eod_prices(prices_df: pd.DataFrame) -> dict[str, Any]:
    """Inspect EOD price coverage: min/max UTC, unique days, recommended start/end windows.

    Uses the same timestamp normalization as the paper runner (UTC-aware).
    Returns schema ops.eod_coverage.v1.
    """
    if prices_df.empty or "timestamp" not in prices_df.columns:
        return {
            "schema_version": SCHEMA_VERSION,
            "n_rows": 0,
            "n_symbols": None,
            "n_unique_days": 0,
            "min_utc": None,
            "max_utc": None,
            "last_10_days": [],
            "last_30_trading_days": None,
            "last_90_trading_days": None,
        }

    ts = pd.to_datetime(prices_df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    ts = ts.dt.normalize()
    dates_sorted = sorted(ts.dt.date.unique())
    n_unique_days = len(dates_sorted)
    min_utc = pd.Timestamp(dates_sorted[0]).tz_localize("UTC") if dates_sorted else None
    max_utc = (
        pd.Timestamp(dates_sorted[-1]).tz_localize("UTC") if dates_sorted else None
    )

    n_symbols = (
        int(prices_df["symbol"].nunique()) if "symbol" in prices_df.columns else None
    )
    last_10_days = (
        [d.isoformat() for d in dates_sorted[-10:]]
        if len(dates_sorted) >= 10
        else [d.isoformat() for d in dates_sorted]
    )

    last_30_trading_days: dict[str, str] | None = None
    last_90_trading_days: dict[str, str] | None = None
    if n_unique_days >= 30:
        end_30 = dates_sorted[-1]
        start_30 = dates_sorted[-30]
        last_30_trading_days = {
            "start": start_30.isoformat(),
            "end": end_30.isoformat(),
        }
    if n_unique_days >= 90:
        end_90 = dates_sorted[-1]
        start_90 = dates_sorted[-90]
        last_90_trading_days = {
            "start": start_90.isoformat(),
            "end": end_90.isoformat(),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(len(prices_df)),
        "n_symbols": n_symbols,
        "n_unique_days": n_unique_days,
        "min_utc": min_utc.isoformat() if min_utc is not None else None,
        "max_utc": max_utc.isoformat() if max_utc is not None else None,
        "last_10_days": last_10_days,
        "last_30_trading_days": last_30_trading_days,
        "last_90_trading_days": last_90_trading_days,
    }


__all__ = ["inspect_eod_prices", "SCHEMA_VERSION"]
