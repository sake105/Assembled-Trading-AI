"""Tests for OPS-8 EOD coverage inspector (inspect_eod_prices)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.assembled_core.ops.inspect_data import SCHEMA_VERSION, inspect_eod_prices

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_prices(
    dates: list[date],
    tz_aware: bool = True,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal prices DataFrame with timestamp and optional symbol."""
    symbols = symbols or ["AAPL"]
    rows = []
    for d in dates:
        ts = pd.Timestamp(d).tz_localize("UTC") if tz_aware else pd.Timestamp(d)
        for sym in symbols:
            rows.append({"timestamp": ts, "symbol": sym, "close": 100.0})
    df = pd.DataFrame(rows)
    return df


def test_inspect_eod_prices_empty() -> None:
    """Empty or no timestamp column returns zeros and no recommendations."""
    empty = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    out = inspect_eod_prices(empty)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["n_rows"] == 0
    assert out["n_unique_days"] == 0
    assert out["min_utc"] is None
    assert out["max_utc"] is None
    assert out["last_30_trading_days"] is None
    assert out["last_90_trading_days"] is None


def test_inspect_eod_prices_utc_aware() -> None:
    """UTC-aware timestamps: correct min/max and n_unique_days."""
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(5)]
    df = _make_prices(dates, tz_aware=True)
    out = inspect_eod_prices(df)
    assert out["n_rows"] == 5
    assert out["n_symbols"] == 1
    assert out["n_unique_days"] == 5
    assert "2024-01-01" in (out["min_utc"] or "")
    assert "2024-01-05" in (out["max_utc"] or "")
    assert out["last_30_trading_days"] is None
    assert out["last_90_trading_days"] is None
    assert len(out["last_10_days"]) == 5


def test_inspect_eod_prices_tz_naive() -> None:
    """Tz-naive timestamps are localized to UTC; min/max correct."""
    base = date(2025, 6, 1)
    dates = [base + timedelta(days=i) for i in range(3)]
    df = _make_prices(dates, tz_aware=False)
    out = inspect_eod_prices(df)
    assert out["n_unique_days"] == 3
    assert out["min_utc"] is not None
    assert out["max_utc"] is not None
    assert "2025-06-01" in (out["min_utc"] or "")
    assert "2025-06-03" in (out["max_utc"] or "")


def test_inspect_eod_prices_recommendations_when_90_days() -> None:
    """When n_unique_days >= 90, last_30 and last_90 recommendations exist."""
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(100)]
    df = _make_prices(dates, tz_aware=True)
    out = inspect_eod_prices(df)
    assert out["n_unique_days"] == 100
    assert out["last_30_trading_days"] is not None
    # last 30 days: start = dates[-30] = 2024-01-01 + 70 days = 2024-03-11, end = dates[-1] = 2024-04-09
    assert out["last_30_trading_days"]["start"] == "2024-03-11"
    assert out["last_30_trading_days"]["end"] == "2024-04-09"
    assert out["last_90_trading_days"] is not None
    # last 90 days: start = dates[-90] = 2024-01-01 + 10 days = 2024-01-11, end = 2024-04-09
    assert out["last_90_trading_days"]["start"] == "2024-01-11"
    assert out["last_90_trading_days"]["end"] == "2024-04-09"


def test_inspect_eod_prices_last_10_days() -> None:
    """last_10_days lists at most 10 most recent dates (isoformat)."""
    base = date(2024, 5, 1)
    dates = [base + timedelta(days=i) for i in range(15)]
    df = _make_prices(dates, tz_aware=True)
    out = inspect_eod_prices(df)
    assert len(out["last_10_days"]) == 10
    assert out["last_10_days"][0] == "2024-05-06"
    assert out["last_10_days"][-1] == "2024-05-15"
