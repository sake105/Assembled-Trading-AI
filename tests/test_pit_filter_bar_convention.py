"""D5: Bar-convention tests for _filter_prices_for_as_of.

Validates that the inclusive <= as_of filter is correct for EOD bar-open
convention: a bar timestamped T is available at close of day T.
"""
import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle_shared import _filter_prices_for_as_of


def _make_prices(dates: list[str], symbols: list[str] | None = None) -> pd.DataFrame:
    syms = symbols or ["AAPL"]
    rows = []
    for d in dates:
        for s in syms:
            rows.append({"timestamp": pd.Timestamp(d, tz="UTC"), "symbol": s, "close": 100.0})
    return pd.DataFrame(rows)


def test_eod_bar_on_as_of_is_included() -> None:
    """A bar whose timestamp equals as_of is included (bar-open convention)."""
    prices = _make_prices(["2024-03-14", "2024-03-15", "2024-03-18"])
    as_of = pd.Timestamp("2024-03-15", tz="UTC")
    filtered, _ = _filter_prices_for_as_of(prices, as_of, mode="eod")
    assert filtered["timestamp"].iloc[0] == as_of


def test_eod_bar_after_as_of_is_excluded() -> None:
    """A bar with timestamp > as_of is excluded."""
    prices = _make_prices(["2024-03-14", "2024-03-15", "2024-03-18"])
    as_of = pd.Timestamp("2024-03-15", tz="UTC")
    filtered, _ = _filter_prices_for_as_of(prices, as_of, mode="eod")
    assert all(t <= as_of for t in filtered["timestamp"])


def test_backtest_returns_full_history_slice() -> None:
    """In backtest mode, all rows <= as_of are returned (not just the last)."""
    prices = _make_prices(["2024-03-13", "2024-03-14", "2024-03-15", "2024-03-18"])
    as_of = pd.Timestamp("2024-03-15", tz="UTC")
    filtered, latest = _filter_prices_for_as_of(prices, as_of, mode="backtest")
    assert len(filtered) == 3
    assert latest is not None and len(latest) == 1
    assert latest["timestamp"].iloc[0] == as_of


def test_no_as_of_returns_all_rows() -> None:
    """as_of=None applies no time filter."""
    prices = _make_prices(["2024-03-13", "2024-03-14", "2024-03-15", "2024-03-18"])
    filtered, _ = _filter_prices_for_as_of(prices, None, mode="eod")
    assert len(filtered) == 1  # eod returns last row per symbol


def test_universe_filter_applied() -> None:
    """Only symbols in universe are returned."""
    prices = _make_prices(["2024-03-15"], symbols=["AAPL", "MSFT", "GOOG"])
    as_of = pd.Timestamp("2024-03-15", tz="UTC")
    filtered, _ = _filter_prices_for_as_of(prices, as_of, universe=["AAPL", "GOOG"], mode="eod")
    assert set(filtered["symbol"]) == {"AAPL", "GOOG"}
