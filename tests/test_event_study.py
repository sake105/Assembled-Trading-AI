"""Unit tests for qa/event_study.py (T6.5)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
)


def _make_prices(symbols: list[str], n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for sym in symbols:
        for i, dt in enumerate(dates):
            rows.append({"timestamp": dt, "symbol": sym, "close": 100.0 + i})
    return pd.DataFrame(rows)


def _make_events(symbol: str, date: str, event_type: str = "earnings") -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": pd.Timestamp(date, tz="UTC"),
        "symbol": symbol,
        "event_type": event_type,
    }])


@pytest.mark.phase12
class TestBuildEventWindowPrices:
    def test_basic_window(self):
        prices = _make_prices(["AAPL"], 30)
        events = _make_events("AAPL", "2024-01-15")
        result = build_event_window_prices(prices, events, window_before=5, window_after=5)
        assert not result.empty
        assert "event_id" in result.columns
        assert "rel_day" in result.columns
        assert result["rel_day"].min() >= -5
        assert result["rel_day"].max() <= 5

    def test_event_day_is_zero(self):
        prices = _make_prices(["AAPL"], 30)
        events = _make_events("AAPL", "2024-01-15")
        result = build_event_window_prices(prices, events, window_before=3, window_after=3)
        assert 0 in result["rel_day"].values

    def test_unknown_symbol_returns_empty(self):
        prices = _make_prices(["AAPL"], 30)
        events = _make_events("MSFT", "2024-01-15")
        result = build_event_window_prices(prices, events, window_before=3, window_after=3)
        assert result.empty or len(result) == 0

    def test_missing_price_col_raises(self):
        prices = _make_prices(["AAPL"], 10).drop(columns=["close"])
        events = _make_events("AAPL", "2024-01-05")
        with pytest.raises(KeyError):
            build_event_window_prices(prices, events)

    def test_multiple_events(self):
        prices = _make_prices(["AAPL"], 60)
        events = pd.DataFrame([
            {"timestamp": pd.Timestamp("2024-01-10", tz="UTC"), "symbol": "AAPL", "event_type": "earnings"},
            {"timestamp": pd.Timestamp("2024-01-20", tz="UTC"), "symbol": "AAPL", "event_type": "news"},
        ])
        result = build_event_window_prices(prices, events, window_before=3, window_after=3)
        assert len(result["event_id"].unique()) == 2

    def test_multiple_symbols(self):
        prices = _make_prices(["AAPL", "GOOG"], 30)
        events = pd.DataFrame([
            {"timestamp": pd.Timestamp("2024-01-15", tz="UTC"), "symbol": "AAPL", "event_type": "e"},
            {"timestamp": pd.Timestamp("2024-01-15", tz="UTC"), "symbol": "GOOG", "event_type": "e"},
        ])
        result = build_event_window_prices(prices, events, window_before=2, window_after=2)
        assert set(result["symbol"].unique()) == {"AAPL", "GOOG"}

    def test_window_clipped_at_data_boundary(self):
        prices = _make_prices(["AAPL"], 10)
        events = _make_events("AAPL", "2024-01-02")  # near start
        result = build_event_window_prices(prices, events, window_before=20, window_after=5)
        assert not result.empty
        assert result["rel_day"].min() >= -1  # clipped by data start


@pytest.mark.phase12
class TestComputeEventReturns:
    def test_returns_present(self):
        prices = _make_prices(["AAPL"], 30)
        events = _make_events("AAPL", "2024-01-15")
        windows = build_event_window_prices(prices, events, window_before=5, window_after=5)
        result = compute_event_returns(windows)
        assert "event_return" in result.columns

    def test_log_vs_simple(self):
        prices = _make_prices(["AAPL"], 30)
        events = _make_events("AAPL", "2024-01-15")
        windows = build_event_window_prices(prices, events, window_before=5, window_after=5)
        log_result = compute_event_returns(windows, return_type="log")
        simple_result = compute_event_returns(windows, return_type="simple")
        assert not log_result["event_return"].equals(simple_result["event_return"])

    def test_abnormal_return_requires_benchmark(self):
        prices = _make_prices(["AAPL"], 30)
        events = _make_events("AAPL", "2024-01-15")
        windows = build_event_window_prices(prices, events, window_before=3, window_after=3)
        result = compute_event_returns(windows)
        assert "abnormal_return" not in result.columns or result["abnormal_return"].isna().all()
