"""Tests for PIT universe wiring: build_universe_history_from_prices + wrap_signal_fn_with_pit_filter."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.universe import (
    build_universe_history_from_prices,
    wrap_signal_fn_with_pit_filter,
)


def _make_prices(symbol_dates: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for sym, dates in symbol_dates.items():
        for d in dates:
            rows.append(
                {"timestamp": pd.Timestamp(d, tz="UTC"), "symbol": sym, "close": 100.0}
            )
    return pd.DataFrame(rows)


def _signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    syms = prices_df["symbol"].unique()
    ts = prices_df["timestamp"].max()
    return pd.DataFrame(
        {"timestamp": ts, "symbol": list(syms), "direction": "LONG", "score": 1.0}
    )


class TestBuildUniverseHistoryFromPrices:
    def test_still_active_symbol_has_nat_end_date(self):
        prices = _make_prices({"AAPL": ["2020-01-01", "2020-01-02", "2020-01-03"]})
        hist = build_universe_history_from_prices(prices)
        aapl = hist[hist["symbol"] == "AAPL"].iloc[0]
        assert pd.isna(aapl["end_date"])
        assert aapl["status"] == "active"

    def test_delisted_symbol_has_end_date(self):
        prices = _make_prices(
            {
                "AAPL": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "OLD": ["2020-01-01", "2020-01-02"],  # last ts < panel max
            }
        )
        hist = build_universe_history_from_prices(prices)
        old_row = hist[hist["symbol"] == "OLD"].iloc[0]
        assert not pd.isna(old_row["end_date"])
        # end_date must be after last date
        assert old_row["end_date"] > pd.Timestamp("2020-01-02", tz="UTC")

    def test_start_date_is_first_appearance(self):
        prices = _make_prices(
            {
                "AAPL": ["2020-01-01", "2020-06-01"],
                "LATE": ["2020-06-01"],
            }
        )
        hist = build_universe_history_from_prices(prices)
        late = hist[hist["symbol"] == "LATE"].iloc[0]
        assert late["start_date"] == pd.Timestamp("2020-06-01", tz="UTC")

    def test_columns_present(self):
        prices = _make_prices({"AAPL": ["2020-01-01"]})
        hist = build_universe_history_from_prices(prices)
        assert set(["symbol", "start_date", "end_date", "status"]).issubset(
            hist.columns
        )


class TestWrapSignalFnWithPitFilter:
    def test_excludes_not_yet_listed_symbol(self):
        prices = _make_prices(
            {
                "AAPL": ["2020-01-01", "2020-06-01"],
                "LATE": ["2020-06-01"],
            }
        )
        history = build_universe_history_from_prices(prices)

        # Signal fn produces AAPL and LATE on 2020-01-01 (before LATE existed)
        def early_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2020-01-01", tz="UTC"),
                    "symbol": ["AAPL", "LATE"],
                    "direction": "LONG",
                    "score": 1.0,
                }
            )

        wrapped = wrap_signal_fn_with_pit_filter(early_signal_fn, history)
        result = wrapped(prices)
        assert "AAPL" in result["symbol"].values
        assert "LATE" not in result["symbol"].values

    def test_includes_symbol_after_listing(self):
        prices = _make_prices(
            {
                "AAPL": ["2020-01-01", "2020-06-01"],
                "LATE": ["2020-06-01"],
            }
        )
        history = build_universe_history_from_prices(prices)

        def late_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2020-06-01", tz="UTC"),
                    "symbol": ["AAPL", "LATE"],
                    "direction": "LONG",
                    "score": 1.0,
                }
            )

        wrapped = wrap_signal_fn_with_pit_filter(late_signal_fn, history)
        result = wrapped(prices)
        assert "AAPL" in result["symbol"].values
        assert "LATE" in result["symbol"].values

    def test_empty_history_passthrough(self, caplog):
        empty_history = pd.DataFrame(columns=["symbol", "start_date", "end_date"])
        wrapped = wrap_signal_fn_with_pit_filter(_signal_fn, empty_history)
        prices = _make_prices({"AAPL": ["2020-01-01"]})
        result = wrapped(prices)
        assert len(result) == 1
        assert "AAPL" in result["symbol"].values

    def test_empty_signals_passthrough(self):
        prices = _make_prices({"AAPL": ["2020-01-01"]})
        history = build_universe_history_from_prices(prices)

        def empty_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

        wrapped = wrap_signal_fn_with_pit_filter(empty_signal_fn, history)
        result = wrapped(prices)
        assert result.empty
