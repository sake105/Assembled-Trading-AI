"""Tests for PIT universe wiring: build_universe_history_from_prices + wrap_signal_fn_with_pit_filter."""

from __future__ import annotations

import logging

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


class TestCoverageGraceDays:
    """DAT-006: tail-gap symbols must not be silently inferred-delisted."""

    def _panel(self) -> pd.DataFrame:
        # Business days only. Panel end = Fri 2020-01-10.
        #   STILL reaches the panel end (always listed)
        #   GAP   last bar Wed 2020-01-08 (2 business days before end)
        #   OLD   last bar Thu 2020-01-02 (far before the grace window)
        return _make_prices(
            {
                "STILL": ["2020-01-06", "2020-01-08", "2020-01-10"],
                "GAP": ["2020-01-06", "2020-01-08"],
                "OLD": ["2020-01-02"],
            }
        )

    def test_strict_default_delists_tail_gap(self):
        # grace=0 (legacy): any symbol missing the final bar => inferred delisted.
        hist = build_universe_history_from_prices(self._panel())
        gap = hist[hist["symbol"] == "GAP"].iloc[0]
        assert not pd.isna(gap["end_date"])

    def test_grace_window_keeps_tail_gap_active(self):
        # grace=5 BDays => threshold = Fri 2020-01-03; GAP (2020-01-08) is inside it.
        hist = build_universe_history_from_prices(self._panel(), coverage_grace_days=5)
        gap = hist[hist["symbol"] == "GAP"].iloc[0]
        assert pd.isna(gap["end_date"])
        assert gap["status"] == "active"

    def test_grace_window_still_delists_far_old(self):
        # OLD (2020-01-02) is beyond the 5-BDay grace window => still delisted.
        hist = build_universe_history_from_prices(self._panel(), coverage_grace_days=5)
        old = hist[hist["symbol"] == "OLD"].iloc[0]
        assert not pd.isna(old["end_date"])

    def test_still_listed_symbol_active_regardless_of_grace(self):
        for grace in (0, 5):
            hist = build_universe_history_from_prices(
                self._panel(), coverage_grace_days=grace
            )
            still = hist[hist["symbol"] == "STILL"].iloc[0]
            assert pd.isna(still["end_date"])

    def test_inferred_delisting_logs_dat006_warning(self, caplog):
        # No logger= filter: capture via root propagation so the assertion holds
        # regardless of whether the module imports as src.assembled_core.* or
        # assembled_core.* (both install paths exist in this repo).
        with caplog.at_level(logging.WARNING):
            build_universe_history_from_prices(self._panel())
        # grace=0 => GAP + OLD delisted, STILL active => "2/3" in the rendered
        # message. Asserting the rendered count forces %-formatting to execute.
        assert "DAT-006" in caplog.text
        assert "2/3" in caplog.text
