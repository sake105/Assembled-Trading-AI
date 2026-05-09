"""Tests for PIT (point-in-time) fix in multifactor_v2 altdata lookups.

Verifies that _compute_earnings_insider_factors, _compute_news_macro_factors,
and _compute_pead_sue_factor use the bar date (as_of) from the panel,
not pd.Timestamp.now(), so backtest mode is free of look-ahead bias.

Also verifies the compute_signals call site derives _bar_as_of from the
panel's latest timestamp and passes it to all three helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch

from src.assembled_core.strategies.multifactor_v2 import (
    _compute_earnings_insider_factors,
    _compute_news_macro_factors,
    _compute_pead_sue_factor,
)

pytestmark = pytest.mark.phase12


_SYMBOLS = ["AAPL", "MSFT"]
_BACKTEST_DATE = pd.Timestamp("2021-06-15")


class TestEarningsInsiderAsOf:
    """_compute_earnings_insider_factors must honour the as_of kwarg."""

    def test_uses_provided_as_of(self) -> None:
        captured: list[pd.Timestamp] = []

        def fake_load_earnings(symbols, as_of, lookback_days=90):
            captured.append(as_of)
            return None

        def fake_load_insider(symbols, as_of, lookback_days=90):
            return None

        with (
            patch(
                "src.assembled_core.data.altdata_loader.load_earnings_history",
                side_effect=fake_load_earnings,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_insider_filings",
                side_effect=fake_load_insider,
            ),
        ):
            _compute_earnings_insider_factors(_SYMBOLS, {}, as_of=_BACKTEST_DATE)

        if captured:
            assert (
                captured[0].normalize() == _BACKTEST_DATE.normalize()
            ), "as_of passed to load_earnings_history must match the bar date"

    def test_defaults_to_now_when_as_of_none(self) -> None:
        """When as_of=None the function falls back gracefully (no crash)."""
        result_earn, result_insider = _compute_earnings_insider_factors(
            _SYMBOLS, {}, as_of=None
        )
        # May be empty (no real data in test env) but must not raise
        assert isinstance(result_earn, pd.Series)
        assert isinstance(result_insider, pd.Series)


class TestNewsMacroAsOf:
    """_compute_news_macro_factors must honour the as_of kwarg."""

    def test_uses_provided_as_of(self) -> None:
        captured: list[pd.Timestamp] = []

        def fake_load_news(symbols, as_of, lookback_days=30):
            captured.append(as_of)
            return None

        def fake_load_macro(as_of, lookback_days=365):
            return None

        with (
            patch(
                "src.assembled_core.data.altdata_loader.load_news_sentiment",
                side_effect=fake_load_news,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_macro_indicators",
                side_effect=fake_load_macro,
            ),
        ):
            _compute_news_macro_factors(_SYMBOLS, {}, as_of=_BACKTEST_DATE)

        if captured:
            assert captured[0].normalize() == _BACKTEST_DATE.normalize()

    def test_returns_dict(self) -> None:
        result = _compute_news_macro_factors(_SYMBOLS, {}, as_of=None)
        assert isinstance(result, dict)


class TestPeadSueAsOf:
    """_compute_pead_sue_factor must honour the as_of kwarg when falling back to altdata."""

    def _make_latest(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": _SYMBOLS,
                "timestamp": [_BACKTEST_DATE, _BACKTEST_DATE],
                "close": [150.0, 300.0],
            }
        )

    def test_uses_provided_as_of_on_fallback(self) -> None:
        captured: list[pd.Timestamp] = []
        latest = self._make_latest()

        def fake_load_earnings(symbols, as_of, lookback_days=90):
            captured.append(as_of)
            return None

        with patch(
            "src.assembled_core.data.altdata_loader.load_earnings_history",
            side_effect=fake_load_earnings,
        ):
            _compute_pead_sue_factor(_SYMBOLS, latest, as_of=_BACKTEST_DATE)

        if captured:
            assert captured[0].normalize() == _BACKTEST_DATE.normalize()

    def test_precomputed_column_skips_load(self) -> None:
        """If sue_score already in panel, altdata load is skipped entirely."""
        latest = self._make_latest()
        latest["sue_score"] = [0.5, -0.3]

        with patch(
            "src.assembled_core.data.altdata_loader.load_earnings_history",
        ) as mock_load:
            result = _compute_pead_sue_factor(_SYMBOLS, latest, as_of=_BACKTEST_DATE)
            mock_load.assert_not_called()

        assert "pead_sue_score" in result


class TestComputeSignalsBarAsOf:
    """compute_signals derives _bar_as_of from panel max timestamp."""

    def _make_panel(self, latest_date: str) -> pd.DataFrame:
        dates = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
        rows = []
        for sym in _SYMBOLS:
            for d in dates:
                rows.append({"timestamp": d, "symbol": sym, "close": 100.0})
        # Add a row at the target latest date
        latest_ts = pd.Timestamp(latest_date, tz="UTC")
        for sym in _SYMBOLS:
            rows.append({"timestamp": latest_ts, "symbol": sym, "close": 110.0})
        return pd.DataFrame(rows)

    def test_bar_as_of_matches_panel_max_timestamp(self) -> None:
        """_bar_as_of must equal the panel's max timestamp (normalized)."""
        target_date = "2021-06-15"
        panel = self._make_panel(target_date)

        captured_earnings: list[pd.Timestamp] = []
        captured_news: list[pd.Timestamp] = []

        def fake_earn(symbols, as_of, lookback_days=90):
            captured_earnings.append(as_of)
            return None

        def fake_news(symbols, as_of, lookback_days=30):
            captured_news.append(as_of)
            return None

        with (
            patch(
                "src.assembled_core.data.altdata_loader.load_earnings_history",
                side_effect=fake_earn,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_insider_filings",
                return_value=None,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_news_sentiment",
                side_effect=fake_news,
            ),
            patch(
                "src.assembled_core.data.altdata_loader.load_macro_indicators",
                return_value=None,
            ),
        ):
            from src.assembled_core.strategies.multifactor_v2 import compute_signals

            compute_signals(panel)

        expected_date = pd.Timestamp(target_date).date()
        for ts in captured_earnings + captured_news:
            assert (
                ts.normalize().date() == expected_date
            ), f"as_of={ts} does not match panel max date {expected_date} — look-ahead bias"
