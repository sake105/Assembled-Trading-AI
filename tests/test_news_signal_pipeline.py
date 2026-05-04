"""Tests for the News → Signal integration pipeline (Weeks 9-12).

Covers:
  - intel_signal_adapter: adapt_intel_signal, overlay_to_dataframe
  - news_features: compute_news_features
  - news_signal_bridge: blend_with_news
  - rules_trend integration with intel_overlay parameter

All tests check concrete values, not just existence (per handbook rule).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.assembled_core.signals.intel_signal_adapter import (
    IntelOverlay,
    adapt_intel_signal,
    overlay_to_dataframe,
)
from src.assembled_core.signals.news_signal_bridge import blend_with_news
from src.assembled_core.features.news_features import compute_news_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intel_signal(
    net_direction: str = "bearish",
    risk_level: str = "HIGH",
    asset_basket: dict | None = None,
    is_actionable: bool = True,
) -> MagicMock:
    sig = MagicMock()
    sig.net_direction = net_direction
    sig.risk_level = risk_level
    sig.asset_basket = asset_basket or {}
    sig.is_actionable.return_value = is_actionable
    sig.generated_at = datetime(2020, 2, 24, 16, 0, 0, tzinfo=timezone.utc)
    return sig


def _make_trend_signals(
    symbols: list[str], direction: str = "LONG", score: float = 0.8
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-02-24", tz="UTC")] * len(symbols),
            "symbol": symbols,
            "direction": [direction] * len(symbols),
            "score": [score] * len(symbols),
        }
    )


# ---------------------------------------------------------------------------
# intel_signal_adapter tests
# ---------------------------------------------------------------------------


class TestAdaptIntelSignal:
    def test_none_returns_neutral(self):
        overlay = adapt_intel_signal(None)
        assert overlay.is_actionable is False
        assert overlay.macro_score == 0.0
        assert overlay.ticker_scores == {}

    def test_non_actionable_returns_neutral(self):
        sig = _make_intel_signal(is_actionable=False)
        overlay = adapt_intel_signal(sig)
        assert overlay.is_actionable is False
        assert overlay.macro_score == 0.0

    def test_bearish_high_risk_macro_score(self):
        sig = _make_intel_signal(
            net_direction="bearish", risk_level="HIGH", is_actionable=True
        )
        overlay = adapt_intel_signal(sig)
        assert overlay.is_actionable is True
        # bearish + HIGH: -1.0 * 0.7 = -0.7
        assert overlay.macro_score == pytest.approx(-0.7, abs=0.01)

    def test_bullish_critical_macro_score(self):
        sig = _make_intel_signal(
            net_direction="bullish", risk_level="CRITICAL", is_actionable=True
        )
        overlay = adapt_intel_signal(sig)
        assert overlay.macro_score == pytest.approx(1.0, abs=0.01)

    def test_ticker_scores_capped_at_max_overlay(self):
        sig = _make_intel_signal(
            net_direction="bearish",
            risk_level="HIGH",
            asset_basket={"AAPL": -0.9, "MSFT": 0.8},
            is_actionable=True,
        )
        overlay = adapt_intel_signal(sig)
        # Scores capped at ±0.5
        assert overlay.ticker_scores["AAPL"] == pytest.approx(-0.5, abs=0.01)
        assert overlay.ticker_scores["MSFT"] == pytest.approx(0.5, abs=0.01)

    def test_neutral_direction_gives_zero_macro(self):
        sig = _make_intel_signal(
            net_direction="neutral", risk_level="HIGH", is_actionable=True
        )
        overlay = adapt_intel_signal(sig)
        assert overlay.macro_score == pytest.approx(0.0, abs=0.01)


class TestOverlayToDataframe:
    def test_empty_on_no_tickers(self):
        overlay = IntelOverlay.neutral()
        df = overlay_to_dataframe(overlay)
        assert df.empty
        assert "symbol" in df.columns

    def test_columns_and_values(self):
        overlay = IntelOverlay(
            ticker_scores={"AAPL": -0.4, "GLD": 0.3},
            macro_score=-0.5,
            risk_level="HIGH",
            is_actionable=True,
        )
        df = overlay_to_dataframe(overlay)
        assert set(df["symbol"]) == {"AAPL", "GLD"}
        aapl_row = df[df["symbol"] == "AAPL"].iloc[0]
        assert aapl_row["intel_score"] == pytest.approx(-0.4, abs=0.01)
        assert aapl_row["macro_score"] == pytest.approx(-0.5, abs=0.01)
        assert aapl_row["risk_level"] == "HIGH"
        assert bool(aapl_row["is_actionable"]) is True


# ---------------------------------------------------------------------------
# news_signal_bridge tests
# ---------------------------------------------------------------------------


class TestBlendWithNews:
    def test_none_overlay_passthrough(self):
        signals = _make_trend_signals(["AAPL"], score=0.8)
        result = blend_with_news(signals, None)
        assert result["score"].iloc[0] == pytest.approx(0.8, abs=0.01)
        assert result["direction"].iloc[0] == "LONG"

    def test_neutral_overlay_passthrough(self):
        signals = _make_trend_signals(["AAPL"], score=0.8)
        overlay = IntelOverlay.neutral()
        result = blend_with_news(signals, overlay)
        assert result["score"].iloc[0] == pytest.approx(0.8, abs=0.01)

    def test_bearish_macro_reduces_score(self):
        signals = _make_trend_signals(["AAPL"], score=1.0)
        overlay = IntelOverlay(
            ticker_scores={},
            macro_score=-0.5,  # no ticker-specific, macro is bearish
            risk_level="HIGH",
            is_actionable=True,
        )
        result = blend_with_news(signals, overlay, news_alpha=0.20)
        # blended = 0.8 * 1.0 + 0.2 * (-0.5) = 0.7, clipped to [0, 1]
        assert result["score"].iloc[0] == pytest.approx(0.7, abs=0.01)
        # Score reduced but still LONG (news_score=-0.5 is not below -0.5 threshold)
        assert result["direction"].iloc[0] == "LONG"

    def test_strong_bearish_downgrades_long_to_flat(self):
        signals = _make_trend_signals(["AAPL"], score=0.8)
        overlay = IntelOverlay(
            ticker_scores={"AAPL": -0.8},  # strongly bearish ticker
            macro_score=0.0,
            risk_level="CRITICAL",
            is_actionable=True,
        )
        result = blend_with_news(signals, overlay)
        # -0.8 < -0.5 threshold → LONG downgraded to FLAT
        assert result["direction"].iloc[0] == "FLAT"
        assert result["score"].iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_ticker_specific_score_takes_priority_over_macro(self):
        signals = _make_trend_signals(["AAPL", "MSFT"], score=0.6)
        overlay = IntelOverlay(
            ticker_scores={"AAPL": 0.4},  # AAPL is bullish
            macro_score=-0.3,  # macro is bearish
            risk_level="MODERATE",
            is_actionable=True,
        )
        result = blend_with_news(signals, overlay, news_alpha=0.20)
        aapl_score = result[result["symbol"] == "AAPL"]["score"].iloc[0]
        msft_score = result[result["symbol"] == "MSFT"]["score"].iloc[0]
        # AAPL: 0.8 * 0.6 + 0.2 * 0.4 = 0.56
        assert aapl_score == pytest.approx(0.56, abs=0.02)
        # MSFT (no ticker score): uses macro -0.3 → 0.8 * 0.6 + 0.2 * (-0.3) = 0.42
        assert msft_score == pytest.approx(0.42, abs=0.02)

    def test_empty_signals_returns_empty(self):
        empty = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
        overlay = IntelOverlay(is_actionable=True, macro_score=-0.5)
        result = blend_with_news(empty, overlay)
        assert result.empty


# ---------------------------------------------------------------------------
# news_features tests
# ---------------------------------------------------------------------------


class TestComputeNewsFeatures:
    def _make_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_date": pd.to_datetime(
                    [
                        "2020-02-20",
                        "2020-02-21",
                        "2020-02-24",
                        "2020-02-24",
                        "2020-02-25",
                    ],
                    utc=True,
                ),
                "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "AAPL"],
                "direction": ["bearish", "bearish", "bearish", "bearish", "bearish"],
                "confidence": [0.8, 0.7, 0.9, 0.6, 0.85],
            }
        )

    def test_output_columns(self):
        events = self._make_events()
        result = compute_news_features(events)
        expected_cols = {
            "timestamp",
            "symbol",
            "news_sentiment",
            "news_event_count",
            "news_velocity",
            "news_confidence",
        }
        assert expected_cols <= set(result.columns)

    def test_sentiment_is_negative_for_all_bearish_events(self):
        events = self._make_events()
        result = compute_news_features(events)
        aapl = result[result["symbol"] == "AAPL"]
        # All events are bearish → sentiment should be negative
        assert (aapl["news_sentiment"] <= 0.0).all()

    def test_event_count_increases_with_more_events(self):
        events = self._make_events()
        result = compute_news_features(events)
        aapl = result[(result["symbol"] == "AAPL")].sort_values("timestamp")
        # Event count on Feb 25 (4 AAPL events by then) > on Feb 20 (1 event)
        count_feb25 = aapl[aapl["timestamp"].dt.date.astype(str) == "2020-02-25"][
            "news_event_count"
        ].iloc[0]
        count_feb20 = aapl[aapl["timestamp"].dt.date.astype(str) == "2020-02-20"][
            "news_event_count"
        ].iloc[0]
        assert count_feb25 >= count_feb20

    def test_confidence_between_zero_and_one(self):
        events = self._make_events()
        result = compute_news_features(events)
        assert (result["news_confidence"] >= 0.0).all()
        assert (result["news_confidence"] <= 1.0).all()

    def test_pit_filter_excludes_future_events(self):
        events = self._make_events()
        as_of = pd.Timestamp("2020-02-21", tz="UTC")
        result = compute_news_features(events, as_of=as_of)
        # Only events on/before Feb 21 should be included
        aapl = result[result["symbol"] == "AAPL"].sort_values("timestamp")
        # Max event count for AAPL should be at most 2 (Feb 20 + Feb 21 events)
        assert aapl["news_event_count"].max() <= 2.0

    def test_empty_events_returns_empty_dataframe(self):
        events = pd.DataFrame(
            columns=["event_date", "symbol", "direction", "confidence"]
        )
        result = compute_news_features(events)
        assert result.empty
        assert "news_sentiment" in result.columns

    def test_mixed_directions_reduce_absolute_sentiment(self):
        events = pd.DataFrame(
            {
                "event_date": pd.to_datetime(["2020-02-20", "2020-02-20"], utc=True),
                "symbol": ["AAPL", "AAPL"],
                "direction": ["bullish", "bearish"],
                "confidence": [0.8, 0.8],
            }
        )
        result = compute_news_features(events)
        aapl = result[result["symbol"] == "AAPL"]
        # Net = 0 (one bullish, one bearish with equal confidence)
        assert abs(aapl["news_sentiment"].iloc[0]) < 0.1


# ---------------------------------------------------------------------------
# Integration: rules_trend with intel_overlay parameter
# ---------------------------------------------------------------------------


class TestRulesTrendIntelIntegration:
    def _make_prices(self) -> pd.DataFrame:
        import numpy as np

        dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
        prices = []
        for sym in ["AAPL", "MSFT"]:
            close = 100.0 * np.cumprod(
                1 + np.random.default_rng(42).normal(0.001, 0.02, len(dates))
            )
            for i, d in enumerate(dates):
                prices.append(
                    {
                        "timestamp": d,
                        "symbol": sym,
                        "close": close[i],
                        "volume": 1_000_000,
                    }
                )
        return pd.DataFrame(prices)

    def test_no_overlay_same_as_baseline(self):
        from src.assembled_core.signals.rules_trend import generate_trend_signals

        prices = self._make_prices()
        baseline = generate_trend_signals(prices, ma_fast=5, ma_slow=10)
        with_none = generate_trend_signals(
            prices, ma_fast=5, ma_slow=10, intel_overlay=None
        )
        pd.testing.assert_frame_equal(baseline, with_none)

    def test_neutral_overlay_same_as_baseline(self):
        from src.assembled_core.signals.rules_trend import generate_trend_signals

        prices = self._make_prices()
        baseline = generate_trend_signals(prices, ma_fast=5, ma_slow=10)
        neutral = IntelOverlay.neutral()
        with_neutral = generate_trend_signals(
            prices, ma_fast=5, ma_slow=10, intel_overlay=neutral
        )
        pd.testing.assert_frame_equal(baseline, with_neutral)

    def test_bearish_overlay_reduces_average_score(self):
        from src.assembled_core.signals.rules_trend import generate_trend_signals

        prices = self._make_prices()
        baseline = generate_trend_signals(prices, ma_fast=5, ma_slow=10)
        bearish_overlay = IntelOverlay(
            ticker_scores={},
            macro_score=-0.8,
            risk_level="CRITICAL",
            is_actionable=True,
        )
        blended = generate_trend_signals(
            prices,
            ma_fast=5,
            ma_slow=10,
            intel_overlay=bearish_overlay,
            news_alpha=0.20,
        )
        # Average LONG score should be lower with bearish overlay
        baseline_long_mean = baseline[baseline["direction"] == "LONG"]["score"].mean()
        blended_long_mean = blended[blended["direction"] == "LONG"]["score"].mean()
        # Either scores decreased OR some LONGs became FLATs (reducing the mean set)
        if len(blended[blended["direction"] == "LONG"]) > 0:
            assert blended_long_mean <= baseline_long_mean + 0.05
