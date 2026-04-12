"""Tests for B2.4 news sentiment + macro regime factors.

Locks PIT safety, 7-day window, volume spike, macro broadcast,
safe-divide and clipping. Synthetic fixtures only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.news_macro_wrapper import (  # noqa: E402
    _macro_regime_raw,
    _news_sentiment_raw,
    _news_volume_spike_raw,
    compute_news_macro_factors,
)

pytestmark = pytest.mark.phase12


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "timestamp", "sentiment_score"])


def _empty_macro() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])


# ---------------------------------------------------------------------------
# PIT safety
# ---------------------------------------------------------------------------


def test_pit_gate_drops_future_news() -> None:
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2026-06-01"),
             "sentiment_score": 0.9},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2026-05-28"),
             "sentiment_score": 0.5},
        ]
    )
    raw = _news_sentiment_raw(as_of, ["AAPL", "MSFT"], news)
    assert pd.isna(raw["AAPL"])
    assert not pd.isna(raw["MSFT"])


def test_pit_gate_drops_future_macro() -> None:
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-06-15"), "macro_code": "GDP_GROWTH",
             "value": 2.5, "country": "US"},
        ]
    )
    raw = _macro_regime_raw(as_of, ["AAPL"], macro, "GDP_GROWTH")
    assert pd.isna(raw["AAPL"])


# ---------------------------------------------------------------------------
# Sentiment mechanics
# ---------------------------------------------------------------------------


def test_sentiment_directionality() -> None:
    """Positive sentiment → higher z than negative."""
    as_of = pd.Timestamp("2026-05-31")
    recent = as_of - pd.Timedelta(days=3)
    news = pd.DataFrame(
        [
            {"symbol": "AAPL", "timestamp": recent, "sentiment_score": 0.9},
            {"symbol": "MSFT", "timestamp": recent, "sentiment_score": -0.8},
            {"symbol": "NVDA", "timestamp": recent, "sentiment_score": 0.0},
        ]
    )
    out = compute_news_macro_factors(as_of, ["AAPL", "MSFT", "NVDA"], news, _empty_macro())
    z = out["news_sentiment_7d_z"]
    assert z["AAPL"] > z["NVDA"] > z["MSFT"]


def test_sentiment_window_boundary() -> None:
    """Data 10 days old is outside the 7-day window."""
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [
            {"symbol": "AAPL", "timestamp": as_of - pd.Timedelta(days=3),
             "sentiment_score": 0.5},
            {"symbol": "MSFT", "timestamp": as_of - pd.Timedelta(days=10),
             "sentiment_score": 0.8},
        ]
    )
    raw = _news_sentiment_raw(as_of, ["AAPL", "MSFT"], news)
    assert not pd.isna(raw["AAPL"])
    assert pd.isna(raw["MSFT"])  # outside 7-day window


# ---------------------------------------------------------------------------
# Volume spike
# ---------------------------------------------------------------------------


def test_volume_spike_detection() -> None:
    """Recent high volume vs 30-day baseline produces spike > 1."""
    as_of = pd.Timestamp("2026-05-31")
    rows = []
    # AAPL: low baseline, then high recent
    for d in range(30, 7, -1):
        rows.append({"symbol": "AAPL",
                      "timestamp": as_of - pd.Timedelta(days=d),
                      "sentiment_score": 0.0, "sentiment_volume": 10})
    for d in range(7, 0, -1):
        rows.append({"symbol": "AAPL",
                      "timestamp": as_of - pd.Timedelta(days=d),
                      "sentiment_score": 0.0, "sentiment_volume": 100})
    # MSFT: steady
    for d in range(30, 0, -1):
        rows.append({"symbol": "MSFT",
                      "timestamp": as_of - pd.Timedelta(days=d),
                      "sentiment_score": 0.0, "sentiment_volume": 50})
    news = pd.DataFrame(rows)
    raw = _news_volume_spike_raw(as_of, ["AAPL", "MSFT"], news)
    assert raw["AAPL"] > raw["MSFT"]  # AAPL has a spike


def test_volume_spike_no_volume_col() -> None:
    """If sentiment_volume column is missing, returns all NaN."""
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [{"symbol": "AAPL", "timestamp": as_of - pd.Timedelta(days=1),
          "sentiment_score": 0.5}]
    )
    raw = _news_volume_spike_raw(as_of, ["AAPL"], news)
    assert pd.isna(raw["AAPL"])


# ---------------------------------------------------------------------------
# Macro regime
# ---------------------------------------------------------------------------


def test_macro_broadcast_to_all_symbols() -> None:
    """Macro value is the same for all symbols (market-wide)."""
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-05-15"), "macro_code": "GDP_GROWTH",
          "value": 2.0, "country": "US"}]
    )
    raw = _macro_regime_raw(as_of, ["AAPL", "MSFT", "NVDA"], macro, "GDP_GROWTH")
    assert raw["AAPL"] == raw["MSFT"] == raw["NVDA"] == 2.0


def test_macro_country_filter() -> None:
    """Only matching country is used."""
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-05-15"), "macro_code": "GDP_GROWTH",
             "value": 3.0, "country": "EU"},
            {"timestamp": pd.Timestamp("2026-05-15"), "macro_code": "GDP_GROWTH",
             "value": 2.0, "country": "US"},
        ]
    )
    raw = _macro_regime_raw(as_of, ["AAPL"], macro, "GDP_GROWTH", country="EU")
    assert raw["AAPL"] == 3.0


def test_macro_uses_latest_filing() -> None:
    """When multiple values exist, the most recent is used."""
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-04-01"), "macro_code": "GDP_GROWTH",
             "value": 1.0, "country": "US"},
            {"timestamp": pd.Timestamp("2026-05-15"), "macro_code": "GDP_GROWTH",
             "value": 2.5, "country": "US"},
        ]
    )
    raw = _macro_regime_raw(as_of, ["AAPL"], macro, "GDP_GROWTH")
    assert raw["AAPL"] == 2.5


# ---------------------------------------------------------------------------
# Clipping and cross-section
# ---------------------------------------------------------------------------


def test_clipping_bounds() -> None:
    """Extreme sentiment gets clipped to +/- 3.0."""
    as_of = pd.Timestamp("2026-05-31")
    recent = as_of - pd.Timedelta(days=2)
    rows = []
    for i in range(11):
        rows.append({"symbol": f"S{i}", "timestamp": recent, "sentiment_score": 0.01 * i})
    rows.append({"symbol": "OUT", "timestamp": recent, "sentiment_score": 100.0})
    news = pd.DataFrame(rows)
    syms = [f"S{i}" for i in range(11)] + ["OUT"]
    out = compute_news_macro_factors(as_of, syms, news, _empty_macro())
    assert out.loc["OUT", "news_sentiment_7d_z"] == pytest.approx(3.0, abs=1e-9)


def test_single_observation_returns_nan() -> None:
    """One valid observation can't be z-scored."""
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [{"symbol": "AAPL", "timestamp": as_of - pd.Timedelta(days=1),
          "sentiment_score": 0.5}]
    )
    out = compute_news_macro_factors(as_of, ["AAPL"], news, _empty_macro())
    assert pd.isna(out.loc["AAPL", "news_sentiment_7d_z"])


def test_identical_macro_zscore_is_zero() -> None:
    """When all symbols have same macro value, z-score = 0.0 (degenerate)."""
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-05-15"), "macro_code": "GDP_GROWTH",
          "value": 2.0, "country": "US"}]
    )
    out = compute_news_macro_factors(
        as_of, ["AAPL", "MSFT", "NVDA"], _empty_news(), macro
    )
    # All identical → degenerate z-score → all 0.0
    for sym in ["AAPL", "MSFT", "NVDA"]:
        assert out.loc[sym, "macro_growth_momentum_z"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_news_column_raises() -> None:
    as_of = pd.Timestamp("2026-05-31")
    bad_news = pd.DataFrame([{"symbol": "AAPL", "timestamp": as_of}])
    with pytest.raises(ValueError, match="news_df"):
        compute_news_macro_factors(as_of, ["AAPL"], bad_news, _empty_macro())


def test_missing_macro_column_raises() -> None:
    as_of = pd.Timestamp("2026-05-31")
    bad_macro = pd.DataFrame([{"timestamp": as_of}])
    with pytest.raises(ValueError, match="macro_df"):
        compute_news_macro_factors(as_of, ["AAPL"], _empty_news(), bad_macro)


def test_non_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="as_of_date"):
        compute_news_macro_factors("2026-05-31", ["AAPL"], _empty_news(), _empty_macro())


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_news_and_macro_returns_all_nan() -> None:
    as_of = pd.Timestamp("2026-05-31")
    out = compute_news_macro_factors(as_of, ["AAPL", "MSFT"], _empty_news(), _empty_macro())
    assert out.shape == (2, 4)
    assert out.isna().all().all()
