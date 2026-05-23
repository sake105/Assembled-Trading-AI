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
    _macro_timeseries_zscore,
    _news_sentiment_raw,
    _news_volume_spike_raw,
    compute_news_macro_factors,
)

pytestmark = pytest.mark.fast


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
            {
                "symbol": "AAPL",
                "timestamp": pd.Timestamp("2026-06-01"),
                "sentiment_score": 0.9,
            },
            {
                "symbol": "MSFT",
                "timestamp": pd.Timestamp("2026-05-28"),
                "sentiment_score": 0.5,
            },
        ]
    )
    raw = _news_sentiment_raw(as_of, ["AAPL", "MSFT"], news)
    assert pd.isna(raw["AAPL"])
    assert not pd.isna(raw["MSFT"])


def test_pit_gate_drops_future_macro() -> None:
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-06-15"),
                "macro_code": "GDP_GROWTH",
                "value": 2.5,
                "country": "US",
            },
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
    out = compute_news_macro_factors(
        as_of, ["AAPL", "MSFT", "NVDA"], news, _empty_macro()
    )
    z = out["news_sentiment_7d_z"]
    assert z["AAPL"] > z["NVDA"] > z["MSFT"]


def test_sentiment_window_boundary() -> None:
    """Data 10 days old is outside the 7-day window."""
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "timestamp": as_of - pd.Timedelta(days=3),
                "sentiment_score": 0.5,
            },
            {
                "symbol": "MSFT",
                "timestamp": as_of - pd.Timedelta(days=10),
                "sentiment_score": 0.8,
            },
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
        rows.append(
            {
                "symbol": "AAPL",
                "timestamp": as_of - pd.Timedelta(days=d),
                "sentiment_score": 0.0,
                "sentiment_volume": 10,
            }
        )
    for d in range(7, 0, -1):
        rows.append(
            {
                "symbol": "AAPL",
                "timestamp": as_of - pd.Timedelta(days=d),
                "sentiment_score": 0.0,
                "sentiment_volume": 100,
            }
        )
    # MSFT: steady
    for d in range(30, 0, -1):
        rows.append(
            {
                "symbol": "MSFT",
                "timestamp": as_of - pd.Timedelta(days=d),
                "sentiment_score": 0.0,
                "sentiment_volume": 50,
            }
        )
    news = pd.DataFrame(rows)
    raw = _news_volume_spike_raw(as_of, ["AAPL", "MSFT"], news)
    assert raw["AAPL"] > raw["MSFT"]  # AAPL has a spike


def test_volume_spike_no_volume_col() -> None:
    """If sentiment_volume column is missing, returns all NaN."""
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "timestamp": as_of - pd.Timedelta(days=1),
                "sentiment_score": 0.5,
            }
        ]
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
        [
            {
                "timestamp": pd.Timestamp("2026-05-15"),
                "macro_code": "GDP_GROWTH",
                "value": 2.0,
                "country": "US",
            }
        ]
    )
    raw = _macro_regime_raw(as_of, ["AAPL", "MSFT", "NVDA"], macro, "GDP_GROWTH")
    assert raw["AAPL"] == raw["MSFT"] == raw["NVDA"] == 2.0


def test_macro_country_filter() -> None:
    """Only matching country is used."""
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-15"),
                "macro_code": "GDP_GROWTH",
                "value": 3.0,
                "country": "EU",
            },
            {
                "timestamp": pd.Timestamp("2026-05-15"),
                "macro_code": "GDP_GROWTH",
                "value": 2.0,
                "country": "US",
            },
        ]
    )
    raw = _macro_regime_raw(as_of, ["AAPL"], macro, "GDP_GROWTH", country="EU")
    assert raw["AAPL"] == 3.0


def test_macro_uses_latest_filing() -> None:
    """When multiple values exist, the most recent is used."""
    as_of = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01"),
                "macro_code": "GDP_GROWTH",
                "value": 1.0,
                "country": "US",
            },
            {
                "timestamp": pd.Timestamp("2026-05-15"),
                "macro_code": "GDP_GROWTH",
                "value": 2.5,
                "country": "US",
            },
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
        rows.append(
            {"symbol": f"S{i}", "timestamp": recent, "sentiment_score": 0.01 * i}
        )
    rows.append({"symbol": "OUT", "timestamp": recent, "sentiment_score": 100.0})
    news = pd.DataFrame(rows)
    syms = [f"S{i}" for i in range(11)] + ["OUT"]
    out = compute_news_macro_factors(as_of, syms, news, _empty_macro())
    assert out.loc["OUT", "news_sentiment_7d_z"] == pytest.approx(3.0, abs=1e-9)


def test_single_observation_returns_nan() -> None:
    """One valid observation can't be z-scored."""
    as_of = pd.Timestamp("2026-05-31")
    news = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "timestamp": as_of - pd.Timedelta(days=1),
                "sentiment_score": 0.5,
            }
        ]
    )
    out = compute_news_macro_factors(as_of, ["AAPL"], news, _empty_macro())
    assert pd.isna(out.loc["AAPL", "news_sentiment_7d_z"])


def test_identical_macro_zscore_is_zero() -> None:
    """When all historical macro values are identical → zero variance → all 0.0.

    Uses yield_curve_spread which is the actual code compute_news_macro_factors
    looks up for macro_growth_momentum_z. With time-series z-scoring, when all
    historical readings are the same (std=0), the z-score is 0.0 (genuine neutral).
    Needs >= MIN_MACRO_OBS=6 history points to pass the stability guard.
    """
    as_of = pd.Timestamp("2026-05-31")
    # 7 rows: 6 history (all 2.0) + 1 latest (also 2.0) → std=0 → z=0.0
    macro = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-15") - pd.Timedelta(days=30 * i),
                "macro_code": "yield_curve_spread",
                "value": 2.0,
                "country": "US",
            }
            for i in range(7)  # i=0 is latest (2026-05-15), i=1..6 are history
        ]
    )
    out = compute_news_macro_factors(
        as_of, ["AAPL", "MSFT", "NVDA"], _empty_news(), macro
    )
    # All historical values identical → zero variance → z=0.0 broadcast to all symbols
    for sym in ["AAPL", "MSFT", "NVDA"]:
        assert out.loc[sym, "macro_growth_momentum_z"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Time-series z-score (_macro_timeseries_zscore)
# ---------------------------------------------------------------------------


def test_macro_timeseries_zscore_direction() -> None:
    """Latest value well above historical mean → positive z; below → negative."""
    as_of_naive = pd.Timestamp("2026-05-31")
    # 12 varied history values centred ~0.5 (variance > 0), latest=1.5 → positive z
    _hist_vals = [0.3, 0.4, 0.6, 0.5, 0.7, 0.4, 0.3, 0.6, 0.5, 0.4, 0.6, 0.5]
    rows = [
        {
            "timestamp": pd.Timestamp("2026-05-15") - pd.Timedelta(days=30 * (i + 1)),
            "macro_code": "yield_curve_spread",
            "value": v,
            "country": "US",
        }
        for i, v in enumerate(_hist_vals)
    ]
    rows.append(
        {
            "timestamp": pd.Timestamp("2026-05-15"),
            "macro_code": "yield_curve_spread",
            "value": 1.5,
            "country": "US",
        }
    )
    macro = pd.DataFrame(rows)
    z_high = _macro_timeseries_zscore(macro, "yield_curve_spread", as_of_naive)
    assert z_high > 0.0

    # Flip latest to well below mean → negative z
    rows[-1]["value"] = -0.5
    macro2 = pd.DataFrame(rows)
    z_low = _macro_timeseries_zscore(macro2, "yield_curve_spread", as_of_naive)
    assert z_low < 0.0


def test_macro_timeseries_zscore_insufficient_history_returns_nan() -> None:
    """Fewer than MIN_MACRO_OBS history points → return NaN (noisy z-score suppressed)."""
    as_of_naive = pd.Timestamp("2026-05-31")
    # 3 rows total: 2 history + 1 latest — less than MIN_MACRO_OBS=6 history points
    macro = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-01"),
                "macro_code": "yield_curve_spread",
                "value": 2.0,
                "country": "US",
            },
            {
                "timestamp": pd.Timestamp("2026-05-08"),
                "macro_code": "yield_curve_spread",
                "value": 2.5,
                "country": "US",
            },
            {
                "timestamp": pd.Timestamp("2026-05-15"),
                "macro_code": "yield_curve_spread",
                "value": 3.0,
                "country": "US",
            },
        ]
    )
    result = _macro_timeseries_zscore(macro, "yield_curve_spread", as_of_naive)
    assert pd.isna(result)  # insufficient history → NaN, not a noisy z-score


def test_macro_timeseries_zscore_pit_gate() -> None:
    """Future rows must be excluded from the historical distribution.

    The fixture is designed so that the correct-PIT path yields an unclipped
    positive z, while a broken-PIT path (future value leaks in as 'latest')
    yields a NEGATIVE z. This makes the assertion discriminating: z > 0.0
    passes only when PIT works correctly.

    Needs >= MIN_MACRO_OBS=6 history rows. Uses 8 history rows + 1 latest.
    """
    as_of_naive = pd.Timestamp("2026-01-31")
    # 8 history rows with wide spread (std ≈ 1.73) so latest=3.5 gives
    # unclipped positive z ≈ 0.87. A broken PIT gate including the future
    # row (value=-5.0) would flip z to negative.
    _hist_vals_pit = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0]  # mean=2.0, std≈1.73
    rows = [
        {
            "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(days=30 * i),
            "macro_code": "cpi_yoy",
            "value": v,
            "country": "US",
        }
        for i, v in enumerate(_hist_vals_pit)
    ]
    rows.append(
        # latest valid: 3.5 → z = (3.5 - 2.0) / 1.73 ≈ +0.87, unclipped
        {
            "timestamp": pd.Timestamp("2026-01-15"),
            "macro_code": "cpi_yoy",
            "value": 3.5,
            "country": "US",
        }
    )
    rows.append(
        # future — must be excluded: if leaked, -5.0 becomes latest → z < 0
        {
            "timestamp": pd.Timestamp("2026-06-01"),
            "macro_code": "cpi_yoy",
            "value": -5.0,
            "country": "US",
        }
    )
    macro = pd.DataFrame(rows)
    z = _macro_timeseries_zscore(macro, "cpi_yoy", as_of_naive)
    # Correct PIT: z ≈ +0.87 (positive, unclipped). Broken PIT: z < 0.
    assert 0.0 < z < 3.0


def test_macro_timeseries_zscore_missing_code_returns_nan() -> None:
    """Code not in macro_df → NaN."""
    as_of_naive = pd.Timestamp("2026-05-31")
    macro = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-01"),
                "macro_code": "cpi_yoy",
                "value": 3.0,
                "country": "US",
            }
        ]
    )
    result = _macro_timeseries_zscore(macro, "nonexistent_code", as_of_naive)
    assert pd.isna(result)


def test_macro_timeseries_zscore_clipped() -> None:
    """Extreme outlier gets clipped to +-3.0."""
    as_of_naive = pd.Timestamp("2026-05-31")
    _hist_vals_clip = [0.0, 0.2, 0.1, 0.0, 0.2, 0.1, 0.0, 0.2, 0.1, 0.0, 0.2]
    rows = [
        {
            "timestamp": pd.Timestamp("2026-05-01") - pd.Timedelta(days=i * 30),
            "macro_code": "cpi_yoy",
            "value": v,
            "country": "US",
        }
        for i, v in enumerate(_hist_vals_clip, 1)
    ]
    rows.append(
        {
            "timestamp": pd.Timestamp("2026-05-01"),
            "macro_code": "cpi_yoy",
            "value": 100.0,
            "country": "US",
        }
    )
    macro = pd.DataFrame(rows)
    z = _macro_timeseries_zscore(macro, "cpi_yoy", as_of_naive)
    assert z == pytest.approx(3.0, abs=1e-9)


def test_macro_timeseries_broadcasts_to_all_symbols() -> None:
    """compute_news_macro_factors broadcasts the scalar to every symbol.

    Uses 7 history rows + 1 latest to exceed MIN_MACRO_OBS=6 guard.
    """
    as_of = pd.Timestamp("2026-05-31")
    # 7 history rows (varied ~0.5 to ensure std>0) + 1 latest (1.5) → z > 0 broadcast
    _hist_vals_bc = [0.3, 0.4, 0.6, 0.5, 0.7, 0.4, 0.6]
    rows = [
        {
            "timestamp": pd.Timestamp("2026-05-01") - pd.Timedelta(days=30 * i),
            "macro_code": "yield_curve_spread",
            "value": v,
            "country": "US",
        }
        for i, v in enumerate(_hist_vals_bc, 1)
    ]
    rows.append(
        {
            "timestamp": pd.Timestamp("2026-05-15"),
            "macro_code": "yield_curve_spread",
            "value": 1.5,
            "country": "US",
        }
    )
    macro = pd.DataFrame(rows)
    out = compute_news_macro_factors(
        as_of, ["AAPL", "MSFT", "NVDA"], _empty_news(), macro
    )
    # All symbols share the same time-series z-score (market-wide broadcast)
    z_vals = out["macro_growth_momentum_z"].values
    assert z_vals[0] == pytest.approx(z_vals[1])
    assert z_vals[1] == pytest.approx(z_vals[2])
    assert z_vals[0] > 0.0  # latest 1.5 > historical mean 0.5


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
        compute_news_macro_factors(
            "2026-05-31", ["AAPL"], _empty_news(), _empty_macro()
        )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_news_and_macro_returns_all_nan() -> None:
    as_of = pd.Timestamp("2026-05-31")
    out = compute_news_macro_factors(
        as_of, ["AAPL", "MSFT"], _empty_news(), _empty_macro()
    )
    assert out.shape == (2, 4)
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# Timezone regression — tz-aware as_of_date vs tz-naive dataframe timestamps
# ---------------------------------------------------------------------------


def test_tz_aware_as_of_does_not_raise() -> None:
    """tz-aware as_of_date must not raise TypeError when comparing with tz-naive data.

    Regression: altdata_loader strips tz from parquet timestamps via
    .dt.tz_localize(None). The helpers used to compare tz-naive df["timestamp"]
    with tz-aware as_of_date → TypeError. This locks the fix.
    """
    as_of_tz = pd.Timestamp("2026-05-15", tz="UTC")  # tz-aware
    syms = ["AAPL", "MSFT"]

    # news_df with tz-naive timestamps (as loaded by altdata_loader)
    news = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pd.to_datetime(["2026-05-10", "2026-05-11"]),  # tz-naive
            "sentiment_score": [0.4, -0.2],
        }
    )
    # macro_df with matching codes so the tz path is also exercised end-to-end.
    # yield_curve_spread → macro_growth_momentum_z; cpi_yoy → macro_inflation_surprise_z.
    macro = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01", "2026-05-08"]),  # tz-naive
            "macro_code": ["yield_curve_spread", "cpi_yoy"],
            "value": [2.5, 3.1],
            "country": ["US", "US"],
        }
    )

    # Must not raise TypeError
    out = compute_news_macro_factors(as_of_tz, syms, news, macro)
    assert out.shape == (2, 4)
    # AAPL and MSFT should have non-NaN sentiment (data is within lookback)
    assert not out["news_sentiment_7d_z"].isna().all()


def test_tz_aware_data_tz_naive_as_of_does_not_raise() -> None:
    """Symmetric case: tz-aware df timestamps with tz-naive as_of_date.

    The helpers strip tz from df["timestamp"] via .dt.tz_localize(None),
    so tz-aware data timestamps must also not raise TypeError.
    """
    as_of_naive = pd.Timestamp("2026-05-15")  # tz-naive
    syms = ["AAPL", "MSFT"]

    # news_df with tz-aware UTC timestamps
    news = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "timestamp": [
                pd.Timestamp("2026-05-10", tz="UTC"),
                pd.Timestamp("2026-05-11", tz="UTC"),
            ],
            "sentiment_score": [0.4, -0.2],
        }
    )
    macro = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-05-01", tz="UTC"),
                pd.Timestamp("2026-05-08", tz="UTC"),
            ],
            "macro_code": ["yield_curve_spread", "cpi_yoy"],
            "value": [2.5, 3.1],
            "country": ["US", "US"],
        }
    )

    # Must not raise TypeError
    out = compute_news_macro_factors(as_of_naive, syms, news, macro)
    assert out.shape == (2, 4)
    assert not out["news_sentiment_7d_z"].isna().all()
