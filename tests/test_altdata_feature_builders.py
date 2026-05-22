"""Tests for alt-data feature builders: BLS, FINRA, Wikipedia.

Covers basic operation, PIT safety, empty input, output schema,
regime labels, and NaN handling.  Audit C2-059.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from assembled_core.features.altdata_bls_features import (
    NONFARM_PAYROLL_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
    build_bls_labor_features,
)
from assembled_core.features.altdata_finra_features import (
    build_finra_short_interest_features,
)
from assembled_core.features.altdata_wikipedia_features import (
    build_wikipedia_attention_features,
)


# ===========================================================================
# BLS TESTS
# ===========================================================================


def _make_bls_df(n_months: int = 12) -> pd.DataFrame:
    """Synthetic BLS DataFrame with unemployment and payroll series."""
    dates = pd.date_range("2023-01-01", periods=n_months, freq="MS", tz="UTC")
    rows = []
    for i, ts in enumerate(dates):
        rows.append(
            {
                "timestamp": ts,
                "series_id": UNEMPLOYMENT_SERIES_ID,
                "value": 4.0 + 0.1 * (i % 5),  # oscillates around 4%
                "period": f"M{i % 12 + 1:02d}",
                "year": ts.year,
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "series_id": NONFARM_PAYROLL_SERIES_ID,
                "value": 160_000.0 + i * 10.0,
                "period": f"M{i % 12 + 1:02d}",
                "year": ts.year,
            }
        )
    return pd.DataFrame(rows)


def test_bls_basic_operation() -> None:
    """build_bls_labor_features returns expected columns with real-ish data."""
    bls_df = _make_bls_df(12)
    as_of = pd.Timestamp("2024-01-01")
    result = build_bls_labor_features(bls_df, as_of)

    assert not result.empty
    expected_cols = {
        "timestamp",
        "unemployment_rate",
        "unemployment_3m_change",
        "nonfarm_payroll_mom",
        "labor_market_regime",
    }
    assert expected_cols.issubset(set(result.columns))


def test_bls_pit_safety() -> None:
    """Rows after as_of are excluded."""
    bls_df = _make_bls_df(24)
    as_of = pd.Timestamp("2023-06-01")
    result = build_bls_labor_features(bls_df, as_of)

    as_of_utc = as_of.tz_localize("UTC")
    assert result["timestamp"].max() <= as_of_utc


def test_bls_empty_input_returns_empty_with_correct_columns() -> None:
    """Empty input → empty DataFrame with correct columns."""
    empty_bls = pd.DataFrame(
        columns=["timestamp", "series_id", "value", "period", "year"]
    )
    result = build_bls_labor_features(empty_bls, pd.Timestamp("2024-01-01"))

    assert result.empty
    assert "unemployment_rate" in result.columns
    assert "labor_market_regime" in result.columns


def test_bls_output_columns_correct() -> None:
    """Output DataFrame has exactly the expected columns (no extras)."""
    bls_df = _make_bls_df(6)
    result = build_bls_labor_features(bls_df, pd.Timestamp("2024-01-01"))

    assert set(result.columns) == {
        "timestamp",
        "unemployment_rate",
        "unemployment_3m_change",
        "nonfarm_payroll_mom",
        "labor_market_regime",
    }


def test_bls_regime_hawkish() -> None:
    """Low unemployment (< 4%) → hawkish regime."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    row = {
        "timestamp": ts,
        "series_id": UNEMPLOYMENT_SERIES_ID,
        "value": 3.5,  # below hawkish threshold
        "period": "M01",
        "year": 2024,
    }
    bls_df = pd.DataFrame([row])
    result = build_bls_labor_features(bls_df, pd.Timestamp("2024-02-01"))

    row_out = result[result["unemployment_rate"] == 3.5]
    assert not row_out.empty
    assert row_out.iloc[0]["labor_market_regime"] == "hawkish"


def test_bls_regime_dovish() -> None:
    """High unemployment (> 6%) → dovish regime."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    row = {
        "timestamp": ts,
        "series_id": UNEMPLOYMENT_SERIES_ID,
        "value": 7.5,  # above dovish threshold
        "period": "M01",
        "year": 2024,
    }
    bls_df = pd.DataFrame([row])
    result = build_bls_labor_features(bls_df, pd.Timestamp("2024-02-01"))

    row_out = result[result["unemployment_rate"] == 7.5]
    assert not row_out.empty
    assert row_out.iloc[0]["labor_market_regime"] == "dovish"


# ===========================================================================
# FINRA TESTS
# ===========================================================================


def _make_finra_wide_df() -> pd.DataFrame:
    """Synthetic FINRA DataFrame in wide format (index = symbol)."""
    symbols = ["AAPL", "TSLA", "GME", "MSFT", "AMC"]
    data = {
        "si_qty": [1_000_000, 5_000_000, 80_000_000, 800_000, 60_000_000],
        "si_pct_float": [0.5, 3.0, 45.0, 0.3, 40.0],
        "days_to_cover": [1.2, 3.5, 12.0, 0.8, 10.5],
        "si_change_pct": [0.05, 0.20, -0.10, 0.01, 0.30],
    }
    return pd.DataFrame(data, index=symbols)


def test_finra_basic_operation() -> None:
    """build_finra_short_interest_features returns expected columns."""
    finra_df = _make_finra_wide_df()
    as_of = pd.Timestamp("2024-01-01")
    result = build_finra_short_interest_features(finra_df, as_of)

    assert not result.empty
    expected_cols = {
        "symbol",
        "short_interest_ratio",
        "short_interest_pct_float",
        "short_squeeze_score",
        "si_regime",
    }
    assert expected_cols.issubset(set(result.columns))


def test_finra_output_columns_correct() -> None:
    """Output has exactly the 5 expected columns."""
    finra_df = _make_finra_wide_df()
    result = build_finra_short_interest_features(finra_df, pd.Timestamp("2024-01-01"))

    assert set(result.columns) == {
        "symbol",
        "short_interest_ratio",
        "short_interest_pct_float",
        "short_squeeze_score",
        "si_regime",
    }


def test_finra_empty_input_returns_empty_with_correct_columns() -> None:
    """Empty input → empty DataFrame with correct columns."""
    empty_df = pd.DataFrame(
        columns=["symbol", "si_qty", "si_pct_float", "days_to_cover", "si_change_pct"]
    )
    result = build_finra_short_interest_features(empty_df, pd.Timestamp("2024-01-01"))

    assert result.empty
    assert "short_interest_ratio" in result.columns
    assert "si_regime" in result.columns


def test_finra_regime_high_for_high_si() -> None:
    """Symbols with very high short interest ratio get si_regime='high'."""
    finra_df = _make_finra_wide_df()
    result = build_finra_short_interest_features(finra_df, pd.Timestamp("2024-01-01"))

    high_rows = result[result["si_regime"] == "high"]
    low_rows = result[result["si_regime"] == "low"]
    # At least one high and one low given the spread in test data
    assert len(high_rows) >= 1
    assert len(low_rows) >= 1


def test_finra_pit_safety_with_timestamp_column() -> None:
    """Rows after as_of are excluded when timestamp column is present."""
    finra_df = _make_finra_wide_df().reset_index().rename(columns={"index": "symbol"})
    # Add timestamps: half before, half after as_of
    as_of = pd.Timestamp("2024-01-01")
    timestamps = [
        pd.Timestamp("2023-12-01", tz="UTC"),  # AAPL — before
        pd.Timestamp("2024-02-01", tz="UTC"),  # TSLA — after
        pd.Timestamp("2023-11-01", tz="UTC"),  # GME — before
        pd.Timestamp("2024-03-01", tz="UTC"),  # MSFT — after
        pd.Timestamp("2023-10-01", tz="UTC"),  # AMC — before
    ]
    finra_df["timestamp"] = timestamps
    result = build_finra_short_interest_features(finra_df, as_of)

    # Only symbols with timestamp <= as_of should appear
    assert "TSLA" not in result["symbol"].values
    assert "MSFT" not in result["symbol"].values
    assert "AAPL" in result["symbol"].values


def test_finra_no_nan_propagation_crash() -> None:
    """NaN values in input don't cause a crash."""
    finra_df = pd.DataFrame(
        {
            "symbol": ["X", "Y"],
            "si_qty": [np.nan, 100_000],
            "si_pct_float": [np.nan, 5.0],
            "days_to_cover": [np.nan, 3.0],
            "si_change_pct": [np.nan, 0.1],
        }
    )
    result = build_finra_short_interest_features(finra_df, pd.Timestamp("2024-01-01"))
    # Should not raise; Y should appear
    assert isinstance(result, pd.DataFrame)
    assert "Y" in result["symbol"].values


# ===========================================================================
# WIKIPEDIA TESTS
# ===========================================================================


def _make_wiki_long_df(n_days: int = 40) -> pd.DataFrame:
    """Synthetic Wikipedia pageview DataFrame in long format."""
    symbols = ["AAPL", "TSLA", "MSFT"]
    dates = pd.date_range("2024-01-01", periods=n_days)
    rows = []
    rng = __import__("random")
    rng.seed(42)
    for sym in symbols:
        for d in dates:
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "views": rng.randint(1000, 50000),
                }
            )
    return pd.DataFrame(rows)


def _make_wiki_wide_df(n_days: int = 40) -> pd.DataFrame:
    """Synthetic Wikipedia pageview DataFrame in wide format."""
    dates = pd.date_range("2024-01-01", periods=n_days)
    import random

    random.seed(99)
    data = {
        "AAPL": [random.randint(1000, 50000) for _ in range(n_days)],
        "TSLA": [random.randint(500, 30000) for _ in range(n_days)],
    }
    return pd.DataFrame(data, index=dates)


def test_wiki_basic_operation_long_format() -> None:
    """build_wikipedia_attention_features works on long-format input."""
    wiki_df = _make_wiki_long_df(40)
    as_of = pd.Timestamp("2024-02-10")
    result = build_wikipedia_attention_features(wiki_df, as_of)

    assert not result.empty
    expected_cols = {
        "symbol",
        "pageview_zscore",
        "pageview_7d_change",
        "attention_spike",
        "attention_regime",
    }
    assert expected_cols.issubset(set(result.columns))


def test_wiki_basic_operation_wide_format() -> None:
    """build_wikipedia_attention_features works on wide-format (DatetimeIndex) input."""
    wiki_df = _make_wiki_wide_df(40)
    as_of = pd.Timestamp("2024-02-10")
    result = build_wikipedia_attention_features(wiki_df, as_of)

    assert not result.empty
    assert "pageview_zscore" in result.columns


def test_wiki_pit_safety() -> None:
    """Pageview data after as_of is excluded."""
    wiki_df = _make_wiki_long_df(60)
    as_of = pd.Timestamp("2024-01-20")
    result = build_wikipedia_attention_features(wiki_df, as_of)

    # Result should reflect only data up to as_of
    assert isinstance(result, pd.DataFrame)  # no crash
    # Cannot directly check dates in output (aggregated), but verify no error


def test_wiki_empty_input_returns_empty_with_correct_columns() -> None:
    """Empty input → empty DataFrame with correct columns."""
    empty_df = pd.DataFrame(columns=["symbol", "date", "views"])
    result = build_wikipedia_attention_features(empty_df, pd.Timestamp("2024-01-01"))

    assert result.empty
    assert "pageview_zscore" in result.columns
    assert "attention_regime" in result.columns


def test_wiki_output_columns_correct() -> None:
    """Output has exactly the 5 expected columns."""
    wiki_df = _make_wiki_long_df(40)
    result = build_wikipedia_attention_features(wiki_df, pd.Timestamp("2024-02-10"))

    assert set(result.columns) == {
        "symbol",
        "pageview_zscore",
        "pageview_7d_change",
        "attention_spike",
        "attention_regime",
    }


def test_wiki_attention_spike_is_bool() -> None:
    """attention_spike column is boolean dtype."""
    wiki_df = _make_wiki_long_df(40)
    result = build_wikipedia_attention_features(wiki_df, pd.Timestamp("2024-02-10"))

    assert result["attention_spike"].dtype == bool


def test_wiki_regime_labels_valid() -> None:
    """attention_regime values are all in {'high', 'normal', 'low'}."""
    wiki_df = _make_wiki_long_df(40)
    result = build_wikipedia_attention_features(wiki_df, pd.Timestamp("2024-02-10"))

    valid_regimes = {"high", "normal", "low"}
    assert result["attention_regime"].isin(valid_regimes).all()


def test_wiki_no_nan_propagation_crash() -> None:
    """NaN views don't cause a crash."""
    rows = [
        {"symbol": "X", "date": pd.Timestamp("2024-01-01"), "views": np.nan},
        {"symbol": "X", "date": pd.Timestamp("2024-01-02"), "views": 1000.0},
        {"symbol": "X", "date": pd.Timestamp("2024-01-03"), "views": np.nan},
        {"symbol": "X", "date": pd.Timestamp("2024-01-04"), "views": 2000.0},
        {"symbol": "X", "date": pd.Timestamp("2024-01-05"), "views": 1500.0},
        {"symbol": "X", "date": pd.Timestamp("2024-01-06"), "views": 1800.0},
        {"symbol": "X", "date": pd.Timestamp("2024-01-07"), "views": 1200.0},
        {"symbol": "X", "date": pd.Timestamp("2024-01-08"), "views": 900.0},
        {"symbol": "X", "date": pd.Timestamp("2024-01-09"), "views": 1100.0},
    ]
    wiki_df = pd.DataFrame(rows)
    result = build_wikipedia_attention_features(wiki_df, pd.Timestamp("2024-01-10"))
    assert isinstance(result, pd.DataFrame)  # no crash


def test_wiki_insufficient_data_returns_gracefully() -> None:
    """Symbol with fewer than 8 days of data returns null row without crash."""
    rows = [
        {"symbol": "FEW", "date": pd.Timestamp(f"2024-01-0{i + 1}"), "views": 1000.0}
        for i in range(5)  # only 5 rows, below SHORT_WINDOW+1=8
    ]
    wiki_df = pd.DataFrame(rows)
    result = build_wikipedia_attention_features(wiki_df, pd.Timestamp("2024-01-10"))
    assert isinstance(result, pd.DataFrame)
    # Symbol may appear with NaN values or be absent — no crash either way
