"""Equivalence tests for vectorized vs legacy event features (Sprint 11.E1).

Tests verify that vectorized and legacy implementations produce identical outputs
for the same inputs, ensuring PIT-safety and deterministic behavior are preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.event_features import (
    add_disclosure_count_feature,
    build_event_feature_panel,
)


def _generate_synthetic_events(
    n_events: int = 500,
    n_symbols: int = 50,
    start_date: pd.Timestamp = pd.Timestamp("2020-01-01", tz="UTC"),
    end_date: pd.Timestamp = pd.Timestamp("2024-12-31", tz="UTC"),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic events for testing (deterministic with seed)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate random event dates
    date_range = pd.date_range(start_date, end_date, freq="D", tz="UTC")
    event_dates = rng.choice(date_range, size=n_events, replace=True)

    # Generate disclosure dates (T+0 to T+5 days after event)
    disclosure_dates = event_dates + pd.to_timedelta(rng.integers(0, 6, size=n_events), unit="D")

    # Generate symbols
    symbols = [f"SYM{i % n_symbols:03d}" for i in range(n_events)]

    # Generate values (optional)
    values = rng.uniform(100.0, 10000.0, size=n_events)

    events = pd.DataFrame({
        "symbol": symbols,
        "event_date": event_dates,
        "disclosure_date": disclosure_dates,
        "value": values,
    })

    return events


def _generate_synthetic_prices(
    n_timestamps: int = 2000,
    n_symbols: int = 200,
    start_date: pd.Timestamp = pd.Timestamp("2020-01-01", tz="UTC"),
    end_date: pd.Timestamp = pd.Timestamp("2024-12-31", tz="UTC"),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic price panel for testing (deterministic with seed)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate timestamps (daily)
    date_range = pd.date_range(start_date, end_date, freq="D", tz="UTC")
    # Limit n_timestamps to available dates
    max_timestamps = len(date_range) * n_symbols
    n_timestamps = min(n_timestamps, max_timestamps)

    # Generate prices panel: each symbol gets same number of timestamps
    timestamps_per_symbol = n_timestamps // n_symbols
    prices_list = []

    for i in range(n_symbols):
        symbol = f"SYM{i:03d}"
        # Select random dates for this symbol
        symbol_dates = rng.choice(date_range, size=min(timestamps_per_symbol, len(date_range)), replace=False)
        symbol_dates = np.sort(symbol_dates)

        for date in symbol_dates:
            prices_list.append({
                "timestamp": date,
                "symbol": symbol,
                "close": 100.0 + rng.uniform(-10, 10),
            })

    prices_df = pd.DataFrame(prices_list)

    return prices_df


def test_build_event_feature_panel_equivalence() -> None:
    """Test that legacy and vectorized implementations produce identical outputs."""
    # Generate synthetic data (deterministic)
    events = _generate_synthetic_events(n_events=500, n_symbols=50, seed=42)
    prices = _generate_synthetic_prices(n_timestamps=2000, n_symbols=200, seed=42)

    # Set as_of to end of date range
    as_of = pd.Timestamp("2024-12-31", tz="UTC")

    # Compute features with legacy method
    result_legacy = build_event_feature_panel(
        events,
        prices,
        as_of=as_of,
        lookback_days=30,
        feature_prefix="event",
        method="legacy",
    )

    # Compute features with vectorized method
    result_vectorized = build_event_feature_panel(
        events,
        prices,
        as_of=as_of,
        lookback_days=30,
        feature_prefix="event",
        method="vectorized",
    )

    # Assert: Same columns
    assert set(result_legacy.columns) == set(result_vectorized.columns), (
        f"Column mismatch: legacy={list(result_legacy.columns)}, "
        f"vectorized={list(result_vectorized.columns)}"
    )

    # Assert: Same number of rows
    assert len(result_legacy) == len(result_vectorized), (
        f"Row count mismatch: legacy={len(result_legacy)}, vectorized={len(result_vectorized)}"
    )

    # Assert: Same sorting (deterministic)
    pd.testing.assert_frame_equal(
        result_legacy[["symbol", "timestamp"]],
        result_vectorized[["symbol", "timestamp"]],
        check_dtype=False,
    )

    # Assert: Same feature values
    feature_cols = ["event_count_30d", "event_sum_30d", "event_mean_30d"]

    for col in feature_cols:
        if col not in result_legacy.columns:
            continue

        legacy_values = result_legacy[col]
        vectorized_values = result_vectorized[col]

        # Handle NaN/NA: both should be NaN or both should be non-NaN
        legacy_isna = legacy_values.isna()
        vectorized_isna = vectorized_values.isna()

        assert legacy_isna.equals(vectorized_isna), (
            f"NaN positions differ for {col}: "
            f"legacy NaN count={legacy_isna.sum()}, vectorized NaN count={vectorized_isna.sum()}"
        )

        # Compare non-NaN values
        if not legacy_isna.all():
            legacy_non_nan = legacy_values[~legacy_isna]
            vectorized_non_nan = vectorized_values[~vectorized_isna]

            if col == "event_count_30d":
                # Integer comparison (exact)
                pd.testing.assert_series_equal(
                    legacy_non_nan,
                    vectorized_non_nan,
                    check_dtype=False,
                )
            else:
                # Float comparison (with tolerance)
                np.testing.assert_allclose(
                    legacy_non_nan.values,
                    vectorized_non_nan.values,
                    rtol=1e-10,
                    atol=1e-10,
                )


def test_add_disclosure_count_feature_equivalence() -> None:
    """Test that legacy and vectorized add_disclosure_count_feature produce identical outputs."""
    # Generate synthetic data (deterministic)
    events = _generate_synthetic_events(n_events=300, n_symbols=30, seed=123)
    prices = _generate_synthetic_prices(n_timestamps=1000, n_symbols=100, seed=123)

    # Set as_of
    as_of = pd.Timestamp("2024-12-31", tz="UTC")

    # Compute features with legacy method
    result_legacy = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        out_col="alt_disclosure_count_30d_v1",
        as_of=as_of,
        method="legacy",
    )

    # Compute features with vectorized method
    result_vectorized = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        out_col="alt_disclosure_count_30d_v1",
        as_of=as_of,
        method="vectorized",
    )

    # Assert: Same columns
    assert set(result_legacy.columns) == set(result_vectorized.columns), (
        f"Column mismatch: legacy={list(result_legacy.columns)}, "
        f"vectorized={list(result_vectorized.columns)}"
    )

    # Assert: Same number of rows
    assert len(result_legacy) == len(result_vectorized), (
        f"Row count mismatch: legacy={len(result_legacy)}, vectorized={len(result_vectorized)}"
    )

    # Sort both by symbol, timestamp for comparison
    result_legacy_sorted = result_legacy.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    result_vectorized_sorted = result_vectorized.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Assert: Same sorting (deterministic)
    pd.testing.assert_frame_equal(
        result_legacy_sorted[["symbol", "timestamp"]],
        result_vectorized_sorted[["symbol", "timestamp"]],
        check_dtype=False,
    )

    # Assert: Same feature values (exact integer comparison)
    pd.testing.assert_series_equal(
        result_legacy_sorted["alt_disclosure_count_30d_v1"],
        result_vectorized_sorted["alt_disclosure_count_30d_v1"],
        check_dtype=False,
    )


def test_equivalence_empty_events() -> None:
    """Test equivalence when events DataFrame is empty."""
    prices = _generate_synthetic_prices(n_timestamps=100, n_symbols=10, seed=456)
    events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])

    as_of = pd.Timestamp("2024-12-31", tz="UTC")

    # Legacy
    result_legacy = build_event_feature_panel(
        events,
        prices,
        as_of=as_of,
        lookback_days=30,
        method="legacy",
    )

    # Vectorized
    result_vectorized = build_event_feature_panel(
        events,
        prices,
        as_of=as_of,
        lookback_days=30,
        method="vectorized",
    )

    # Assert: Same outputs
    pd.testing.assert_frame_equal(
        result_legacy,
        result_vectorized,
        check_dtype=False,
    )


def test_equivalence_no_events_for_symbols() -> None:
    """Test equivalence when no events exist for any symbols in prices."""
    # Generate prices with symbols SYM000-SYM099
    prices = _generate_synthetic_prices(n_timestamps=500, n_symbols=100, seed=789)

    # Generate events with different symbols (SYM200-SYM299)
    events = _generate_synthetic_events(n_events=100, n_symbols=50, seed=789)
    # Shift symbols to be different from prices
    events["symbol"] = events["symbol"].str.replace("SYM", "EVT")

    as_of = pd.Timestamp("2024-12-31", tz="UTC")

    # Legacy
    result_legacy = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        as_of=as_of,
        method="legacy",
    )

    # Vectorized
    result_vectorized = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        as_of=as_of,
        method="vectorized",
    )

    # Assert: All counts should be zero
    assert (result_legacy["alt_disclosure_count_30d_v1"] == 0).all()
    assert (result_vectorized["alt_disclosure_count_30d_v1"] == 0).all()

    # Assert: Same outputs
    pd.testing.assert_frame_equal(
        result_legacy,
        result_vectorized,
        check_dtype=False,
    )


def test_equivalence_deterministic_multiple_runs() -> None:
    """Test that both implementations are deterministic (same input -> same output)."""
    events = _generate_synthetic_events(n_events=200, n_symbols=20, seed=999)
    prices = _generate_synthetic_prices(n_timestamps=500, n_symbols=50, seed=999)
    as_of = pd.Timestamp("2024-12-31", tz="UTC")

    # Run legacy twice
    result_legacy_1 = build_event_feature_panel(
        events, prices, as_of=as_of, lookback_days=30, method="legacy"
        )
    result_legacy_2 = build_event_feature_panel(
        events, prices, as_of=as_of, lookback_days=30, method="legacy"
        )

    # Run vectorized twice
    result_vectorized_1 = build_event_feature_panel(
        events, prices, as_of=as_of, lookback_days=30, method="vectorized"
    )
    result_vectorized_2 = build_event_feature_panel(
        events, prices, as_of=as_of, lookback_days=30, method="vectorized"
    )

    # Assert: Legacy is deterministic
    pd.testing.assert_frame_equal(
        result_legacy_1,
        result_legacy_2,
        check_dtype=False,
    )

    # Assert: Vectorized is deterministic
    pd.testing.assert_frame_equal(
        result_vectorized_1,
        result_vectorized_2,
        check_dtype=False,
    )
