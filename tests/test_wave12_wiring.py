"""Tests for wave-12 module wiring into trading_cycle.py.

Covers:
  Step 2.8 — features.mean_reversion_factors (compute_mean_reversion_factors)
  Step 2.9 — features.interaction_features (compute_interaction_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.mean_reversion_factors import compute_mean_reversion_factors
from src.assembled_core.features.interaction_features import (
    compute_interaction_features,
    DEFAULT_INTERACTIONS,
)


# ---------------------------------------------------------------------------
# compute_mean_reversion_factors (Step 2.8)
# ---------------------------------------------------------------------------

def _make_panel(n_symbols: int = 3, n_days: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_symbols)]:
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n_days))
        ts = pd.date_range("2024-01-01", periods=n_days, freq="B")
        for t, p in zip(ts, prices):
            rows.append({"symbol": sym, "timestamp": t, "close": float(p)})
    return pd.DataFrame(rows)


def test_mr_factors_returns_df():
    df = _make_panel()
    result = compute_mean_reversion_factors(df)
    assert isinstance(result, pd.DataFrame)


def test_mr_factors_has_expected_columns():
    df = _make_panel()
    result = compute_mean_reversion_factors(df)
    for col in ["mr_zscore_reversal_3d", "mr_rsi_extreme_uptrend", "mr_bollinger_squeeze_break"]:
        assert col in result.columns, f"Missing: {col}"


def test_mr_factors_row_count():
    df = _make_panel(n_symbols=3, n_days=80)
    result = compute_mean_reversion_factors(df)
    assert len(result) == len(df)


def test_mr_factors_all_symbols_present():
    df = _make_panel(n_symbols=4)
    result = compute_mean_reversion_factors(df)
    assert set(result["symbol"].unique()) == {"S0", "S1", "S2", "S3"}


def test_mr_factors_missing_close_raises():
    df = pd.DataFrame({"symbol": ["A"], "timestamp": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(KeyError):
        compute_mean_reversion_factors(df)


def test_mr_factors_empty_df():
    df = pd.DataFrame(columns=["symbol", "timestamp", "close"])
    result = compute_mean_reversion_factors(df)
    assert isinstance(result, pd.DataFrame)


def test_mr_factors_no_nan_at_tail():
    df = _make_panel(n_symbols=2, n_days=100)
    result = compute_mean_reversion_factors(df)
    # Last row per symbol: at least zscore and squeeze should be non-NaN after warmup
    for sym, grp in result.groupby("symbol"):
        last = grp.iloc[-1]
        # At least one mr factor should be non-NaN at tail
        mr_cols = ["mr_zscore_reversal_3d", "mr_rsi_extreme_uptrend", "mr_bollinger_squeeze_break"]
        assert not all(pd.isna(last[c]) for c in mr_cols), f"All NaN at tail for {sym}"


# ---------------------------------------------------------------------------
# compute_interaction_features (Step 2.9)
# ---------------------------------------------------------------------------

def _make_feature_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "symbol": [f"S{i % 5}" for i in range(n)],
        "momentum_12m_excl_1m": rng.standard_normal(n),
        "rv_20": np.abs(rng.standard_normal(n)) + 0.01,
        "trend_strength_50": rng.standard_normal(n),
        "trend_strength_200": rng.standard_normal(n),
        "trend_strength_20": rng.standard_normal(n),
        "volume_ratio_20": np.abs(rng.standard_normal(n)) + 0.5,
        "rsi_14": rng.uniform(20, 80, n),
        "reversal_1d": rng.standard_normal(n),
        "fraction_above_ma50": rng.uniform(0, 1, n),
    })


def test_interaction_features_returns_df():
    df = _make_feature_df()
    result = compute_interaction_features(df)
    assert isinstance(result, pd.DataFrame)


def test_interaction_features_adds_ix_columns():
    df = _make_feature_df()
    result = compute_interaction_features(df)
    ix_cols = [c for c in result.columns if c.startswith("ix_")]
    assert len(ix_cols) > 0


def test_interaction_features_preserves_original_columns():
    df = _make_feature_df()
    original_cols = set(df.columns)
    result = compute_interaction_features(df)
    assert original_cols.issubset(set(result.columns))


def test_interaction_features_row_count_unchanged():
    df = _make_feature_df(n=30)
    result = compute_interaction_features(df)
    assert len(result) == len(df)


def test_interaction_features_custom_interactions():
    df = _make_feature_df()
    custom = [("my_test_ix", "rv_20", "trend_strength_50", "multiply")]
    result = compute_interaction_features(df, interactions=custom)
    assert "my_test_ix" in result.columns


def test_interaction_features_missing_col_skipped():
    # Only some of the default interactions have columns available
    df = pd.DataFrame({
        "symbol": ["A", "B"],
        "rv_20": [0.2, 0.3],
        "momentum_12m_excl_1m": [0.1, -0.1],
    })
    result = compute_interaction_features(df)
    # Should not crash; may produce only some ix_ columns
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
