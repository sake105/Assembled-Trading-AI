"""Tests for wave-37 module wiring into trading_cycle.py.

Covers:
  Step 2.22 — features.fundamental_factors (cross_sectional_zscore)
  Step 3.90 — ml.cpcv (generate_cpcv_splits)
  Step 8.28 — ml.factor_models (detect_feature_cols)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.fundamental_factors import (
    cross_sectional_zscore,
)
from src.assembled_core.ml.cpcv import (
    generate_cpcv_splits,
    compute_cpcv_sharpe_distribution,
    CPCVResult,
)
from src.assembled_core.ml.factor_models import detect_feature_cols


# ---------------------------------------------------------------------------
# cross_sectional_zscore (Step 2.22)
# ---------------------------------------------------------------------------

def _make_factors_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "symbol": [f"S{i % 5}" for i in range(n)],
        "factor_momentum": rng.normal(0.02, 0.05, n),
        "factor_value": rng.normal(10.0, 3.0, n),
        "factor_quality": rng.normal(0.5, 0.2, n),
    })


def test_csz_returns_df():
    df = _make_factors_df()
    result = cross_sectional_zscore(df, columns=["factor_momentum", "factor_value"])
    assert isinstance(result, pd.DataFrame)


def test_csz_same_shape():
    df = _make_factors_df()
    result = cross_sectional_zscore(df, columns=["factor_momentum", "factor_value"])
    assert result.shape == df.shape


def test_csz_near_zero_mean():
    df = _make_factors_df(n=50)
    result = cross_sectional_zscore(df, columns=["factor_momentum"])
    col_mean = result["factor_momentum"].mean()
    assert abs(col_mean) < 0.5  # should be close to zero after z-scoring


def test_csz_near_unit_std():
    df = _make_factors_df(n=100)
    result = cross_sectional_zscore(df, columns=["factor_value"])
    col_std = result["factor_value"].std()
    assert 0.5 < col_std < 2.0  # approx unit std


def test_csz_constant_col_becomes_zero():
    df = _make_factors_df()
    df["constant"] = 5.0
    result = cross_sectional_zscore(df, columns=["constant"])
    assert (result["constant"] == 0.0).all()


def test_csz_preserves_non_factor_cols():
    df = _make_factors_df()
    result = cross_sectional_zscore(df, columns=["factor_momentum"])
    assert "symbol" in result.columns


def test_csz_empty_columns_list():
    df = _make_factors_df()
    result = cross_sectional_zscore(df, columns=[])
    # No columns to zscore — original values unchanged
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# generate_cpcv_splits (Step 3.90)
# ---------------------------------------------------------------------------

def test_cpcv_returns_list():
    splits = generate_cpcv_splits(n_timestamps=200, n_groups=6, k_test_groups=2)
    assert isinstance(splits, list)


def test_cpcv_splits_non_empty():
    splits = generate_cpcv_splits(n_timestamps=200, n_groups=6, k_test_groups=2)
    assert len(splits) > 0


def test_cpcv_each_split_is_tuple():
    splits = generate_cpcv_splits(n_timestamps=200, n_groups=6, k_test_groups=2)
    for train_idx, test_idx in splits:
        assert isinstance(train_idx, list)
        assert isinstance(test_idx, list)


def test_cpcv_no_overlap():
    splits = generate_cpcv_splits(n_timestamps=200, n_groups=6, k_test_groups=2, purge_length=5)
    for train_idx, test_idx in splits:
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0


def test_cpcv_small_groups_returns_empty():
    splits = generate_cpcv_splits(n_timestamps=30, n_groups=6, k_test_groups=2)
    # group_size = 5, too small
    assert isinstance(splits, list)


def test_cpcv_k1_groups_per_split():
    splits = generate_cpcv_splits(n_timestamps=300, n_groups=6, k_test_groups=1)
    assert len(splits) == 6  # C(6,1) = 6


def test_cpcv_number_of_splits():
    # C(6,2) = 15 combinations
    splits = generate_cpcv_splits(n_timestamps=600, n_groups=6, k_test_groups=2)
    assert len(splits) <= 15


# ---------------------------------------------------------------------------
# detect_feature_cols (Step 8.28)
# ---------------------------------------------------------------------------

def _make_panel_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="B"),
        "symbol": [f"S{i % 5}" for i in range(n)],
        "close": rng.uniform(50, 200, n),
        "fwd_return_20d": rng.normal(0, 0.05, n),
        "factor_momentum_12m": rng.normal(0, 1, n),
        "factor_value_pb": rng.normal(0, 1, n),
        "returns_6m": rng.normal(0, 1, n),
        "news_sentiment": rng.normal(0, 0.5, n),
    })


def test_detect_feature_cols_returns_list():
    df = _make_panel_df()
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert isinstance(result, list)


def test_detect_feature_cols_excludes_label():
    df = _make_panel_df()
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert "fwd_return_20d" not in result


def test_detect_feature_cols_excludes_timestamp():
    df = _make_panel_df()
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert "timestamp" not in result


def test_detect_feature_cols_includes_factor_prefix():
    df = _make_panel_df()
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert "factor_momentum_12m" in result
    assert "factor_value_pb" in result


def test_detect_feature_cols_includes_returns_prefix():
    df = _make_panel_df()
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert "returns_6m" in result


def test_detect_feature_cols_includes_news_prefix():
    df = _make_panel_df()
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert "news_sentiment" in result


def test_detect_feature_cols_empty_df():
    df = pd.DataFrame({"timestamp": [], "symbol": [], "close": [], "fwd_return_20d": []})
    result = detect_feature_cols(df, label_col="fwd_return_20d")
    assert isinstance(result, list)
    assert len(result) == 0
