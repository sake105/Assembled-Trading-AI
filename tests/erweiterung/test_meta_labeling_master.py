"""Tests für meta_labeling_master."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erweiterung.ml.meta_labeling_master import (
    MetaLabelingConfig,
    apply_meta_gate,
    build_features,
    triple_barrier_simple,
    walk_forward_meta_predictions,
)


def _make_returns(n: int = 600, seed: int = 0, vol: float = 0.012) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0003, vol, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_triple_barrier_returns_finite_labels():
    r = _make_returns(n=300)
    labels = triple_barrier_simple(r, MetaLabelingConfig(horizon_days=15))
    valid = labels.dropna()
    assert not valid.empty
    assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})


def test_triple_barrier_uptrend_yields_more_positives():
    rng = np.random.default_rng(1)
    r = pd.Series(
        rng.normal(0.002, 0.005, 300),  # strong drift
        index=pd.date_range("2022-01-01", periods=300, freq="B"),
    )
    labels = triple_barrier_simple(r, MetaLabelingConfig(horizon_days=20))
    valid = labels.dropna()
    # In strong uptrend should have more TP-hits than SL
    assert (valid == 1.0).sum() > (valid == -1.0).sum()


def test_build_features_columns():
    r = _make_returns(n=400)
    feat = build_features(r)
    assert "trailing_vol" in feat.columns
    assert "trailing_sharpe" in feat.columns
    assert "drawdown_pct" in feat.columns
    assert "return_lag1" in feat.columns


def test_build_features_with_macro():
    r = _make_returns(n=400)
    rng = np.random.default_rng(2)
    macro = pd.DataFrame(
        {
            "vix_close": np.full(400, 17.0) + rng.normal(0, 2, 400),
            "yield_curve_spread": np.full(400, 0.5),
            "hy_spread": np.full(400, 3.5),
        },
        index=r.index,
    )
    feat = build_features(r, macro_panel=macro)
    assert "vix" in feat.columns
    assert "yc_spread" in feat.columns
    assert "hy_spread" in feat.columns


def test_walk_forward_meta_predictions_runs():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 1200
    r = pd.Series(
        rng.normal(0.0005, 0.01, n),
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    feat = build_features(r)
    labels = triple_barrier_simple(r, MetaLabelingConfig(horizon_days=10))
    preds = walk_forward_meta_predictions(
        feat,
        labels,
        MetaLabelingConfig(train_window=300, test_window=100),
    )
    assert not preds.empty
    assert "proba" in preds.columns
    assert "predicted" in preds.columns


def test_apply_meta_gate_zeros_when_predicted_zero():
    r = pd.Series(
        [0.01, -0.02, 0.005, -0.001],
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )
    preds = pd.DataFrame(
        {"predicted": [0, 1, 0, 1]},
        index=r.index,
    )
    gated = apply_meta_gate(r, preds)
    np.testing.assert_array_almost_equal(gated.values, [0, -0.02, 0, -0.001])


def test_apply_meta_gate_handles_index_misalignment():
    r = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    preds = pd.DataFrame(
        {"predicted": [1, 1]},
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    gated = apply_meta_gate(r, preds)
    # only overlap counts
    assert len(gated.dropna()) == 2


def test_walk_forward_returns_empty_when_too_short():
    pytest.importorskip("sklearn")
    r = _make_returns(n=200)
    feat = build_features(r)
    labels = triple_barrier_simple(r)
    preds = walk_forward_meta_predictions(
        feat,
        labels,
        MetaLabelingConfig(train_window=300, test_window=100),
    )
    assert preds.empty
