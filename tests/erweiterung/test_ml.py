"""Tests for erweiterung.ml."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.ml import conformal_prediction, stacking_ensemble, triple_barrier


def linear_fit_predict(X_tr, y_tr, X_te):
    """Tiny linear regressor for tests."""
    Xb = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta, *_ = np.linalg.lstsq(Xb, y_tr, rcond=None)
    Xeb = np.column_stack([np.ones(len(X_te)), X_te])
    return Xeb @ beta


def test_split_conformal_coverage():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, (n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)
    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:], y[400:]
    ci = conformal_prediction.split_conformal_regression(
        linear_fit_predict, X_train, y_train, X_test, alpha=0.1
    )
    coverage = ci.coverage_check(y_test)
    # Empirically, conformal coverage should be close to 0.9
    assert 0.80 <= coverage <= 0.99


def test_conformal_intervals_finite():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0, 1, (200, 2))
    y_tr = X_tr.sum(axis=1) + rng.normal(0, 0.3, 200)
    X_te = rng.normal(0, 1, (50, 2))
    ci = conformal_prediction.split_conformal_regression(
        linear_fit_predict, X_tr, y_tr, X_te, alpha=0.1
    )
    assert (ci.upper >= ci.lower).all()
    assert np.isfinite(ci.upper).all()


def test_conformal_to_signal():
    ci = conformal_prediction.ConformalIntervals(
        point_estimates=np.array([0.1, -0.2, 0.5]),
        lower=np.array([0.05, -0.5, 0.1]),
        upper=np.array([0.15, 0.1, 1.0]),
        alpha=0.1,
    )
    sig = conformal_prediction.conformal_to_signal(ci)
    assert sig[0] == +1  # lower > 0
    assert sig[1] == 0
    assert sig[2] == +1


def test_triple_barrier_basic():
    prices = pd.Series(
        [100, 101, 102, 103, 99, 101, 105, 100, 95, 110, 105],
        index=pd.date_range("2024-01-01", periods=11, freq="D"),
    )
    cfg = triple_barrier.TripleBarrierConfig(
        take_profit_pct=0.03, stop_loss_pct=0.03, horizon_days=5
    )
    out = triple_barrier.triple_barrier_labels(prices, config=cfg)
    assert "label" in out.columns
    assert set(out["label"].unique()).issubset({-1, 0, +1})


def test_sample_uniqueness():
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
            "t1": pd.to_datetime(["2024-01-04", "2024-01-04", "2024-01-08"]),
            "label": [1, -1, 1],
        }
    )
    uniq = triple_barrier.sample_uniqueness(bars)
    assert len(uniq) == 3
    # uniqueness should be in (0, 1], NaN/inf not allowed
    valid = uniq.dropna()
    assert (valid > 0).all()
    assert (valid <= 1.0 + 1e-9).all()
    # First two samples fully overlap with each other -> their uniqueness < 1
    assert valid.iloc[0] < 1.0
    assert valid.iloc[1] < 1.0


def test_make_meta_labels():
    side = pd.Series([1, -1, 1, -1])
    tb = pd.Series([1, -1, -1, 1])
    meta = triple_barrier.make_meta_labels(side, tb)
    assert (meta == [1, 1, 0, 0]).all()


def test_stacking_basic():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)

    base_models = [
        stacking_ensemble.BaseModel("lin1", linear_fit_predict),
        stacking_ensemble.BaseModel("lin2", linear_fit_predict),
    ]

    def meta_fit_predict(Xtr, ytr, Xte):
        # use mean of base predictions
        return Xte.mean(axis=1)

    stk = stacking_ensemble.StackingRegressor(base_models, meta_fit_predict, n_splits=3)
    stk.fit(X[:150], y[:150])
    preds = stk.predict_full(X[:150], y[:150], X[150:])
    assert len(preds) == 50
    assert np.isfinite(preds).all()
