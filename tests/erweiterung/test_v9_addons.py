"""Tests für v9-Add-Ons: Bayesian-HPO, Multi-Objective, MaxEnt-Bootstrap, BMA,
Hampel-Filter, Realized-Beta."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.backtest.maxent_bootstrap import (
    bootstrap_confidence_interval,
    maxent_bootstrap_ensemble,
    maxent_bootstrap_sample,
)
from erweiterung.bayesian.model_averaging import BMAResult, bma_predict
from erweiterung.factors.realized_beta import (
    beta_components,
    beta_hedge_size,
    realized_beta_daily,
    rolling_beta,
)
from erweiterung.optimization.bayesian_hpo import bayesian_optimize
from erweiterung.optimization.multi_objective import (
    crowding_distance,
    is_dominated,
    non_dominated_sort,
    pareto_front_indices,
)
from erweiterung.robust_stats.hampel_filter import (
    hampel_filter,
    rolling_zscore_outliers,
    winsorize_series,
)


# ----- Bayesian HPO -----


def test_bayesian_optimize_minimum():
    # Convex quadratic objective — should find minimum near 0
    def obj(x):
        return float(x[0] ** 2 + 0.5 * (x[1] - 1) ** 2)

    res = bayesian_optimize(obj, bounds=[(-2, 2), (-2, 2)], n_iter=15, n_init=5)
    # Should be near (0, 1)
    assert abs(res.best_params[0]) < 1.0
    assert abs(res.best_params[1] - 1.0) < 1.0


# ----- Multi-Objective -----


def test_is_dominated():
    p = np.array([1.0, 2.0])
    others = np.array([[2.0, 3.0], [0.5, 1.5]])
    # Maximize: p=(1,2) is dominated by (2,3); not by (0.5,1.5)
    assert is_dominated(p, others, maximize=True) is True


def test_pareto_front():
    objs = np.array([[3, 1], [1, 3], [2, 2], [4, 4], [0, 0]])
    front = pareto_front_indices(objs, maximize=True)
    assert 3 in front  # (4,4) dominates all
    assert 4 not in front  # (0,0) dominated


def test_non_dominated_sort():
    objs = np.array([[3, 1], [1, 3], [2, 2], [4, 4], [0, 0]])
    fronts = non_dominated_sort(objs, maximize=True)
    assert len(fronts) >= 1
    assert 3 in fronts[0]  # (4,4) in best front


def test_crowding_distance():
    objs = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    dist = crowding_distance(objs, list(range(4)))
    assert len(dist) == 4
    # boundary points have inf
    assert np.isinf(dist[0])
    assert np.isinf(dist[-1])


# ----- MaxEnt Bootstrap -----


def test_maxent_bootstrap_sample():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.01, 200))
    sample = maxent_bootstrap_sample(s, seed=42)
    assert len(sample) == 200
    # Mean should be close to original
    assert abs(sample.mean() - s.mean()) < 0.01


def test_maxent_bootstrap_ensemble():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.01, 100))
    ens = maxent_bootstrap_ensemble(s, n_samples=10)
    assert ens.shape == (100, 10)


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.001, 0.01, 300))

    def stat(x):
        return float(x.mean())

    res = bootstrap_confidence_interval(s, stat, n_samples=50)
    assert "ci_low" in res
    assert res["ci_low"] < res["ci_high"]


# ----- BMA -----


def test_bma_basic():
    rng = np.random.default_rng(0)
    n = 100
    y_true = pd.Series(rng.normal(0, 1, n))
    # Two models — first better than second
    pred1 = y_true + rng.normal(0, 0.5, n)
    pred2 = y_true + rng.normal(0, 1.5, n)
    df = pd.DataFrame({"good": pred1, "bad": pred2})
    res = bma_predict(df, n_params={"good": 5, "bad": 5}, y_true=y_true)
    assert isinstance(res, BMAResult)
    # Better model should get higher weight
    assert res.model_weights["good"] > res.model_weights["bad"]


# ----- Hampel Filter -----


def test_hampel_filter_replaces_outliers():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 100))
    s.iloc[50] = 1000  # huge outlier
    filtered = hampel_filter(s, window=11, n_sigma=3.0)
    assert abs(filtered.iloc[50]) < 10  # outlier corrected


def test_hampel_filter_with_mask():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 100))
    s.iloc[20] = 100
    filtered, mask = hampel_filter(s, window=11, return_mask=True)
    assert mask.iloc[20]


def test_winsorize():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 200))
    s.iloc[0] = 100
    s.iloc[-1] = -100
    w = winsorize_series(s, lower_q=0.01, upper_q=0.99)
    assert w.max() < 100
    assert w.min() > -100


def test_rolling_zscore_outliers():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 100))
    s.iloc[50] = 50  # outlier
    mask = rolling_zscore_outliers(s, window=20, threshold=3.0)
    assert mask.iloc[50]


# ----- Realized Beta -----


def test_rolling_beta_recovers():
    rng = np.random.default_rng(0)
    n = 500
    m = pd.Series(rng.normal(0, 0.01, n))
    a = 1.5 * m + pd.Series(rng.normal(0, 0.005, n))
    b = rolling_beta(a, m, window=100)
    valid = b.dropna()
    # Mean should be near 1.5
    assert abs(valid.mean() - 1.5) < 0.5


def test_realized_beta_daily():
    rng = np.random.default_rng(0)
    n = 200
    m = pd.Series(rng.normal(0, 0.01, n))
    a = 1.2 * m + pd.Series(rng.normal(0, 0.005, n))
    rb = realized_beta_daily(a, m, window=30)
    valid = rb.dropna()
    assert abs(valid.mean() - 1.2) < 0.5


def test_beta_components():
    rng = np.random.default_rng(0)
    n = 300
    m = pd.Series(rng.normal(0, 0.01, n))
    a = 1.2 * m + pd.Series(rng.normal(0, 0.005, n))
    bc = beta_components(a, m, window=100)
    assert "upside_beta" in bc.columns
    assert "downside_beta" in bc.columns


def test_beta_hedge_size():
    hs = beta_hedge_size(asset_notional=100000, beta=1.5)
    assert hs == -150000  # short 1.5x notional to neutralize
