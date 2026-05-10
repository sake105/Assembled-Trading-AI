"""Tests für v3-Module (classical_ml, timeseries_tools, dl_advanced, risk_metrics,
stress_test, report, strategies)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.classical_ml.boost_wrappers import (
    fit_predict_random_forest,
    time_series_cv_score,
)
from erweiterung.dl_advanced.reservoir_computing import ESNConfig, fit_predict_esn
from erweiterung.risk_metrics.advanced_metrics import (
    burke_ratio,
    comprehensive_metrics,
    omega_ratio,
    pain_index,
    stutzer_index,
    treynor_ratio,
    ulcer_index,
    upside_potential_ratio,
)
from erweiterung.strategies.templates import (
    StrategyConfig,
    low_vol_strategy,
    trend_following,
)
from erweiterung.stress_test.historical_replay import (
    STANDARD_CRISES,
    replay_all_crises,
    stress_score,
)
from erweiterung.stress_test.monte_carlo import MCConfig, path_metrics, simulate_paths
from erweiterung.timeseries_tools.change_points import binary_segmentation, cusum_filter
from erweiterung.timeseries_tools.entropy import (
    approximate_entropy,
    sample_entropy,
    shannon_entropy,
)
from erweiterung.timeseries_tools.fractional_diff import (
    fractional_diff_ffd,
    get_weights_ffd,
)
from erweiterung.timeseries_tools.hurst_dfa import (
    detrended_fluctuation_analysis,
    hurst_rs,
    variance_ratio_test,
)


# ---- Risk metrics ----


def test_omega_ratio():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    o = omega_ratio(r, 0.0)
    assert o > 0


def test_treynor_ratio():
    rng = np.random.default_rng(0)
    n = 500
    market = pd.Series(rng.normal(0, 0.01, n))
    asset = 1.2 * market + pd.Series(rng.normal(0.0005, 0.005, n))
    t = treynor_ratio(asset, market, rf=0.0)
    assert np.isfinite(t)


def test_ulcer_pain():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    eq = (1 + r).cumprod()
    u = ulcer_index(eq)
    p = pain_index(eq)
    assert u >= 0
    assert p >= 0


def test_burke_ratio():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    eq = (1 + r).cumprod()
    b = burke_ratio(r, eq)
    assert np.isfinite(b)


def test_stutzer():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    s = stutzer_index(r)
    assert np.isfinite(s)


def test_upside_potential():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    u = upside_potential_ratio(r)
    assert u > 0


def test_comprehensive_metrics():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    bench = pd.Series(rng.normal(0.0003, 0.01, 500))
    out = comprehensive_metrics(r, bench)
    for key in ("omega_ratio_0", "ulcer_index", "burke_ratio", "treynor_ratio"):
        assert key in out


# ---- Hurst / DFA / Variance-Ratio ----


def test_hurst_random_walk():
    rng = np.random.default_rng(42)
    # Brownian motion -> H ~ 0.5
    rw = np.cumsum(rng.standard_normal(2000))
    h = hurst_rs(np.diff(rw), max_window=200)
    assert 0.3 < h < 0.7


def test_dfa_random():
    rng = np.random.default_rng(42)
    s = rng.standard_normal(2000)
    h = detrended_fluctuation_analysis(s, max_window=200)
    assert 0.3 < h < 0.7


def test_variance_ratio():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000)
    out = variance_ratio_test(r, lags=(2, 4, 8))
    assert "vr_2" in out
    # i.i.d. -> VR ~ 1
    assert 0.7 < out["vr_2"] < 1.3


# ---- Fractional Diff ----


def test_get_weights_ffd():
    w = get_weights_ffd(0.4)
    assert w[0] == 1.0
    assert len(w) > 1
    assert abs(w[-1]) < 1e-4


def test_fractional_diff():
    rng = np.random.default_rng(0)
    # 2000 obs + threshold 1e-3 keeps weight-window manageable
    s = pd.Series(np.cumsum(rng.standard_normal(2000)))
    diff = fractional_diff_ffd(s, d=0.5, threshold=1e-3)
    valid = diff.dropna()
    assert len(valid) > 0
    assert valid.std() > 0


# ---- Entropy ----


def test_sample_entropy():
    rng = np.random.default_rng(0)
    # Random series should have positive entropy
    s = rng.standard_normal(500)
    e = sample_entropy(s, m=2)
    assert np.isfinite(e) and e > 0


def test_approximate_entropy():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(500)
    e = approximate_entropy(s, m=2)
    assert np.isfinite(e)


def test_shannon_entropy():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(1000)
    e = shannon_entropy(s, n_bins=20)
    assert e > 0


# ---- Change Points ----


def test_cusum_filter():
    s = pd.Series(np.concatenate([np.zeros(50), np.ones(50) * 0.05, np.zeros(50)]))
    events = cusum_filter(s, threshold=0.02)
    # Should detect at least one event near the break
    assert len(events) > 0


def test_binary_segmentation():
    s = np.concatenate([np.zeros(50), np.ones(50), np.zeros(50)])
    bps = binary_segmentation(s, n_breakpoints=2, min_segment=20)
    # Should find approximately 50 and 100
    assert len(bps) > 0


# ---- DL Advanced ----


def test_reservoir_computing_runs():
    rng = np.random.default_rng(0)
    train = rng.standard_normal(500)
    test = rng.standard_normal(100)
    pred = fit_predict_esn(train, test, config=ESNConfig(n_reservoir=50, sparsity=0.1))
    assert pred.shape == (100,)
    assert np.isfinite(pred).all()


# ---- Classical ML ----


def test_random_forest():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 200)
    pred = fit_predict_random_forest(X[:150], y[:150], X[150:], n_estimators=20)
    assert pred.shape == (50,)
    # Predictions should correlate with truth
    assert np.corrcoef(pred, y[150:])[0, 1] > 0.3


def test_time_series_cv():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(0, 1, (n, 2))
    y = X.sum(axis=1) + rng.normal(0, 0.5, n)
    out = time_series_cv_score(
        fit_predict_random_forest, X, y, n_splits=4, metric="rmse"
    )
    assert "mean" in out
    assert out["mean"] > 0


# ---- Stress Test ----


def test_monte_carlo_paths():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    paths = simulate_paths(r, MCConfig(n_paths=100, horizon=60, method="bootstrap"))
    assert paths.shape == (100, 60)
    m = path_metrics(paths)
    assert "terminal_mean" in m and "max_dd_mean" in m


def test_monte_carlo_methods():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    for m in ("bootstrap", "block", "stationary", "normal"):
        cfg = MCConfig(n_paths=50, horizon=30, method=m, seed=42)
        paths = simulate_paths(r, cfg)
        assert paths.shape == (50, 30)


def test_historical_replay():
    dates = pd.date_range("2007-01-01", "2024-12-31", freq="B")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
    out = replay_all_crises(r, STANDARD_CRISES[:3])
    assert "max_drawdown" in out.columns
    summary = stress_score(out)
    assert "worst_drawdown" in summary


# ---- Strategies ----


def test_trend_following():
    rng = np.random.default_rng(0)
    n = 100
    rows = []
    for d in pd.date_range("2024-01-01", periods=n):
        for sym in ("A", "B", "C", "D"):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "return": rng.normal(0, 0.01),
                    "momentum_12_1": rng.normal(),
                }
            )
    panel = pd.DataFrame(rows)
    ret = trend_following(panel, StrategyConfig(quantile=0.5, transaction_cost_bps=5.0))
    assert len(ret) > 0


def test_low_vol_strategy():
    rng = np.random.default_rng(0)
    n = 100
    rows = []
    for d in pd.date_range("2024-01-01", periods=n):
        for sym in ("A", "B", "C", "D"):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "return": rng.normal(0, 0.01),
                    "rolling_vol_60": rng.uniform(0.005, 0.02),
                }
            )
    panel = pd.DataFrame(rows)
    ret = low_vol_strategy(panel, StrategyConfig(quantile=0.5))
    assert len(ret) > 0
