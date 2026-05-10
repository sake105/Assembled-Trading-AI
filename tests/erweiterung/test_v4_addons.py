"""Tests für v4-Add-Ons (Quality, Wavelet, Resampled-EF, Copulas, Online-Learning, Kalman)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.bayesian.copulas import (
    fit_clayton_copula,
    fit_gumbel_copula,
    kendalls_tau,
    lower_tail_dependence,
    sample_clayton,
    sample_gaussian_copula,
    upper_tail_dependence,
)
from erweiterung.factors.quality_minus_junk import (
    profitability_score,
    quality_score,
    safety_score,
)
from erweiterung.online_learning.adwin import ADWIN
from erweiterung.online_learning.passive_aggressive import (
    PAConfig,
    PassiveAggressiveRegressor,
    online_predict_sequence,
)
from erweiterung.online_learning.recursive_lr import (
    RecursiveLeastSquares,
    online_rls_predict,
)
from erweiterung.portfolio.resampled_efficient_frontier import (
    REFConfig,
    resampled_efficient_frontier,
    select_target_vol_portfolio,
)
from erweiterung.state_space.kalman_beta import (
    KalmanBetaConfig,
    kalman_filter_beta,
    kalman_pairs_hedge_ratio,
)
from erweiterung.timeseries_tools.wavelet import (
    rolling_wavelet_energy_ratio,
    wavelet_decompose,
    wavelet_energy_per_scale,
)


# ----- Quality Factor -----


def test_profitability_score():
    rng = np.random.default_rng(0)
    n_dates = 5
    n_syms = 10
    rows = []
    for d in pd.date_range("2024-01-01", periods=n_dates):
        for sym in [f"S{i}" for i in range(n_syms)]:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "gross_profit": rng.uniform(1e6, 1e8),
                    "total_assets": rng.uniform(1e7, 1e9),
                    "net_income": rng.uniform(1e5, 1e7),
                    "equity": rng.uniform(1e7, 1e9),
                }
            )
    panel = pd.DataFrame(rows)
    score = profitability_score(panel)
    assert score.notna().any()


def test_safety_score():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2024-01-01", periods=5):
        for sym in [f"S{i}" for i in range(10)]:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "beta": rng.uniform(0.5, 1.5),
                    "debt": rng.uniform(1e6, 1e8),
                    "equity": rng.uniform(1e6, 1e8),
                    "roe_vol": rng.uniform(0.05, 0.3),
                }
            )
    panel = pd.DataFrame(rows)
    score = safety_score(panel)
    assert score.notna().any()


def test_quality_score_runs():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2024-01-01", periods=3):
        for sym in [f"S{i}" for i in range(10)]:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "gross_profit": rng.uniform(1e6, 1e8),
                    "total_assets": rng.uniform(1e7, 1e9),
                    "net_income": rng.uniform(1e5, 1e7),
                    "equity": rng.uniform(1e7, 1e9),
                }
            )
    panel = pd.DataFrame(rows)
    out = quality_score(panel)
    assert out.notna().any()


# ----- Wavelet -----


def test_wavelet_decompose_haar_fallback():
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.standard_normal(256)))
    decomp = wavelet_decompose(s, wavelet="haar", level=3)
    assert "approximation" in decomp
    assert len(decomp["details"]) >= 1


def test_wavelet_energy():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(256))
    decomp = wavelet_decompose(s, wavelet="haar", level=3)
    e = wavelet_energy_per_scale(decomp)
    assert "approx_energy" in e
    assert e["total_energy"] > 0


def test_rolling_wavelet_energy():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(500))
    out = rolling_wavelet_energy_ratio(
        s, window=128, wavelet="haar", level=3, target_level=0
    )
    valid = out.dropna()
    assert valid.shape[0] > 0


# ----- Resampled EF -----


def test_resampled_ef_runs():
    rng = np.random.default_rng(0)
    n = 100
    n_assets = 4
    df = pd.DataFrame(
        rng.normal(0.001, 0.02, (n, n_assets)),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    out = resampled_efficient_frontier(
        df, REFConfig(n_bootstrap=20, n_frontier_points=10, max_weight=0.5)
    )
    assert "frontier_df" in out
    front = out["frontier_df"]
    assert "volatility" in front.columns
    assert (front["volatility"] >= 0).all()


def test_select_target_vol():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(0.001, 0.02, (100, 3)),
        columns=["A", "B", "C"],
    )
    out = resampled_efficient_frontier(
        df, REFConfig(n_bootstrap=10, n_frontier_points=5, max_weight=0.6)
    )
    w = select_target_vol_portfolio(out["frontier_df"], target_vol=0.02)
    assert isinstance(w, pd.Series)


# ----- Copulas -----


def test_kendalls_tau():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    tau = kendalls_tau(x, y)
    assert abs(tau + 1.0) < 1e-9  # perfect anti-correlation


def test_clayton_fit_and_sample():
    rng = np.random.default_rng(42)
    u, v = sample_clayton(theta=2.0, n=500, rng=rng)
    fit = fit_clayton_copula(u, v)
    assert fit.family == "Clayton"
    assert fit.param > 0
    # Lower-tail dependence should be positive for Clayton
    lambda_l = lower_tail_dependence("Clayton", fit.param)
    assert lambda_l > 0


def test_gumbel_fit():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=500)
    v = u**0.7 + rng.uniform(-0.1, 0.1, size=500)
    v = np.clip(v, 0.01, 0.99)
    fit = fit_gumbel_copula(u, v)
    assert fit.family == "Gumbel"
    assert fit.param >= 1.0


def test_gaussian_copula_sample():
    rng = np.random.default_rng(0)
    u, v = sample_gaussian_copula(rho=0.7, n=1000, rng=rng)
    assert (u >= 0).all() and (u <= 1).all()
    assert (v >= 0).all() and (v <= 1).all()


def test_tail_dependence_consistency():
    # Clayton should have positive lower TD, zero upper TD
    assert lower_tail_dependence("Clayton", 2.0) > 0
    assert upper_tail_dependence("Clayton", 2.0) == 0
    # Gumbel: opposite
    assert lower_tail_dependence("Gumbel", 2.0) == 0
    assert upper_tail_dependence("Gumbel", 2.0) > 0


# ----- Online Learning -----


def test_passive_aggressive_basic():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 3))
    true_w = np.array([0.5, -0.3, 0.7])
    y = X @ true_w + rng.normal(0, 0.1, n)
    pa = PassiveAggressiveRegressor(n_features=3, config=PAConfig(epsilon=0.05, C=1.0))
    for t in range(n):
        pa.partial_fit(X[t], y[t])
    # weights should converge near true_w
    assert np.linalg.norm(pa.w - true_w) < 1.0


def test_pa_online_predict():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1, (n, 2))
    y = X.sum(axis=1) + rng.normal(0, 0.1, n)
    preds = online_predict_sequence(X, y)
    assert len(preds) == n


def test_recursive_least_squares():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 3))
    true_w = np.array([0.5, -0.3, 0.7])
    y = X @ true_w + rng.normal(0, 0.05, n)
    rls = RecursiveLeastSquares(n_features=3, lam=0.99)
    for t in range(n):
        rls.partial_fit(X[t], y[t])
    assert np.linalg.norm(rls.beta - true_w) < 0.5


def test_online_rls_predict():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1, (n, 2))
    y = X.sum(axis=1) + rng.normal(0, 0.05, n)
    preds = online_rls_predict(X, y)
    assert len(preds) == n


def test_adwin_no_drift():
    rng = np.random.default_rng(0)
    adwin = ADWIN(delta=0.01)
    drifts = []
    for _ in range(200):
        drifts.append(adwin.update(rng.normal(0, 1)))
    # i.i.d. -> few or no drifts
    assert sum(drifts) <= 5


def test_adwin_detects_mean_shift():
    adwin = ADWIN(delta=0.01)
    detected = False
    for i in range(200):
        val = 0.0 if i < 100 else 5.0  # mean shift
        if adwin.update(val):
            detected = True
            break
    assert detected


# ----- Kalman Beta -----


def test_kalman_constant_beta():
    rng = np.random.default_rng(0)
    n = 300
    market = pd.Series(rng.normal(0, 0.01, n))
    asset = 1.5 * market + pd.Series(rng.normal(0, 0.005, n))
    res = kalman_filter_beta(asset, market, KalmanBetaConfig(process_variance=1e-6))
    final_beta = float(res["beta"].iloc[-1])
    assert abs(final_beta - 1.5) < 0.5


def test_kalman_time_varying():
    rng = np.random.default_rng(0)
    n = 500
    market = pd.Series(rng.normal(0, 0.01, n))
    # beta drifts from 0.5 to 2.0
    betas_true = np.linspace(0.5, 2.0, n)
    asset_vals = betas_true * market.values + rng.normal(0, 0.005, n)
    asset = pd.Series(asset_vals)
    res = kalman_filter_beta(asset, market, KalmanBetaConfig(process_variance=1e-3))
    # Final beta should be near 2.0
    assert res["beta"].iloc[-1] > 1.0
    # Initial should be near 0.5 (after some warm-up)
    assert res["beta"].iloc[100] < 1.5


def test_kalman_pairs_hedge():
    rng = np.random.default_rng(0)
    n = 200
    x = pd.Series(np.cumsum(rng.normal(0, 0.5, n)))
    y = 1.2 * x + pd.Series(rng.normal(0, 0.3, n))
    res = kalman_pairs_hedge_ratio(y, x)
    assert "hedge_ratio" in res.columns
    assert "spread" in res.columns
