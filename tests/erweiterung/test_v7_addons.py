"""Tests für v7-Add-Ons: Mutual-Info, Robust Regression, Spillover, LASSO, Cornish-Fisher, LPPL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.factors.lasso_selection import (
    cv_optimal_alpha,
    lasso_factor_selection,
    lasso_path,
)
from erweiterung.info_theory.mutual_info import (
    kl_divergence,
    mrmr_feature_selection,
    mutual_info_histogram,
    normalized_mutual_info,
    transfer_entropy,
)
from erweiterung.risk.cornish_fisher_var import (
    cornish_fisher_var,
    rolling_var_comparison,
)
from erweiterung.risk.lppl_bubble import bubble_likelihood_score, fit_lppl
from erweiterung.robust_stats.robust_regression import (
    huber_regression,
    median_absolute_deviation,
    ransac_regression,
)
from erweiterung.timeseries_tools.spillover_index import (
    rolling_total_spillover,
    spillover_indices,
)


# ----- Mutual Info -----


def test_mutual_info_correlation():
    rng = np.random.default_rng(0)
    # Strongly dependent
    x = rng.normal(0, 1, 1000)
    y = x + rng.normal(0, 0.3, 1000)
    mi = mutual_info_histogram(x, y, n_bins=20)
    # Independent
    z = rng.normal(0, 1, 1000)
    mi_indep = mutual_info_histogram(x, z, n_bins=20)
    assert mi > mi_indep


def test_mutual_info_independence_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    mi = mutual_info_histogram(x, y, n_bins=10)
    assert mi < 0.1  # near 0


def test_normalized_mi():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    nmi = normalized_mutual_info(x, x)
    assert nmi > 0.5  # self-similarity


def test_kl_divergence():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.5, 0.3, 0.2])
    assert abs(kl_divergence(p, q)) < 1e-9
    q2 = np.array([0.2, 0.5, 0.3])
    assert kl_divergence(p, q2) > 0


def test_transfer_entropy_directional():
    rng = np.random.default_rng(0)
    n = 1000
    # X drives Y: y_t = 0.5 x_{t-1} + noise
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * x[t - 1] + 0.5 * rng.normal()
    te_xy = transfer_entropy(x, y, lag=1, n_bins=6)
    te_yx = transfer_entropy(y, x, lag=1, n_bins=6)
    assert te_xy > te_yx  # x → y should be stronger


def test_mrmr_selection():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "good1": rng.normal(0, 1, n),
            "good2": rng.normal(0, 1, n),
            "redundant": rng.normal(0, 1, n),
            "noise": rng.normal(0, 1, n),
        }
    )
    df["redundant"] = df["good1"] + 0.05 * rng.normal(0, 1, n)  # redundant with good1
    y = pd.Series(0.5 * df["good1"] + 0.5 * df["good2"] + 0.2 * rng.normal(0, 1, n))
    selected = mrmr_feature_selection(df, y, n_select=2, n_bins=15)
    assert "good1" in selected or "good2" in selected
    assert len(selected) == 2


# ----- Robust Regression -----


def test_huber_regression_with_outliers():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    true_beta = np.array([0.5, 2.0])
    y = X @ true_beta + rng.normal(0, 0.3, n)
    # Inject outliers
    y[:10] += 20  # 5% extreme outliers
    beta_huber = huber_regression(X, y, delta=1.345)
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    # Huber should be closer to true_beta than OLS
    assert abs(beta_huber[1] - 2.0) < abs(beta_ols[1] - 2.0)


def test_ransac_basic():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    true_beta = np.array([0.5, 2.0])
    y = X @ true_beta + rng.normal(0, 0.2, n)
    y[150:] += 10  # 25% strong outliers
    res = ransac_regression(X, y, threshold=1.0, n_iter=100)
    assert res["n_inliers"] > 0
    assert abs(res["beta"][1] - 2.0) < 0.5


def test_mad():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mad = median_absolute_deviation(x)
    assert mad > 0


# ----- Spillover Index -----


def test_spillover_indices():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), columns=["A", "B", "C"])
    out = spillover_indices(df, p=1, horizon=5)
    assert "total_spillover_pct" in out
    assert 0 <= out["total_spillover_pct"] <= 100
    assert out["pairwise_matrix"].shape == (3, 3)


def test_rolling_spillover_runs():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), columns=["A", "B", "C"])
    rs = rolling_total_spillover(df, window=80, p=1, horizon=3, step=10)
    assert len(rs) > 0


# ----- LASSO -----


def test_lasso_selects_relevant():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(0, 1, (n, 8)), columns=[f"X{i}" for i in range(8)])
    # Only X0, X1 are predictive
    y = pd.Series(0.5 * X["X0"] + 0.3 * X["X1"] + rng.normal(0, 0.3, n))
    res = lasso_factor_selection(X, y, alpha=0.05)
    assert len(res.selected_features) < 8  # some pruned
    # Strong features should be among selected
    assert "X0" in res.selected_features


def test_lasso_path():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (200, 4)), columns=["A", "B", "C", "D"])
    y = pd.Series(X["A"] + 0.5 * X["B"] + rng.normal(0, 0.3, 200))
    path = lasso_path(X, y, alphas=[0.001, 0.01, 0.1, 1.0])
    assert "alpha" in path.columns
    assert len(path) >= 1


def test_cv_optimal_alpha():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (300, 4)), columns=["A", "B", "C", "D"])
    y = pd.Series(X["A"] + rng.normal(0, 0.3, 300))
    res = cv_optimal_alpha(X, y, alphas=[0.001, 0.01, 0.1])
    if "best_alpha" in res:
        assert res["best_alpha"] in [0.001, 0.01, 0.1]


# ----- Cornish-Fisher VaR -----


def test_cornish_fisher_vs_gauss():
    rng = np.random.default_rng(0)
    # Heavy-tail (t-distribution)
    r = pd.Series(rng.standard_t(df=4, size=1000) / 100)
    out = cornish_fisher_var(r, alpha=0.99)
    assert "var_cf" in out
    # With negative skew or excess kurt, CF-VaR should differ from Gauss
    assert out["excess_kurt"] > 0  # heavy-tail
    assert abs(out["var_cf"] - out["var_gauss"]) > 0


def test_rolling_var_comparison_runs():
    rng = np.random.default_rng(0)
    r = pd.Series(
        rng.normal(0, 0.01, 800),
        index=pd.date_range("2020-01-01", periods=800),
    )
    out = rolling_var_comparison(r, window=200, alpha=0.95)
    assert "var_cf" in out.columns
    assert len(out) > 0


# ----- LPPL Bubble -----


def test_lppl_fit_runs():
    n = 100
    # Construct a fake bubble: exponential growth + log-periodic oscillation
    t = np.arange(n)
    t_c = 150
    tau = t_c - t
    log_price = 10 + (-0.5) * tau**0.5 + 0.1 * tau**0.5 * np.cos(8 * np.log(tau) - 1.0)
    s = pd.Series(log_price)
    fit = fit_lppl(s)
    assert 0 < fit.beta < 1
    assert fit.rmse < 1.0


def test_bubble_likelihood_score():
    n = 100
    t = np.arange(n)
    t_c = 150
    tau = t_c - t
    log_price = 10 + (-0.5) * tau**0.5 + 0.1 * tau**0.5 * np.cos(8 * np.log(tau) - 1.0)
    s = pd.Series(log_price)
    out = bubble_likelihood_score(s)
    assert "score" in out
    assert 0 <= out["score"] <= 1
