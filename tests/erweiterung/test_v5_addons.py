"""Tests für v5-Add-Ons: Options-Pricing, Heston, DCC-GARCH, OU, Shrinkage,
Max-Div, RMT-Denoising, Bandit, Graph-Methods, Brinson-Attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.attribution.brinson import (
    brinson_attribution,
    factor_attribution,
    pnl_decomposition,
)
from erweiterung.graph_methods.correlation_graph import (
    asset_degrees_in_mst,
    cluster_diversification_weights,
    correlation_distance,
    mst_kruskal,
    spectral_clustering_assets,
)
from erweiterung.meta.bandit_allocator import (
    EpsilonGreedy,
    ThompsonGaussian,
    UCB1,
    run_bandit_on_strategy_returns,
)
from erweiterung.options_pricing.black_scholes import (
    BSParams,
    bs_greeks,
    bs_price,
    implied_volatility,
    iv_smile,
)
from erweiterung.options_pricing.heston import (
    HestonParams,
    heston_char_function,
    heston_price_cos,
    heston_simulate_paths,
)
from erweiterung.portfolio.covariance_shrinkage import (
    constant_correlation_target_shrinkage,
    ledoit_wolf_shrinkage,
    rie_clip_eigenvalues,
)
from erweiterung.portfolio.max_diversification import (
    diversification_ratio,
    max_diversification_weights,
    max_sharpe_weights_analytical,
    max_sharpe_weights_constrained,
)
from erweiterung.portfolio.rmt_denoising import (
    denoise_covariance,
    fit_marchenko_pastur,
    signal_to_noise_ratio,
)
from erweiterung.timeseries_tools.ornstein_uhlenbeck import (
    fit_ornstein_uhlenbeck,
    is_mean_reverting,
    ou_simulate,
)
from erweiterung.volatility.dcc_garch import dcc_covariance_at, fit_dcc_garch


# ----- Black-Scholes -----


def test_bs_call_price_in_the_money():
    p = BSParams(spot=110, strike=100, time_to_expiry=0.5, risk_free=0.05)
    px = bs_price(p, sigma=0.20, is_call=True)
    assert px > 10  # at least intrinsic 10
    assert px < 110


def test_bs_put_call_parity():
    p = BSParams(spot=100, strike=100, time_to_expiry=0.5, risk_free=0.05)
    c = bs_price(p, sigma=0.20, is_call=True)
    pp = bs_price(p, sigma=0.20, is_call=False)
    # C - P = S - K*exp(-rT)
    parity_lhs = c - pp
    parity_rhs = 100 - 100 * np.exp(-0.05 * 0.5)
    assert abs(parity_lhs - parity_rhs) < 1e-6


def test_bs_greeks_signs():
    p = BSParams(spot=100, strike=100, time_to_expiry=0.5, risk_free=0.05)
    g = bs_greeks(p, sigma=0.25, is_call=True)
    assert 0 < g["delta"] < 1  # call delta
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_implied_vol_roundtrip():
    p = BSParams(spot=100, strike=100, time_to_expiry=0.5, risk_free=0.05)
    true_vol = 0.30
    market_px = bs_price(p, sigma=true_vol, is_call=True)
    iv = implied_volatility(market_px, p, is_call=True)
    assert abs(iv - true_vol) < 1e-4


def test_iv_smile_basic():
    spot = 100
    strikes = np.array([90, 95, 100, 105, 110])
    prices = np.array([bs_price(BSParams(spot, k, 0.5, 0.05), 0.20) for k in strikes])
    iv = iv_smile(spot, strikes, prices, 0.5)
    assert (np.abs(iv - 0.20) < 1e-3).all()


# ----- Heston -----


def test_heston_char_function_at_zero():
    p = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04, spot=100)
    val = heston_char_function(np.array([0.0]), p, T=0.5)
    # φ(0) = 1 by definition (up to discount); for our formulation should be ~spot factor
    assert np.isfinite(val[0])


def test_heston_price_runs():
    p = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04, spot=100)
    price = heston_price_cos(p, strike=100, T=0.5, is_call=True)
    assert np.isfinite(price)
    assert 0 < price < 100


def test_heston_simulate_paths():
    p = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04, spot=100)
    S, v = heston_simulate_paths(p, T=0.5, n_steps=50, n_paths=100, seed=42)
    assert S.shape == (100, 51)
    assert (S > 0).all()
    assert (v >= 0).all()


# ----- DCC-GARCH -----


def test_dcc_garch_fits():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(0, 0.01, n)
    r1 = pd.Series(base + rng.normal(0, 0.005, n), name="A")
    r2 = pd.Series(0.5 * base + rng.normal(0, 0.005, n), name="B")
    df = pd.concat([r1, r2], axis=1)
    fit = fit_dcc_garch(df)
    assert 0 <= fit.alpha <= 1
    assert 0 <= fit.beta <= 1
    cov_last = dcc_covariance_at(fit, len(df) - 1)
    assert cov_last.shape == (2, 2)
    assert cov_last[0, 0] > 0


# ----- Ornstein-Uhlenbeck -----


def test_ou_fit_recovers_mean():
    rng = np.random.default_rng(0)
    n = 1000
    theta = 0.5
    mu = 2.0
    sigma = 0.3
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    fit = fit_ornstein_uhlenbeck(pd.Series(x))
    assert abs(fit.mu - mu) < 0.5
    assert fit.theta > 0


def test_ou_simulate_runs():
    rng_seed = 42
    fit = type("F", (), {"theta": 0.5, "mu": 0.0, "sigma": 0.1})()  # mock
    path = ou_simulate(fit, n_steps=100, seed=rng_seed)
    assert path.shape == (100,)


def test_is_mean_reverting():
    rng = np.random.default_rng(0)
    n = 500
    # OU process: mean-reverting
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] + 0.3 * (1.0 - x[t - 1]) + 0.1 * rng.standard_normal()
    assert is_mean_reverting(pd.Series(x)) is True
    # Random walk: not mean-reverting
    rw = np.cumsum(rng.standard_normal(n))
    assert is_mean_reverting(pd.Series(rw)) is False


# ----- Covariance Shrinkage -----


def test_ledoit_wolf_shrinkage():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 5)), columns=[f"A{i}" for i in range(5)])
    shrunk, alpha = ledoit_wolf_shrinkage(df)
    assert 0 <= alpha <= 1
    assert shrunk.shape == (5, 5)
    # Diagonal positive
    assert (np.diag(shrunk) > 0).all()


def test_constant_correlation_shrinkage():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (200, 4)), columns=["A", "B", "C", "D"])
    shrunk, alpha = constant_correlation_target_shrinkage(df)
    assert shrunk.shape == (4, 4)


def test_rie_clip():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (100, 10)))
    clean = rie_clip_eigenvalues(df)
    assert clean.shape == (10, 10)


# ----- Max-Diversification + Max-Sharpe -----


def test_max_diversification():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (200, 4)), columns=["A", "B", "C", "D"])
    cov = df.cov()
    w = max_diversification_weights(cov, long_only=True, max_weight=0.5)
    # SLSQP with equality constraint should reach 1e-4 precision
    assert abs(w.sum() - 1.0) < 1e-4
    dr = diversification_ratio(w, cov)
    assert dr >= 1.0


def test_max_sharpe_analytical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0.0005, 0.01, (200, 4)), columns=["A", "B", "C", "D"])
    cov = df.cov()
    mu = df.mean()
    w = max_sharpe_weights_analytical(mu, cov)
    assert abs(w.sum() - 1.0) < 1e-6


def test_max_sharpe_constrained():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0.001, 0.01, (200, 4)), columns=["A", "B", "C", "D"])
    cov = df.cov()
    mu = df.mean()
    w = max_sharpe_weights_constrained(mu, cov, long_only=True, max_weight=0.4)
    assert (w >= -1e-9).all()
    assert (w <= 0.4 + 1e-9).all()


# ----- RMT Denoising -----


def test_marchenko_pastur_fit():
    rng = np.random.default_rng(0)
    n = 50
    T = 200
    X = rng.normal(0, 1, (T, n))
    Xc = X - X.mean(axis=0)
    corr = np.corrcoef(Xc.T)
    eigvals = np.linalg.eigvalsh(corr)
    mp = fit_marchenko_pastur(eigvals, T=T, N=n)
    assert mp.lambda_plus > mp.lambda_minus
    assert 0 < mp.q < 1


def test_denoise_correlation():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (200, 10)))
    cov = denoise_covariance(df)
    assert cov.shape == (10, 10)
    # diagonal preserved
    assert (np.diag(cov.values) > 0).all()


def test_signal_to_noise_ratio():
    rng = np.random.default_rng(0)
    # i.i.d. random -> SNR should be near 0
    X = rng.normal(0, 1, (500, 50))
    eigvals = np.linalg.eigvalsh(np.corrcoef(X.T))
    snr = signal_to_noise_ratio(eigvals, T=500, N=50)
    assert 0 <= snr <= 1


# ----- Bandit -----


def test_epsilon_greedy():
    rng = np.random.default_rng(0)
    bandit = EpsilonGreedy(n_arms=3, epsilon=0.1, seed=42)
    # Arm 0 best, Arm 1 mid, Arm 2 worst
    true_means = [0.5, 0.0, -0.5]
    for _ in range(500):
        a = bandit.select()
        r = true_means[a] + rng.normal(0, 0.5)
        bandit.update(a, r)
    means = [bandit.state.mean_reward(i) for i in range(3)]
    # Best arm should be selected most often
    assert bandit.state.counts[0] > bandit.state.counts[2]


def test_ucb1_picks_best():
    rng = np.random.default_rng(0)
    bandit = UCB1(n_arms=3)
    true_means = [1.0, 0.0, -1.0]
    for _ in range(500):
        a = bandit.select()
        r = true_means[a] + rng.normal(0, 0.3)
        bandit.update(a, r)
    assert bandit.state.counts[0] > bandit.state.counts[2]


def test_thompson_basic():
    bandit = ThompsonGaussian(n_arms=3, prior_std=1.0, obs_std=1.0, seed=0)
    for _ in range(100):
        a = bandit.select()
        bandit.update(a, 1.0 if a == 0 else 0.0)
    # Arm 0 should be preferred
    assert bandit.state.counts[0] > bandit.state.counts[1]


def test_bandit_on_strategy_returns():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.01, n),
            "B": rng.normal(-0.001, 0.01, n),
        }
    )
    out = run_bandit_on_strategy_returns(df, algorithm="ucb1")
    assert "chosen" in out.columns
    assert len(out) == n


# ----- Graph Methods -----


def test_correlation_distance():
    corr = pd.DataFrame(
        [[1.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    d = correlation_distance(corr)
    assert d.iat[0, 0] == 0
    assert d.iat[0, 1] > 0


def test_mst_kruskal_n_minus_1():
    corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.5, 0.1],
            [0.9, 1.0, 0.6, 0.2],
            [0.5, 0.6, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    d = correlation_distance(corr)
    mst = mst_kruskal(d)
    assert len(mst) == 3  # n - 1


def test_mst_degrees():
    corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.5, 0.1],
            [0.9, 1.0, 0.6, 0.2],
            [0.5, 0.6, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    d = correlation_distance(corr)
    mst = mst_kruskal(d)
    deg = asset_degrees_in_mst(mst)
    assert sum(deg.values()) == 2 * len(mst)


def test_spectral_clustering():
    rng = np.random.default_rng(0)
    n = 10
    X = rng.normal(0, 1, (200, n))
    corr = pd.DataFrame(
        np.corrcoef(X.T),
        index=[f"A{i}" for i in range(n)],
        columns=[f"A{i}" for i in range(n)],
    )
    clusters = spectral_clustering_assets(corr, n_clusters=3)
    assert len(clusters) == n
    assert all(0 <= v < 3 for v in clusters.values())


def test_cluster_diversification_weights():
    cluster_assignment = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2}
    w = cluster_diversification_weights(cluster_assignment, n_clusters=3)
    assert abs(w.sum() - 1.0) < 1e-9


# ----- Brinson Attribution -----


def test_brinson_attribution():
    res = brinson_attribution(
        portfolio_weights={"Tech": 0.5, "Energy": 0.3, "Finance": 0.2},
        benchmark_weights={"Tech": 0.4, "Energy": 0.3, "Finance": 0.3},
        portfolio_returns={"Tech": 0.05, "Energy": 0.02, "Finance": 0.01},
        benchmark_returns={"Tech": 0.04, "Energy": 0.02, "Finance": 0.015},
    )
    assert "Tech" in res.allocation_effect
    # Check that sum of effects ≈ active return
    total_eff = (
        sum(res.allocation_effect.values())
        + sum(res.selection_effect.values())
        + sum(res.interaction_effect.values())
    )
    assert abs(total_eff - res.total_active_return) < 1e-9


def test_factor_attribution():
    rng = np.random.default_rng(0)
    n = 200
    factor_returns = pd.DataFrame(
        {"MKT": rng.normal(0.001, 0.01, n), "SMB": rng.normal(0, 0.005, n)}
    )
    alpha_true = 0.0005
    asset = pd.Series(
        alpha_true
        + 1.2 * factor_returns["MKT"]
        - 0.3 * factor_returns["SMB"]
        + rng.normal(0, 0.005, n)
    )
    res = factor_attribution(asset, factor_returns)
    assert "alpha" in res
    assert "factor_loadings" in res
    assert abs(res["factor_loadings"]["MKT"] - 1.2) < 0.3


def test_pnl_decomposition():
    rng = np.random.default_rng(0)
    n = 50
    weights = pd.DataFrame(
        rng.uniform(0, 1, (n, 3)),
        columns=["A", "B", "C"],
    )
    weights = weights.div(weights.sum(axis=1), axis=0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), columns=["A", "B", "C"])
    pnl = pnl_decomposition(weights, returns)
    assert pnl.shape == (n, 3)
