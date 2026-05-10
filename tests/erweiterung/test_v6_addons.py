"""Tests für v6-Add-Ons: Causal, Crisis-Composite, VAR, Cost, Sector-Rotation, Sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.causal_inference.double_ml import (
    double_ml,
    propensity_score_matching,
)
from erweiterung.cost_analytics.implementation_shortfall import (
    TradeContext,
    aggregate_trade_costs,
    implementation_shortfall,
    slippage_decomposition,
)
from erweiterung.portfolio.position_sizing import (
    atr_position_size,
    average_true_range,
    equal_risk_contribution_sizes,
    heat_based_sizing,
    volatility_targeted_size,
)
from erweiterung.risk.crisis_composite import (
    composite_crisis_index,
    crisis_state,
    exposure_multiplier_from_crisis,
)
from erweiterung.strategies.sector_rotation import (
    faber_trend_filter,
    relative_strength_ranking,
    sector_rotation_returns,
    top_n_sector_strategy,
)
from erweiterung.timeseries_tools.var_model import (
    fit_var,
    granger_causality_var,
    impulse_response,
    select_lag_order,
)


# ----- Double-ML -----


def test_double_ml_recovers_effect():
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.normal(0, 1, (n, 2))
    T = 0.5 * X[:, 0] + rng.normal(0, 0.5, n)
    true_beta = 2.0
    Y = true_beta * T + 0.3 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.3, n)
    res = double_ml(Y, T, X, n_folds=5)
    # Tightened from 0.5 to 0.1: DML on linear DGP with n=1000 should reach ~0.05 SE
    assert abs(res.treatment_effect - true_beta) < 0.1
    # Confidence interval should cover true beta
    assert res.confidence_interval[0] <= true_beta <= res.confidence_interval[1]


def test_propensity_matching_runs():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(0, 1, (n, 2)), columns=["x1", "x2"])
    # Treatment depends on X
    logit = 0.5 * X["x1"] - 0.3 * X["x2"]
    p = 1 / (1 + np.exp(-logit))
    T = pd.Series(rng.binomial(1, p))
    Y = pd.Series(2 * T + 0.5 * X["x1"] + rng.normal(0, 0.3, n))
    res = propensity_score_matching(T, Y, X)
    assert "ate" in res
    assert res["n_treated"] > 5


# ----- Crisis Composite -----


def test_composite_crisis_runs():
    rng = np.random.default_rng(0)
    n = 800
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    market = pd.Series(rng.normal(0, 0.01, n), index=dates)
    apc = pd.Series(rng.uniform(0.2, 0.8, n), index=dates)
    out = composite_crisis_index(market, correlation_apc=apc)
    assert "crisis_score" in out.columns
    assert out["crisis_score"].between(0, 1).all()


def test_crisis_state_with_hysteresis():
    score = pd.Series(
        [0.1, 0.2, 0.5, 0.8, 0.85, 0.6, 0.4, 0.2, 0.1],
        index=pd.date_range("2024-01-01", periods=9),
    )
    state = crisis_state(score, threshold_high=0.7, threshold_low=0.3)
    assert "crisis" in state.values
    # Should not flip back to normal at score=0.6 due to hysteresis
    assert state.iloc[5] != "normal"


def test_exposure_multiplier():
    state = pd.Series(["normal", "warning", "crisis", "normal"])
    mult = exposure_multiplier_from_crisis(state)
    assert mult.iloc[0] == 1.0
    assert mult.iloc[2] == 0.0


# ----- VAR -----


def test_var_fit():
    rng = np.random.default_rng(0)
    n = 300
    # Generate VAR(1) data
    y = np.zeros((n, 2))
    A = np.array([[0.5, 0.2], [0.1, 0.4]])
    for t in range(1, n):
        y[t] = A @ y[t - 1] + rng.multivariate_normal([0, 0], 0.01 * np.eye(2))
    df = pd.DataFrame(y, columns=["a", "b"])
    fit = fit_var(df, p=1)
    assert fit.n_vars == 2
    assert fit.p == 1


def test_select_lag_order():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), columns=["a", "b"])
    out = select_lag_order(df, max_p=4)
    assert "best_p_bic" in out


def test_granger_causality_var():
    rng = np.random.default_rng(0)
    n = 300
    # a Granger-causes b
    a = rng.normal(0, 1, n)
    b = np.zeros(n)
    for t in range(1, n):
        b[t] = 0.6 * a[t - 1] + rng.normal(0, 0.5)
    df = pd.DataFrame({"a": a, "b": b})
    res = granger_causality_var(df, cause="a", effect="b", p=1)
    assert "F" in res
    if "p_value" in res and res["p_value"] is not None and not np.isnan(res["p_value"]):
        assert res["p_value"] < 0.05


def test_impulse_response():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), columns=["a", "b"])
    fit = fit_var(df, p=1)
    irf = impulse_response(fit, horizon=10)
    assert irf.shape == (11, 2, 2)


# ----- Implementation Shortfall -----


def test_implementation_shortfall():
    ctx = TradeContext(
        side=1,  # buy
        intended_shares=1000,
        decision_price=100.0,
        arrival_price=100.10,  # slipped up by decision time
        avg_execution_price=100.30,  # impact during exec
        close_price=100.40,
        filled_shares=950,
        commission=10.0,
    )
    out = implementation_shortfall(ctx)
    assert out["total_cost_usd"] > 0
    # Delay cost positive (arrival > decision for a buy)
    assert out["delay_cost_usd"] > 0
    # Trading cost positive (avg_exec > arrival for buy)
    assert out["trading_cost_usd"] > 0


def test_slippage_decomposition():
    out = slippage_decomposition(
        target_price=100.0, fill_price=100.15, bid_ask_mid=100.10, side=1
    )
    assert out["half_spread_usd"] > 0
    assert out["adverse_selection_usd"] > 0
    assert out["total_slippage_usd"] > 0


def test_aggregate_trade_costs():
    df = pd.DataFrame(
        {"cost_bps": [5, 10, 15, 8, 6], "notional": [1e6, 2e6, 1.5e6, 3e6, 1e6]}
    )
    out = aggregate_trade_costs(df)
    assert "weighted_avg_cost_bps" in out
    assert out["weighted_avg_cost_bps"] > 0


# ----- Sector Rotation -----


def test_faber_trend_filter():
    rng = np.random.default_rng(0)
    # uptrend
    p = pd.Series(100 + np.cumsum(rng.normal(0.1, 0.5, 300)))
    sig = faber_trend_filter(p, lookback=50)
    # Should be mostly 1
    assert sig.iloc[100:].mean() > 0.5


def test_relative_strength_ranking():
    rng = np.random.default_rng(0)
    n_sec = 5
    n = 400
    prices = pd.DataFrame(
        100 * (1 + rng.normal(0.0005, 0.01, (n, n_sec))).cumprod(axis=0),
        columns=[f"S{i}" for i in range(n_sec)],
    )
    ranks = relative_strength_ranking(prices, lookback=252, skip=21)
    valid = ranks.dropna()
    assert valid.shape[0] > 0


def test_top_n_sector_strategy():
    rng = np.random.default_rng(0)
    n_sec = 4
    n = 400
    prices = pd.DataFrame(
        100 * (1 + rng.normal(0.0005, 0.01, (n, n_sec))).cumprod(axis=0),
        columns=[f"S{i}" for i in range(n_sec)],
    )
    w = top_n_sector_strategy(prices, n_top=2, lookback=200, skip=21)
    # Weights should sum to ≤ 1
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_sector_rotation_returns():
    rng = np.random.default_rng(0)
    n = 100
    w = pd.DataFrame(
        rng.uniform(0, 1, (n, 3)),
        columns=["A", "B", "C"],
        index=pd.date_range("2024-01-01", periods=n),
    )
    w = w.div(w.sum(axis=1), axis=0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (n, 3)),
        columns=["A", "B", "C"],
        index=w.index,
    )
    out = sector_rotation_returns(w, ret)
    assert len(out) == n


# ----- Position Sizing -----


def test_heat_based_sizing():
    out = heat_based_sizing(equity=100000, entry_price=50, stop_price=45, heat_pct=0.02)
    assert out["dollar_risk"] == 2000  # 2% of 100k
    assert out["shares"] == 400  # 2000 / 5
    assert out["ratio_of_equity"] < 1


def test_average_true_range():
    rng = np.random.default_rng(0)
    n = 100
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    atr = average_true_range(high, low, close, window=14)
    valid = atr.dropna()
    assert (valid > 0).all()


def test_atr_position_size():
    out = atr_position_size(equity=100000, entry_price=50, atr=1.5, risk_pct=0.01)
    assert out["shares"] > 0
    assert out["dollar_risk"] == 1000


def test_vol_targeted_size():
    out = volatility_targeted_size(
        capital=100000, asset_vol_annualized=0.30, target_vol_annualized=0.15
    )
    assert out["leverage"] == 0.5
    assert out["notional"] == 50000


def test_erc_sizes():
    vols = pd.Series([0.20, 0.30, 0.40], index=["A", "B", "C"])
    out = equal_risk_contribution_sizes(
        capital=100000, asset_vols=vols, target_total_vol=0.15
    )
    assert len(out) == 3
    assert (out > 0).all()
