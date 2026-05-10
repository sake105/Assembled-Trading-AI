"""Tests für v8-Add-Ons: TAR, Permutation-Importance, Trend-Following, Active-Risk,
Pairs-Arbitrage, Spectral-Analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.attribution.active_risk import (
    active_risk_decomposition,
    active_share,
    concentration_metrics,
    information_ratio,
    tracking_error,
    turnover_ratio,
)
from erweiterung.ml.permutation_importance import (
    permutation_importance,
    shapley_sampling_values,
)
from erweiterung.strategies.pairs_arbitrage import (
    aggregate_pair_pnl,
    cointegration_engle_granger,
    trade_pair,
)
from erweiterung.strategies.trend_following import (
    cta_multi_asset_strategy,
    donchian_breakout,
    dual_ma_crossover,
    time_series_momentum_signal,
)
from erweiterung.timeseries_tools.spectral_analysis import (
    cross_spectrum,
    periodogram,
    power_spectral_density_welch,
)
from erweiterung.timeseries_tools.tar_model import (
    fit_setar,
    linearity_test_tsay,
    setar_forecast,
)


# ----- TAR -----


def test_setar_fit():
    rng = np.random.default_rng(0)
    n = 500
    s = np.zeros(n)
    for t in range(1, n):
        if s[t - 1] <= 0:
            s[t] = 0.5 * s[t - 1] + rng.normal()
        else:
            s[t] = -0.3 * s[t - 1] + rng.normal()
    fit = fit_setar(pd.Series(s), p=1, delay=1)
    assert fit.p == 1
    assert fit.delay == 1


def test_setar_forecast():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500))
    fit = fit_setar(s, p=1, delay=1)
    forecast = setar_forecast(fit, s)
    assert np.isfinite(forecast)


def test_tsay_linearity():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500))
    res = linearity_test_tsay(s, p=1, delay=1)
    assert "F" in res


# ----- Permutation Importance -----


def test_permutation_importance():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(0, 1, (n, 3))
    y = 2 * X[:, 0] + 0.1 * rng.normal(0, 1, n)

    def model_predict(Xq):
        # use OLS fit on training data X, y (closure)
        Xb = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        Xeb = np.column_stack([np.ones(len(Xq)), Xq])
        return Xeb @ beta

    imp = permutation_importance(model_predict, X, y, n_repeats=5)
    # Feature 0 should be most important
    top_feature = int(imp.iloc[0]["feature_idx"])
    assert top_feature == 0


def test_shapley_sampling():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1, (n, 2))

    def predict(Xq):
        return Xq.sum(axis=1)

    phi = shapley_sampling_values(predict, X, n_samples=20)
    assert phi.shape == (n, 2)


# ----- Trend Following -----


def test_tsm_signal():
    rng = np.random.default_rng(0)
    # uptrend
    p = pd.Series(100 + np.cumsum(rng.normal(0.05, 0.5, 400)))
    sig = time_series_momentum_signal(p, lookback=252)
    valid = sig.iloc[300:]
    assert valid.mean() > 0  # mostly positive


def test_donchian_breakout():
    rng = np.random.default_rng(0)
    p = pd.Series(100 + np.cumsum(rng.normal(0.1, 0.5, 200)))
    sig = donchian_breakout(p, lookback=20)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_dual_ma_crossover():
    rng = np.random.default_rng(0)
    p = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))
    sig = dual_ma_crossover(p, fast=50, slow=200)
    assert sig.isin([0.0, 1.0]).all()


def test_cta_multi_asset():
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        100 * (1 + rng.normal(0.0005, 0.01, (300, 3))).cumprod(axis=0),
        columns=["A", "B", "C"],
    )
    w = cta_multi_asset_strategy(panel, method="tsm", vol_lookback=30)
    assert w.shape == (300, 3)


# ----- Active Risk -----


def test_tracking_error():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    b = pd.Series(rng.normal(0.0005, 0.01, 500))
    te = tracking_error(r, b)
    assert te > 0


def test_active_share():
    p = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
    b = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    sh = active_share(p, b)
    assert 0 <= sh <= 1


def test_information_ratio():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    b = pd.Series(rng.normal(0.0005, 0.01, 500))
    ir = information_ratio(r, b)
    assert np.isfinite(ir) or pd.isna(ir)


def test_active_risk_decomposition():
    rng = np.random.default_rng(0)
    n = 300
    fac = pd.DataFrame({"MKT": rng.normal(0, 0.01, n), "SMB": rng.normal(0, 0.005, n)})
    b = pd.Series(rng.normal(0.0005, 0.01, n))
    r = b + 0.5 * fac["MKT"] + rng.normal(0, 0.003, n)
    res = active_risk_decomposition(r, b, fac)
    assert "total_active_risk_ann" in res


def test_turnover():
    rng = np.random.default_rng(0)
    n = 50
    w = pd.DataFrame(
        rng.uniform(0, 1, (n, 3)),
        columns=["A", "B", "C"],
    )
    w = w.div(w.sum(axis=1), axis=0)
    t = turnover_ratio(w)
    assert (t >= 0).all()


def test_concentration():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    m = concentration_metrics(w)
    assert "hhi" in m
    assert m["effective_n"] > 1


# ----- Pairs Arbitrage -----


def test_engle_granger_cointegrated():
    rng = np.random.default_rng(0)
    n = 500
    common = rng.normal(0, 1, n).cumsum()
    y = pd.Series(common + rng.normal(0, 0.3, n))
    x = pd.Series(common + rng.normal(0, 0.3, n))
    res = cointegration_engle_granger(y, x)
    assert "beta" in res


def test_trade_pair_runs():
    rng = np.random.default_rng(0)
    n = 500
    common = rng.normal(0, 1, n).cumsum()
    y = pd.Series(100 + common + rng.normal(0, 0.5, n))
    x = pd.Series(100 + common + rng.normal(0, 0.5, n))
    positions, trades = trade_pair(y, x, rolling_window=30)
    assert isinstance(positions, pd.DataFrame)
    if trades:
        summary = aggregate_pair_pnl(trades)
        assert "total_pnl" in summary


# ----- Spectral Analysis -----


def test_periodogram_detects_period():
    n = 256
    t = np.arange(n)
    # Strong 16-period component
    s = pd.Series(
        np.cos(2 * np.pi * t / 16) + 0.3 * np.random.default_rng(0).normal(0, 1, n)
    )
    p = periodogram(s)
    # Dominant period should be near 16
    assert abs(p.dominant_period - 16) < 4


def test_cross_spectrum():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(0, 1, n)
    x = pd.Series(base + rng.normal(0, 0.3, n))
    y = pd.Series(base + rng.normal(0, 0.3, n))
    res = cross_spectrum(x, y)
    assert "coherence_squared" in res
    assert (res["coherence_squared"] >= 0).all()


def test_welch_psd():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 256))
    out = power_spectral_density_welch(s, n_segments=4)
    assert "psd" in out
    assert (out["psd"] >= 0).all()
