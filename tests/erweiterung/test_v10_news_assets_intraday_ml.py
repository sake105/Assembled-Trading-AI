"""Tests für v10-Add-Ons:
- news_impact (8 Module)
- asset_specific (4 Module)
- intraday (4 Module)
- ml_advanced (3 Module)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.asset_specific.bonds_nelson_siegel import (
    fit_nelson_siegel,
    fit_panel_ns,
    nelson_siegel_basis,
    yield_curve_inversion_signal,
)
from erweiterung.asset_specific.commodities_roll_yield import (
    backwardation_cross_section,
    curve_steepness,
    momentum_in_commodity_curve,
    roll_yield,
)
from erweiterung.asset_specific.crypto_funding import (
    annualized_funding_rate,
    crypto_mean_reversion_signal,
    funding_zscore,
    long_squeeze_risk,
    perpetual_basis,
)
from erweiterung.asset_specific.fx_carry import (
    carry_crash_indicator,
    carry_ranking,
)
from erweiterung.intraday.jump_detection import (
    bipower_variation,
    jump_intensity,
    lee_mykland_test,
    split_continuous_jump_variance,
)
from erweiterung.intraday.lee_ready import (
    lee_ready_classify,
    order_flow_imbalance,
    rolling_ofi_imbalance_ratio,
    tick_rule_classify,
)
from erweiterung.intraday.order_book_proxies import (
    close_position_in_range,
    imbalance_composite,
    money_flow_index,
    on_balance_volume,
)
from erweiterung.intraday.two_scale_rv import (
    realized_kernel_variance,
    rolling_intraday_volatility_panel,
    two_scale_realized_variance,
)
from erweiterung.ml_advanced.gaussian_process import (
    fit_gp,
    gp_marginal_log_likelihood,
    gp_predict,
    grid_search_hyperparams,
    matern_kernel,
    rbf_kernel,
)
from erweiterung.ml_advanced.particle_filter import (
    ParticleFilter,
    stoch_vol_particle_filter_example,
)
from erweiterung.news_impact.cross_asset_spillover import (
    co_mention_matrix,
    propagate_news_to_followers,
    sentiment_spillover_matrix,
)
from erweiterung.news_impact.decay_model import (
    cumulative_news_impact_signal,
    expected_impact,
    fit_news_decay_model,
)
from erweiterung.news_impact.event_clustering import (
    event_clusters_per_day,
    event_size_distribution,
)
from erweiterung.news_impact.news_surprise import (
    compute_surprise,
    cross_section_surprise_rank,
    rolling_sentiment_baseline,
    standardized_surprise,
    surprise_to_signal,
)
from erweiterung.news_impact.reactivity_index import (
    reactivity_panel,
)
from erweiterung.news_impact.sentiment_divergence import (
    compute_divergence_panel,
    detect_extreme_divergence,
)
from erweiterung.news_impact.time_of_day_impact import (
    classify_time_of_day,
    split_returns_by_session,
)
from erweiterung.news_impact.topic_drift import (
    detect_topic_change_points,
    jensen_shannon_divergence,
    topic_drift_signal,
    topic_persistence,
)


def _make_news_panel(n: int = 200, n_sym: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_sym)]:
        for d in pd.date_range("2024-01-01", periods=n, tz="UTC"):
            if rng.random() < 0.3:
                rows.append(
                    {
                        "date": d,
                        "symbol": sym,
                        "sentiment": float(rng.uniform(-1, 1)),
                        "headline": f"news about {sym} on {d.date()}",
                    }
                )
    return pd.DataFrame(rows)


def _make_returns_panel(n: int = 200, n_sym: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_sym)]:
        for d in pd.date_range("2024-01-01", periods=n, tz="UTC"):
            rows.append(
                {"date": d, "symbol": sym, "return": float(rng.normal(0, 0.01))}
            )
    return pd.DataFrame(rows)


# ===== News Impact =====


def test_decay_model_fits():
    news = _make_news_panel(200, 5, 0)
    rets = _make_returns_panel(200, 5, 0)
    fit = fit_news_decay_model(news, rets, horizons=(1, 3, 5, 10))
    assert fit.decay_lambda > 0
    assert fit.half_life_days > 0
    impact = expected_impact(fit, sentiment=0.5, h=5)
    assert np.isfinite(impact)


def test_cumulative_news_impact():
    from erweiterung.news_impact.decay_model import DecayFit

    news = _make_news_panel(100, 3, 0)
    fit = DecayFit(
        alpha=0.0,
        beta=0.01,
        decay_lambda=0.3,
        half_life_days=2.3,
        r_squared=0.1,
        n_obs=100,
    )
    out = cumulative_news_impact_signal(news, fit)
    assert "news_impact_signal" in out.columns


def test_news_surprise():
    news = _make_news_panel(200, 3, 0)
    with_baseline = rolling_sentiment_baseline(news, window=30)
    assert "baseline" in with_baseline.columns
    with_surprise = compute_surprise(with_baseline)
    assert "surprise" in with_surprise.columns
    standardized = standardized_surprise(news, window=30)
    assert "surprise_z" in standardized.columns
    ranked = cross_section_surprise_rank(with_surprise)
    assert "surprise_rank_pct" in ranked.columns
    sig = surprise_to_signal(standardized, threshold=1.5)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_reactivity_index():
    news = _make_news_panel(300, 4, 0)
    rets = _make_returns_panel(300, 4, 0)
    panel = reactivity_panel(news, rets, min_news=10)
    assert "beta" in panel.columns if not panel.empty else True


def test_co_mention_matrix():
    df = pd.DataFrame(
        {
            "headline": [
                "Apple and Microsoft announce partnership",
                "Apple and Microsoft announce partnership",
                "Apple and Microsoft announce partnership",
                "Tesla updates roadmap",
            ],
            "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        }
    )
    M = co_mention_matrix(df, min_co_mentions=1)
    # AAPL-MSFT should have co-mention
    assert M.loc["AAPL", "MSFT"] > 0


def test_spillover_matrix():
    news = _make_news_panel(200, 3, 0)
    rets = _make_returns_panel(200, 3, 0)
    M = sentiment_spillover_matrix(news, rets, horizon_days=3)
    if not M.empty:
        assert M.shape[0] == M.shape[1]


def test_propagate_news():
    news = _make_news_panel(50, 2, 0)
    M = pd.DataFrame([[1.0, 0.4], [0.3, 1.0]], index=["S0", "S1"], columns=["S0", "S1"])
    out = propagate_news_to_followers(news, M, threshold=0.2)
    assert isinstance(out, pd.DataFrame)


def test_topic_drift():
    # Fake topic-distributions
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        rng.dirichlet([1, 1, 1, 1], n),
        columns=[f"topic_{i}" for i in range(4)],
        index=pd.date_range("2024-01-01", periods=n, tz="UTC"),
    )
    sig = topic_drift_signal(df, baseline_window=30)
    assert isinstance(sig, pd.Series)
    persistence = topic_persistence(df)
    assert isinstance(persistence, pd.Series)
    changes = detect_topic_change_points(sig)
    assert isinstance(changes, list)


def test_jensen_shannon():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert jensen_shannon_divergence(p, q) == 0
    p2 = np.array([1.0, 0.0])
    q2 = np.array([0.0, 1.0])
    assert jensen_shannon_divergence(p2, q2) > 0.5


def test_time_of_day_classification():
    times = pd.Series(
        pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00", "2024-01-01 17:00"])
    )
    labels = classify_time_of_day(times)
    assert labels.iloc[0] == "pre_market"
    assert labels.iloc[1] == "intraday"
    assert labels.iloc[2] == "after_hours"


def test_split_returns_by_session():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["A"] * 5,
            "open": [100, 102, 101, 103, 104],
            "close": [102, 101, 103, 104, 105],
        }
    )
    out = split_returns_by_session(df)
    assert "overnight_return" in out.columns
    assert "intraday_return" in out.columns


def test_event_clusters():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 6),
            "headline": [
                "Apple earnings beat",
                "Apple beats Q3 earnings",
                "Apple Q3 results strong",
                "Microsoft new product",
                "Microsoft announces cloud",
                "Microsoft cloud growth",
            ],
        }
    )
    clustered = event_clusters_per_day(df, distance_threshold=0.7)
    assert "event_cluster_id" in clustered.columns
    sizes = event_size_distribution(clustered)
    assert "n_articles" in sizes.columns


def test_sentiment_divergence():
    news = _make_news_panel(200, 3, 0)
    social = _make_news_panel(200, 3, 99)
    panel = compute_divergence_panel(news, social, rolling_window=20)
    assert "divergence_z" in panel.columns
    extreme = detect_extreme_divergence(panel, threshold=1.0)
    assert "abs_divergence" in extreme.columns


# ===== Asset Specific =====


def test_crypto_funding_zscore():
    rng = np.random.default_rng(0)
    fr = pd.Series(rng.normal(0.0001, 0.0002, 300))
    z = funding_zscore(fr, lookback=60)
    valid = z.dropna()
    assert valid.shape[0] > 0


def test_annualized_funding():
    fr = pd.Series([0.0001, 0.0002, 0.0001])
    ann = annualized_funding_rate(fr, n_per_day=3)
    expected = 0.0001 * 3 * 365
    assert abs(ann.iloc[0] - expected) < 1e-9


def test_perpetual_basis():
    perp = pd.Series([101, 102, 103])
    spot = pd.Series([100, 100, 100])
    basis = perpetual_basis(perp, spot)
    assert basis.iloc[0] > 0


def test_long_squeeze_risk():
    rng = np.random.default_rng(0)
    fr = pd.Series(rng.normal(0.0002, 0.0001, 200))
    oi = pd.Series(rng.uniform(1e6, 5e6, 200))
    risk = long_squeeze_risk(fr, oi, lookback=30)
    assert (risk >= 0).all()


def test_crypto_mean_reversion():
    rng = np.random.default_rng(0)
    fr = pd.Series(rng.normal(0, 0.001, 200))
    fr.iloc[100] = 0.005  # extreme spike
    sig = crypto_mean_reversion_signal(fr, threshold_z=2.0, lookback=50)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_fx_carry_ranking():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "currency": ["USD", "EUR", "JPY", "AUD", "CAD"],
            "interest_rate": [4.0, 3.0, 0.1, 4.5, 4.2],
        }
    )
    res = carry_ranking(df, n_long=2, n_short=2)
    assert "position" in res.columns


def test_carry_crash_indicator():
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 500))
    ret.iloc[400:420] = rng.normal(0, 0.05, 20)  # crash period
    ind = carry_crash_indicator(ret, vol_window=30, threshold=2.0)
    assert ind.iloc[400:420].sum() > 0


def test_nelson_siegel_basis():
    tau = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    B = nelson_siegel_basis(tau, lam=0.0609)
    assert B.shape == (5, 3)
    # Level should be all-ones
    assert (B[:, 0] == 1).all()


def test_ns_fit():
    tau = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    yields = np.array([4.0, 4.2, 4.5, 4.7, 5.0])
    fit = fit_nelson_siegel(yields, tau)
    assert "level" in fit
    assert "slope" in fit
    assert fit["r_squared"] > 0.5  # should fit well


def test_ns_panel():
    rng = np.random.default_rng(0)
    tau = np.array([0.25, 1.0, 5.0, 10.0])
    n = 60
    panel = pd.DataFrame(
        rng.uniform(2, 5, (n, 4)),
        columns=tau,
        index=pd.date_range("2024-01-01", periods=n, freq="MS"),
    )
    factors = fit_panel_ns(panel, tau)
    assert "level" in factors.columns


def test_yield_curve_inversion():
    factors = pd.DataFrame({"slope": [0.5, -0.2, 0.1, -0.3]})
    sig = yield_curve_inversion_signal(factors)
    assert sig.tolist() == [0, 1, 0, 1]


def test_commodities_roll_yield():
    near = pd.Series([100, 101, 102])
    far = pd.Series([99, 100, 101])
    ry = roll_yield(near, far, days_between=30)
    assert (ry > 0).all()  # backwardation


def test_curve_steepness():
    df = pd.DataFrame(
        {"c1": [100], "c2": [101], "c3": [102], "c4": [103]},
        index=[pd.Timestamp("2024-01-01")],
    )
    slope = curve_steepness(df, [30, 60, 90, 120])
    assert slope.iloc[0] > 0


def test_backwardation_cross_section():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 5),
            "commodity": ["A", "B", "C", "D", "E"],
            "roll_yield": [0.1, 0.05, 0.0, -0.05, -0.1],
        }
    )
    res = backwardation_cross_section(df, n_long=2, n_short=2)
    assert "position" in res.columns


def test_momentum_commodities():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "date": pd.concat(
                [pd.Series(pd.date_range("2024-01-01", periods=n))] * 2
            ).reset_index(drop=True),
            "commodity": ["A"] * n + ["B"] * n,
            "near_price": np.concatenate(
                [
                    100 + np.cumsum(rng.normal(0.05, 0.5, n)),
                    100 + np.cumsum(rng.normal(-0.02, 0.5, n)),
                ]
            ),
        }
    )
    mom = momentum_in_commodity_curve(df, lookback=200, skip=20)
    assert "momentum" in mom.columns


# ===== Intraday =====


def test_two_scale_rv():
    rng = np.random.default_rng(0)
    n = 500
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, n)))
    res = two_scale_realized_variance(prices, sparse_step=20)
    assert res.tsrv >= 0


def test_realized_kernel_variance():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.005, 200)
    rkv = realized_kernel_variance(rets, bandwidth=5)
    assert rkv >= 0


def test_rolling_intraday_panel():
    rng = np.random.default_rng(0)
    # Multi-day intraday panel
    n_per_day = 300
    days = 3
    idx = []
    prices = []
    for d in range(days):
        start = pd.Timestamp(f"2024-01-0{d + 1} 09:30")
        for i in range(n_per_day):
            idx.append(start + pd.Timedelta(minutes=i))
            prices.append(100 + rng.normal(0, 0.05))
    s = pd.Series(prices, index=idx)
    panel = rolling_intraday_volatility_panel(pd.DataFrame(s), sparse_step=20)
    assert isinstance(panel, pd.Series)


def test_bipower_variation():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 500)
    bv = bipower_variation(rets)
    assert bv >= 0


def test_lee_mykland():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 600)
    rets[400] = 0.10  # extreme jump
    res = lee_mykland_test(rets, window=200, alpha=0.01)
    assert res.n_jumps >= 1  # should detect


def test_jump_intensity():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 600)
    intensity = jump_intensity(rets, window=200, alpha=0.01)
    assert 0 <= intensity <= 1


def test_split_continuous_jump():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 200)
    res = split_continuous_jump_variance(rets)
    assert "total_rv" in res
    assert "continuous_rv" in res
    assert "jump_rv" in res
    assert res["jump_share"] >= 0


def test_tick_rule():
    prices = pd.Series([100, 101, 100, 102, 102, 101])
    sign = tick_rule_classify(prices)
    assert sign.isin([-1, 1, 0]).all()


def test_lee_ready_with_quotes():
    prices = pd.Series([100.5, 101.5, 99.5])
    bid = pd.Series([100, 101, 99])
    ask = pd.Series([101, 102, 100])
    sign = lee_ready_classify(prices, bid, ask)
    # 100.5 vs mid=100.5 → tied → tick-rule
    assert len(sign) == 3


def test_order_flow_imbalance():
    prices = pd.Series([100, 101, 100, 102])
    volumes = pd.Series([1000, 2000, 1500, 1200])
    ofi = order_flow_imbalance(prices, volumes)
    assert len(ofi) == 4


def test_rolling_ofi():
    rng = np.random.default_rng(0)
    signed = pd.Series(rng.choice([-1, 1], size=200) * rng.integers(1000, 5000, 200))
    total = signed.abs()
    ratio = rolling_ofi_imbalance_ratio(signed, total, window=20)
    valid = ratio.dropna()
    assert ((valid >= -1) & (valid <= 1)).all()


def test_close_position_in_range():
    high = pd.Series([102, 103, 105])
    low = pd.Series([99, 100, 101])
    close = pd.Series([101.5, 102.5, 104])
    cpr = close_position_in_range(high, low, close)
    assert ((cpr >= 0) & (cpr <= 1)).all()


def test_on_balance_volume():
    close = pd.Series([100, 101, 100, 102])
    volume = pd.Series([1000, 2000, 1500, 1200])
    obv = on_balance_volume(close, volume)
    assert len(obv) == 4


def test_money_flow_index():
    rng = np.random.default_rng(0)
    n = 50
    high = pd.Series(100 + rng.uniform(0, 2, n))
    low = pd.Series(100 - rng.uniform(0, 2, n))
    close = pd.Series(100 + rng.normal(0, 0.5, n))
    volume = pd.Series(rng.integers(1000, 5000, n))
    mfi = money_flow_index(high, low, close, volume, window=14)
    valid = mfi.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_imbalance_composite():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "open": 100 + rng.normal(0, 1, n),
            "high": 102 + rng.normal(0, 0.5, n),
            "low": 98 + rng.normal(0, 0.5, n),
            "close": 100 + rng.normal(0, 1, n),
            "volume": rng.integers(1000, 10000, n),
        }
    )
    out = imbalance_composite(df)
    assert "cpr" in out.columns
    assert "vw_body" in out.columns


# ===== Advanced ML =====


def test_gp_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (50, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 50)
    k = rbf_kernel(length_scale=1.0, variance=1.0)
    fit = fit_gp(X, y, k, noise=0.01)
    X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
    mean, var = gp_predict(fit, X_test)
    assert mean.shape == (20,)
    assert (var > 0).all()


def test_gp_marginal_log_likelihood():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (50, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 50)
    fit = fit_gp(X, y, rbf_kernel(1.0, 1.0), noise=0.01)
    ll = gp_marginal_log_likelihood(fit)
    assert np.isfinite(ll)


def test_grid_search_hyperparams():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (50, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 50)
    best = grid_search_hyperparams(X, y, [0.5, 1.0, 2.0], [0.5, 1.0], [0.01, 0.1])
    assert "length_scale" in best


def test_matern_kernel():
    X1 = np.array([[0.0], [1.0]])
    X2 = np.array([[0.0], [1.0]])
    k = matern_kernel(1.0, 1.5, 1.0)
    K = k(X1, X2)
    assert K.shape == (2, 2)
    assert K[0, 0] > 0


def test_particle_filter_basic():
    rng = np.random.default_rng(0)

    def transition(x, t, rng):
        return 0.9 * x + 0.5 * rng.standard_normal()

    def likelihood(x, y, t):
        return -0.5 * (y - x) ** 2

    pf = ParticleFilter(n_particles=200, transition=transition, likelihood=likelihood)
    pf.initialize(lambda r: r.standard_normal(), rng)
    for t in range(50):
        pf.step(0.5 + 0.1 * rng.standard_normal(), t, rng)
    mean = pf.posterior_mean()
    assert np.isfinite(mean)


def test_stoch_vol_particle_filter():
    rng = np.random.default_rng(0)
    n = 200
    h_true = np.zeros(n)
    for t in range(1, n):
        h_true[t] = 0.95 * h_true[t - 1] + 0.2 * rng.standard_normal()
    returns = np.exp(h_true / 2) * rng.standard_normal(n)
    res = stoch_vol_particle_filter_example(returns, n_particles=100, seed=0)
    assert "posterior_h" in res
    assert len(res["posterior_h"]) == n
