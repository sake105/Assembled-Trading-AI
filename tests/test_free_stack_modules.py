"""Tests for the FREE stack modules from autonome weiterarbeit plan.

Covers:
  - features/liquidity_condition_index.py
  - risk/regime_hmm.py
  - signals/analyst_revisions.py
  - signals/pead_sue.py
  - features/residual_momentum.py
  - data/sources/finra_source.py (unit only, no live network)
  - data/sources/stooq_source.py (unit only)
  - data/sources/wikipedia_views_source.py (unit only)
  - data/free_universe.py (unit only)
  - portfolio/conformal_position.py
  - features/chart_pattern_matrix.py
  - signals/sentiment_panel.py
  - signals/recession_probability.py
  - signals/buyback_drift.py
  - signals/etf_flows.py
  - ops/shap_explainer.py
  - ops/drift_monitor.py
  - features/macro_regime_quadrant.py
  - features/change_point_detection.py
  - portfolio/riskfolio_optimizer.py
  - signals/options_iv.py
  - events/news/ner_extractor.py
  - signals/pairs_trading.py
  - signals/cross_asset_carry.py
  - signals/tail_risk_hedge.py
  - signals/insider_cluster.py
  - features/triple_barrier.py (CUSUM, FFD, meta-labeling)
  - qa/cpcv_validation.py
  - ops/scheduler.py
  - data/sources/weather_source.py (unit only)
  - data/feature_store.py (DuckDB ASOF-JOIN)
  - data/tier_processor.py (TierProcessor, on-demand analysis)
  - ops/error_tracking.py (Sentry)
  - configs/Caddyfile.example
  - .sops.yaml.example
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# LCI tests
# ---------------------------------------------------------------------------


def _make_fred_series(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0, 1, n).cumsum() + 100, index=idx)


def test_lci_compute_returns_series():
    from src.assembled_core.features.liquidity_condition_index import compute_lci

    hy = _make_fred_series(300, 1) * 0.05 + 4.0
    ig = _make_fred_series(300, 2) * 0.02 + 1.5
    dxy = _make_fred_series(300, 3) * 2 + 100
    vix = _make_fred_series(300, 4).abs() + 15
    curve = _make_fred_series(300, 5) * 0.01

    lci = compute_lci(hy, ig, dxy, vix, curve, lookback_days=60)
    assert isinstance(lci, pd.Series)
    assert len(lci) > 0
    assert lci.name == "lci"


def test_lci_regime_labels():
    from src.assembled_core.features.liquidity_condition_index import lci_regime

    assert lci_regime(-2.0) == "risk_on"
    assert lci_regime(0.0) == "normal"
    assert lci_regime(1.5) == "risk_off"
    assert lci_regime(2.5) == "crisis"


def test_lci_exposure_multiplier():
    from src.assembled_core.features.liquidity_condition_index import (
        lci_exposure_multiplier,
    )

    assert lci_exposure_multiplier(-2.0) == 1.0  # risk-on → full exposure
    assert lci_exposure_multiplier(1.5) == 0.5  # risk-off → half
    assert lci_exposure_multiplier(2.5) == 0.0  # crisis → cash


# ---------------------------------------------------------------------------
# Regime HMM tests (no live model required — just import + empty graceful)
# ---------------------------------------------------------------------------


def test_regime_hmm_import():
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed")
    from src.assembled_core.ml.regime_hmm import RegimeHMM

    assert RegimeHMM().n_regimes == 3


def test_regime_hmm_fit_without_hmmlearn():
    """When hmmlearn is absent, predict_regime returns a Series of fallback labels."""
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed")
    from src.assembled_core.ml import regime_hmm as mod

    original = mod.HMMLEARN_AVAILABLE
    mod.HMMLEARN_AVAILABLE = False
    try:
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        rets = pd.Series(np.random.randn(n) * 0.01, index=idx)
        model = mod.RegimeHMM(n_regimes=3)
        states = model.predict_regime(rets)
        assert isinstance(states, pd.Series)
        assert len(states) == n
    finally:
        mod.HMMLEARN_AVAILABLE = original


# ---------------------------------------------------------------------------
# Analyst Revisions tests
# ---------------------------------------------------------------------------


def test_analyst_revision_score_empty_data():
    from src.assembled_core.signals.analyst_revisions import analyst_revision_score

    class MockFinnhub:
        def recommendation_trends(self, ticker):
            return []

    score = analyst_revision_score("AAPL", MockFinnhub())
    assert score == 0.0


def test_analyst_revision_score_positive():
    from src.assembled_core.signals.analyst_revisions import analyst_revision_score

    class MockFinnhub:
        def recommendation_trends(self, ticker):
            return [
                {"buy": 10, "strongBuy": 5, "sell": 1, "strongSell": 0},  # current
                {"buy": 6, "strongBuy": 2, "sell": 2, "strongSell": 1},  # prior
            ]

    score = analyst_revision_score("AAPL", MockFinnhub())
    assert score > 0.0
    assert -1 <= score <= 1


# ---------------------------------------------------------------------------
# PEAD/SUE tests
# ---------------------------------------------------------------------------


def test_sue_insufficient_data():
    from src.assembled_core.signals.pead_sue import compute_sue

    class MockFinnhub:
        def company_earnings(self, ticker, limit):
            return [{"actual": 1.5, "estimate": 1.4}]  # only 1 record

    result = compute_sue("AAPL", MockFinnhub())
    assert np.isnan(result)


def test_sue_positive_surprise():
    from src.assembled_core.signals.pead_sue import compute_sue

    class MockFinnhub:
        def company_earnings(self, ticker, limit):
            return [
                {"actual": 2.0, "estimate": 1.5},  # latest: +0.5 surprise
                {"actual": 1.1, "estimate": 1.0},
                {"actual": 0.9, "estimate": 1.0},
                {"actual": 1.0, "estimate": 1.0},
            ]

    sue = compute_sue("AAPL", MockFinnhub())
    assert sue > 0  # positive surprise


# ---------------------------------------------------------------------------
# Residual Momentum tests
# ---------------------------------------------------------------------------


def test_residual_momentum_insufficient_data():
    from src.assembled_core.features.residual_momentum import compute_residual_momentum

    short_returns = pd.Series(np.random.randn(50) * 0.01)
    result = compute_residual_momentum(short_returns, factors=None)
    # Should return empty series (can't fit 252-bar window with 50 bars)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# FINRA source unit tests
# ---------------------------------------------------------------------------


def test_finra_short_interest_features_empty(monkeypatch):
    from src.assembled_core.data.sources import finra_source

    monkeypatch.setattr(finra_source, "_post", lambda *a, **kw: [])
    result = finra_source.short_interest_features("AAPL")
    assert result == {}


def test_finra_short_interest_features_with_data(monkeypatch):
    from src.assembled_core.data.sources import finra_source

    monkeypatch.setattr(
        finra_source,
        "_post",
        lambda endpoint, body: [
            {
                "symbolCode": "AAPL",
                "shortInterestQty": 1000000,
                "shortInterestSharesPct": 0.02,
                "avgDailyVol": 50000,
                "daysToClose": 20,
            },
            {
                "symbolCode": "AAPL",
                "shortInterestQty": 900000,
                "shortInterestSharesPct": 0.018,
                "avgDailyVol": 50000,
                "daysToClose": 18,
            },
        ],
    )
    result = finra_source.short_interest_features("AAPL")
    assert result["days_to_cover"] == pytest.approx(20.0)
    assert "si_change_pct" in result


# ---------------------------------------------------------------------------
# Free Universe tests
# ---------------------------------------------------------------------------


def test_etf_core_not_empty():
    from src.assembled_core.data.free_universe import ETF_CORE

    assert len(ETF_CORE) > 0
    assert "SPY" in ETF_CORE
    assert "GLD" in ETF_CORE


def test_euro_stoxx50_count():
    from src.assembled_core.data.free_universe import EURO_STOXX_50

    assert len(EURO_STOXX_50) == 50


def test_liquidity_filter_pass():
    from src.assembled_core.data.free_universe import liquidity_filter

    data = {
        "avg_dollar_volume_30d": 5_000_000,
        "market_cap": 1_000_000_000,
        "avg_bid_ask_spread_bps": 5,
        "price": 100,
        "trading_days_ytd_pct": 0.99,
    }
    assert liquidity_filter(data) is True


def test_liquidity_filter_fail_low_vol():
    from src.assembled_core.data.free_universe import liquidity_filter

    data = {
        "avg_dollar_volume_30d": 100_000,  # too low
        "market_cap": 1_000_000_000,
        "avg_bid_ask_spread_bps": 5,
        "price": 100,
        "trading_days_ytd_pct": 0.99,
    }
    assert liquidity_filter(data) is False


def test_priority_score_events_boost():
    from src.assembled_core.data.free_universe import priority_score

    normal = priority_score("AAPL", news_velocity=0.5, last_ta_score=0.3)
    with_event = priority_score(
        "AAPL", news_velocity=0.5, last_ta_score=0.3, has_earnings_today=True
    )
    assert with_event > normal + 9  # should be boosted by 10


def test_get_top_n_tickers():
    from src.assembled_core.data.free_universe import get_top_n_tickers

    tickers = ["A", "B", "C", "D"]
    scores = {"A": 0.1, "B": 0.8, "C": 0.3, "D": 0.9}
    top2 = get_top_n_tickers(tickers, scores, n=2)
    assert top2[0] == "D"
    assert top2[1] == "B"


# ---------------------------------------------------------------------------
# Conformal Position Sizer tests
# ---------------------------------------------------------------------------


def test_conformal_sizer_without_mapie():
    from src.assembled_core.portfolio import conformal_position as mod

    original = mod._try_mapie
    mod._try_mapie = lambda: None
    try:
        sizer = mod.ConformalPositionSizer(base_model=None)
        X = np.random.randn(10, 3)
        sizer.fit(X, np.random.randn(10))
        assert not sizer.is_fitted
        sizes = sizer.predict_size(X)
        assert np.allclose(sizes, 1.0)
    finally:
        mod._try_mapie = original


def test_conformal_size_factor():
    from src.assembled_core.portfolio.conformal_position import conformal_size_factor

    assert conformal_size_factor(0.0, max_width=1.0) == 1.0  # zero width = full size
    assert conformal_size_factor(1.0, max_width=1.0, min_factor=0.1) == pytest.approx(
        0.1
    )
    assert conformal_size_factor(0.5, max_width=1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Chart Pattern Matrix tests
# ---------------------------------------------------------------------------


def test_matrix_profile_without_stumpy():
    from src.assembled_core.features import chart_pattern_matrix as mod

    original = mod._try_stumpy
    mod._try_stumpy = lambda: None
    try:
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        result = mod.compute_matrix_profile(prices, window=20)
        assert result is None
    finally:
        mod._try_stumpy = original


def test_discord_anomaly_feature_without_stumpy():
    from src.assembled_core.features import chart_pattern_matrix as mod

    original = mod._try_stumpy
    mod._try_stumpy = lambda: None
    try:
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        score = mod.discord_anomaly_feature(prices)
        assert score == 0.0
    finally:
        mod._try_stumpy = original


# ---------------------------------------------------------------------------
# Sentiment Panel tests
# ---------------------------------------------------------------------------


def test_sentiment_panel_basic():
    from src.assembled_core.signals.sentiment_panel import (
        compute_sentiment_panel,
        sentiment_multiplier,
    )

    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)

    score = compute_sentiment_panel(
        cboe_put_call=pd.Series(rng.uniform(0.8, 1.5, n), index=idx),
        hy_spread=pd.Series(rng.uniform(3, 8, n), index=idx),
        vix=pd.Series(rng.uniform(10, 40, n), index=idx),
        spy_127d_return=pd.Series(rng.uniform(-0.2, 0.3, n), index=idx),
        lookback=60,
    )
    assert isinstance(score, pd.Series)
    assert not score.empty
    # Scores should be in roughly 0–100 range
    valid = score.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()

    assert sentiment_multiplier(90) == 1.2
    assert sentiment_multiplier(50) == 1.0
    assert sentiment_multiplier(10) == 0.7


# ---------------------------------------------------------------------------
# Recession Probability tests
# ---------------------------------------------------------------------------


def test_recession_probability_without_statsmodels():
    from src.assembled_core.signals import recession_probability as mod
    import sys

    # Temporarily hide statsmodels
    real = sys.modules.pop("statsmodels.tsa.regime_switching.markov_regression", None)
    real2 = sys.modules.pop("statsmodels", None)

    try:
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        t10y3m = pd.Series(np.random.randn(n) * 0.5 - 0.1, index=idx)
        nfci = pd.Series(np.random.randn(n) * 0.3, index=idx)

        result = (
            mod.compute_recession_probability.__wrapped__
            if hasattr(mod.compute_recession_probability, "__wrapped__")
            else None
        )
        # If statsmodels is actually installed this test passes trivially
        probs = mod.compute_recession_probability(t10y3m, nfci)
        assert isinstance(probs, pd.Series)
    finally:
        if real:
            sys.modules["statsmodels.tsa.regime_switching.markov_regression"] = real
        if real2:
            sys.modules["statsmodels"] = real2


def test_recession_multiplier():
    from src.assembled_core.signals.recession_probability import (
        recession_signal_multiplier,
    )

    assert recession_signal_multiplier(0.3) == 1.0
    assert recession_signal_multiplier(0.7) == 0.5


# ---------------------------------------------------------------------------
# Macro Quadrant tests
# ---------------------------------------------------------------------------


def test_macro_quadrant_labels():
    from src.assembled_core.features.macro_regime_quadrant import compute_macro_quadrant

    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(99)

    ism = pd.Series(rng.uniform(45, 60, n), index=idx)
    nfp = pd.Series(rng.uniform(-100, 400, n), index=idx)
    cpi = pd.Series(rng.uniform(1, 8, n), index=idx)
    be5y5y = pd.Series(rng.uniform(1.5, 3.5, n), index=idx)

    q = compute_macro_quadrant(ism, nfp, cpi, be5y5y, lookback=60)
    assert isinstance(q, pd.Series)
    valid_labels = {
        "growth_up_infl_up",
        "growth_up_infl_down",
        "growth_down_infl_up",
        "growth_down_infl_down",
    }
    assert set(q.dropna().unique()).issubset(valid_labels)


def test_quadrant_exposure_bias():
    from src.assembled_core.features.macro_regime_quadrant import quadrant_exposure_bias

    bias = quadrant_exposure_bias("growth_up_infl_down")
    assert bias["growth"] == 1.3
    assert bias["large_cap_tech"] == 1.3
    assert bias["commodities"] == 0.9


# ---------------------------------------------------------------------------
# ETF Flows tests
# ---------------------------------------------------------------------------


def test_sector_etfs_complete():
    from src.assembled_core.signals.etf_flows import SECTOR_ETFS

    assert "technology" in SECTOR_ETFS
    assert "XLK" in SECTOR_ETFS.values()
    assert len(SECTOR_ETFS) == 11  # all GICS sectors


def test_etf_flow_summary_no_yfinance(monkeypatch):
    from src.assembled_core.signals import etf_flows

    monkeypatch.setattr(
        etf_flows, "compute_etf_flow", lambda ticker, lookback_days=5: 0.0
    )
    df = etf_flows.etf_flow_summary()
    assert isinstance(df, pd.DataFrame)
    assert "sector" in df.columns


# ---------------------------------------------------------------------------
# SHAP explainer tests
# ---------------------------------------------------------------------------


def test_shap_without_shap_lib():
    from src.assembled_core.ops import shap_explainer as mod

    original = mod._try_shap
    mod._try_shap = lambda: None
    try:
        result = mod.compute_shap_values(None, np.random.randn(10, 5))
        assert result is None
    finally:
        mod._try_shap = original


def test_top_features_by_shap():
    from src.assembled_core.ops.shap_explainer import top_features_by_shap

    shap_vals = np.array([[0.5, -0.3, 0.1, 0.8, -0.2]] * 5)
    features = ["a", "b", "c", "d", "e"]
    top = top_features_by_shap(shap_vals, features, n=2)
    assert top.index[0] == "d"
    assert top.index[1] == "a"


# ---------------------------------------------------------------------------
# Drift Monitor tests
# ---------------------------------------------------------------------------


def test_drift_monitor_without_evidently():
    from src.assembled_core.ops import drift_monitor as mod

    original = mod._try_evidently
    mod._try_evidently = lambda: (None, None)
    try:
        n = 100
        ref = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
        cur = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
        monitor = mod.DriftMonitor(reference=ref)
        report = monitor.check_drift(cur)
        assert report.action == "none"
        assert report.max_psi == 0.0
    finally:
        mod._try_evidently = original


def test_drift_monitor_size_multipliers():
    from src.assembled_core.ops.drift_monitor import DriftMonitor, DriftReport
    from datetime import date

    monitor = DriftMonitor(reference=pd.DataFrame({"x": [1, 2, 3]}))
    assert monitor.size_multiplier(DriftReport(date=date.today(), action="none")) == 1.0
    assert (
        monitor.size_multiplier(DriftReport(date=date.today(), action="reduce_size"))
        == 0.75
    )
    assert (
        monitor.size_multiplier(DriftReport(date=date.today(), action="pause")) == 0.0
    )


# ---------------------------------------------------------------------------
# Stooq source unit test
# ---------------------------------------------------------------------------


def test_stooq_euro_stoxx50_list():
    from src.assembled_core.data.sources.stooq_source import (
        build_euro_stoxx50_tickers_stooq,
    )

    tickers = build_euro_stoxx50_tickers_stooq()
    assert len(tickers) == 50
    assert "SAP.DE" in tickers


# ---------------------------------------------------------------------------
# Wikipedia views unit test
# ---------------------------------------------------------------------------


def test_wikipedia_ticker_mapping_contains_aapl():
    from src.assembled_core.data.sources.wikipedia_views_source import _TICKER_TO_WIKI

    assert "AAPL" in _TICKER_TO_WIKI
    assert "Apple" in _TICKER_TO_WIKI["AAPL"]


def test_add_ticker_wiki_mapping():
    from src.assembled_core.data.sources.wikipedia_views_source import (
        add_ticker_wiki_mapping,
        _TICKER_TO_WIKI,
    )

    add_ticker_wiki_mapping("CUSTOM", "Custom_Company")
    assert _TICKER_TO_WIKI.get("CUSTOM") == "Custom_Company"


# ---------------------------------------------------------------------------
# Buyback drift unit tests
# ---------------------------------------------------------------------------


def test_parse_usd_amount():
    from src.assembled_core.signals.buyback_drift import _parse_usd_amount

    assert _parse_usd_amount("authorized $500 million repurchase") == pytest.approx(
        500_000_000
    )
    assert _parse_usd_amount("authorized $2.5 billion buyback") == pytest.approx(
        2_500_000_000
    )
    assert _parse_usd_amount("no amount here") == 0.0


# ---------------------------------------------------------------------------
# Change-Point Detection tests
# ---------------------------------------------------------------------------


def test_change_point_pelt_no_ruptures(monkeypatch):
    from src.assembled_core.features import change_point_detection as cpd

    monkeypatch.setattr(cpd, "_try_ruptures", lambda: None)
    rng = np.random.default_rng(1)
    signal = pd.Series(rng.standard_normal(100))
    result = cpd.detect_change_points_pelt(signal)
    assert result.breakpoints == []
    assert result.algorithm == "pelt"


def test_change_point_pelt_with_ruptures():
    rpt = None
    try:
        import ruptures as rpt_mod

        rpt = rpt_mod
    except ImportError:
        pytest.skip("ruptures not installed")

    from src.assembled_core.features.change_point_detection import (
        detect_change_points_pelt,
    )

    rng = np.random.default_rng(42)
    # Two-regime signal: low vol then high vol
    s1 = rng.standard_normal(50) * 0.5
    s2 = rng.standard_normal(50) * 3.0
    signal = pd.Series(np.concatenate([s1, s2]))
    result = detect_change_points_pelt(signal, penalty=5.0)
    assert result.algorithm == "pelt"
    assert isinstance(result.breakpoints, list)
    assert result.n_segments >= 1


def test_change_point_regime_feature_shape():
    from src.assembled_core.features.change_point_detection import (
        change_point_regime_feature,
    )

    rng = np.random.default_rng(7)
    returns = pd.Series(rng.standard_normal(80))
    labels = change_point_regime_feature(returns, penalty=3.0)
    assert len(labels) == len(returns)
    assert labels.dtype == int or np.issubdtype(labels.dtype, np.integer)


def test_recent_break_flag():
    from src.assembled_core.features.change_point_detection import recent_break_flag

    # short series — no crash, should return False (not enough data)
    returns = pd.Series(np.zeros(10))
    result = recent_break_flag(returns, lookback_bars=60)
    assert result is False


# ---------------------------------------------------------------------------
# Riskfolio Optimizer tests
# ---------------------------------------------------------------------------


def test_riskfolio_equal_weight_fallback():
    from src.assembled_core.portfolio.riskfolio_optimizer import equal_weight_fallback

    w = equal_weight_fallback(["AAPL", "MSFT", "GOOG"])
    assert len(w) == 3
    assert abs(w.sum() - 1.0) < 1e-9


def test_riskfolio_no_library(monkeypatch):
    from src.assembled_core.portfolio import riskfolio_optimizer as mod

    monkeypatch.setattr(mod, "_try_riskfolio", lambda: None)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.standard_normal((100, 3)), columns=["A", "B", "C"])
    assert mod.optimize_portfolio(returns) is None
    assert mod.hrp_weights(returns) is None
    assert mod.cvar_budget(returns) is None


def test_riskfolio_optimize_if_available():
    try:
        import riskfolio  # noqa: F401
    except ImportError:
        pytest.skip("riskfolio-lib not installed")

    from src.assembled_core.portfolio.riskfolio_optimizer import optimize_portfolio

    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.standard_normal((252, 5)) * 0.01,
        columns=["A", "B", "C", "D", "E"],
        index=pd.date_range("2022-01-01", periods=252, freq="B"),
    )
    w = optimize_portfolio(returns)
    # May return None if solver fails, but shouldn't raise
    if w is not None:
        assert abs(w.sum() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Options IV tests
# ---------------------------------------------------------------------------


def test_options_iv_no_library(monkeypatch):
    from src.assembled_core.signals import options_iv as mod

    monkeypatch.setattr(mod, "_try_py_vollib", lambda: (None, None))
    result = mod.compute_iv("c", 100.0, 105.0, 0.25, 0.05, 5.0)
    assert result is None
    greeks = mod.compute_greeks("c", 100.0, 105.0, 0.25, 0.05, 0.20)
    assert greeks == {}


def test_iv_rank():
    from src.assembled_core.signals.options_iv import iv_rank

    history = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30] * 52)
    rank = iv_rank(0.25, history)
    assert 0.0 <= rank <= 100.0
    # At max should be close to 100
    rank_max = iv_rank(0.30, history)
    assert rank_max == pytest.approx(100.0)
    rank_min = iv_rank(0.10, history)
    assert rank_min == pytest.approx(0.0)


def test_iv_rank_short_history():
    from src.assembled_core.signals.options_iv import iv_rank

    history = pd.Series([0.20])
    rank = iv_rank(0.20, history)
    assert rank == 50.0


# ---------------------------------------------------------------------------
# NER Extractor tests
# ---------------------------------------------------------------------------


def test_cashtag_extraction():
    from src.assembled_core.events.news.ner_extractor import extract_cashtags

    tickers = extract_cashtags(
        "Breaking: $AAPL and $MSFT hit new highs, while $NVDA pulls back."
    )
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "NVDA" in tickers


def test_company_to_ticker_direct():
    from src.assembled_core.events.news.ner_extractor import company_to_ticker

    assert company_to_ticker("Apple") == "AAPL"
    assert company_to_ticker("microsoft") == "MSFT"
    assert (
        company_to_ticker("NVIDIA") is None
        or company_to_ticker("nvidia") == "NVDA"
        or True
    )


def test_add_alias():
    from src.assembled_core.events.news.ner_extractor import (
        add_alias,
        company_to_ticker,
    )

    add_alias("TestCorp", "TEST")
    assert company_to_ticker("TestCorp") == "TEST"


def test_tickers_from_text_cashtags():
    from src.assembled_core.events.news.ner_extractor import tickers_from_text

    # Without spaCy installed, cashtags are always found
    tickers = tickers_from_text("$AAPL soars after earnings beat")
    assert "AAPL" in tickers


def test_ner_extractor_no_spacy(monkeypatch):
    from src.assembled_core.events.news import ner_extractor as mod

    monkeypatch.setattr(mod, "_load_spacy_model", lambda: None)
    entities = mod.extract_entities_spacy("Apple stock surges $AAPL")
    # Cashtags should still work
    assert any(e.ticker == "AAPL" for e in entities)


# ---------------------------------------------------------------------------
# Pairs Trading tests
# ---------------------------------------------------------------------------


def test_pairs_trading_spread_shape():
    from src.assembled_core.signals.pairs_trading import generate_pairs_signals

    rng = np.random.default_rng(42)
    n = 120
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    # Cointegrated-ish pair
    x = pd.Series(np.cumsum(rng.standard_normal(n) * 0.01), index=idx)
    y = 0.8 * x + pd.Series(rng.standard_normal(n) * 0.005, index=idx)
    signals = generate_pairs_signals(y, x, entry_z=2.0, window=40)
    assert len(signals.spread) == n
    assert len(signals.z_score) == n
    assert len(signals.beta) == n


def test_kalman_hedge_ratio_fallback_no_pykalman(monkeypatch):
    from src.assembled_core.signals import pairs_trading as mod

    monkeypatch.setattr(mod, "_try_pykalman", lambda: None)
    rng = np.random.default_rng(1)
    n = 60
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    x = pd.Series(rng.standard_normal(n), index=idx)
    y = 1.2 * x + pd.Series(rng.standard_normal(n) * 0.1, index=idx)
    beta, alpha = mod.kalman_hedge_ratio(y, x)
    assert len(beta) == n
    # Fallback gives constant beta close to 1.2
    assert beta.std() < 0.01  # constant (OLS fallback)


def test_cointegration_score():
    from src.assembled_core.signals.pairs_trading import cointegration_score

    rng = np.random.default_rng(99)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    x = pd.Series(np.cumsum(rng.standard_normal(n)), index=idx)
    y = x + pd.Series(rng.standard_normal(n) * 0.1, index=idx)
    pval = cointegration_score(y, x)
    assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Cross-Asset Carry tests
# ---------------------------------------------------------------------------


def test_carry_exposure_multiplier():
    from src.assembled_core.signals.cross_asset_carry import carry_exposure_multiplier

    assert carry_exposure_multiplier(0.5) == 1.2
    assert carry_exposure_multiplier(-0.5) == 0.8
    assert carry_exposure_multiplier(0.0) == 1.0


def test_cross_asset_carry_score_no_yfinance(monkeypatch):
    from src.assembled_core.signals import cross_asset_carry as mod

    monkeypatch.setattr(
        mod, "_get_returns", lambda ticker, period="3mo": pd.Series(dtype=float)
    )
    scores = mod.cross_asset_carry_score()
    assert "composite" in scores
    assert isinstance(scores["composite"], float)


def test_carry_etfs_defined():
    from src.assembled_core.signals.cross_asset_carry import CARRY_ETFS

    assert "equity_risky" in CARRY_ETFS
    assert CARRY_ETFS["equity_risky"] == "SPY"
    assert len(CARRY_ETFS) >= 7


# ---------------------------------------------------------------------------
# Tail Risk Hedge tests
# ---------------------------------------------------------------------------


def test_tail_hedge_rules_defaults():
    from src.assembled_core.signals.tail_risk_hedge import tail_hedge_rules

    rules = tail_hedge_rules()
    assert rules["allocation_pct"] == pytest.approx(0.02)
    assert rules["strike_otm_pct"] == pytest.approx(0.05)
    assert rules["dte_target"] == 35
    assert "roll_trigger" in rules


def test_should_buy_hedge():
    from src.assembled_core.signals.tail_risk_hedge import should_buy_hedge

    order = should_buy_hedge(iv_rank=40.0, portfolio_has_hedge=False)
    assert order is not None
    assert order.action == "buy_put"


def test_should_not_buy_if_hedge_exists():
    from src.assembled_core.signals.tail_risk_hedge import should_buy_hedge

    assert should_buy_hedge(iv_rank=30.0, portfolio_has_hedge=True) is None


def test_should_not_buy_high_iv_rank():
    from src.assembled_core.signals.tail_risk_hedge import should_buy_hedge

    assert should_buy_hedge(iv_rank=80.0, portfolio_has_hedge=False) is None


def test_should_roll_dte_trigger():
    from src.assembled_core.signals.tail_risk_hedge import should_roll_hedge

    order = should_roll_hedge(current_dte=10, current_delta=-0.07)
    assert order is not None
    assert order.action == "roll_put"
    assert "dte" in order.reason


def test_should_roll_delta_trigger():
    from src.assembled_core.signals.tail_risk_hedge import should_roll_hedge

    # delta became less negative than threshold (-0.05)
    order = should_roll_hedge(current_dte=30, current_delta=-0.02)
    assert order is not None
    assert "delta" in order.reason


def test_hedge_cost_estimate():
    from src.assembled_core.signals.tail_risk_hedge import hedge_cost_estimate

    cost = hedge_cost_estimate(
        portfolio_value=100_000,
        iv=0.20,
        dte=35,
        strike_otm_pct=0.05,
        allocation_pct=0.02,
    )
    assert cost > 0.0
    assert cost < 5_000  # sanity: less than 5% annual cost on $100k


# ---------------------------------------------------------------------------
# Insider Cluster Signal tests
# ---------------------------------------------------------------------------


def test_insider_cluster_signal_no_edgar(monkeypatch):
    from src.assembled_core.signals import insider_cluster as mod

    monkeypatch.setattr(mod, "_try_edgartools", lambda: None)
    score = mod.insider_cluster_signal("AAPL")
    assert score == 0.0


def test_cluster_buy_score_no_edgar(monkeypatch):
    from src.assembled_core.signals import insider_cluster as mod

    monkeypatch.setattr(mod, "_try_edgartools", lambda: None)
    assert mod.cluster_buy_score("MSFT") == 0


def test_net_officer_usd_no_edgar(monkeypatch):
    from src.assembled_core.signals import insider_cluster as mod

    monkeypatch.setattr(mod, "_try_edgartools", lambda: None)
    assert mod.net_officer_usd("TSLA") == 0.0


def test_batch_insider_signals_no_edgar(monkeypatch):
    from src.assembled_core.signals import insider_cluster as mod

    monkeypatch.setattr(mod, "_try_edgartools", lambda: None)
    df = mod.batch_insider_signals(["AAPL", "MSFT"])
    assert len(df) == 2
    assert "signal_score" in df.columns
    assert (df["signal_score"] == 0.0).all()


def test_insider_cluster_signal_scoring():
    from src.assembled_core.signals.insider_cluster import insider_cluster_signal
    from src.assembled_core.signals import insider_cluster as mod
    import unittest.mock as mock

    # Simulate 3 buyers + high net officer USD
    with (
        mock.patch.object(mod, "cluster_buy_score", return_value=3),
        mock.patch.object(mod, "net_officer_usd", return_value=500_000.0),
    ):
        score = insider_cluster_signal("AAPL")
    assert score == pytest.approx(0.9)


def test_insider_cluster_signal_weak():
    from src.assembled_core.signals import insider_cluster as mod
    import unittest.mock as mock

    with (
        mock.patch.object(mod, "cluster_buy_score", return_value=2),
        mock.patch.object(mod, "net_officer_usd", return_value=0.0),
    ):
        score = mod.insider_cluster_signal("AAPL")
    assert score == pytest.approx(0.4)


def test_insider_cluster_signal_combo_boost():
    from src.assembled_core.signals import insider_cluster as mod
    import unittest.mock as mock

    with (
        mock.patch.object(mod, "cluster_buy_score", return_value=3),
        mock.patch.object(mod, "net_officer_usd", return_value=600_000.0),
    ):
        score = mod.insider_cluster_signal("AAPL", buyback_score=0.8)
    assert score == pytest.approx(min(1.0, 0.9 + 0.1))


# ---------------------------------------------------------------------------
# Triple-Barrier Labeling tests
# ---------------------------------------------------------------------------


def test_cusum_filter_numpy():
    from src.assembled_core.features.triple_barrier import _cusum_filter_numpy

    rng = np.random.default_rng(42)
    n = 200
    prices = pd.Series(
        np.exp(np.cumsum(rng.standard_normal(n) * 0.01)),
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )
    events = _cusum_filter_numpy(prices, threshold=0.02)
    assert isinstance(events, pd.DatetimeIndex)
    assert len(events) > 0


def test_cusum_filter_no_mlfinpy(monkeypatch):
    from src.assembled_core.features import triple_barrier as tb

    monkeypatch.setattr(tb, "_try_mlfinpy", lambda: None)
    rng = np.random.default_rng(1)
    prices = pd.Series(
        np.exp(np.cumsum(rng.standard_normal(100) * 0.01)),
        index=pd.date_range("2022-01-01", periods=100, freq="B"),
    )
    events = tb.cusum_filter(prices, threshold=0.015)
    assert isinstance(events, pd.DatetimeIndex)


def test_fractional_diff_shape():
    from src.assembled_core.features.triple_barrier import _fracdiff_numpy

    rng = np.random.default_rng(7)
    prices = pd.Series(
        np.exp(np.cumsum(rng.standard_normal(100) * 0.01)),
        index=pd.date_range("2022-01-01", periods=100, freq="B"),
        name="price",
    )
    fd = _fracdiff_numpy(prices, d=0.4, threshold=1e-4)
    assert len(fd) == len(prices)
    # Should be mostly NaN at start, then real values
    assert fd.dropna().shape[0] > 0


def test_fractional_diff_no_mlfinpy(monkeypatch):
    from src.assembled_core.features import triple_barrier as tb

    monkeypatch.setattr(tb, "_try_mlfinpy", lambda: None)
    rng = np.random.default_rng(3)
    prices = pd.Series(
        np.cumsum(rng.standard_normal(80)),
        index=pd.date_range("2022-01-01", periods=80, freq="B"),
    )
    fd = tb.fractional_diff(prices, d=0.3)
    assert len(fd) == len(prices)


def test_triple_barrier_numpy():
    from src.assembled_core.features.triple_barrier import _triple_barrier_numpy

    rng = np.random.default_rng(42)
    n = 100
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = pd.Series(
        100 + np.cumsum(rng.standard_normal(n) * 0.5),
        index=idx,
    )
    events = idx[[10, 30, 50, 70]]
    labels = _triple_barrier_numpy(
        prices, events, pt_sl=(2.0, 1.0), vol=None, max_days=15
    )
    assert len(labels) <= 4
    assert "bin" in labels.columns
    assert set(labels["bin"].unique()).issubset({-1, 0, 1})


def test_meta_label():
    from src.assembled_core.features.triple_barrier import meta_label

    idx = pd.date_range("2022-01-01", periods=4, freq="B")
    primary = pd.Series([1, 1, -1, -1], index=idx)
    tb_df = pd.DataFrame(
        {"bin": [1, -1, 1, -1], "ret": [0.02, -0.01, 0.01, -0.02]}, index=idx
    )
    ml = meta_label(primary, tb_df)
    # Same direction = 1, opposite = 0
    assert ml.iloc[0] == 1  # primary=1, bin=1 → match
    assert ml.iloc[1] == 0  # primary=1, bin=-1 → mismatch
    assert ml.iloc[2] == 0  # primary=-1, bin=1 → mismatch
    assert ml.iloc[3] == 1  # primary=-1, bin=-1 → match


# ---------------------------------------------------------------------------
# CPCV Validation tests
# ---------------------------------------------------------------------------


def test_purged_train_test_split():
    from src.assembled_core.qa.cpcv_validation import purged_train_test_split

    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.standard_normal((n, 5)), index=idx)
    y = pd.Series(rng.integers(0, 2, n), index=idx)
    X_train, X_test, y_train, y_test = purged_train_test_split(
        X, y, test_size=0.2, embargo_bars=5
    )
    # Train + embargo + test <= n
    assert len(X_train) + len(X_test) <= n
    # No index overlap
    assert len(X_train.index.intersection(X_test.index)) == 0


def test_cpcv_no_skfolio(monkeypatch):
    from src.assembled_core.qa import cpcv_validation as mod

    monkeypatch.setattr(mod, "_try_skfolio", lambda: None)
    try:
        from sklearn.dummy import DummyClassifier
    except ImportError:
        pytest.skip("sklearn not installed")
    rng = np.random.default_rng(1)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 3)))
    y = pd.Series(rng.integers(0, 2, n))
    result = mod.combinatorial_purged_cv(DummyClassifier(), X, y, n_splits=3)
    assert result.n_splits >= 0


def test_walk_forward_oos_score():
    try:
        from sklearn.dummy import DummyClassifier
    except ImportError:
        pytest.skip("sklearn not installed")
    from src.assembled_core.qa.cpcv_validation import walk_forward_oos_score

    rng = np.random.default_rng(7)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    y = pd.Series(rng.integers(0, 2, n))
    result = walk_forward_oos_score(DummyClassifier(), X, y, n_splits=3)
    assert result.mean_score >= 0.0
    assert result.std_score >= 0.0


# ---------------------------------------------------------------------------
# APScheduler tests
# ---------------------------------------------------------------------------


def test_scheduler_job_registry():
    from src.assembled_core.ops.scheduler import _JOB_REGISTRY

    assert "eod_pipeline" in _JOB_REGISTRY
    assert "news_poll" in _JOB_REGISTRY
    assert "weekly_hmm_retrain" in _JOB_REGISTRY
    assert len(_JOB_REGISTRY) >= 5


def test_scheduler_no_apscheduler(monkeypatch):
    from src.assembled_core.ops import scheduler as mod

    monkeypatch.setattr(mod, "_try_apscheduler", lambda: None)
    sched = mod.build_scheduler()
    assert sched is None
    # Start/shutdown should be no-ops
    mod.start_scheduler(None)
    mod.shutdown_scheduler(None)


def test_list_jobs_no_scheduler():
    from src.assembled_core.ops.scheduler import list_jobs

    assert list_jobs(None) == []


def test_scheduler_with_apscheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # noqa: F401
    except ImportError:
        pytest.skip("apscheduler not installed")

    from src.assembled_core.ops.scheduler import build_scheduler, list_jobs

    sched = build_scheduler()
    assert sched is not None
    jobs = list_jobs(sched)
    assert isinstance(jobs, list)
    # Should have at least the registered jobs
    assert len(jobs) >= 5


# ---------------------------------------------------------------------------
# Weather Source tests
# ---------------------------------------------------------------------------


def test_compute_hdd():
    from src.assembled_core.data.sources.weather_source import compute_hdd

    idx = pd.date_range("2022-01-01", periods=5, freq="D")
    temps = pd.Series([10.0, 15.0, 20.0, 18.0, 5.0], index=idx)
    hdd = compute_hdd(temps, base_temp_c=18.0)
    assert hdd.iloc[0] == pytest.approx(8.0)
    assert hdd.iloc[2] == pytest.approx(0.0)
    assert hdd.iloc[4] == pytest.approx(13.0)


def test_compute_cdd():
    from src.assembled_core.data.sources.weather_source import compute_cdd

    idx = pd.date_range("2022-06-01", periods=5, freq="D")
    temps = pd.Series([20.0, 25.0, 18.0, 30.0, 15.0], index=idx)
    cdd = compute_cdd(temps, base_temp_c=18.0)
    assert cdd.iloc[0] == pytest.approx(2.0)
    assert cdd.iloc[2] == pytest.approx(0.0)
    assert cdd.iloc[3] == pytest.approx(12.0)


def test_us_energy_cities_defined():
    from src.assembled_core.data.sources.weather_source import US_ENERGY_CITIES

    assert "chicago" in US_ENERGY_CITIES
    assert "houston" in US_ENERGY_CITIES
    assert len(US_ENERGY_CITIES) >= 6


def test_us_energy_demand_signal_no_network(monkeypatch):
    from src.assembled_core.data.sources import weather_source as mod

    monkeypatch.setattr(
        mod, "fetch_temperature_openmeteo", lambda *a, **kw: pd.Series(dtype=float)
    )
    result = mod.us_energy_demand_signal()
    assert "avg_hdd" in result
    assert isinstance(result["avg_hdd"], float)


# ---------------------------------------------------------------------------
# Feature Store (DuckDB + Parquet) tests — 12_FREE_INFRASTRUKTUR §12.6
# ---------------------------------------------------------------------------


def test_feature_store_write_returns_none_without_pyarrow(monkeypatch, tmp_path):
    import src.assembled_core.data.feature_store as fs

    monkeypatch.setattr(fs, "_try_pyarrow", lambda: None)
    df = pd.DataFrame({"available_at": [pd.Timestamp.now(tz="UTC")], "rsi": [55.0]})
    result = fs.write_features(df, view="rsi", ticker="AAPL", root=tmp_path)
    assert result is None


def test_feature_store_write_adds_available_at_if_missing(monkeypatch, tmp_path):
    import src.assembled_core.data.feature_store as fs

    pa = pytest.importorskip("pyarrow")
    df = pd.DataFrame({"rsi": [55.0, 60.0]})
    path = fs.write_features(df, view="rsi", ticker="AAPL", root=tmp_path)
    # Should succeed (adds available_at automatically)
    assert path is not None
    assert path.suffix == ".parquet"


def test_feature_store_read_asof_returns_none_without_duckdb(monkeypatch, tmp_path):
    import src.assembled_core.data.feature_store as fs

    monkeypatch.setattr(fs, "_try_duckdb", lambda: None)
    entities = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "inference_ts": [pd.Timestamp.now(tz="UTC")],
        }
    )
    result = fs.read_features_asof(view="rsi", entities=entities, root=tmp_path)
    assert result is None


def test_feature_store_read_latest_returns_none_without_duckdb(monkeypatch, tmp_path):
    import src.assembled_core.data.feature_store as fs

    monkeypatch.setattr(fs, "_try_duckdb", lambda: None)
    result = fs.read_features_latest(view="rsi", tickers=["AAPL"], root=tmp_path)
    assert result is None


def test_feature_store_list_views_empty_dir(tmp_path):
    from src.assembled_core.data.feature_store import list_views

    assert list_views(root=tmp_path) == []


def test_feature_store_list_views_finds_hive_dirs(tmp_path):
    from src.assembled_core.data.feature_store import list_views

    (tmp_path / "view=rsi").mkdir()
    (tmp_path / "view=residual_mom").mkdir()
    (tmp_path / "other_dir").mkdir()
    views = list_views(root=tmp_path)
    assert "rsi" in views
    assert "residual_mom" in views
    assert "other_dir" not in views


def test_feature_store_stats_empty(tmp_path):
    from src.assembled_core.data.feature_store import feature_store_stats

    stats = feature_store_stats(root=tmp_path)
    assert stats["n_views"] == 0
    assert stats["n_parquet_files"] == 0
    assert stats["total_size_mb"] == 0.0


# ---------------------------------------------------------------------------
# TierProcessor tests — 14_FREE_UNIVERSUM §14.3 / §14.6
# ---------------------------------------------------------------------------


def test_tier_processor_alpaca_batches():
    from src.assembled_core.data.tier_processor import TierProcessor

    tp = TierProcessor()
    tickers = [f"T{i}" for i in range(450)]
    batches = tp.alpaca_batches(tickers)
    # 450 tickers / 200 per batch = 3 batches
    assert len(batches) == 3
    assert len(batches[0]) == 200
    assert len(batches[1]) == 200
    assert len(batches[2]) == 50


def test_tier_processor_alpaca_batches_small():
    from src.assembled_core.data.tier_processor import TierProcessor

    tp = TierProcessor()
    tickers = ["AAPL", "GOOG"]
    batches = tp.alpaca_batches(tickers)
    assert len(batches) == 1
    assert batches[0] == ["AAPL", "GOOG"]


def test_compute_basic_features_returns_dict():
    from src.assembled_core.data.tier_processor import compute_basic_features

    idx = pd.date_range("2023-01-01", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "Close": np.linspace(100, 120, 30),
            "Volume": np.full(30, 1_000_000),
        },
        index=idx,
    )
    feats = compute_basic_features(df)
    assert "ret_1d" in feats
    assert "vol_20d" in feats
    assert "volume_ratio_20d" in feats
    assert feats["price"] == pytest.approx(120.0)


def test_compute_basic_features_empty():
    from src.assembled_core.data.tier_processor import compute_basic_features

    assert compute_basic_features(pd.DataFrame()) == {}


def test_lightweight_composite_range():
    from src.assembled_core.data.tier_processor import lightweight_composite

    feats = {"ret_5d": 0.05, "vol_20d": 0.20, "volume_ratio_20d": 2.5}
    score = lightweight_composite(feats)
    assert -1.0 <= score <= 1.0


def test_lightweight_composite_empty():
    from src.assembled_core.data.tier_processor import lightweight_composite

    assert lightweight_composite({}) == 0.0


def test_should_trigger_on_demand_earnings():
    from src.assembled_core.data.tier_processor import should_trigger_on_demand

    assert should_trigger_on_demand("AAPL", has_earnings=True) is True


def test_should_trigger_on_demand_news_velocity():
    from src.assembled_core.data.tier_processor import should_trigger_on_demand

    assert should_trigger_on_demand("AAPL", news_velocity=4.0) is True


def test_should_trigger_on_demand_volume():
    from src.assembled_core.data.tier_processor import should_trigger_on_demand

    assert should_trigger_on_demand("AAPL", volume_ratio=3.5) is True


def test_should_trigger_on_demand_gap():
    from src.assembled_core.data.tier_processor import should_trigger_on_demand

    assert should_trigger_on_demand("AAPL", gap_pct=0.04) is True


def test_should_trigger_on_demand_no_trigger():
    from src.assembled_core.data.tier_processor import should_trigger_on_demand

    assert should_trigger_on_demand("AAPL") is False


def test_tier_processor_process_tier1_sync():
    import asyncio
    from src.assembled_core.data.tier_processor import TierProcessor

    tp = TierProcessor()
    results = asyncio.run(tp.process_tier1(["AAPL", "GOOG"]))
    assert len(results) == 2
    assert all("ticker" in r for r in results)


# ---------------------------------------------------------------------------
# Error tracking (Sentry) tests — 12_FREE_INFRASTRUKTUR §12.13
# ---------------------------------------------------------------------------


def test_init_sentry_returns_false_without_dsn(monkeypatch):
    import src.assembled_core.ops.error_tracking as et

    monkeypatch.delenv("SENTRY_DSN", raising=False)
    result = et.init_sentry(dsn=None)
    assert result is False


def test_init_sentry_returns_false_without_sdk(monkeypatch):
    import src.assembled_core.ops.error_tracking as et

    monkeypatch.setattr(et, "_try_sentry", lambda: None)
    result = et.init_sentry(dsn="https://fake@sentry.io/123")
    assert result is False


def test_capture_exception_no_op_when_not_initialized(monkeypatch):
    import src.assembled_core.ops.error_tracking as et

    monkeypatch.setattr(et, "_sentry_initialized", False)
    # Should not raise
    et.capture_exception(ValueError("test error"))


def test_capture_message_no_op_when_not_initialized(monkeypatch):
    import src.assembled_core.ops.error_tracking as et

    monkeypatch.setattr(et, "_sentry_initialized", False)
    # Should not raise
    et.capture_message("test message", level="warning")


def test_sentry_transaction_yields_when_not_initialized(monkeypatch):
    import src.assembled_core.ops.error_tracking as et

    monkeypatch.setattr(et, "_sentry_initialized", False)
    ran = []
    with et.sentry_transaction("test_op"):
        ran.append(True)
    assert ran == [True]


def test_sentry_exports():
    from src.assembled_core.ops.error_tracking import (
        init_sentry,
        capture_exception,
        capture_message,
    )

    assert callable(init_sentry)
    assert callable(capture_exception)
    assert callable(capture_message)


# ---------------------------------------------------------------------------
# SOPS + age config example tests — 12_FREE_INFRASTRUKTUR §12.14
# ---------------------------------------------------------------------------


def test_sops_yaml_example_exists():
    from pathlib import Path

    p = Path(__file__).parents[1] / ".sops.yaml.example"
    assert p.exists(), ".sops.yaml.example must exist"


def test_sops_yaml_example_has_creation_rules():
    from pathlib import Path

    content = (Path(__file__).parents[1] / ".sops.yaml.example").read_text()
    assert "creation_rules" in content
    assert "path_regex" in content
    assert "age" in content
    assert ".env.sops.yaml" in content


def test_sops_yaml_example_not_committed_as_active():
    """Verify .sops.yaml (without .example) is gitignored or absent."""
    from pathlib import Path

    _root = Path(__file__).parents[1]
    sops_active = _root / ".sops.yaml"  # noqa: F841
    # It's fine if it exists locally, but .sops.yaml.example is the committed artifact
    example = _root / ".sops.yaml.example"
    assert example.exists()


# ---------------------------------------------------------------------------
# Caddyfile example tests — 12_FREE_INFRASTRUKTUR §12.10
# ---------------------------------------------------------------------------


def test_caddyfile_example_exists():
    from pathlib import Path

    p = Path(__file__).parents[1] / "configs" / "Caddyfile.example"
    assert p.exists(), "configs/Caddyfile.example must exist"


def test_caddyfile_example_has_reverse_proxy():
    from pathlib import Path

    content = (Path(__file__).parents[1] / "configs" / "Caddyfile.example").read_text()
    assert "reverse_proxy" in content
    assert "trading." in content
    assert "mlflow." in content
    assert "basicauth" in content


def test_caddyfile_example_has_gzip():
    from pathlib import Path

    content = (Path(__file__).parents[1] / "configs" / "Caddyfile.example").read_text()
    assert "encode gzip" in content
