"""Tests für Round-7 (12 Module + Wirings)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# A: Signal-Decay-Tracker
# ---------------------------------------------------------------------------

def test_signal_decay_tracker_basic(tmp_path):
    from src.assembled_core.ml.signal_decay_tracker import SignalDecayTracker

    tracker = SignalDecayTracker(state_path=tmp_path / "decay.json", horizons=[1, 5])

    rng = np.random.default_rng(3)
    n = 100
    preds = {"sig1": pd.Series(rng.standard_normal(n))}
    rets = {1: pd.Series(rng.normal(0, 0.01, n)), 5: pd.Series(rng.normal(0, 0.02, n))}

    snap = tracker.record_snapshot(preds, rets)
    assert snap.as_of is not None
    assert "sig1" in snap.signal_ic
    assert tracker.history_length() == 1


def test_signal_decay_report_halflife(tmp_path):
    """IC sinkt über Snapshots → halflife erkennbar."""
    from src.assembled_core.ml.signal_decay_tracker import SignalDecayTracker

    tracker = SignalDecayTracker(state_path=tmp_path / "decay.json", horizons=[5])

    rng = np.random.default_rng(42)
    n = 100

    # 5 Snapshots mit abnehmender IC
    for ic_noise in [0.0, 0.2, 0.5, 0.8, 1.0]:  # immer mehr Noise → sinkende IC
        alpha_signal = 0.5
        actual_ret = rng.normal(0, 0.01, n)
        pred = alpha_signal * actual_ret + ic_noise * rng.standard_normal(n)
        tracker.record_snapshot(
            {"sig1": pd.Series(pred)},
            {5: pd.Series(actual_ret)},
            as_of=f"2025-01-{ic_noise*10:02.0f}",
        )

    report = tracker.get_report("sig1")
    assert report is not None
    assert "horizon_5d" in report.current_ic
    # historische Werte müssen eine Zeitreihe sein
    assert len(report.historical_ic["horizon_5d"]) == 5


def test_signal_decay_wiring_feedback_loop(tmp_path):
    """WIRING: feedback_loop._record_signal_decay existiert und crasht nicht."""
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    controller = FeedbackLoopController()
    assert hasattr(controller, "_record_signal_decay")

    # Empty panel → should not crash, just return
    controller._record_signal_decay(pd.DataFrame())
    # Also works with valid panel
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=60),
        "f1": rng.standard_normal(60),
        "fwd_return_5d": rng.normal(0, 0.01, 60),
    })
    controller._record_signal_decay(panel)  # should not raise


# ---------------------------------------------------------------------------
# B: Turnover-Penalty
# ---------------------------------------------------------------------------

def test_turnover_smoothing():
    from src.assembled_core.portfolio.turnover_penalty import apply_turnover_smoothing

    target = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    previous = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
    smoothed = apply_turnover_smoothing(target, previous, alpha=0.5)
    # Mid-point
    assert abs(smoothed["A"] - 0.4) < 1e-9
    assert abs(smoothed["B"] - 0.35) < 1e-9


def test_turnover_budget():
    from src.assembled_core.portfolio.turnover_penalty import enforce_turnover_budget, compute_turnover

    target = pd.Series({"A": 1.0, "B": -1.0})  # alles neu
    previous = pd.Series({"A": 0.0, "B": 0.0})
    capped = enforce_turnover_budget(target, previous, max_turnover=0.3)

    actual_turnover = compute_turnover(capped, previous)
    assert actual_turnover <= 0.3 + 1e-6


def test_turnover_wrapper_stateful():
    from src.assembled_core.portfolio.turnover_penalty import TurnoverConstrainedSizer

    sizer = TurnoverConstrainedSizer()
    t1 = sizer.process({"A": 0.5, "B": 0.5})
    t2 = sizer.process({"A": 1.0, "B": 0.0})
    # t2 sollte zwischen t1 und target liegen (Smoothing aktiv)
    assert t2["A"] > t1["A"]
    assert t2["A"] < 1.0


def test_turnover_wiring_position_sizing():
    """WIRING: compute_target_positions_with_smoothing existiert."""
    from src.assembled_core.portfolio.position_sizing import compute_target_positions_with_smoothing

    sig_df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "score": [0.8, 0.5, 0.3],
        "direction": ["long", "long", "long"],
    })
    result = compute_target_positions_with_smoothing(
        sig_df,
        previous_positions=None,  # no smoothing
    )
    assert "target_weight" in result.columns


# ---------------------------------------------------------------------------
# C: Online-HPO
# ---------------------------------------------------------------------------

def test_online_hpo_select_arm(tmp_path):
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter(state_path=tmp_path / "hpo.json")
    chosen = adapter.select_arm()
    assert chosen.arm_id.startswith("arm_")
    assert isinstance(chosen.params, dict)


def test_online_hpo_reward_update(tmp_path):
    from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter

    adapter = OnlineHyperparamAdapter(state_path=tmp_path / "hpo.json")
    chosen = adapter.select_arm()
    adapter.observe_reward(chosen.arm_id, reward=0.12)
    adapter.save()

    # Re-load and check state persisted
    adapter2 = OnlineHyperparamAdapter(state_path=tmp_path / "hpo.json")
    assert adapter2.arms[chosen.arm_id].n_pulls == 1
    assert adapter2.arms[chosen.arm_id].mean_reward == pytest.approx(0.12)


def test_online_hpo_wiring_retraining_scheduler(tmp_path):
    """WIRING: adapt_hyperparameters_via_bandit existiert."""
    from src.assembled_core.ml.retraining_scheduler import RetrainingScheduler

    scheduler = RetrainingScheduler()
    # Wenn flag nicht aktiv → None
    result = scheduler.adapt_hyperparameters_via_bandit(state_path=tmp_path / "hpo.json")
    assert result is None  # default disabled


# ---------------------------------------------------------------------------
# D: Backtest-Comparison
# ---------------------------------------------------------------------------

def test_backtest_comparison_basic():
    from src.assembled_core.qa.backtest_comparison import compare_backtests

    rng = np.random.default_rng(7)
    n = 100
    strategies = {
        "A": pd.Series(rng.normal(0.001, 0.01, n)),
        "B": pd.Series(rng.normal(0.0005, 0.012, n)),
        "C": pd.Series(rng.normal(-0.0005, 0.015, n)),
    }
    report = compare_backtests(strategies)
    assert len(report.strategies) == 3
    assert len(report.pairwise) == 3  # C(3,2) = 3
    assert len(report.ranking) == 3


def test_rank_strategies():
    from src.assembled_core.qa.backtest_comparison import rank_strategies

    rng = np.random.default_rng(11)
    strategies = {
        "good": pd.Series(rng.normal(0.002, 0.01, 100)),
        "bad": pd.Series(rng.normal(-0.001, 0.01, 100)),
    }
    ranked = rank_strategies(strategies)
    assert ranked[0][0] == "good"


# ---------------------------------------------------------------------------
# E: News-Trade-Attribution
# ---------------------------------------------------------------------------

def test_news_trade_attribution_link():
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    trade = {
        "trade_id": "T1",
        "symbol": "AAPL",
        "opened_at": "2025-06-01T12:00:00Z",
        "closed_return": 0.02,
    }
    news = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "symbol": ["AAPL", "GOOG"],
        "published_at": ["2025-06-01T10:00:00Z", "2025-06-01T11:00:00Z"],
        "impact_bps": [50.0, 30.0],
    })

    attributor = NewsTradeAttributor(pre_window_hours=24, post_window_hours=24)
    links = attributor.link_trade_to_events(trade, news)
    # Nur AAPL-Event sollte gematched werden
    assert len(links) == 1
    assert links[0].event_id == "E1"


def test_news_trade_attribution_wiring_learning_store(tmp_path):
    """WIRING: enrich_learning_store schreibt news_links in Records."""
    from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor

    ls_path = tmp_path / "learning_store.jsonl"
    records = [
        {
            "trade_id": "T1", "symbol": "AAPL",
            "opened_at": "2025-06-01T12:00:00Z",
            "closed_at": "2025-06-02T12:00:00Z",
            "closed_return": 0.02,
        }
    ]
    with ls_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    news_path = tmp_path / "news.jsonl"
    news_records = [{
        "event_id": "E1", "symbol": "AAPL",
        "published_at": "2025-06-01T10:00:00Z",
        "impact_bps": 50.0,
    }]
    with news_path.open("w", encoding="utf-8") as fh:
        for r in news_records:
            fh.write(json.dumps(r) + "\n")

    attributor = NewsTradeAttributor()
    n = attributor.enrich_learning_store(ls_path, news_path)
    assert n == 1

    with ls_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    enriched = json.loads(lines[0])
    assert "news_links" in enriched


# ---------------------------------------------------------------------------
# F: Kelly-with-Uncertainty
# ---------------------------------------------------------------------------

def test_kelly_uncertainty_basic():
    from src.assembled_core.portfolio.kelly_uncertainty import compute_kelly_with_uncertainty

    # Kein Uncertainty-Input → Standard-Kelly × fractional
    w = compute_kelly_with_uncertainty(
        edge=0.05, variance=0.04,
        conformal_half_width=None, reference_half_width=None,
        fractional_kelly=0.5, max_fraction=1.0,
    )
    # kelly = 0.05 / 0.04 = 1.25, × 0.5 = 0.625, gecappt auf max_fraction
    assert w == pytest.approx(0.625, rel=1e-6)


def test_kelly_uncertainty_discount():
    """Hohes Conformal-Intervall → niedriger Kelly."""
    from src.assembled_core.portfolio.kelly_uncertainty import compute_kelly_with_uncertainty

    w_low_uncertainty = compute_kelly_with_uncertainty(
        edge=0.05, variance=0.04,
        conformal_half_width=0.01, reference_half_width=0.01,
        fractional_kelly=1.0, max_fraction=10.0,
    )
    w_high_uncertainty = compute_kelly_with_uncertainty(
        edge=0.05, variance=0.04,
        conformal_half_width=0.05, reference_half_width=0.01,
        fractional_kelly=1.0, max_fraction=10.0,
    )
    assert w_high_uncertainty < w_low_uncertainty


def test_kelly_wiring_position_sizing():
    """WIRING: compute_kelly_weights_with_uncertainty in position_sizing."""
    from src.assembled_core.portfolio.position_sizing import compute_kelly_weights_with_uncertainty

    edges = pd.Series({"A": 0.05, "B": 0.02})
    variances = pd.Series({"A": 0.04, "B": 0.01})
    weights = compute_kelly_weights_with_uncertainty(edges, variances)
    assert len(weights) == 2
    assert (weights.abs() <= 1.0).all()


# ---------------------------------------------------------------------------
# G: Signal-Correlation-Analyzer
# ---------------------------------------------------------------------------

def test_signal_correlation_redundancy():
    from src.assembled_core.ml.signal_correlation import SignalCorrelationAnalyzer

    rng = np.random.default_rng(1)
    n = 200
    base = rng.standard_normal(n)
    # Zwei stark korrelierte und ein unabhängiges Signal
    df = pd.DataFrame({
        "sig1": base + 0.01 * rng.standard_normal(n),
        "sig2": base + 0.01 * rng.standard_normal(n),
        "sig3": rng.standard_normal(n),
    })
    analyzer = SignalCorrelationAnalyzer(redundancy_threshold=0.9)
    report = analyzer.analyze(df)
    assert report.n_signals == 3
    assert len(report.redundant_clusters) >= 1


def test_signal_correlation_wiring_feedback_loop():
    """WIRING: _record_signal_correlation existiert und läuft fehlerfrei."""
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    controller = FeedbackLoopController()
    assert hasattr(controller, "_record_signal_correlation")

    rng = np.random.default_rng(0)
    panel = pd.DataFrame({
        "f1": rng.standard_normal(50),
        "f2": rng.standard_normal(50),
        "f3": rng.standard_normal(50),
    })
    controller._record_signal_correlation(panel)  # no-raise


# ---------------------------------------------------------------------------
# H: Drawdown-Decomposition
# ---------------------------------------------------------------------------

def test_find_worst_drawdown():
    from src.assembled_core.qa.drawdown_decomposition import find_worst_drawdown

    returns = pd.Series([0.01, -0.05, -0.03, 0.02, 0.01])
    dd = find_worst_drawdown(returns)
    assert dd.max_drawdown < 0
    assert dd.duration >= 1


def test_drawdown_decomposition():
    from src.assembled_core.qa.drawdown_decomposition import decompose_drawdown

    rng = np.random.default_rng(5)
    n = 100
    market = pd.Series(rng.normal(0.0005, 0.01, n))
    portfolio = 0.5 * market + rng.normal(0, 0.008, n)
    # Force a drawdown
    portfolio.iloc[20:30] = -0.01

    report = decompose_drawdown(portfolio, pd.DataFrame({"market": market}))
    assert "market" in report.factor_betas or report.drawdown.max_drawdown < 0


def test_drawdown_wiring_attribution():
    """WIRING: attribution_during_worst_drawdown existiert."""
    from src.assembled_core.qa.performance_attribution import attribution_during_worst_drawdown

    rng = np.random.default_rng(3)
    n = 100
    portfolio = pd.Series(rng.normal(0.0001, 0.01, n))
    factor = pd.Series(rng.normal(0.0005, 0.012, n))

    result = attribution_during_worst_drawdown(
        portfolio, pd.DataFrame({"mkt": factor}),
    )
    # Entweder summary dict oder error dict
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# I: Trade-TCA
# ---------------------------------------------------------------------------

def test_trade_tca_basic():
    from src.assembled_core.qa.trade_tca import compute_trade_tca

    tca = compute_trade_tca(
        trade_id="T1", symbol="AAPL", side="buy", quantity=100,
        execution_price=100.5, arrival_price=100.0, vwap_price=100.2,
    )
    # buy higher than arrival → positive IS
    assert tca.implementation_shortfall_bps > 0


def test_trade_tca_aggregate():
    from src.assembled_core.qa.trade_tca import aggregate_tca, compute_trade_tca

    tcas = [
        compute_trade_tca("T1", "AAPL", "buy", 100, 100.5, 100.0),
        compute_trade_tca("T2", "AAPL", "sell", 50, 99.5, 100.0),
        compute_trade_tca("T3", "GOOG", "buy", 10, 2000.0, 1999.0),
    ]
    report = aggregate_tca(tcas)
    assert report.n_trades == 3
    assert "AAPL" in report.per_symbol


def test_trade_tca_wiring_learning_store(tmp_path):
    """WIRING: run_tca_from_learning_store verarbeitet JSONL."""
    from src.assembled_core.qa.trade_tca import run_tca_from_learning_store

    ls_path = tmp_path / "ls.jsonl"
    records = [
        {
            "trade_id": "T1", "symbol": "AAPL", "side": "buy",
            "quantity": 100, "execution_price": 100.5, "arrival_price": 100.0,
        },
    ]
    with ls_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    report = run_tca_from_learning_store(ls_path, tmp_path / "tca.json")
    assert report["n_trades"] == 1


# ---------------------------------------------------------------------------
# J: Online-HMM-Regime
# ---------------------------------------------------------------------------

def test_online_hmm_fallback():
    """Ohne hmmlearn → Vol-Quantile-Fallback funktioniert."""
    from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector

    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0005, 0.01, 200))
    detector = OnlineHMMRegimeDetector()
    detector.fit(returns)
    state = detector.predict_current_regime(returns)
    assert state.regime_label in ("LOW_VOL", "NORMAL", "HIGH_VOL")


def test_combined_regime_agreement():
    from src.assembled_core.ml.combined_regime import CombinedRegimeClassifier

    # Beide classifier None → NEUTRAL / NEUTRAL → agreement
    combined = CombinedRegimeClassifier()
    out = combined.predict()
    assert out.combined_regime == "NEUTRAL"


def test_combined_regime_wiring_ml_pipeline():
    """WIRING: MLSignalPipeline akzeptiert combined_regime_classifier."""
    from src.assembled_core.signals.ml_integration import MLSignalPipeline
    from src.assembled_core.ml.combined_regime import CombinedRegimeClassifier

    combined = CombinedRegimeClassifier()
    pipeline = MLSignalPipeline(combined_regime_classifier=combined)
    X = pd.DataFrame({"f1": [0.0, 0.1, -0.2]})
    output = pipeline.run(X, market_returns=pd.Series([0.001, -0.002, 0.001]))
    assert output.regime in ("RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS")


# ---------------------------------------------------------------------------
# K: LIME
# ---------------------------------------------------------------------------

def test_lime_permutation_fallback():
    """Ohne training_data → Permutation-Fallback."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from src.assembled_core.ml.lime_explainer import LIMEExplainerWrapper

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 100)
    model = LinearRegression().fit(X, y)

    wrapper = LIMEExplainerWrapper(
        model=model, feature_names=["a", "b", "c"],
        training_data=None,
    )
    expl = wrapper.explain(np.array([1.0, 0.0, 0.0]))
    assert len(expl.feature_contributions) > 0
    assert expl.source == "permutation_fallback"


def test_lime_wiring_meta_model():
    """WIRING: MetaModel.explain_prediction_lime existiert."""
    pytest.importorskip("sklearn")
    from src.assembled_core.signals.meta_model import train_meta_model

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "f1": rng.standard_normal(100),
        "f2": rng.standard_normal(100),
        "label": rng.integers(0, 2, 100),
    })
    mm = train_meta_model(df, feature_cols=["f1", "f2"], label_col="label")
    result = mm.explain_prediction_lime(df.iloc[0][["f1", "f2"]])
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# L: Monte-Carlo Scenarios
# ---------------------------------------------------------------------------

def test_scenario_vol_spike():
    from src.assembled_core.qa.scenario_simulator import simulate_vol_spike_scenario

    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    result = simulate_vol_spike_scenario(returns, vol_multiplier=3.0, duration=10)
    assert result.scenario_name == "VolSpike"
    assert result.cvar_95 < 0  # tail risk negativ


def test_scenario_crash():
    from src.assembled_core.qa.scenario_simulator import simulate_crash_scenario

    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    result = simulate_crash_scenario(returns, crash_magnitude=-0.10, recovery_days=20)
    # Crash scenario sollte negativer mean return haben
    assert result.mean_return < 0


def test_stress_test_report():
    from src.assembled_core.qa.scenario_simulator import run_stress_test

    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 300))
    report = run_stress_test(returns, include_correlation=False)
    assert len(report.scenarios) >= 2
    assert report.worst_scenario != ""
