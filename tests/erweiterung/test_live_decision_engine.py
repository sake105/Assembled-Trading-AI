"""Tests für LiveDecisionEngine — inkrementelle State-Updates + Latency."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from erweiterung.live.live_decision_engine import (
    LiveDecisionEngine,
)


def _make_wide_returns(n_days: int = 500, n_symbols: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="UTC")
    cols = [f"S{i}" for i in range(n_symbols)]
    eq = pd.DataFrame(
        rng.normal(0.0005, 0.012, (n_days, n_symbols)), index=idx, columns=cols
    )
    xa_cols = [f"X{i}" for i in range(6)]
    xa = pd.DataFrame(
        rng.normal(0.0003, 0.008, (n_days, 6)), index=idx, columns=xa_cols
    )
    return eq, xa


def test_engine_bootstrap_initializes_state():
    eq, xa = _make_wide_returns(n_days=400)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    s = engine.state_summary()
    assert s["n_eq_history_days"] > 0
    assert s["n_xa_history_days"] > 0
    assert s["last_date"] is not None


def test_decide_next_returns_required_fields():
    eq, xa = _make_wide_returns(n_days=400)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    out = engine.decide_next()
    assert "sa_leverage" in out
    assert "xa_top_weights" in out
    assert "xa_hybrid_weights" in out
    assert "eq_top_weights" in out
    assert "decision_latency_ms" in out


def test_decide_next_latency_under_10ms():
    """LATENCY SLA: decide_next() muss < 10ms sein."""
    eq, xa = _make_wide_returns(n_days=504)  # 2y bootstrap
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    # Warm-up
    engine.decide_next()
    # Multiple runs to get stable median
    latencies = []
    for _ in range(20):
        out = engine.decide_next()
        latencies.append(out["decision_latency_ms"])
    median_lat = float(np.median(latencies))
    assert median_lat < 10.0, f"Median latency {median_lat:.2f} ms exceeds 10ms SLA"


def test_update_with_new_day_increments_state():
    eq, xa = _make_wide_returns(n_days=400)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    s0 = engine.state_summary()
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    new_eq_ret = pd.Series(0.001, index=eq.columns)
    new_xa_ret = pd.Series(0.0005, index=xa.columns)
    engine.update_with_new_day(next_date, new_eq_ret, new_xa_ret)
    s1 = engine.state_summary()
    assert s1["n_eq_history_days"] == s0["n_eq_history_days"] + 1
    assert s1["n_xa_history_days"] == s0["n_xa_history_days"] + 1
    assert s1["last_date"] != s0["last_date"]


def test_update_latency_under_10ms():
    """LATENCY SLA: update_with_new_day() muss < 10ms sein."""
    eq, xa = _make_wide_returns(n_days=504)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    new_eq_ret = pd.Series(0.001, index=eq.columns)
    new_xa_ret = pd.Series(0.0005, index=xa.columns)
    # Warm-up
    engine.update_with_new_day(next_date, new_eq_ret, new_xa_ret)
    # Multiple updates
    latencies = []
    for i in range(20):
        d = next_date + pd.Timedelta(days=i + 1)
        t0 = time.perf_counter()
        engine.update_with_new_day(d, new_eq_ret, new_xa_ret)
        latencies.append((time.perf_counter() - t0) * 1000)
    median_lat = float(np.median(latencies))
    assert (
        median_lat < 10.0
    ), f"Median update latency {median_lat:.2f} ms exceeds 10ms SLA"


def test_save_load_state(tmp_path):
    eq, xa = _make_wide_returns(n_days=300)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    state_path = tmp_path / "engine_state.pkl"
    engine.save_state(state_path)

    engine2 = LiveDecisionEngine()
    engine2.load_state(state_path)
    s1 = engine.state_summary()
    s2 = engine2.state_summary()
    assert s1["n_eq_history_days"] == s2["n_eq_history_days"]
    assert s1["last_date"] == s2["last_date"]


def test_max_history_truncation():
    """State darf nicht unbegrenzt wachsen."""
    eq, xa = _make_wide_returns(n_days=300)
    engine = LiveDecisionEngine()
    engine.state.max_history = 100  # Force tight buffer
    engine.bootstrap_from_history(eq, xa)
    next_date = eq.index[-1]
    new_eq_ret = pd.Series(0.001, index=eq.columns)
    new_xa_ret = pd.Series(0.0005, index=xa.columns)
    for i in range(50):
        d = next_date + pd.Timedelta(days=i + 1)
        engine.update_with_new_day(d, new_eq_ret, new_xa_ret)
    s = engine.state_summary()
    assert s["n_eq_history_days"] <= 100
    assert s["n_xa_history_days"] <= 100


def test_xa_rebalance_triggers_at_month_end():
    """Top-Weights sollten sich an Monatsende neu bestimmen."""
    eq, xa = _make_wide_returns(n_days=300)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    initial_weights = engine.state.xa_mom_top_weights.copy()
    # Trigger 25 daily updates -> should force rebalance at ~day 21
    next_date = eq.index[-1]
    for i in range(25):
        d = next_date + pd.Timedelta(days=i + 1)
        new_eq = pd.Series(
            np.random.default_rng(i).normal(0, 0.01, len(eq.columns)), index=eq.columns
        )
        new_xa = pd.Series(
            np.random.default_rng(i + 1000).normal(0, 0.01, len(xa.columns)),
            index=xa.columns,
        )
        engine.update_with_new_day(d, new_eq, new_xa)
    final_weights = engine.state.xa_mom_top_weights
    # State should reset (rebalance happened)
    assert engine.state.days_since_xa_rebalance < 21
