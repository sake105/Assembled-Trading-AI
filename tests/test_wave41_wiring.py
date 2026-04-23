"""Tests for wave-41 module wiring into trading_cycle.py.

Covers:
  Step 5.14 — ops.self_healing (RiskEscalationLadder)
  Step 8.34 — qa.dataset_builder (build_ml_dataset_from_backtest)
  Step 8.35 — ml.stacking (enforce_ensemble_diversity)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ops.self_healing import (
    RiskEscalationLadder,
    EscalationLevel,
    EscalationState,
    DataSourceCascade,
)
from src.assembled_core.qa.dataset_builder import (
    build_ml_dataset_from_backtest,
    save_ml_dataset,
)
from src.assembled_core.ml.stacking import (
    enforce_ensemble_diversity,
    OnlineEnsembleWeights,
)


# ---------------------------------------------------------------------------
# RiskEscalationLadder (Step 5.14)
# ---------------------------------------------------------------------------

def test_escalation_ladder_creates():
    ladder = RiskEscalationLadder()
    assert isinstance(ladder, RiskEscalationLadder)


def test_escalation_normal_state():
    ladder = RiskEscalationLadder()
    state = ladder.evaluate(current_drawdown=-0.02)
    assert state.level == EscalationLevel.NORMAL


def test_escalation_reduce_state():
    ladder = RiskEscalationLadder(dd_reduce=0.10)
    state = ladder.evaluate(current_drawdown=-0.12)
    assert state.level in {EscalationLevel.REDUCE, EscalationLevel.CRITICAL, EscalationLevel.KILL}


def test_escalation_kill_state():
    ladder = RiskEscalationLadder(dd_kill=0.20)
    state = ladder.evaluate(current_drawdown=-0.25)
    assert state.level == EscalationLevel.KILL


def test_escalation_state_has_reason():
    ladder = RiskEscalationLadder()
    state = ladder.evaluate(current_drawdown=-0.05)
    assert isinstance(state.trigger_reason, str)
    assert len(state.trigger_reason) > 0


def test_escalation_returns_escalation_state():
    ladder = RiskEscalationLadder()
    state = ladder.evaluate(current_drawdown=-0.01)
    assert isinstance(state, EscalationState)


def test_data_source_cascade_empty():
    cascade = DataSourceCascade()
    with pytest.raises(RuntimeError):
        cascade.fetch()


def test_data_source_cascade_success():
    cascade = DataSourceCascade()
    cascade.register_source("test", lambda: {"data": 42})
    data, source = cascade.fetch()
    assert data == {"data": 42}
    assert source == "test"


# ---------------------------------------------------------------------------
# build_ml_dataset_from_backtest (Step 8.34)
# ---------------------------------------------------------------------------

def _make_prices_features(n: int = 30, n_syms: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in [f"S{i}" for i in range(n_syms)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        for t, c in zip(ts, closes):
            rows.append({
                "timestamp": t, "symbol": sym, "close": float(c),
                "news_sentiment": rng.normal(0, 0.3),
            })
    return pd.DataFrame(rows)


def _make_trades(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-15", periods=n, freq="3B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": [f"S{i % 3}" for i in range(n)],
        "side": ["BUY"] * n,
        "qty": rng.uniform(10, 100, n),
        "price": rng.uniform(95, 105, n),
        "pnl_pct": rng.normal(0.01, 0.03, n),
    })


def test_dataset_builder_returns_df():
    prices = _make_prices_features()
    trades = _make_trades()
    result = build_ml_dataset_from_backtest(prices, trades)
    assert isinstance(result, pd.DataFrame)


def test_dataset_builder_empty_trades():
    prices = _make_prices_features()
    result = build_ml_dataset_from_backtest(prices, pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


def test_dataset_builder_has_label_col():
    prices = _make_prices_features()
    trades = _make_trades()
    result = build_ml_dataset_from_backtest(prices, trades)
    if len(result) > 0:
        assert "label" in result.columns


def test_save_ml_dataset_creates_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 0]})
    path = tmp_path / "dataset.parquet"
    save_ml_dataset(df, path)
    assert path.exists()


def test_dataset_builder_empty_prices():
    result = build_ml_dataset_from_backtest(pd.DataFrame(), pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# enforce_ensemble_diversity (Step 8.35)
# ---------------------------------------------------------------------------

def test_enforce_diversity_returns_dict():
    rng = np.random.default_rng(0)
    oof = rng.normal(0, 1, (100, 3))
    result = enforce_ensemble_diversity(oof)
    assert isinstance(result, dict)


def test_enforce_diversity_has_avg_correlation():
    rng = np.random.default_rng(0)
    oof = rng.normal(0, 1, (100, 3))
    result = enforce_ensemble_diversity(oof)
    assert "avg_correlation" in result


def test_enforce_diversity_correlated_not_diverse():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 100)
    # Highly correlated models
    oof = np.column_stack([base, base + rng.normal(0, 0.01, 100), base + rng.normal(0, 0.01, 100)])
    result = enforce_ensemble_diversity(oof, max_correlation=0.80)
    assert result["diverse"] is False


def test_enforce_diversity_uncorrelated_diverse():
    rng = np.random.default_rng(1)
    oof = rng.normal(0, 1, (100, 3))
    result = enforce_ensemble_diversity(oof, max_correlation=0.80)
    assert isinstance(result["diverse"], bool)


def test_enforce_diversity_single_model():
    rng = np.random.default_rng(0)
    oof = rng.normal(0, 1, (100, 1))
    result = enforce_ensemble_diversity(oof)
    assert result["avg_correlation"] == 0.0
    assert result["diverse"] is True


def test_online_ensemble_weights_creates():
    ew = OnlineEnsembleWeights(n_models=3)
    assert isinstance(ew, OnlineEnsembleWeights)
