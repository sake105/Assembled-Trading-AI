"""Tests for wave-43 module wiring into trading_cycle.py.

Covers:
  Step 7.71 — ops.paper_ledger (load_ledger_state / mark_to_market_equity / write_ledger_snapshot)
  Step 8.38 — qa.factor_ranking (build_factor_ranking)
  Step 8.39 — ml.meta_labeling (MetaLabeler)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ops.paper_ledger import (
    load_ledger_state,
    save_ledger_state,
    mark_to_market_equity,
    write_ledger_snapshot,
    simulate_fills,
    apply_fills_to_ledger,
)
from src.assembled_core.qa.factor_ranking import build_factor_ranking


# ---------------------------------------------------------------------------
# paper_ledger (Step 7.71)
# ---------------------------------------------------------------------------

def test_load_ledger_state_fresh(tmp_path):
    state = load_ledger_state(tmp_path / "nonexistent.json", start_capital=50000.0)
    assert isinstance(state, dict)
    assert state["cash"] == 50000.0


def test_load_ledger_state_no_positions_by_default(tmp_path):
    state = load_ledger_state(tmp_path / "nope.json")
    assert "positions" in state
    assert isinstance(state["positions"], dict)
    assert len(state["positions"]) == 0


def test_save_and_load_ledger_state(tmp_path):
    path = tmp_path / "ledger.json"
    state = load_ledger_state(path, start_capital=10000.0)
    state["cash"] = 9500.0
    save_ledger_state(state, path)
    loaded = load_ledger_state(path)
    assert abs(loaded["cash"] - 9500.0) < 0.01


def test_mark_to_market_no_positions():
    state = {"cash": 10000.0, "positions": {}}
    prices = pd.DataFrame(columns=["symbol", "close"])
    equity = mark_to_market_equity(state, prices)
    assert abs(equity - 10000.0) < 0.01


def test_mark_to_market_with_positions():
    state = {
        "cash": 8000.0,
        "positions": {"AAPL": {"qty": 10.0}},
    }
    prices = pd.DataFrame({"symbol": ["AAPL"], "close": [150.0]})
    equity = mark_to_market_equity(state, prices)
    assert abs(equity - 9500.0) < 0.01


def test_write_ledger_snapshot_creates_file(tmp_path):
    state = {"cash": 10000.0, "positions": {}}
    path = write_ledger_snapshot(tmp_path, state, equity=10000.0)
    assert path.exists()


def test_write_ledger_snapshot_content(tmp_path):
    state = {"cash": 9000.0, "positions": {"BTC": {"qty": 1.0}}}
    write_ledger_snapshot(tmp_path, state, equity=12000.0)
    snap = json.loads((tmp_path / "ledger_snapshot.json").read_text())
    assert snap["equity"] == 12000.0
    assert snap["cash"] == 9000.0


def test_simulate_fills_empty():
    orders = pd.DataFrame(columns=["symbol", "side", "qty", "price"])
    prices = pd.DataFrame(columns=["symbol", "close"])
    fills = simulate_fills(orders, prices)
    assert isinstance(fills, list)
    assert len(fills) == 0


# ---------------------------------------------------------------------------
# factor_ranking (Step 8.38)
# ---------------------------------------------------------------------------

def _make_ic_csv(tmp_path: Path, name: str) -> Path:
    df = pd.DataFrame({
        "factor": ["momentum", "value", "quality"],
        "mean_ic": [0.04, 0.02, 0.03],
        "std_ic": [0.08, 0.06, 0.07],
        "ic_ir": [0.50, 0.33, 0.43],
        "hit_ratio": [0.55, 0.52, 0.53],
        "count": [252, 252, 252],
    })
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _make_rank_ic_csv(tmp_path: Path, name: str) -> Path:
    df = pd.DataFrame({
        "factor": ["momentum", "value", "quality"],
        "mean_ic": [0.05, 0.02, 0.03],
        "std_ic": [0.09, 0.06, 0.07],
        "ic_ir": [0.56, 0.33, 0.43],
        "hit_ratio": [0.56, 0.51, 0.54],
        "count": [252, 252, 252],
    })
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_factor_ranking_returns_df(tmp_path):
    ic_path = _make_ic_csv(tmp_path, "ic_summary.csv")
    rank_path = _make_rank_ic_csv(tmp_path, "rank_ic_summary.csv")
    result = build_factor_ranking([ic_path], [rank_path])
    assert isinstance(result, pd.DataFrame)


def test_factor_ranking_has_factor_name(tmp_path):
    ic_path = _make_ic_csv(tmp_path, "ic_summary.csv")
    rank_path = _make_rank_ic_csv(tmp_path, "rank_ic_summary.csv")
    result = build_factor_ranking([ic_path], [rank_path])
    assert "factor_name" in result.columns or "factor" in result.columns


def test_factor_ranking_returns_three_factors(tmp_path):
    ic_path = _make_ic_csv(tmp_path, "ic_summary.csv")
    rank_path = _make_rank_ic_csv(tmp_path, "rank_ic_summary.csv")
    result = build_factor_ranking([ic_path], [rank_path])
    assert len(result) == 3


def test_factor_ranking_missing_ic_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        build_factor_ranking([], [])


def test_factor_ranking_nonexistent_paths_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_factor_ranking([tmp_path / "no.csv"], [tmp_path / "also_no.csv"])


# ---------------------------------------------------------------------------
# MetaLabeler (Step 8.39)
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn", reason="scikit-learn required for meta_labeling")

from src.assembled_core.ml.meta_labeling import MetaLabeler, MetaLabelRecord  # noqa: E402


def test_meta_labeler_creates():
    ml = MetaLabeler()
    assert isinstance(ml, MetaLabeler)


def test_meta_labeler_model_type():
    ml = MetaLabeler(model_type="gradient_boosting")
    assert ml.model_type == "gradient_boosting"


def test_meta_labeler_threshold():
    ml = MetaLabeler(confidence_threshold=0.60)
    assert abs(ml.confidence_threshold - 0.60) < 1e-9


def test_meta_labeler_unfitted_predict_returns_half():
    rng = np.random.default_rng(0)
    ml = MetaLabeler()
    features = pd.DataFrame({
        "primary_signal": rng.normal(0, 1, 5),
        "primary_direction": [1, -1, 1, -1, 1],
        "news_sentiment_mean": rng.normal(0, 0.1, 5),
        "news_velocity": rng.uniform(0, 1, 5),
        "regime_state": [0, 0, 1, 1, 0],
        "vix_proxy": rng.uniform(15, 30, 5),
    })
    preds = ml.predict_confidence(features)
    assert isinstance(preds, pd.Series)
    assert len(preds) == 5
    assert (preds == 0.5).all()


def test_meta_labeler_fit_and_predict():
    rng = np.random.default_rng(42)
    n = 120
    dataset = pd.DataFrame({
        "primary_signal": rng.normal(0, 1, n),
        "primary_direction": rng.choice([-1, 1], n),
        "news_sentiment_mean": rng.normal(0, 0.1, n),
        "news_velocity": rng.uniform(0, 1, n),
        "regime_state": rng.choice([0, 1], n),
        "vix_proxy": rng.uniform(15, 30, n),
        "meta_label": rng.choice([0, 1], n),
    })
    ml = MetaLabeler()
    ml.fit(dataset)
    assert ml._model is not None
    features = dataset[MetaLabeler.FEATURE_NAMES]
    preds = ml.predict_confidence(features)
    assert len(preds) == n
    assert ((preds >= 0.0) & (preds <= 1.0)).all()
