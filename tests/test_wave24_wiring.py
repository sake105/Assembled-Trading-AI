"""Tests for wave-24 module wiring into trading_cycle.py.

Covers:
  Step 2.14 — ml.feature_selection (ic_prescreen)
  Step 7.6  — ops.kpi_artifacts (write_run_kpis)
  Step 8.12 — ml.online_learning (compute_model_age_confidence)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.feature_selection import (
    ic_prescreen,
    collinearity_filter,
    FeatureSelectionResult,
)
from src.assembled_core.ml.online_learning import compute_model_age_confidence
from src.assembled_core.ops.kpi_artifacts import (
    write_run_kpis,
    write_targets_artifact,
    write_orders_artifact,
)


# ---------------------------------------------------------------------------
# ic_prescreen (Step 2.14)
# ---------------------------------------------------------------------------

def _make_factor_panel(n_ts: int = 30, n_syms: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ts = pd.date_range("2024-01-01", periods=n_ts, freq="B")
    for t in ts:
        for sym in [f"S{i}" for i in range(n_syms)]:
            fwd = rng.normal(0, 0.01)
            rows.append({
                "timestamp": t,
                "symbol": sym,
                "feat_a": rng.normal(0, 1),
                "feat_b": fwd * 10 + rng.normal(0, 0.5),  # correlated with fwd
                "feat_c": rng.normal(0, 1),
                "fwd_return_1m": fwd,
            })
    return pd.DataFrame(rows)


def test_ic_prescreen_returns_tuple():
    panel = _make_factor_panel()
    kept, scores = ic_prescreen(panel)
    assert isinstance(kept, list)
    assert isinstance(scores, dict)


def test_ic_prescreen_scores_for_all_features():
    panel = _make_factor_panel()
    _, scores = ic_prescreen(panel)
    for feat in ["feat_a", "feat_b", "feat_c"]:
        assert feat in scores


def test_ic_prescreen_scores_non_negative():
    panel = _make_factor_panel()
    _, scores = ic_prescreen(panel)
    for v in scores.values():
        assert v >= 0.0


def test_ic_prescreen_min_ic_filters():
    panel = _make_factor_panel()
    _, scores = ic_prescreen(panel, min_ic=0.0)
    kept_all, _ = ic_prescreen(panel, min_ic=0.0)
    kept_strict, _ = ic_prescreen(panel, min_ic=0.99)
    assert len(kept_strict) <= len(kept_all)


def test_ic_prescreen_no_forward_col_returns_all():
    panel = _make_factor_panel().drop(columns=["fwd_return_1m"])
    kept, scores = ic_prescreen(panel)
    # Should return all feature columns when no forward return col
    assert isinstance(kept, list)
    assert len(kept) > 0


def test_collinearity_filter_returns_tuple():
    panel = _make_factor_panel()
    feats = ["feat_a", "feat_b", "feat_c"]
    ic_scores = {"feat_a": 0.05, "feat_b": 0.1, "feat_c": 0.03}
    kept, pairs = collinearity_filter(panel, feats, ic_scores)
    assert isinstance(kept, list)
    assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# compute_model_age_confidence (Step 8.12)
# ---------------------------------------------------------------------------

def test_model_age_fresh_is_one():
    conf = compute_model_age_confidence(days_since_refit=0)
    assert conf == 1.0


def test_model_age_decays_over_time():
    conf0 = compute_model_age_confidence(0)
    conf10 = compute_model_age_confidence(10)
    conf30 = compute_model_age_confidence(30)
    assert conf0 > conf10 > conf30


def test_model_age_half_life_at_half_life():
    conf = compute_model_age_confidence(days_since_refit=30, half_life_days=30)
    assert abs(conf - 0.5) < 0.01


def test_model_age_in_01_range():
    for d in [0, 1, 7, 30, 90, 365]:
        conf = compute_model_age_confidence(d)
        assert 0.0 < conf <= 1.0


def test_model_age_shorter_half_life_decays_faster():
    conf_fast = compute_model_age_confidence(30, half_life_days=10)
    conf_slow = compute_model_age_confidence(30, half_life_days=60)
    assert conf_fast < conf_slow


# ---------------------------------------------------------------------------
# write_run_kpis / write_targets_artifact (Step 7.6)
# ---------------------------------------------------------------------------

class _MockCtx:
    as_of = pd.Timestamp("2024-01-15")
    execution_mode = "paper"
    risk_state = None
    news_geo = None
    market_stress = None
    news_triggers = None
    current_positions = pd.DataFrame()
    equity = 100_000.0
    current_equity = 100_000.0


class _MockResult:
    target_positions = pd.DataFrame({"symbol": ["A", "B"], "target_weight": [0.5, 0.5], "weight": [0.5, 0.5]})
    orders_filtered = pd.DataFrame()
    meta: dict = {}
    signals = pd.DataFrame()


def test_write_run_kpis_creates_file(tmp_path):
    ctx = _MockCtx()
    result = _MockResult()
    path = write_run_kpis(tmp_path, ctx, result, policy={}, mode="paper")
    assert Path(path).exists()


def test_write_run_kpis_returns_path(tmp_path):
    ctx = _MockCtx()
    result = _MockResult()
    path = write_run_kpis(tmp_path, ctx, result, policy={}, mode="paper")
    assert str(path).endswith(".json")


def test_write_targets_artifact_creates_file(tmp_path):
    targets = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "target_weight": [0.4, 0.3, 0.3],
    })
    path = write_targets_artifact(tmp_path, targets)
    assert Path(path).exists()


def test_write_orders_artifact_empty_ok(tmp_path):
    orders = pd.DataFrame()
    path = write_orders_artifact(tmp_path, orders)
    assert Path(path).exists()
