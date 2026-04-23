"""Tests for wave-34 module wiring into trading_cycle.py.

Covers:
  Step 3.88 — signals.risk_aware_combiner (RiskAwareSignalCombiner.combine)
  Step 8.22 — qa.labeling (label_daily_records)
  Step 8.23 — ml.feature_importance_tracker (FeatureImportanceTracker)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.signals.risk_aware_combiner import (
    RiskAwareSignalCombiner,
    CombinerState,
)
from src.assembled_core.qa.labeling import (
    label_daily_records,
    label_trades,
)
from src.assembled_core.ml.feature_importance_tracker import (
    FeatureImportanceTracker,
    ImportanceSnapshot,
)


# ---------------------------------------------------------------------------
# RiskAwareSignalCombiner (Step 3.88)
# ---------------------------------------------------------------------------

def _make_signal_df(n: int = 20, n_sigs: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"sig_{i}": rng.uniform(-1, 1, n) for i in range(n_sigs)},
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


def test_combiner_returns_series():
    combiner = RiskAwareSignalCombiner()
    sig_df = _make_signal_df()
    result = combiner.combine(sig_df, current_regime="NEUTRAL")
    assert isinstance(result, pd.Series)


def test_combiner_length_matches_input():
    combiner = RiskAwareSignalCombiner()
    sig_df = _make_signal_df(n=30)
    result = combiner.combine(sig_df, current_regime="RISK_ON")
    assert len(result) == 30


def test_combiner_valid_regimes():
    combiner = RiskAwareSignalCombiner()
    sig_df = _make_signal_df()
    for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"]:
        result = combiner.combine(sig_df, current_regime=regime)
        assert isinstance(result, pd.Series)


def test_combiner_unknown_regime_fallback():
    combiner = RiskAwareSignalCombiner()
    sig_df = _make_signal_df()
    # Unknown regime should fall back to NEUTRAL without raising
    result = combiner.combine(sig_df, current_regime="UNKNOWN_REGIME")
    assert isinstance(result, pd.Series)
    assert len(result) == len(sig_df)


def test_combiner_no_weights_returns_zeros():
    combiner = RiskAwareSignalCombiner()
    sig_df = _make_signal_df()
    result = combiner.combine(sig_df, current_regime="NEUTRAL")
    # No fitted weights → default_weight=0 → all zeros
    assert (result == 0.0).all()


def test_combiner_with_equal_weights():
    combiner = RiskAwareSignalCombiner()
    n_sigs = 3
    # Manually set equal weights for NEUTRAL
    for i in range(n_sigs):
        combiner.state.weights[(f"sig_{i}", "NEUTRAL")] = 1.0 / n_sigs
    sig_df = _make_signal_df(n=10, n_sigs=n_sigs)
    result = combiner.combine(sig_df, current_regime="NEUTRAL")
    expected = sig_df.mean(axis=1)
    np.testing.assert_allclose(result.values, expected.values, atol=1e-10)


def test_combiner_state_default_weight():
    state = CombinerState()
    assert state.default_weight == 0.0


def test_combiner_single_signal():
    combiner = RiskAwareSignalCombiner()
    sig_df = pd.DataFrame({"sig_0": [0.5, -0.3, 0.1]})
    result = combiner.combine(sig_df, current_regime="NEUTRAL")
    assert isinstance(result, pd.Series)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# label_daily_records (Step 8.22)
# ---------------------------------------------------------------------------

def _make_equity_df(n: int = 30, seed: int = 0, trend: float = 50.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="B")
    equity = 100000.0 + np.cumsum(rng.normal(trend, 200, n))
    return pd.DataFrame({"timestamp": ts, "equity": equity})


def test_label_daily_returns_df():
    eq_df = _make_equity_df()
    result = label_daily_records(eq_df, horizon_days=5)
    assert isinstance(result, pd.DataFrame)


def test_label_daily_has_label_column():
    eq_df = _make_equity_df()
    result = label_daily_records(eq_df, horizon_days=5)
    assert "label" in result.columns


def test_label_daily_row_count_preserved():
    eq_df = _make_equity_df(n=40)
    result = label_daily_records(eq_df, horizon_days=5)
    assert len(result) == 40


def test_label_daily_binary_values():
    eq_df = _make_equity_df()
    result = label_daily_records(eq_df, horizon_days=5)
    valid = result["label"].dropna()
    assert valid.isin([0, 0.0, 1, 1.0]).all()


def test_label_daily_strong_uptrend_positive():
    # Very strong uptrend: many labels should be 1
    eq_df = _make_equity_df(n=30, trend=500.0)
    result = label_daily_records(eq_df, horizon_days=5, success_threshold=0.001)
    valid = result["label"].dropna()
    if len(valid) > 0:
        assert valid.mean() > 0.0


def test_label_daily_empty_returns_empty():
    result = label_daily_records(pd.DataFrame(), horizon_days=5)
    assert isinstance(result, pd.DataFrame)


def test_label_daily_missing_timestamp_raises():
    df = pd.DataFrame({"equity": [100, 200, 300]})
    with pytest.raises(ValueError):
        label_daily_records(df, horizon_days=5)


# ---------------------------------------------------------------------------
# FeatureImportanceTracker (Step 8.23)
# ---------------------------------------------------------------------------

def test_fi_tracker_creates_empty(tmp_path):
    tracker = FeatureImportanceTracker(state_path=tmp_path / "fi.json")
    assert isinstance(tracker._snapshots, list)
    assert len(tracker._snapshots) == 0


def test_fi_tracker_missing_path_ok(tmp_path):
    tracker = FeatureImportanceTracker(state_path=tmp_path / "nonexistent" / "fi.json")
    assert tracker._snapshots == []


def test_fi_tracker_history_window(tmp_path):
    tracker = FeatureImportanceTracker(state_path=tmp_path / "fi.json", history_window=6)
    assert tracker.history_window == 6


def test_fi_tracker_persists_snapshot(tmp_path):
    path = tmp_path / "fi.json"
    tracker = FeatureImportanceTracker(state_path=path)
    snap = ImportanceSnapshot(
        as_of="2024-01-15",
        importances={"feat_a": 0.05, "feat_b": 0.02},
        baseline_score=0.3,
        n_features=2,
        n_samples=100,
    )
    tracker._snapshots.append(snap)
    tracker._save()
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data["snapshots"]) == 1
    assert data["snapshots"][0]["as_of"] == "2024-01-15"


def test_fi_tracker_loads_saved_snapshots(tmp_path):
    path = tmp_path / "fi.json"
    tracker = FeatureImportanceTracker(state_path=path)
    for i in range(3):
        snap = ImportanceSnapshot(
            as_of=f"2024-01-{i+1:02d}",
            importances={"feat": float(i) * 0.01},
            baseline_score=0.2 + i * 0.05,
            n_features=1,
            n_samples=50,
        )
        tracker._snapshots.append(snap)
    tracker._save()
    tracker2 = FeatureImportanceTracker(state_path=path)
    assert len(tracker2._snapshots) == 3


def test_fi_tracker_history_window_trims(tmp_path):
    path = tmp_path / "fi.json"
    tracker = FeatureImportanceTracker(state_path=path, history_window=3)
    for i in range(5):
        snap = ImportanceSnapshot(
            as_of=f"2024-01-{i+1:02d}",
            importances={"feat": float(i) * 0.01},
            baseline_score=0.1,
            n_features=1,
            n_samples=20,
        )
        tracker._snapshots.append(snap)
    tracker._save()
    tracker2 = FeatureImportanceTracker(state_path=path, history_window=3)
    assert len(tracker2._snapshots) <= 3
