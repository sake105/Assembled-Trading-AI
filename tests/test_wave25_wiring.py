"""Tests for wave-25 module wiring into trading_cycle.py.

Covers:
  Step 2.15 — ml.triple_barrier (build_triple_barrier_labels)
  Step 3.9  — ml.adversarial_validation (run_adversarial_validation)
  Step 7.62 — ops.run_manifest (write_run_manifest)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.triple_barrier import (
    build_triple_barrier_labels,
    apply_triple_barrier,
    compute_daily_volatility,
)
from src.assembled_core.ops.run_manifest import (
    write_run_manifest,
    RunManifest,
    compute_config_hash,
)


# ---------------------------------------------------------------------------
# build_triple_barrier_labels (Step 2.15)
# ---------------------------------------------------------------------------

def _make_panel(n: int = 80, n_syms: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in [f"S{i}" for i in range(n_syms)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = 100.0 + np.cumsum(rng.normal(0.1, 0.5, n))
        for t, p in zip(ts, prices):
            rows.append({"timestamp": t, "symbol": sym, "close": float(p)})
    return pd.DataFrame(rows)


def test_triple_barrier_labels_returns_df():
    panel = _make_panel()
    result = build_triple_barrier_labels(panel)
    assert isinstance(result, pd.DataFrame)


def test_triple_barrier_labels_adds_columns():
    panel = _make_panel()
    result = build_triple_barrier_labels(panel, horizon_days=5)
    assert "tb_label_5d" in result.columns
    assert "tb_ret_5d" in result.columns


def test_triple_barrier_labels_values_valid():
    panel = _make_panel()
    result = build_triple_barrier_labels(panel, horizon_days=5)
    valid_labels = {-1.0, 0.0, 1.0}
    actual = set(result["tb_label_5d"].dropna().unique())
    assert actual.issubset(valid_labels)


def test_triple_barrier_labels_pit_nan_at_end():
    panel = _make_panel(n=60, n_syms=1)
    result = build_triple_barrier_labels(panel, horizon_days=5)
    sym_result = result[result["symbol"] == "S0"].tail(5)
    # Last horizon_days rows should have NaN labels (PIT guarantee)
    assert sym_result["tb_label_5d"].isna().any()


def test_triple_barrier_labels_row_count_preserved():
    panel = _make_panel()
    result = build_triple_barrier_labels(panel)
    assert len(result) == len(panel)


def test_apply_triple_barrier_returns_df():
    rng = np.random.default_rng(0)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 0.5, 60)))
    vol = compute_daily_volatility(prices)
    result = apply_triple_barrier(prices, vol, horizon_days=5)
    assert isinstance(result, pd.DataFrame)


def test_apply_triple_barrier_has_label_column():
    rng = np.random.default_rng(1)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 0.5, 60)))
    vol = compute_daily_volatility(prices)
    result = apply_triple_barrier(prices, vol, horizon_days=5)
    assert "label" in result.columns


def test_compute_daily_volatility_non_negative():
    rng = np.random.default_rng(2)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 0.5, 60)))
    vol = compute_daily_volatility(prices)
    valid_vol = vol.dropna()
    assert (valid_vol >= 0).all()


# ---------------------------------------------------------------------------
# run_adversarial_validation (Step 3.9)
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn", reason="sklearn required for adversarial_validation")

from src.assembled_core.ml.adversarial_validation import (
    run_adversarial_validation,
    AdversarialResult,
)


def _make_feature_df(n: int = 50, n_cols: int = 6, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n_cols)) + shift
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(n_cols)])


def test_adversarial_returns_result():
    train = _make_feature_df(50)
    test = _make_feature_df(50)
    result = run_adversarial_validation(train, test)
    assert isinstance(result, AdversarialResult)


def test_adversarial_auc_in_range():
    train = _make_feature_df(50)
    test = _make_feature_df(50)
    result = run_adversarial_validation(train, test)
    assert 0.0 <= result.auc <= 1.0


def test_adversarial_no_drift_near_05():
    # Same distribution — AUC should be close to 0.5
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 6))
    train = pd.DataFrame(data[:50], columns=[f"f{i}" for i in range(6)])
    test = pd.DataFrame(data[50:], columns=[f"f{i}" for i in range(6)])
    result = run_adversarial_validation(train, test)
    assert result.auc < 0.85  # soft upper bound


def test_adversarial_extreme_drift_high_auc():
    # Train and test from very different distributions
    train = _make_feature_df(60, shift=0.0)
    test = _make_feature_df(60, shift=10.0)
    result = run_adversarial_validation(train, test)
    assert result.auc > 0.8


def test_adversarial_top_drift_features():
    train = _make_feature_df(50)
    test = _make_feature_df(50, shift=5.0)
    result = run_adversarial_validation(train, test, top_k_features=3)
    assert len(result.top_drift_features) <= 3


# ---------------------------------------------------------------------------
# write_run_manifest (Step 7.62)
# ---------------------------------------------------------------------------

def test_run_manifest_creates_file(tmp_path):
    p = write_run_manifest(
        run_id="test_run",
        date="2024-01-15",
        started_at_utc="2024-01-15T09:00:00+00:00",
        status="success",
        manifests_dir=tmp_path,
    )
    assert Path(p).exists()


def test_run_manifest_readable(tmp_path):
    p = write_run_manifest(
        run_id="test_run",
        date="2024-01-15",
        started_at_utc="2024-01-15T09:00:00+00:00",
        manifests_dir=tmp_path,
    )
    with open(p) as f:
        data = json.load(f)
    assert data["run_id"] == "test_run"
    assert data["date"] == "2024-01-15"


def test_run_manifest_latest_pointer(tmp_path):
    write_run_manifest(
        run_id="r1",
        date="2024-01-15",
        started_at_utc="2024-01-15T09:00:00+00:00",
        manifests_dir=tmp_path,
    )
    latest = tmp_path / "r1" / "manifest.latest.json"
    assert latest.exists()


def test_run_manifest_with_metrics(tmp_path):
    p = write_run_manifest(
        run_id="r2",
        date="2024-01-16",
        started_at_utc="2024-01-16T09:00:00+00:00",
        metrics={"n_orders": 5, "sharpe": 1.2},
        manifests_dir=tmp_path,
    )
    with open(p) as f:
        data = json.load(f)
    assert data["metrics"]["n_orders"] == 5


def test_compute_config_hash_is_str():
    h = compute_config_hash({"a": 1, "b": [1, 2, 3]})
    assert isinstance(h, str)
    assert len(h) > 0


def test_compute_config_hash_deterministic():
    cfg = {"key": "value", "n": 42}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2
