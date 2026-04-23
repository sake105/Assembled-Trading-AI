"""Tests for wave-23 module wiring into trading_cycle.py.

Covers:
  Step 2.13 — ml.feature_clustering (cluster_features_by_correlation)
  Step 7.8  — ops.shadow_mode (write_shadow_snapshot / read_shadow_snapshot)
  Step 7.9  — ops.experience_log (append_experience / compute_experience_summary)
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.feature_clustering import (
    cluster_features_by_correlation,
    FeatureClusterResult,
)
from src.assembled_core.ops.shadow_mode import (
    write_shadow_snapshot,
    read_shadow_snapshot,
)
from src.assembled_core.ops.experience_log import (
    append_experience,
    load_experience,
    compute_experience_summary,
)


# ---------------------------------------------------------------------------
# cluster_features_by_correlation (Step 2.13)
# ---------------------------------------------------------------------------

def _make_feature_df(n_rows: int = 80, n_cols: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_cols))
    # Create correlated groups
    data[:, 1] = data[:, 0] + rng.normal(0, 0.1, n_rows)
    data[:, 2] = data[:, 0] + rng.normal(0, 0.1, n_rows)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(n_cols)])


def test_feature_clustering_returns_result():
    df = _make_feature_df()
    result = cluster_features_by_correlation(df)
    assert isinstance(result, FeatureClusterResult)


def test_feature_clustering_n_original_matches():
    df = _make_feature_df(n_cols=8)
    result = cluster_features_by_correlation(df)
    assert result.n_original_features == 8


def test_feature_clustering_n_clusters_leq_original():
    df = _make_feature_df()
    result = cluster_features_by_correlation(df)
    assert result.n_clusters <= result.n_original_features


def test_feature_clustering_correlated_reduces():
    # Highly correlated features should cluster together
    rng = np.random.default_rng(5)
    n = 80
    base = rng.standard_normal(n)
    df = pd.DataFrame({
        "a": base, "b": base + rng.normal(0, 0.01, n),
        "c": base + rng.normal(0, 0.01, n), "d": rng.standard_normal(n),
    })
    result = cluster_features_by_correlation(df)
    assert result.n_clusters < 4


def test_feature_clustering_representatives_valid():
    df = _make_feature_df()
    result = cluster_features_by_correlation(df)
    for rep in result.representatives.values():
        assert rep in df.columns


def test_feature_clustering_get_selected_features():
    df = _make_feature_df()
    result = cluster_features_by_correlation(df)
    selected = result.get_selected_features()
    assert isinstance(selected, list)
    assert len(selected) == result.n_clusters


def test_feature_clustering_single_col_graceful():
    df = pd.DataFrame({"a": np.random.default_rng(0).standard_normal(50)})
    result = cluster_features_by_correlation(df, feature_cols=["a"])
    assert result.n_clusters >= 1


# ---------------------------------------------------------------------------
# write_shadow_snapshot / read_shadow_snapshot (Step 7.8)
# ---------------------------------------------------------------------------

def test_shadow_snapshot_writes_file(tmp_path):
    p = write_shadow_snapshot(
        "test_module",
        {"value": 42, "ok": True},
        snapshot_date=date(2024, 1, 15),
        shadow_root=tmp_path,
    )
    assert p.exists()


def test_shadow_snapshot_readable(tmp_path):
    write_shadow_snapshot(
        "test_module",
        {"x": 1.5},
        snapshot_date=date(2024, 1, 15),
        shadow_root=tmp_path,
    )
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = read_shadow_snapshot(files[0])
    assert data["payload"]["x"] == 1.5


def test_shadow_snapshot_envelope_fields(tmp_path):
    p = write_shadow_snapshot(
        "cycle_meta",
        {"regime": "bull"},
        snapshot_date=date(2024, 3, 1),
        shadow_root=tmp_path,
    )
    data = read_shadow_snapshot(p)
    for key in ["module", "snapshot_date", "written_at", "payload"]:
        assert key in data


def test_shadow_snapshot_invalid_module_raises():
    with pytest.raises(ValueError):
        write_shadow_snapshot("bad/module", {})


def test_shadow_snapshot_run_id_in_filename(tmp_path):
    p = write_shadow_snapshot(
        "mymod",
        {},
        snapshot_date=date(2024, 1, 1),
        run_id="r1",
        shadow_root=tmp_path,
    )
    assert "r1" in p.name


# ---------------------------------------------------------------------------
# append_experience / compute_experience_summary (Step 7.9)
# ---------------------------------------------------------------------------

def test_append_experience_returns_entry(tmp_path):
    log_p = tmp_path / "exp.jsonl"
    entry = {"cycle_date": "2024-01-15", "execution_mode": "paper", "n_orders": 5}
    result = append_experience(entry, log_path=log_p)
    assert isinstance(result, dict)
    assert "timestamp_utc" in result


def test_append_experience_file_created(tmp_path):
    log_p = tmp_path / "exp.jsonl"
    append_experience({"cycle_date": "2024-01-15"}, log_path=log_p)
    assert log_p.exists()


def test_load_experience_returns_df(tmp_path):
    log_p = tmp_path / "exp.jsonl"
    append_experience({"cycle_date": "2024-01-15", "n_orders": 3}, log_path=log_p)
    append_experience({"cycle_date": "2024-01-16", "n_orders": 7}, log_path=log_p)
    df = load_experience(log_path=log_p)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_experience_empty_file_returns_empty_df(tmp_path):
    log_p = tmp_path / "empty.jsonl"
    log_p.touch()
    df = load_experience(log_path=log_p)
    assert isinstance(df, pd.DataFrame)


def test_compute_experience_summary_empty_log(tmp_path):
    log_p = tmp_path / "empty.jsonl"
    log_p.touch()
    summary = compute_experience_summary(log_path=log_p)
    assert summary["total_cycles"] == 0


def test_compute_experience_summary_with_entries(tmp_path):
    log_p = tmp_path / "exp.jsonl"
    for i in range(5):
        append_experience({
            "cycle_date": f"2024-01-{i+1:02d}",
            "broker_equity": 100000.0 + i * 500,
        }, log_path=log_p)
    summary = compute_experience_summary(log_path=log_p)
    assert summary["total_cycles"] == 5
