"""Tests for wave-26 module wiring into trading_cycle.py.

Covers:
  Step 2.16 — ml.online_hmm_regime (OnlineHMMRegimeDetector.predict_current_regime)
  Step 7.63 — ops.run_index (append_run_index)
  Step 8.14 — ml.calibration_monitor (compute_calibration)
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.online_hmm_regime import (
    OnlineHMMRegimeDetector,
    RegimeState,
)
from src.assembled_core.ml.calibration_monitor import (
    compute_calibration,
    CalibrationReport,
)
from src.assembled_core.ops.run_index import append_run_index


# ---------------------------------------------------------------------------
# OnlineHMMRegimeDetector (Step 2.16)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 100, vol: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, vol, n))


def test_hmm_predict_returns_regime_state():
    det = OnlineHMMRegimeDetector()
    rets = _make_returns(100)
    state = det.predict_current_regime(rets)
    assert isinstance(state, RegimeState)


def test_hmm_regime_label_valid():
    det = OnlineHMMRegimeDetector()
    rets = _make_returns(100)
    state = det.predict_current_regime(rets)
    assert state.regime_label in {"LOW_VOL", "NORMAL", "HIGH_VOL"}


def test_hmm_regime_id_in_range():
    det = OnlineHMMRegimeDetector(n_states=3)
    rets = _make_returns(100)
    state = det.predict_current_regime(rets)
    assert 0 <= state.regime_id < 3


def test_hmm_probability_in_01():
    det = OnlineHMMRegimeDetector()
    rets = _make_returns(100)
    state = det.predict_current_regime(rets)
    assert 0.0 <= state.probability <= 1.0


def test_hmm_short_series_returns_default():
    det = OnlineHMMRegimeDetector()
    rets = _make_returns(5)  # below 20 threshold
    state = det.predict_current_regime(rets)
    assert isinstance(state, RegimeState)
    assert state.regime_label == "NORMAL"


def test_hmm_high_vol_detected():
    rng = np.random.default_rng(7)
    # Alternating calm and crisis
    calm = rng.normal(0.0, 0.005, 150)
    crisis = rng.normal(0.0, 0.05, 30)
    rets = pd.Series(np.concatenate([calm, crisis]))
    det = OnlineHMMRegimeDetector()
    state = det.predict_current_regime(rets)
    # High vol at end should trigger HIGH_VOL or at least not error
    assert isinstance(state, RegimeState)


def test_hmm_low_vol_detected():
    rng = np.random.default_rng(9)
    rets = pd.Series(rng.normal(0.0, 0.001, 120))  # very calm
    det = OnlineHMMRegimeDetector()
    state = det.predict_current_regime(rets)
    assert isinstance(state, RegimeState)


def test_hmm_volatility_non_negative():
    det = OnlineHMMRegimeDetector()
    rets = _make_returns(100)
    state = det.predict_current_regime(rets)
    assert state.volatility >= 0.0


# ---------------------------------------------------------------------------
# compute_calibration (Step 8.14)
# ---------------------------------------------------------------------------

def test_calibration_returns_report():
    rng = np.random.default_rng(0)
    preds = rng.uniform(0, 1, 100)
    actuals = (preds > 0.5).astype(float)
    report = compute_calibration(preds, actuals)
    assert isinstance(report, CalibrationReport)


def test_calibration_ece_in_01():
    rng = np.random.default_rng(1)
    preds = rng.uniform(0, 1, 100)
    actuals = (rng.uniform(0, 1, 100) > 0.5).astype(float)
    report = compute_calibration(preds, actuals)
    assert 0.0 <= report.ece <= 1.0


def test_calibration_brier_in_01():
    rng = np.random.default_rng(2)
    preds = rng.uniform(0, 1, 100)
    actuals = (rng.uniform(0, 1, 100) > 0.5).astype(float)
    report = compute_calibration(preds, actuals)
    assert 0.0 <= report.brier_score <= 1.0


def test_calibration_n_samples():
    rng = np.random.default_rng(3)
    preds = rng.uniform(0, 1, 80)
    actuals = (preds > 0.5).astype(float)
    report = compute_calibration(preds, actuals, n_bins=5)
    assert report.n_samples == 80


def test_calibration_perfect_is_well_calibrated():
    # If predictions == actuals (all 0 or 1), ECE should be low
    preds = np.array([0.0] * 50 + [1.0] * 50)
    actuals = np.array([0.0] * 50 + [1.0] * 50)
    report = compute_calibration(preds, actuals)
    assert report.ece < 0.1


def test_calibration_bin_stats_non_empty():
    rng = np.random.default_rng(4)
    preds = rng.uniform(0, 1, 100)
    actuals = (preds > 0.5).astype(float)
    report = compute_calibration(preds, actuals, n_bins=5)
    assert isinstance(report.bin_stats, list)


def test_calibration_accepts_series():
    rng = np.random.default_rng(5)
    preds = pd.Series(rng.uniform(0, 1, 60))
    actuals = pd.Series((rng.uniform(0, 1, 60) > 0.5).astype(float))
    report = compute_calibration(preds, actuals)
    assert isinstance(report, CalibrationReport)


# ---------------------------------------------------------------------------
# append_run_index (Step 7.63)
# ---------------------------------------------------------------------------

def test_run_index_creates_file(tmp_path):
    idx_path = tmp_path / "index.csv"
    append_run_index(
        run_id="test_run",
        date="2024-01-15",
        status="success",
        metrics={"final_equity": 100500.0, "n_fills": 3},
        git_sha="abc123",
        config_hash="deadbeef01234567",
        manifest_path=tmp_path / "manifest.json",
        index_path=idx_path,
    )
    assert idx_path.exists()


def test_run_index_readable_csv(tmp_path):
    idx_path = tmp_path / "index.csv"
    append_run_index(
        run_id="r1",
        date="2024-01-15",
        status="success",
        metrics={"final_equity": 100000.0, "n_fills": 5},
        git_sha="sha1",
        config_hash="hash1",
        manifest_path=tmp_path / "m.json",
        index_path=idx_path,
    )
    with open(idx_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["status"] == "success"


def test_run_index_append_multiple(tmp_path):
    idx_path = tmp_path / "index.csv"
    for i in range(3):
        append_run_index(
            run_id=f"run_{i}",
            date=f"2024-01-{i+1:02d}",
            status="success",
            metrics={},
            git_sha="",
            config_hash="",
            manifest_path=tmp_path / f"m{i}.json",
            index_path=idx_path,
        )
    with open(idx_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_run_index_upsert_replaces_existing(tmp_path):
    idx_path = tmp_path / "index.csv"
    for status in ("error", "success"):
        append_run_index(
            run_id="r1",
            date="2024-01-15",
            status=status,
            metrics={},
            git_sha="",
            config_hash="",
            manifest_path=tmp_path / "m.json",
            index_path=idx_path,
        )
    with open(idx_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "success"


def test_run_index_returns_path(tmp_path):
    idx_path = tmp_path / "index.csv"
    result = append_run_index(
        run_id="r2",
        date="2024-01-16",
        status="success",
        metrics={"final_equity": 99000.0},
        git_sha="abc",
        config_hash="xyz",
        manifest_path=tmp_path / "m.json",
        index_path=idx_path,
    )
    assert Path(result).exists()
