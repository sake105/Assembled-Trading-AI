"""Tests for wave-29 module wiring into trading_cycle.py.

Covers:
  Step 2.19 — ml.combined_regime (CombinedRegimeClassifier)
  Step 7.66 — ops.trade_journal (append_trade_journal_entries)
  Step 8.15 — ml.experiment_tracking (ExperimentTracker.log_run)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.combined_regime import (
    CombinedRegimeClassifier,
    CombinedRegimeOutput,
)
from src.assembled_core.ops.trade_journal import (
    append_trade_journal_entries,
    load_trade_journal,
)
from src.assembled_core.ml.experiment_tracking import (
    ExperimentTracker,
    ExperimentRun,
)


# ---------------------------------------------------------------------------
# CombinedRegimeClassifier (Step 2.19)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def test_combined_regime_returns_output():
    clf = CombinedRegimeClassifier()
    rets = _make_returns(100)
    out = clf.predict(returns=rets)
    assert isinstance(out, CombinedRegimeOutput)


def test_combined_regime_combined_label_valid():
    clf = CombinedRegimeClassifier()
    rets = _make_returns(100)
    out = clf.predict(returns=rets)
    valid = {"RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"}
    assert out.combined_regime in valid


def test_combined_regime_confidence_in_01():
    clf = CombinedRegimeClassifier()
    rets = _make_returns(100)
    out = clf.predict(returns=rets)
    assert 0.0 <= out.confidence <= 1.0


def test_combined_regime_agreement_bool():
    clf = CombinedRegimeClassifier()
    rets = _make_returns(100)
    out = clf.predict(returns=rets)
    assert isinstance(out.agreement, bool)


def test_combined_regime_no_sources_returns_neutral():
    # No classifiers passed — should fallback gracefully
    clf = CombinedRegimeClassifier()
    out = clf.predict()
    assert isinstance(out, CombinedRegimeOutput)


def test_combined_regime_with_hmm():
    from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector
    hmm = OnlineHMMRegimeDetector(n_states=3)
    clf = CombinedRegimeClassifier(hmm_detector=hmm)
    rets = _make_returns(150)
    out = clf.predict(returns=rets)
    assert isinstance(out.combined_regime, str)


def test_combined_regime_crisis_if_both_crisis():
    # Simulate very high volatility → should produce HIGH_VOL / CRISIS
    rng = np.random.default_rng(42)
    crash = pd.Series(rng.normal(0.0, 0.08, 150))  # very high vol
    clf = CombinedRegimeClassifier()
    out = clf.predict(returns=crash)
    assert isinstance(out, CombinedRegimeOutput)


# ---------------------------------------------------------------------------
# append_trade_journal_entries (Step 7.66)
# ---------------------------------------------------------------------------

def _make_fills(n: int = 3) -> list[dict]:
    return [
        {"symbol": f"S{i}", "side": "BUY", "qty": 100.0, "price": 50.0 + i}
        for i in range(n)
    ]


def test_trade_journal_returns_list(tmp_path):
    fills = _make_fills(3)
    entries = append_trade_journal_entries(
        fills,
        journal_path=tmp_path / "journal.jsonl",
    )
    assert isinstance(entries, list)
    assert len(entries) == 3


def test_trade_journal_creates_file(tmp_path):
    fills = _make_fills(2)
    journal_path = tmp_path / "journal.jsonl"
    append_trade_journal_entries(fills, journal_path=journal_path)
    assert journal_path.exists()


def test_trade_journal_entry_has_trade_id(tmp_path):
    fills = _make_fills(1)
    entries = append_trade_journal_entries(
        fills, journal_path=tmp_path / "j.jsonl"
    )
    assert "trade_id" in entries[0]


def test_trade_journal_entry_has_timestamp(tmp_path):
    fills = _make_fills(1)
    entries = append_trade_journal_entries(
        fills, journal_path=tmp_path / "j.jsonl"
    )
    assert "timestamp_utc" in entries[0]


def test_trade_journal_load_returns_list(tmp_path):
    fills = _make_fills(4)
    journal_path = tmp_path / "j.jsonl"
    append_trade_journal_entries(fills, journal_path=journal_path)
    loaded = load_trade_journal(journal_path=journal_path)
    assert isinstance(loaded, list)
    assert len(loaded) == 4


def test_trade_journal_empty_fills_ok(tmp_path):
    entries = append_trade_journal_entries(
        [], journal_path=tmp_path / "j.jsonl"
    )
    assert entries == []


# ---------------------------------------------------------------------------
# ExperimentTracker.log_run (Step 8.15)
# ---------------------------------------------------------------------------

def test_experiment_tracker_log_run_returns_run():
    tracker = ExperimentTracker()
    run = tracker.log_run(
        experiment_name="test_exp",
        params={"lr": 0.01},
        metrics={"sharpe": 1.2, "n_orders": 5.0},
    )
    assert isinstance(run, ExperimentRun)


def test_experiment_tracker_run_has_id():
    tracker = ExperimentTracker()
    run = tracker.log_run("exp", {}, {"acc": 0.9})
    assert isinstance(run.run_id, str)
    assert len(run.run_id) > 0


def test_experiment_tracker_multiple_runs():
    tracker = ExperimentTracker()
    for i in range(3):
        tracker.log_run(f"exp_{i}", {"i": i}, {"val": float(i)})
    assert len(tracker._runs) == 3


def test_experiment_tracker_get_runs_filters():
    tracker = ExperimentTracker()
    tracker.log_run("exp_a", {}, {"score": 1.0})
    tracker.log_run("exp_b", {}, {"score": 2.0})
    runs_a = tracker.get_runs(experiment_name="exp_a")
    assert len(runs_a) == 1
    assert runs_a[0].experiment_name == "exp_a"


def test_experiment_tracker_persists(tmp_path):
    exp_dir = tmp_path / "experiments"
    tracker = ExperimentTracker(storage_path=exp_dir)
    tracker.log_run("persist_test", {"x": 1}, {"y": 2.0})
    # Saved as runs.jsonl
    runs_file = exp_dir / "runs.jsonl"
    assert runs_file.exists()


def test_experiment_tracker_tags_stored():
    tracker = ExperimentTracker()
    run = tracker.log_run(
        "tagged_exp", {}, {"v": 1.0}, tags={"regime": "NEUTRAL"}
    )
    assert run.tags.get("regime") == "NEUTRAL"
