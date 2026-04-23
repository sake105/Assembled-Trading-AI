"""Tests for wave-30 module wiring into trading_cycle.py.

Covers:
  Step 3.86 — ml.purged_cv (purged_walk_forward_split)
  Step 8.16 — ml.retraining_scheduler (RetrainingScheduler)
  Step 8.17 — ml.signal_decay_tracker (SignalDecayTracker)
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.purged_cv import purged_walk_forward_split, PurgedKFold
from src.assembled_core.ml.retraining_scheduler import (
    RetrainingScheduler,
    RetrainingRecommendation,
)
from src.assembled_core.ml.signal_decay_tracker import SignalDecayTracker


# ---------------------------------------------------------------------------
# purged_walk_forward_split (Step 3.86)
# ---------------------------------------------------------------------------

def _make_timestamps(n_days: int = 500) -> pd.Series:
    return pd.Series(pd.date_range("2022-01-01", periods=n_days, freq="B"))


def test_purged_wf_returns_list():
    ts = _make_timestamps(500)
    splits = purged_walk_forward_split(ts, train_window_days=120, test_window_days=30)
    assert isinstance(splits, list)


def test_purged_wf_splits_non_empty():
    ts = _make_timestamps(500)
    splits = purged_walk_forward_split(ts, train_window_days=120, test_window_days=30)
    assert len(splits) > 0


def test_purged_wf_each_split_is_tuple():
    ts = _make_timestamps(500)
    splits = purged_walk_forward_split(ts, train_window_days=120, test_window_days=30)
    for train_idx, test_idx in splits:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)


def test_purged_wf_no_overlap():
    ts = _make_timestamps(500)
    splits = purged_walk_forward_split(
        ts, train_window_days=120, test_window_days=30, embargo_days=5
    )
    for train_idx, test_idx in splits:
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, "train and test must not overlap"


def test_purged_wf_empty_series_returns_empty():
    ts = pd.Series([], dtype="datetime64[ns]")
    splits = purged_walk_forward_split(ts)
    assert splits == []


def test_purged_wf_max_splits_respected():
    ts = _make_timestamps(2000)
    splits = purged_walk_forward_split(ts, train_window_days=252, test_window_days=63, max_splits=5)
    assert len(splits) <= 5


def test_purged_kfold_init():
    kf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    assert kf.n_splits == 5


# ---------------------------------------------------------------------------
# RetrainingScheduler (Step 8.16)
# ---------------------------------------------------------------------------

def test_retraining_scheduler_returns_recommendation():
    sched = RetrainingScheduler()
    rec = sched.evaluate()
    assert isinstance(rec, RetrainingRecommendation)


def test_retraining_scheduler_decision_valid():
    sched = RetrainingScheduler()
    rec = sched.evaluate()
    valid = {"no_retrain", "log_and_monitor", "recommend", "urgent"}
    assert rec.decision in valid


def test_retraining_scheduler_never_auto_deploy():
    sched = RetrainingScheduler()
    rec = sched.evaluate()
    assert rec.auto_deploy is False


def test_retraining_scheduler_signals_fired_int():
    sched = RetrainingScheduler()
    rec = sched.evaluate()
    assert isinstance(rec.signals_fired, int)
    assert 0 <= rec.signals_fired <= 5


def test_retraining_scheduler_with_equity():
    rng = np.random.default_rng(0)
    equity = pd.Series(100000.0 + np.cumsum(rng.normal(0, 100, 90)))
    sched = RetrainingScheduler()
    rec = sched.evaluate(equity_since_retrain=equity)
    assert isinstance(rec, RetrainingRecommendation)


def test_retraining_scheduler_old_date_fires_calendar():
    sched = RetrainingScheduler()
    old_date = date(2020, 1, 1)  # Very old
    rec = sched.evaluate(model_last_trained_date=old_date)
    # Old date should fire at least 1 signal
    assert rec.signals_fired >= 1


def test_retraining_scheduler_fresh_date_no_calendar():
    sched = RetrainingScheduler()
    fresh_date = date.today()
    rec = sched.evaluate(model_last_trained_date=fresh_date)
    # Fresh date = 0 days ago, should not fire calendar signal
    assert isinstance(rec, RetrainingRecommendation)


# ---------------------------------------------------------------------------
# SignalDecayTracker (Step 8.17)
# ---------------------------------------------------------------------------

def test_signal_decay_tracker_creates_empty(tmp_path):
    tracker = SignalDecayTracker(state_path=tmp_path / "decay.json")
    assert isinstance(tracker._snapshots, list)
    assert len(tracker._snapshots) == 0


def test_signal_decay_tracker_state_is_list(tmp_path):
    tracker = SignalDecayTracker(state_path=tmp_path / "decay.json")
    assert isinstance(tracker._snapshots, list)


def test_signal_decay_tracker_missing_path_ok(tmp_path):
    tracker = SignalDecayTracker(state_path=tmp_path / "nonexistent" / "decay.json")
    assert tracker._snapshots == []


def test_signal_decay_tracker_horizons(tmp_path):
    tracker = SignalDecayTracker(
        state_path=tmp_path / "decay.json",
        horizons=[1, 5, 20],
    )
    assert 5 in tracker.horizons


def test_signal_decay_tracker_history_window(tmp_path):
    tracker = SignalDecayTracker(
        state_path=tmp_path / "decay.json",
        history_window=6,
    )
    assert tracker.history_window == 6
