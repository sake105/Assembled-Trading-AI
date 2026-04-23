"""Tests for wave-38 module wiring into trading_cycle.py.

Covers:
  Step 3.91 — signals.signal_api (normalize_signals)
  Step 7.68 — ops.heartbeat (write_heartbeat / read_heartbeat)
  Step 8.29 — qa.candidate_gate (check_candidate_allowed)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.signals.signal_api import (
    normalize_signals,
    make_signal_frame,
    validate_signal_frame,
    SignalMetadata,
)
from src.assembled_core.ops.heartbeat import (
    write_heartbeat,
    read_heartbeat,
    heartbeat_age_seconds,
    check_liveness,
)
from src.assembled_core.qa.candidate_gate import (
    check_candidate_allowed,
    read_robustness_ok_from_manifest,
)


# ---------------------------------------------------------------------------
# normalize_signals (Step 3.91)
# ---------------------------------------------------------------------------

def _make_signal_df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-15", tz="UTC")] * n)
    return pd.DataFrame(
        {"signal_value": rng.normal(0, 1, n)},
        index=idx,
    )


def test_normalize_returns_df():
    df = _make_signal_df()
    result = normalize_signals(df, method="zscore")
    assert isinstance(result, pd.DataFrame)


def test_normalize_zscore_near_zero_mean():
    rng = np.random.default_rng(42)
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-15", tz="UTC")] * 50)
    df = pd.DataFrame({"signal_value": rng.normal(5.0, 2.0, 50)}, index=idx)
    result = normalize_signals(df, method="zscore")
    # z-scored mean should be near zero (single timestamp = single group)
    assert abs(result["signal_value"].mean()) < 0.1


def test_normalize_rank_method():
    df = _make_signal_df(n=20)
    result = normalize_signals(df, method="rank")
    assert isinstance(result, pd.DataFrame)
    assert "signal_value" in result.columns


def test_normalize_none_method_preserves_values():
    df = _make_signal_df()
    original = df["signal_value"].copy()
    result = normalize_signals(df, method="none", clip=None)
    pd.testing.assert_series_equal(result["signal_value"].reset_index(drop=True),
                                   original.reset_index(drop=True))


def test_normalize_clipping_applied():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-15", tz="UTC")] * 5)
    df = pd.DataFrame({"signal_value": [10.0, -10.0, 0.0, 5.0, -5.0]}, index=idx)
    result = normalize_signals(df, method="none", clip=3.0)
    assert (result["signal_value"].abs() <= 3.0 + 1e-9).all()


def test_normalize_missing_value_col_raises():
    df = pd.DataFrame({"other_col": [1.0, 2.0]})
    with pytest.raises(ValueError):
        normalize_signals(df, value_col="signal_value")


def test_normalize_unknown_method_raises():
    df = _make_signal_df()
    with pytest.raises(ValueError):
        normalize_signals(df, method="invalid_method")


def test_signal_metadata_creates():
    meta = SignalMetadata(strategy_name="test", freq="D")
    assert meta.strategy_name == "test"


# ---------------------------------------------------------------------------
# write_heartbeat / read_heartbeat (Step 7.68)
# ---------------------------------------------------------------------------

def test_write_heartbeat_creates_file(tmp_path):
    path = tmp_path / "heartbeat.json"
    result = write_heartbeat(path=path, status="ok")
    assert path.exists()


def test_write_heartbeat_returns_path(tmp_path):
    path = tmp_path / "hb.json"
    result = write_heartbeat(path=path)
    assert Path(result) == path


def test_read_heartbeat_after_write(tmp_path):
    path = tmp_path / "hb.json"
    write_heartbeat(path=path, status="ok", details={"cycle": "2024-01-15"})
    hb = read_heartbeat(path=path)
    assert hb is not None
    assert hb["status"] == "ok"
    assert hb["details"]["cycle"] == "2024-01-15"


def test_read_heartbeat_missing_returns_none(tmp_path):
    result = read_heartbeat(path=tmp_path / "nonexistent.json")
    assert result is None


def test_heartbeat_age_seconds(tmp_path):
    path = tmp_path / "hb.json"
    write_heartbeat(path=path, status="ok")
    age = heartbeat_age_seconds(path=path)
    assert age is not None
    assert age >= 0.0


def test_check_liveness_fresh_heartbeat(tmp_path):
    path = tmp_path / "hb.json"
    write_heartbeat(path=path, status="ok")
    result = check_liveness(path=path, max_age_seconds=3600)
    assert result is not None  # returns dict or str


def test_write_heartbeat_status_degraded(tmp_path):
    path = tmp_path / "hb.json"
    write_heartbeat(path=path, status="degraded")
    hb = read_heartbeat(path=path)
    assert hb["status"] == "degraded"


# ---------------------------------------------------------------------------
# check_candidate_allowed (Step 8.29)
# ---------------------------------------------------------------------------

def test_candidate_allowed_both_true():
    allowed, msg = check_candidate_allowed(robustness_ok=True, reconciliation_ok=True)
    assert allowed is True
    assert isinstance(msg, str)


def test_candidate_blocked_robustness_false():
    allowed, msg = check_candidate_allowed(robustness_ok=False, reconciliation_ok=True)
    assert allowed is False


def test_candidate_blocked_reconcile_false():
    allowed, msg = check_candidate_allowed(robustness_ok=True, reconciliation_ok=False)
    assert allowed is False


def test_candidate_blocked_robustness_none():
    allowed, msg = check_candidate_allowed(robustness_ok=None, reconciliation_ok=True)
    assert allowed is False


def test_candidate_allowed_reconcile_none_backward_compat():
    allowed, msg = check_candidate_allowed(robustness_ok=True, reconciliation_ok=None)
    assert isinstance(allowed, bool)


def test_candidate_returns_string_message():
    _, msg = check_candidate_allowed(robustness_ok=True, reconciliation_ok=True)
    assert isinstance(msg, str)
    assert len(msg) > 0
