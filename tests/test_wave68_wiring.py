"""Tests for wave-68 module wiring into trading_cycle.py.

Covers:
  Step 2.41 — data.pit_guard (PITGuard / PITViolationError)
  Step 2.42 — data.realism_meta (build_realism_label)
  Step 2.43 — data.latency (apply_source_latency / filter_events_as_of)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.assembled_core.data.pit_guard import PITGuard, PITViolationError
from src.assembled_core.data.realism_meta import build_realism_label
from src.assembled_core.data.latency import apply_source_latency, filter_events_as_of


# ---------------------------------------------------------------------------
# pit_guard (Step 2.41)
# ---------------------------------------------------------------------------

def test_pit_guard_creates():
    guard = PITGuard(as_of=pd.Timestamp("2024-06-01", tz="UTC"))
    assert isinstance(guard, PITGuard)


def test_pit_guard_warn_mode():
    guard = PITGuard(as_of=pd.Timestamp("2024-06-01", tz="UTC"), mode="warn")
    assert guard.mode == "warn"


def test_pit_guard_validate_empty():
    guard = PITGuard(as_of=pd.Timestamp("2024-06-01", tz="UTC"), mode="warn")
    df = pd.DataFrame(columns=["timestamp"])
    result = guard.validate(df)
    assert result is True


def test_pit_guard_validate_clean():
    guard = PITGuard(as_of=pd.Timestamp("2024-06-01", tz="UTC"), mode="assert")
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-05-01", "2024-05-15"]).tz_localize("UTC")})
    result = guard.validate(df)
    assert result is True


def test_pit_guard_violation_warn_mode():
    guard = PITGuard(as_of=pd.Timestamp("2024-01-01", tz="UTC"), mode="warn")
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]).tz_localize("UTC")})
    # warn mode should not raise
    result = guard.validate(df)
    assert result is False


# ---------------------------------------------------------------------------
# realism_meta (Step 2.42)
# ---------------------------------------------------------------------------

def test_build_realism_label_returns_dict():
    label = build_realism_label()
    assert isinstance(label, dict)


def test_build_realism_label_fields():
    label = build_realism_label(
        calendar_mode="nyse",
        cost_model_mode="policy",
        data_source="synthetic",
    )
    assert "data_source" in label
    assert label["data_source"] == "synthetic"


def test_build_realism_label_realism_level():
    label = build_realism_label()
    assert "realism_level" in label
    assert isinstance(label["realism_level"], str)


def test_build_realism_label_synthetic_is_low():
    label = build_realism_label(data_source="synthetic")
    assert label["realism_level"] in ("low", "minimal", "none", "synthetic")


# ---------------------------------------------------------------------------
# latency (Step 2.43)
# ---------------------------------------------------------------------------

def test_apply_source_latency_empty():
    df = pd.DataFrame(columns=["timestamp"])
    result = apply_source_latency(df)
    assert isinstance(result, pd.DataFrame)


def test_apply_source_latency_adds_disclosure_date():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-05-01", "2024-05-02"]).tz_localize("UTC"),
        "source": ["finnhub", "edgar"],
    })
    result = apply_source_latency(df, days=1)
    assert isinstance(result, pd.DataFrame)
    assert "disclosure_date" in result.columns


def test_apply_source_latency_shifts_by_days():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-05-01"]).tz_localize("UTC"),
    })
    result = apply_source_latency(df, days=2)
    assert result["disclosure_date"].iloc[0] >= pd.Timestamp("2024-05-01", tz="UTC")


def test_filter_events_as_of_empty():
    df = pd.DataFrame(columns=["timestamp"])
    result = filter_events_as_of(df, as_of=pd.Timestamp("2024-06-01", tz="UTC"))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
