"""Tests for EVT Tail-VaR wrapper in risk_metrics (Sprint 3 / C9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.risk_metrics import compute_evt_tail_var

# scipy is an optional dependency; skip when unavailable so the CI-critical
# lane stays green on minimal environments.
scipy = pytest.importorskip("scipy")


def _fat_tail_returns(n: int = 2000, seed: int = 7) -> pd.Series:
    """Student-t-like returns with heavy left tail."""
    rng = np.random.default_rng(seed)
    # Student-t with df=3 → fat tails
    x = rng.standard_t(df=3, size=n) * 0.01
    return pd.Series(x)


def test_fat_tail_returns_yield_positive_tail_var() -> None:
    r = _fat_tail_returns()
    metrics = compute_evt_tail_var(r)
    assert metrics["evt_var_95"] > 0
    assert metrics["evt_var_99"] >= metrics["evt_var_95"]
    assert metrics["evt_var_999"] >= metrics["evt_var_99"]
    assert metrics["evt_cvar_99"] >= metrics["evt_var_99"]


def test_insufficient_data_returns_zeros() -> None:
    r = pd.Series([0.01, -0.02, 0.005])  # way below min_exceedances
    metrics = compute_evt_tail_var(r)
    assert metrics["evt_var_95"] == 0.0
    assert metrics["evt_var_99"] == 0.0
    assert metrics["evt_shape_xi"] == 0.0


def test_metric_schema_keys_stable() -> None:
    r = _fat_tail_returns()
    metrics = compute_evt_tail_var(r)
    expected = {
        "evt_var_95",
        "evt_var_99",
        "evt_var_999",
        "evt_cvar_95",
        "evt_cvar_99",
        "evt_shape_xi",
        "evt_return_period_100y",
    }
    assert set(metrics.keys()) == expected


def test_accepts_numpy_array() -> None:
    r = _fat_tail_returns().to_numpy()
    metrics = compute_evt_tail_var(r)
    assert isinstance(metrics, dict)
    assert metrics["evt_var_95"] > 0
