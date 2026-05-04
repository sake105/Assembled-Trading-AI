"""Tests for correlation_guard tail-dependence tightening (Sprint 3 / C8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.correlation_guard import apply_correlation_guard

# copula tail-dep uses the Clayton MLE which requires scipy.optimize; skip
# when scipy is not installed so the CI-critical lane stays green.
scipy = pytest.importorskip("scipy")


def _build_prices(
    symbols: list[str], n_bars: int = 120, seed: int = 11, tight: bool = True
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    if tight:
        # Common driver → strong correlation AND tail co-movement
        driver = rng.normal(0, 0.02, size=n_bars)
        for sym in symbols:
            idio = rng.normal(0, 0.003, size=n_bars)
            rets = driver + idio
            px = 100.0 * np.cumprod(1.0 + rets)
            for b in range(n_bars):
                rows.append(
                    {
                        "timestamp": base + pd.Timedelta(days=b),
                        "symbol": sym,
                        "close": float(px[b]),
                    }
                )
    else:
        for sym in symbols:
            rets = rng.normal(0, 0.01, size=n_bars)
            px = 100.0 * np.cumprod(1.0 + rets)
            for b in range(n_bars):
                rows.append(
                    {
                        "timestamp": base + pd.Timedelta(days=b),
                        "symbol": sym,
                        "close": float(px[b]),
                    }
                )
    return pd.DataFrame(rows)


def test_tail_dep_disabled_by_default() -> None:
    syms = ["AAA", "BBB", "CCC"]
    prices = _build_prices(syms, tight=True)
    weights = {s: 0.33 for s in syms}
    policy = {
        "correlation_guard": {
            "enabled": True,
            "threshold": 0.70,
            "max_cluster_weight": 0.4,
            "lookback_days": 60,
        }
    }
    adjusted, reasons = apply_correlation_guard(weights, prices, policy)
    # No tail_dependence block → reasons should not mention tail_dep
    assert not any("tail_dep" in r for r in reasons)
    assert not any("avg_lower_tail_dep" in r for r in reasons)


def test_tail_dep_forced_trigger_tightens_cluster_cap() -> None:
    """When trigger is set below 0 it always fires, proving the code path
    actually tightens max_cluster_weight. Using trigger=-1.0 guarantees
    avg_td > trigger regardless of the underlying Clayton fit result."""
    syms = ["AAA", "BBB", "CCC"]
    prices = _build_prices(syms, tight=True)
    weights = {s: 0.33 for s in syms}

    # Baseline: no tail-dep tightening → cluster cap is 0.80, sum=0.99 → scale=0.808
    policy_base = {
        "correlation_guard": {
            "enabled": True,
            "threshold": 0.70,
            "max_cluster_weight": 0.80,
            "lookback_days": 60,
        }
    }
    _, reasons_base = apply_correlation_guard(weights, prices, policy_base)
    base_scale = None
    for r in reasons_base:
        if "scaled by" in r:
            base_scale = float(r.split("scaled by ")[-1])
            break

    # With tail-dep enabled + trigger=-1 (always fires) + factor=2 → cap halves to 0.40
    policy_tight = {
        "correlation_guard": {
            "enabled": True,
            "threshold": 0.70,
            "max_cluster_weight": 0.80,
            "lookback_days": 60,
            "tail_dependence": {
                "enabled": True,
                "trigger": -1.0,
                "tightening_factor": 2.0,
            },
        }
    }
    adjusted, reasons_tight = apply_correlation_guard(weights, prices, policy_tight)
    assert any("avg_lower_tail_dep" in r for r in reasons_tight)
    # Tightened scale must be strictly smaller than the baseline scale
    tight_scale = None
    for r in reasons_tight:
        if "scaled by" in r:
            tight_scale = float(r.split("scaled by ")[-1])
            break
    assert base_scale is not None and tight_scale is not None
    assert tight_scale < base_scale


def test_tail_dep_respects_disabled_guard() -> None:
    syms = ["AAA", "BBB"]
    prices = _build_prices(syms, tight=True)
    weights = {s: 0.5 for s in syms}
    policy = {
        "correlation_guard": {
            "enabled": False,
            "tail_dependence": {
                "enabled": True,
                "trigger": 0.0,
                "tightening_factor": 2.0,
            },
        }
    }
    adjusted, reasons = apply_correlation_guard(weights, prices, policy)
    assert adjusted == weights
    assert reasons == []
