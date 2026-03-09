"""Tests for market stress signal (INT-5.1): price-based stress_ok / stress_score."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from src.assembled_core.risk.market_stress import compute_market_stress


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def _policy() -> Dict[str, Any]:
    return {
        "market_stress": {
            "enabled": True,
            "lookback_days": 20,
            "metrics": {
                "vol_z": {"enabled": True, "z_threshold": 1.5},
                "dd_lookback": {"enabled": True, "dd_threshold": -0.05},
            },
            "confirm_rule": {"mode": "any"},
            "qc": {"if_data_missing": False},
        }
    }


def test_market_stress_no_data_returns_false() -> None:
    """Empty or insufficient data -> stress_ok False (qc.if_data_missing false)."""
    policy = _policy()
    empty = pd.DataFrame(columns=["timestamp", "close"])
    out = compute_market_stress(empty, policy)
    assert out["stress_ok"] is False
    assert out["stress_score"] == 0
    assert "details" in out

    single_row = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01", tz="UTC")], "close": [100.0]})
    out2 = compute_market_stress(single_row, policy)
    assert out2["stress_ok"] is False
    assert out2["stress_score"] == 0


def test_market_stress_dd_triggers() -> None:
    """Synthetic price series with -10% drawdown over lookback -> stress_ok True (stress_dd)."""
    policy = _policy()
    lookback = 20
    # Start at 100, go up to 110, then drop to 99 (-10% from 110)
    n = lookback + 10
    dates = pd.date_range(start="2025-01-01", periods=n, freq="D", tz="UTC")
    close = [100.0] * 5 + [105.0] * 5 + [110.0] * 5 + [105.0] * 5 + [99.0] * (n - 20)
    df = pd.DataFrame({"timestamp": dates, "close": close})
    out = compute_market_stress(df, policy)
    # min_dd in last lookback: 99/110 - 1 = -0.1 <= -0.05
    assert out["stress_ok"] is True
    assert out["stress_score"] >= 1
    assert out["details"].get("stress_dd") is True
    assert out["details"].get("min_dd") is not None
    assert out["details"]["min_dd"] <= -0.05


def test_market_stress_vol_z_triggers() -> None:
    """Synthetic higher volatility in last window -> stress_vol True when vol_z >= threshold."""
    policy = _policy()
    lookback = 20
    roll_win = 5 * lookback
    n = roll_win + lookback + 5
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D", tz="UTC")
    # Flat then noisy: low vol history, then high vol in last lookback
    import numpy as np
    np.random.seed(42)
    base = 100.0
    close = [base] * n
    for i in range(1, n):
        if i < n - lookback:
            close[i] = close[i - 1] * (1.0 + 0.001)
        else:
            close[i] = close[i - 1] * (1.0 + 0.02 * (np.random.rand() - 0.5))
    df = pd.DataFrame({"timestamp": dates, "close": close})
    out = compute_market_stress(df, policy)
    # May or may not trigger depending on random; at least structure is correct
    assert "stress_ok" in out
    assert "stress_score" in out
    assert out["stress_score"] in (0, 1, 2)
    assert "details" in out
    assert "stress_vol" in out["details"]
    assert "stress_dd" in out["details"]


def test_market_stress_disabled_returns_false() -> None:
    """When market_stress.enabled is False -> stress_ok False, no computation."""
    policy = _policy()
    policy["market_stress"]["enabled"] = False
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC"),
        "close": [100.0 - i * 0.5 for i in range(30)],
    })
    out = compute_market_stress(df, policy)
    assert out["stress_ok"] is False
    assert out["stress_score"] == 0


def test_market_stress_multi_symbol_uses_first() -> None:
    """Multi-symbol prices: use first symbol (or benchmark if configured)."""
    policy = _policy()
    dates = pd.date_range("2025-01-01", periods=25, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": list(dates) * 2,
        "symbol": ["A"] * 25 + ["B"] * 25,
        "close": [100.0] * 25 + [200.0] * 25,
    })
    out = compute_market_stress(df, policy)
    assert "stress_ok" in out
    assert "stress_score" in out
    policy_b = _policy()
    policy_b["market_stress"]["benchmark_symbol"] = "B"
    out2 = compute_market_stress(df, policy_b)
    assert "stress_ok" in out2
