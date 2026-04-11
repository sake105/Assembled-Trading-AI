"""Tests for portfolio/bl_sizing.py (Sprint 3 / Plan W11)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.portfolio.bl_sizing import (  # noqa: E402
    apply_bl_sizing,
    apply_bl_sizing_from_policy,
)

# BL optimiser needs scipy.optimize.
pytest.importorskip("scipy")


def _panel(symbols: list[str], n_bars: int = 120, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0005, 0.012 + 0.002 * i, size=n_bars)
        px = 100.0 * np.cumprod(1.0 + rets)
        for b in range(n_bars):
            rows.append({
                "timestamp": base + pd.Timedelta(days=b),
                "symbol": sym,
                "close": float(px[b]),
            })
    return pd.DataFrame(rows)


def test_empty_scores_return_empty() -> None:
    adjusted, reasons = apply_bl_sizing({}, pd.DataFrame())
    assert adjusted == {}
    assert reasons == []


def test_fallback_to_score_on_missing_prices() -> None:
    scores = {"AAA": 0.4, "BBB": 0.2}
    adjusted, reasons = apply_bl_sizing(scores, pd.DataFrame(), target_invested_pct=1.0)
    assert set(adjusted.keys()) == {"AAA", "BBB"}
    total = sum(adjusted.values())
    assert abs(total - 1.0) < 1e-9
    assert any("insufficient" in r or "falling back" in r for r in reasons)


def test_bl_sizing_produces_scaled_weights() -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}

    adjusted, reasons = apply_bl_sizing(
        scores,
        prices,
        lookback_days=100,
        target_invested_pct=0.75,
        max_position=0.35,
    )
    assert set(adjusted.keys()) == set(symbols)
    total = sum(abs(w) for w in adjusted.values())
    assert abs(total - 0.75) < 1e-6
    assert any("BL posterior" in r for r in reasons)


def test_from_policy_disabled_returns_copy() -> None:
    scores = {"AAA": 0.5, "BBB": 0.5}
    adjusted, reasons = apply_bl_sizing_from_policy(
        scores, pd.DataFrame(), {"bl_sizing": {"enabled": False}}
    )
    assert adjusted == scores
    assert adjusted is not scores
    assert reasons == []


def test_from_policy_enabled_applies() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
    policy = {
        "bl_sizing": {
            "enabled": True,
            "lookback_days": 100,
            "target_invested_pct": 0.9,
            "max_position": 0.5,
        }
    }
    adjusted, reasons = apply_bl_sizing_from_policy(scores, prices, policy)
    total = sum(adjusted.values())
    assert abs(total - 0.9) < 1e-6
    assert any("BL posterior" in r for r in reasons)
