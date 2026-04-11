"""Tests for portfolio/hrp_sizing.py (Sprint 3 / Plan W10)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.portfolio.hrp_sizing import (  # noqa: E402
    apply_hrp_sizing,
    apply_hrp_sizing_from_policy,
)

# HRP uses scipy linkage; skip when missing.
pytest.importorskip("scipy")


def _panel(symbols: list[str], n_bars: int = 120, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    driver = rng.normal(0.0, 0.01, size=n_bars)
    for i, sym in enumerate(symbols):
        idio = rng.normal(0.0, 0.005 + 0.001 * i, size=n_bars)
        rets = driver + idio
        px = 100.0 * np.cumprod(1.0 + rets)
        for b in range(n_bars):
            rows.append({
                "timestamp": base + pd.Timedelta(days=b),
                "symbol": sym,
                "close": float(px[b]),
            })
    return pd.DataFrame(rows)


def test_empty_scores_return_empty() -> None:
    adjusted, reasons = apply_hrp_sizing({}, pd.DataFrame())
    assert adjusted == {}
    assert reasons == []


def test_insufficient_prices_falls_back_to_score() -> None:
    scores = {"AAA": 0.4, "BBB": 0.2}
    # No price panel at all → fallback to score, scaled to target_invested_pct=1
    adjusted, reasons = apply_hrp_sizing(scores, pd.DataFrame())
    assert set(adjusted.keys()) == {"AAA", "BBB"}
    total = sum(adjusted.values())
    assert abs(total - 1.0) < 1e-9
    assert any("insufficient" in r for r in reasons)


def test_hrp_blend_produces_valid_weights() -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}

    adjusted, reasons = apply_hrp_sizing(
        scores,
        prices,
        lookback_days=100,
        blend=0.7,
        target_invested_pct=0.8,
    )
    assert set(adjusted.keys()) == set(symbols)
    total = sum(adjusted.values())
    assert abs(total - 0.8) < 1e-6
    for w in adjusted.values():
        assert w >= 0.0
    assert any("blended HRP" in r for r in reasons)


def test_blend_zero_equals_score_only() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}

    adjusted, _ = apply_hrp_sizing(
        scores,
        prices,
        blend=0.0,
        target_invested_pct=1.0,
    )
    # With blend=0 we should recover normalized score weights
    for s in symbols:
        assert abs(adjusted[s] - scores[s]) < 1e-6


def test_from_policy_disabled_returns_copy() -> None:
    scores = {"AAA": 0.5, "BBB": 0.5}
    adjusted, reasons = apply_hrp_sizing_from_policy(
        scores, pd.DataFrame(), {"hrp_sizing": {"enabled": False}}
    )
    assert adjusted == scores
    assert adjusted is not scores
    assert reasons == []


def test_from_policy_enabled_applies_blend() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
    policy = {
        "hrp_sizing": {
            "enabled": True,
            "lookback_days": 100,
            "blend": 0.5,
            "target_invested_pct": 0.9,
        }
    }
    adjusted, reasons = apply_hrp_sizing_from_policy(scores, prices, policy)
    total = sum(adjusted.values())
    assert abs(total - 0.9) < 1e-6
    assert any("blended HRP" in r for r in reasons)
