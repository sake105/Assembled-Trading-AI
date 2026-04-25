"""Tests for liquidity-scoring wiring (Sprint 1 / W2).

Covers:
- compute_liquidity_scores tier classification by ADV
- apply_liquidity_adjusted_sizing zeroes below threshold, scales by score,
  preserves gross exposure after renormalisation
- Amihud lambda / Roll spread basic sanity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.liquidity_scoring import (
    LiquidityScore,
    apply_liquidity_adjusted_sizing,
    compute_amihud_lambda,
    compute_liquidity_scores,
    compute_roll_spread,
)


def _mk_panel(symbol: str, n: int = 80, price: float = 100.0,
              volume: float = 1_000_000.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    rets = rng.normal(0, 0.01, n)
    closes = price * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": [symbol] * n,
            "close": closes,
            "volume": [volume] * n,
        }
    )


def test_amihud_lambda_higher_for_illiquid() -> None:
    rets = np.array([0.01, -0.02, 0.015, -0.01, 0.005, 0.012])
    dv_liquid = np.array([1e9] * 6)
    dv_illiquid = np.array([1e5] * 6)
    lam_l = compute_amihud_lambda(rets, dv_liquid)
    lam_i = compute_amihud_lambda(rets, dv_illiquid)
    assert lam_i > lam_l


def test_amihud_lambda_insufficient_returns_inf() -> None:
    rets = np.array([0.01, 0.02])
    dv = np.array([1e6, 1e6])
    assert compute_amihud_lambda(rets, dv) == np.inf


def test_roll_spread_short_series_zero() -> None:
    assert compute_roll_spread(np.array([100.0, 101.0])) == 0.0


def test_liquidity_scores_mega_tier_for_high_adv() -> None:
    # Price 100 * volume 2M = 200M dollar volume → mega tier
    panel = _mk_panel("MEGA", price=100.0, volume=2_000_000.0)
    scores = compute_liquidity_scores(panel)
    assert len(scores) == 1
    assert scores[0].tier == "mega"
    assert 0.0 <= scores[0].score <= 1.0


def test_liquidity_scores_micro_tier_for_low_adv() -> None:
    # Price 5 * volume 50k = 250k dollar volume → micro tier
    panel = _mk_panel("MICRO", price=5.0, volume=50_000.0)
    scores = compute_liquidity_scores(panel)
    assert len(scores) == 1
    assert scores[0].tier == "micro"


def test_liquidity_scores_empty_panel() -> None:
    panel = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
    assert compute_liquidity_scores(panel) == []


def test_apply_zeroes_below_threshold() -> None:
    weights = {"AAA": 0.1, "BBB": 0.1}
    scores = [
        LiquidityScore("AAA", 1e-9, 5.0, 5e8, 0.9, "mega"),
        LiquidityScore("BBB", 1e-3, 50.0, 1e5, 0.05, "micro"),
    ]
    adj = apply_liquidity_adjusted_sizing(
        weights, scores, alpha=0.5, min_score_threshold=0.1
    )
    assert adj["BBB"] == 0.0
    assert adj["AAA"] > 0.0


def test_apply_renormalises_gross_exposure() -> None:
    weights = {"AAA": 0.1, "BBB": 0.1}
    scores = [
        LiquidityScore("AAA", 0, 0, 0, 1.0, "mega"),
        LiquidityScore("BBB", 0, 0, 0, 0.5, "large"),
    ]
    adj = apply_liquidity_adjusted_sizing(weights, scores, alpha=0.5)
    before = sum(abs(v) for v in weights.values())
    after = sum(abs(v) for v in adj.values())
    # Gross exposure is preserved by the renormalisation step.
    assert abs(before - after) < 1e-9


def test_apply_noop_when_all_scores_one() -> None:
    weights = {"AAA": 0.1, "BBB": 0.2}
    scores = [
        LiquidityScore("AAA", 0, 0, 0, 1.0, "mega"),
        LiquidityScore("BBB", 0, 0, 0, 1.0, "mega"),
    ]
    adj = apply_liquidity_adjusted_sizing(weights, scores, alpha=0.5)
    assert abs(adj["AAA"] - 0.1) < 1e-9
    assert abs(adj["BBB"] - 0.2) < 1e-9


def test_apply_unknown_symbol_default_midrange() -> None:
    weights = {"XXX": 0.1}
    adj = apply_liquidity_adjusted_sizing(weights, [], alpha=0.5)
    # No scores → default 0.5 → kept (above threshold 0.1) then renormalised
    # back to preserve gross exposure.
    assert abs(adj["XXX"] - 0.1) < 1e-9


def test_alpha_zero_equals_noop_before_renorm() -> None:
    # With alpha=0, weight^score_factor = weight, and renormalisation should
    # still yield the original exposures.
    weights = {"AAA": 0.2, "BBB": 0.1}
    scores = [
        LiquidityScore("AAA", 0, 0, 0, 0.9, "mega"),
        LiquidityScore("BBB", 0, 0, 0, 0.4, "mid"),
    ]
    adj = apply_liquidity_adjusted_sizing(weights, scores, alpha=0.0)
    assert abs(adj["AAA"] - 0.2) < 1e-9
    assert abs(adj["BBB"] - 0.1) < 1e-9
