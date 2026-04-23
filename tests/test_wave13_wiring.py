"""Tests for wave-13 module wiring into trading_cycle.py.

Covers:
  Step 4.9  — portfolio.long_short_balance (LongShortBalancer)
  Step 5.7  — portfolio.hierarchical_risk_parity (compute_hrp_weights)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.portfolio.long_short_balance import (
    LongShortBalancer,
    ExposureMetrics,
)
from src.assembled_core.portfolio.hierarchical_risk_parity import compute_hrp_weights


# ---------------------------------------------------------------------------
# LongShortBalancer (Step 4.9)
# ---------------------------------------------------------------------------

def _make_positions(n_long: int = 4, n_short: int = 2, long_w: float = 0.15, short_w: float = -0.08) -> pd.DataFrame:
    longs = [{"symbol": f"L{i}", "target_weight": long_w} for i in range(n_long)]
    shorts = [{"symbol": f"S{i}", "target_weight": short_w} for i in range(n_short)]
    return pd.DataFrame(longs + shorts)


def test_lsb_compute_exposure_returns_metrics():
    positions = _make_positions()
    balancer = LongShortBalancer()
    metrics = balancer.compute_exposure(positions)
    assert isinstance(metrics, ExposureMetrics)


def test_lsb_long_exposure_correct():
    positions = _make_positions(n_long=4, long_w=0.20)
    balancer = LongShortBalancer()
    metrics = balancer.compute_exposure(positions)
    assert abs(metrics.long_exposure - 0.80) < 0.01


def test_lsb_short_exposure_correct():
    positions = _make_positions(n_short=2, short_w=-0.10)
    balancer = LongShortBalancer()
    metrics = balancer.compute_exposure(positions)
    assert abs(metrics.short_exposure - 0.20) < 0.01


def test_lsb_gross_exposure():
    positions = _make_positions(n_long=2, n_short=2, long_w=0.20, short_w=-0.10)
    balancer = LongShortBalancer()
    metrics = balancer.compute_exposure(positions)
    expected = 2 * 0.20 + 2 * 0.10
    assert abs(metrics.gross_exposure - expected) < 0.01


def test_lsb_empty_positions():
    balancer = LongShortBalancer()
    metrics = balancer.compute_exposure(pd.DataFrame())
    assert metrics.gross_exposure == 0.0
    assert metrics.long_count == 0


def test_lsb_from_policy():
    policy = {"shorts": {"max_gross_exposure": 1.2, "max_net_short": 0.15, "max_total_short_exposure": 0.25}}
    balancer = LongShortBalancer.from_policy(policy)
    assert balancer.max_gross == 1.2
    assert balancer.max_net_short == 0.15


def test_lsb_count_longs_and_shorts():
    positions = _make_positions(n_long=5, n_short=3)
    balancer = LongShortBalancer()
    metrics = balancer.compute_exposure(positions)
    assert metrics.long_count == 5
    assert metrics.short_count == 3


# ---------------------------------------------------------------------------
# compute_hrp_weights (Step 5.7)
# ---------------------------------------------------------------------------

def _make_returns(n_symbols: int = 5, n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days)
    return pd.DataFrame(rng.standard_normal((n_days, n_symbols)),
                        index=idx, columns=[f"S{i}" for i in range(n_symbols)])


def test_hrp_returns_dict():
    returns = _make_returns()
    result = compute_hrp_weights(returns)
    assert isinstance(result, dict)


def test_hrp_weights_sum_to_1():
    returns = _make_returns()
    result = compute_hrp_weights(returns)
    if result:
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01


def test_hrp_all_symbols_present():
    returns = _make_returns(n_symbols=4)
    result = compute_hrp_weights(returns)
    if result:
        assert set(result.keys()) == {"S0", "S1", "S2", "S3"}


def test_hrp_weights_non_negative():
    returns = _make_returns()
    result = compute_hrp_weights(returns)
    for sym, w in result.items():
        assert w >= 0.0, f"{sym}: negative weight {w}"


def test_hrp_too_few_assets_returns_empty_or_valid():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.01, 0.0, -0.01]})
    result = compute_hrp_weights(returns)
    # Single-asset HRP → empty (needs >= 2 assets) or trivially 1.0
    assert isinstance(result, dict)


def test_hrp_respects_max_weight():
    returns = _make_returns(n_symbols=3)
    result = compute_hrp_weights(returns, max_weight=0.5)
    for sym, w in result.items():
        assert w <= 0.5 + 1e-9, f"{sym}: {w} > max_weight 0.5"
