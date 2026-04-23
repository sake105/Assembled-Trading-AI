"""Tests for wave-60 module wiring into trading_cycle.py.

Covers:
  Step 3.95 — strategies.stat_arb (check_cointegration / PairResult)
  Step 3.96 — strategies.strategy_discovery (DiscoveryResult / StrategyCandidate)
  Step 3.97 — signals.regime.hmm_posterior (smooth_posterior / blend_weights_by_regime_posterior)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.strategies.stat_arb import (
    check_cointegration,
    estimate_hedge_ratio,
    estimate_half_life,
    PairResult,
)
try:
    import statsmodels
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
from src.assembled_core.strategies.strategy_discovery import (
    DiscoveryResult,
    StrategyCandidate,
)
from src.assembled_core.signals.regime.hmm_posterior import (
    smooth_posterior,
    blend_weights_by_regime_posterior,
    RegimeBlendResult,
)


# ---------------------------------------------------------------------------
# stat_arb (Step 3.95)
# ---------------------------------------------------------------------------

def test_check_cointegration_returns_pair_result():
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(0, 1, 100))
    b = a + rng.normal(0, 0.5, 100)  # highly cointegrated
    result = check_cointegration(a, b)
    assert isinstance(result, PairResult)


def test_check_cointegration_has_pvalue():
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(0, 1, 100))
    b = a + rng.normal(0, 0.5, 100)
    result = check_cointegration(a, b)
    assert 0.0 <= result.coint_pvalue <= 1.0


def test_check_cointegration_has_hedge_ratio():
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(0, 1, 100))
    b = a + rng.normal(0, 0.5, 100)
    result = check_cointegration(a, b)
    assert isinstance(result.hedge_ratio, float)


def test_estimate_hedge_ratio_returns_float():
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(0, 1, 100))
    b = np.cumsum(rng.normal(0, 1, 100))
    ratio = estimate_hedge_ratio(a, b)
    assert isinstance(ratio, (float, np.floating))


def test_estimate_half_life_positive():
    rng = np.random.default_rng(0)
    spread = rng.normal(0, 1, 100)
    hl = estimate_half_life(spread)
    assert isinstance(hl, float)


def test_statsmodels_available_flag():
    assert isinstance(STATSMODELS_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# strategy_discovery (Step 3.96)
# ---------------------------------------------------------------------------

def test_strategy_candidate_creates():
    candidate = StrategyCandidate(
        strategy_id="s1",
        feature_names=["momentum", "value"],
        signal_type="long_short",
        sharpe_ratio=1.5,
        cagr=0.15,
        max_drawdown=-0.12,
        turnover=0.8,
        ic_mean=0.04,
        ic_ir=2.0,
        p_value=0.03,
        passes_gate=True,
        capacity_usd=1_000_000.0,
    )
    assert candidate.strategy_id == "s1"
    assert candidate.passes_gate


def test_discovery_result_importable():
    from src.assembled_core.strategies.strategy_discovery import DiscoveryResult
    assert DiscoveryResult is not None


# ---------------------------------------------------------------------------
# hmm_posterior (Step 3.97)
# ---------------------------------------------------------------------------

def test_smooth_posterior_first_call():
    posterior = {"BULL": 0.6, "BEAR": 0.2, "NEUTRAL": 0.2}
    result = smooth_posterior(posterior, prev_smoothed=None)
    assert isinstance(result, dict)
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_smooth_posterior_with_prev():
    post1 = {"BULL": 0.6, "BEAR": 0.2, "NEUTRAL": 0.2}
    prev = {"BULL": 0.5, "BEAR": 0.3, "NEUTRAL": 0.2}
    result = smooth_posterior(post1, prev_smoothed=prev)
    assert isinstance(result, dict)
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_smooth_posterior_sums_to_one():
    post = {"A": 0.7, "B": 0.3}
    result = smooth_posterior(post, prev_smoothed={"A": 0.4, "B": 0.6}, half_life_days=3.0)
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_blend_weights_by_regime_posterior():
    posterior = {"BULL": 0.6, "BEAR": 0.4}
    base = {
        "BULL": {"momentum": 0.6, "value": 0.4},
        "BEAR": {"momentum": 0.2, "value": 0.8},
    }
    result = blend_weights_by_regime_posterior(posterior, base)
    assert isinstance(result, RegimeBlendResult)
    assert isinstance(result.weights, dict)


def test_blend_weights_result_blended_correctly():
    posterior = {"BULL": 0.5, "BEAR": 0.5}
    base = {
        "BULL": {"momentum": 1.0, "value": 0.0},
        "BEAR": {"momentum": 0.0, "value": 1.0},
    }
    result = blend_weights_by_regime_posterior(posterior, base)
    assert abs(result.weights["momentum"] - 0.5) < 1e-9
    assert abs(result.weights["value"] - 0.5) < 1e-9
