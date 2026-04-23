"""Tests for wave-57 module wiring into trading_cycle.py.

Covers:
  Step 5.100 — portfolio.multi_period (compute_trade_speed / multi_period_optimize)
  Step 5.101 — portfolio.multiasset_allocator (RegimeDetector / allocate_by_regime)
  Step 5.102 — portfolio.strategy_allocator (AllocationConfig / StrategyAllocator)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.portfolio.multi_period import (
    compute_trade_speed,
    garleanu_pedersen_target,
    multi_period_optimize,
    MultiPeriodResult,
    SCIPY_AVAILABLE,
)
from src.assembled_core.portfolio.multiasset_allocator import (
    RegimeDetectorConfig,
    RegimeDetector,
    RegimeAllocation,
    allocate_by_regime,
)
from src.assembled_core.portfolio.strategy_allocator import (
    AllocationConfig,
    EnsembleResult,
)


# ---------------------------------------------------------------------------
# multi_period (Step 5.100)
# ---------------------------------------------------------------------------

def test_compute_trade_speed_returns_float():
    speed = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.001)
    assert isinstance(speed, float)


def test_compute_trade_speed_higher_cost_slower():
    speed_low = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.0001)
    speed_high = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.01)
    assert speed_low >= speed_high


def test_compute_trade_speed_in_range():
    speed = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.001)
    assert 0.0 < speed <= 1.0


def test_scipy_available_flag():
    assert isinstance(SCIPY_AVAILABLE, bool)


def test_garleanu_pedersen_target_returns_result():
    aim = {"A": 0.4, "B": 0.4, "C": 0.2}
    current = {"A": 0.3, "B": 0.5, "C": 0.2}
    result = garleanu_pedersen_target(aim, current, trade_speed=0.5)
    assert isinstance(result, MultiPeriodResult)
    assert isinstance(result.target_weights, dict)


def test_multi_period_optimize_empty_path():
    cov = pd.DataFrame([[0.01, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    current = {"A": 0.5, "B": 0.5}
    result = multi_period_optimize([], cov, current)
    assert isinstance(result, MultiPeriodResult)
    assert result.periods_ahead == 0


def test_multi_period_optimize_one_period():
    rng = np.random.default_rng(0)
    symbols = ["A", "B", "C"]
    cov = pd.DataFrame(
        np.diag([0.01, 0.01, 0.01]),
        index=symbols, columns=symbols,
    )
    rets = [pd.Series(rng.normal(0.001, 0.01, 3), index=symbols)]
    current = {"A": 0.33, "B": 0.33, "C": 0.34}
    result = multi_period_optimize(rets, cov, current)
    assert isinstance(result, MultiPeriodResult)
    assert len(result.target_weights) > 0


# ---------------------------------------------------------------------------
# multiasset_allocator (Step 5.101)
# ---------------------------------------------------------------------------

def test_regime_detector_config_creates():
    cfg = RegimeDetectorConfig()
    assert isinstance(cfg, RegimeDetectorConfig)


def test_regime_detector_config_defaults():
    cfg = RegimeDetectorConfig()
    assert cfg.vix_bull_threshold > 0
    assert cfg.hysteresis_bars >= 1


def test_regime_detector_creates():
    detector = RegimeDetector()
    assert isinstance(detector, RegimeDetector)


def test_regime_detector_update():
    detector = RegimeDetector()
    regime = detector.update(vix=18.0, breadth=0.65, spy_close=450.0)
    assert isinstance(regime, str)
    assert regime in ("bull", "sideways", "bear", "crisis")


def test_regime_allocation_creates():
    alloc = RegimeAllocation(equity=0.8, tlt=0.1, gld=0.1)
    assert alloc.equity == 0.8
    assert isinstance(alloc.as_dict(), dict)


def test_allocate_by_regime_bull():
    equity_weights = {"AAPL": 0.5, "MSFT": 0.5}
    result = allocate_by_regime("bull", equity_weights)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_allocate_by_regime_crisis():
    equity_weights = {"AAPL": 0.5, "MSFT": 0.5}
    result = allocate_by_regime("crisis", equity_weights)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# strategy_allocator (Step 5.102)
# ---------------------------------------------------------------------------

def test_allocation_config_creates():
    cfg = AllocationConfig()
    assert isinstance(cfg, AllocationConfig)


def test_allocation_config_defaults():
    cfg = AllocationConfig()
    assert cfg.method in ("weighted_average", "majority_vote", "regime_conditional")
    assert cfg.min_strategies_required >= 1


def test_allocation_config_custom_method():
    cfg = AllocationConfig(method="majority_vote", weights={"a": 0.5, "b": 0.5})
    assert cfg.method == "majority_vote"
    assert len(cfg.weights) == 2
