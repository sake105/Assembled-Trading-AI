"""Tests for wave-142 module wiring into trading_cycle.py.

Covers:
  Step sig.1   — signals.ensemble (apply_meta_filter / apply_meta_scaling)
  Step strat.1 — strategies.base (Strategy / StrategyRegistry)
  Step strat.2 — strategies.stat_arb.cointegration (PairCandidate / screen_pairs)
"""

from __future__ import annotations

import pytest

from src.assembled_core.signals.ensemble import apply_meta_filter, apply_meta_scaling
from src.assembled_core.strategies.base import (
    StrategySignal,
    Strategy,
    StrategyRegistry,
)
from src.assembled_core.strategies.stat_arb.cointegration import (
    PairCandidate,
    screen_pairs,
    test_cointegration as _test_cointegration,
)


# ---------------------------------------------------------------------------
# signals.ensemble (Step sig.1)
# ---------------------------------------------------------------------------

def test_apply_meta_filter_importable():
    assert apply_meta_filter is not None


def test_apply_meta_scaling_importable():
    assert apply_meta_scaling is not None


# ---------------------------------------------------------------------------
# strategies.base (Step strat.1)
# ---------------------------------------------------------------------------

def test_strategy_importable():
    assert Strategy is not None


def test_strategy_signal_importable():
    assert StrategySignal is not None


def test_strategy_registry_creates():
    registry = StrategyRegistry()
    assert isinstance(registry, StrategyRegistry)


# ---------------------------------------------------------------------------
# strategies.stat_arb.cointegration (Step strat.2)
# ---------------------------------------------------------------------------

def test_pair_candidate_importable():
    assert PairCandidate is not None


def test_pair_candidate_creates():
    pc = PairCandidate(
        stock_a="AAPL",
        stock_b="MSFT",
        hedge_ratio=1.2,
        half_life=15.0,
        p_value=0.03,
        correlation=0.85,
        spread_std=0.02,
    )
    assert pc.is_significant is True
    assert pc.stock_a == "AAPL"


def test_screen_pairs_importable():
    assert screen_pairs is not None


def test_cointegration_fn_importable():
    assert _test_cointegration is not None
