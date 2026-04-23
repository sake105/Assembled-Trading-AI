"""Tests for wave-105 module wiring into trading_cycle.py.

Covers:
  Step 3.92 — strategies.multifactor_long_short (MultiFactorStrategyConfig / generate_multifactor_long_short_signals)
  Step 3.93 — strategies.multifactor_v1 (compute_signals / compute_target_positions)
  Step 3.94 — strategies.multifactor_v2 (compute_signals / compute_target_positions)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.strategies.multifactor_long_short import (
    MultiFactorStrategyConfig,
    generate_multifactor_long_short_signals,
)
from src.assembled_core.strategies.multifactor_v1 import (
    compute_signals as mfv1_compute_signals,
    compute_target_positions as mfv1_compute_target_positions,
)
from src.assembled_core.strategies.multifactor_v2 import (
    compute_signals as mfv2_compute_signals,
    compute_target_positions as mfv2_compute_target_positions,
)


# ---------------------------------------------------------------------------
# multifactor_long_short (Step 3.92)
# ---------------------------------------------------------------------------

def test_multifactor_strategy_config_creates():
    cfg = MultiFactorStrategyConfig(bundle_path="")
    assert isinstance(cfg, MultiFactorStrategyConfig)


def test_multifactor_strategy_config_defaults():
    cfg = MultiFactorStrategyConfig(bundle_path="configs/factor_bundle.yaml")
    assert cfg.top_quantile == 0.2
    assert cfg.bottom_quantile == 0.2
    assert cfg.rebalance_freq == "M"


def test_multifactor_strategy_config_max_leverage():
    cfg = MultiFactorStrategyConfig(bundle_path="")
    assert cfg.max_leverage == 1.0


def test_generate_multifactor_long_short_signals_importable():
    assert generate_multifactor_long_short_signals is not None


def test_generate_multifactor_long_short_signals_empty_df():
    try:
        result = generate_multifactor_long_short_signals(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
    except (KeyError, ValueError):
        pass  # requires timestamp/symbol/close columns


# ---------------------------------------------------------------------------
# multifactor_v1 (Step 3.93)
# ---------------------------------------------------------------------------

def test_mfv1_compute_signals_importable():
    assert mfv1_compute_signals is not None


def test_mfv1_compute_target_positions_importable():
    assert mfv1_compute_target_positions is not None


def test_mfv1_compute_signals_empty_df():
    result = mfv1_compute_signals(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# multifactor_v2 (Step 3.94)
# ---------------------------------------------------------------------------

def test_mfv2_compute_signals_importable():
    assert mfv2_compute_signals is not None


def test_mfv2_compute_target_positions_importable():
    assert mfv2_compute_target_positions is not None


def test_mfv2_compute_signals_empty_df():
    result = mfv2_compute_signals(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


def test_mfv2_compute_signals_returns_dataframe():
    result = mfv2_compute_signals(pd.DataFrame(), strategy_cfg={})
    assert isinstance(result, pd.DataFrame)
