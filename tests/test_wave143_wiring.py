"""Tests for wave-143 module wiring into trading_cycle.py.

Covers:
  Step strat.3 — strategies.stat_arb.pair_signals (PairSignalGenerator)
  Step strat.4 — strategies.stat_arb.pca_arb (PCAFactorModel / compute_pca_factors)
  Step util.1  — utils.dataframe (ensure_cols / coerce_price_types)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.assembled_core.strategies.stat_arb.pair_signals import (
    PairPosition,
    PairSignal,
    PairSignalGenerator,
)
from src.assembled_core.strategies.stat_arb.pca_arb import (
    PCAFactorModel,
    PCASignal,
    compute_pca_factors,
)
from src.assembled_core.utils.dataframe import ensure_cols, coerce_price_types


# ---------------------------------------------------------------------------
# strategies.stat_arb.pair_signals (Step strat.3)
# ---------------------------------------------------------------------------

def test_pair_position_importable():
    assert PairPosition is not None


def test_pair_signal_generator_creates():
    psg = PairSignalGenerator(hedge_ratio=1.5, lookback=30)
    assert psg.hedge_ratio == 1.5
    assert psg.lookback == 30


def test_pair_signal_importable():
    assert PairSignal is not None


# ---------------------------------------------------------------------------
# strategies.stat_arb.pca_arb (Step strat.4)
# ---------------------------------------------------------------------------

def test_pca_factor_model_importable():
    assert PCAFactorModel is not None


def test_pca_signal_importable():
    assert PCASignal is not None


def test_compute_pca_factors_importable():
    assert compute_pca_factors is not None


def test_compute_pca_factors_insufficient_data():
    returns = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (5, 3)))
    result = compute_pca_factors(returns, n_components=2, min_obs=60)
    assert result is None  # too few observations


# ---------------------------------------------------------------------------
# utils.dataframe (Step util.1)
# ---------------------------------------------------------------------------

def test_ensure_cols_importable():
    assert ensure_cols is not None


def test_coerce_price_types_importable():
    assert coerce_price_types is not None


def test_ensure_cols_passthrough():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = ensure_cols(df, ["a", "b"])
    assert list(result.columns) == ["a", "b"]
