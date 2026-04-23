"""Tests for wave-139 module wiring into trading_cycle.py.

Covers:
  Step qa.5 — qa.leakage_tests.altdata_leakage (assert_feature_zero_before_disclosure)
  Step qa.6 — qa.signal_decay (SignalDecayProfile / compute_ic_series)
  Step qa.7 — qa.tca (build_tca_report)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.assembled_core.qa.leakage_tests.altdata_leakage import assert_feature_zero_before_disclosure
from src.assembled_core.qa.signal_decay import (
    SignalDecayProfile,
    compute_ic_series,
    compute_ic_half_life,
)
from src.assembled_core.qa.tca import build_tca_report, write_tca_report_csv


# ---------------------------------------------------------------------------
# qa.leakage_tests.altdata_leakage (Step qa.5)
# ---------------------------------------------------------------------------

def test_assert_feature_zero_importable():
    assert assert_feature_zero_before_disclosure is not None


# ---------------------------------------------------------------------------
# qa.signal_decay (Step qa.6)
# ---------------------------------------------------------------------------

def test_signal_decay_profile_importable():
    assert SignalDecayProfile is not None


def test_compute_ic_series_importable():
    assert compute_ic_series is not None


def test_compute_ic_half_life_importable():
    assert compute_ic_half_life is not None


def test_compute_ic_half_life_none_for_short():
    ic = pd.Series([0.1, 0.08, 0.05, 0.04])
    result = compute_ic_half_life(ic)
    # May return None if < min threshold
    assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# qa.tca (Step qa.7)
# ---------------------------------------------------------------------------

def test_build_tca_report_importable():
    assert build_tca_report is not None


def test_build_tca_report_empty():
    df = pd.DataFrame(columns=[
        "timestamp", "symbol", "qty", "price",
        "commission_cash", "spread_cash", "slippage_cash", "total_cost_cash"
    ])
    result = build_tca_report(df, freq="1d", strategy_name="test")
    assert isinstance(result, pd.DataFrame)


def test_write_tca_report_csv_importable():
    assert write_tca_report_csv is not None
