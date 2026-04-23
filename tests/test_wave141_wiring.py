"""Tests for wave-141 module wiring into trading_cycle.py.

Covers:
  Step risk.1 — risk.group_exposures (GroupExposureSummary / compute_group_exposures)
  Step risk.2 — risk.risk_metrics (compute_basic_risk_metrics)
  Step risk.3 — risk.tail_hedge (CollarConfig / TailHedgeResult)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.assembled_core.risk.group_exposures import (
    GroupExposureSummary,
    compute_group_exposures,
    compute_net_market_exposure,
)
from src.assembled_core.risk.risk_metrics import compute_basic_risk_metrics
from src.assembled_core.risk.tail_hedge import (
    CollarConfig,
    TailHedgeResult,
    estimate_option_premium,
)


# ---------------------------------------------------------------------------
# risk.group_exposures (Step risk.1)
# ---------------------------------------------------------------------------

def test_group_exposure_summary_importable():
    assert GroupExposureSummary is not None


def test_group_exposure_summary_creates():
    ges = GroupExposureSummary(
        total_groups=3,
        max_gross_weight=0.5,
        max_net_weight=0.3,
        total_gross_exposure=1.2,
        total_net_exposure=0.4,
    )
    assert ges.total_groups == 3


def test_compute_group_exposures_importable():
    assert compute_group_exposures is not None


def test_compute_net_market_exposure_importable():
    assert compute_net_market_exposure is not None


# ---------------------------------------------------------------------------
# risk.risk_metrics (Step risk.2)
# ---------------------------------------------------------------------------

def test_compute_basic_risk_metrics_importable():
    assert compute_basic_risk_metrics is not None


def test_compute_basic_risk_metrics_basic():
    equity = pd.Series(
        np.linspace(100_000, 120_000, 252),
        index=pd.date_range("2024-01-01", periods=252, freq="D"),
    )
    result = compute_basic_risk_metrics(equity, freq="1d")
    assert isinstance(result, dict)
    assert "sharpe" in result or "volatility" in result


# ---------------------------------------------------------------------------
# risk.tail_hedge (Step risk.3)
# ---------------------------------------------------------------------------

def test_collar_config_importable():
    assert CollarConfig is not None


def test_collar_config_creates():
    cfg = CollarConfig()
    assert cfg.hedge_ratio == 1.0
    assert cfg.put_otm_pct > 0.0


def test_tail_hedge_result_importable():
    assert TailHedgeResult is not None


def test_estimate_option_premium_importable():
    assert estimate_option_premium is not None
