"""Tests for wave-55 module wiring into trading_cycle.py.

Covers:
  Step 8.57 — risk.antifragility (compute_antifragility_score / compute_portfolio_antifragility)
  Step 8.58 — risk.stressed_var (marchenko_pastur_bounds / clean_covariance_rmt / RMTResult)
  Step 8.59 — risk.profit_targets (ProfitTargetConfig / check_profit_targets)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.antifragility import (
    compute_antifragility_score,
    compute_portfolio_antifragility,
)
from src.assembled_core.risk.stressed_var import (
    marchenko_pastur_bounds,
    clean_covariance_rmt,
    RMTResult,
    StressedVaRResult,
)
from src.assembled_core.risk.profit_targets import (
    ProfitTargetConfig,
    PositionRecord,
    check_profit_targets,
    build_position_records,
)


# ---------------------------------------------------------------------------
# antifragility (Step 8.57)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 80) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    port = pd.Series(rng.normal(0, 0.01, n), index=idx)
    market = pd.Series(rng.normal(0, 0.012, n), index=idx)
    return port, market


def test_compute_antifragility_score_returns_series():
    port, market = _make_returns()
    result = compute_antifragility_score(port, market, window=20)
    assert isinstance(result, pd.Series)


def test_compute_antifragility_score_length():
    port, market = _make_returns()
    result = compute_antifragility_score(port, market, window=20)
    assert len(result) == len(port)


def test_compute_antifragility_score_range():
    port, market = _make_returns()
    result = compute_antifragility_score(port, market, window=20).dropna()
    assert (result >= -1.0).all()
    assert (result <= 1.0).all()


def test_compute_portfolio_antifragility_returns_float():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    asset_returns = pd.DataFrame(
        rng.normal(0, 0.01, (80, 3)),
        index=idx,
        columns=["A", "B", "C"],
    )
    market = pd.Series(rng.normal(0, 0.012, 80), index=idx)
    weights = {"A": 0.4, "B": 0.4, "C": 0.2}
    result = compute_portfolio_antifragility(weights, asset_returns, market)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# stressed_var (Step 8.58)
# ---------------------------------------------------------------------------

def test_marchenko_pastur_bounds_returns_tuple():
    lo, hi = marchenko_pastur_bounds(n_obs=252, n_assets=50)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo >= 0
    assert hi > lo


def test_marchenko_pastur_bounds_scaling():
    lo1, hi1 = marchenko_pastur_bounds(n_obs=252, n_assets=50)
    lo2, hi2 = marchenko_pastur_bounds(n_obs=252, n_assets=50, sigma_sq=2.0)
    assert abs(hi2 - 2 * hi1) < 1e-9


def test_clean_covariance_rmt_returns_result():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (100, 10)))
    result = clean_covariance_rmt(returns)
    assert isinstance(result, RMTResult)


def test_rmt_result_has_cleaned_covariance():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (100, 5)))
    result = clean_covariance_rmt(returns)
    assert result.cleaned_covariance.shape == (5, 5)


def test_rmt_result_noise_threshold_positive():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (100, 5)))
    result = clean_covariance_rmt(returns)
    assert result.noise_threshold > 0


# ---------------------------------------------------------------------------
# profit_targets (Step 8.59)
# ---------------------------------------------------------------------------

def test_profit_target_config_creates():
    cfg = ProfitTargetConfig()
    assert isinstance(cfg, ProfitTargetConfig)


def test_profit_target_config_has_tiers():
    cfg = ProfitTargetConfig()
    assert len(cfg.tiers) >= 2


def test_profit_target_config_tiers_ascending():
    cfg = ProfitTargetConfig()
    thresholds = [t[0] for t in cfg.tiers]
    assert thresholds == sorted(thresholds)


def test_position_record_creates():
    pos = PositionRecord(symbol="AAPL", entry_price=150.0, is_long=True)
    assert pos.symbol == "AAPL"
    assert pos.is_long


def test_check_profit_targets_no_trigger():
    cfg = ProfitTargetConfig()
    pos = PositionRecord(symbol="AAPL", entry_price=150.0, is_long=True)
    positions = {"AAPL": pos}
    prices = {"AAPL": 151.0}  # +0.67% — below first tier
    result = check_profit_targets(positions, prices, config=cfg)
    assert "AAPL" not in result


def test_check_profit_targets_triggers_tier():
    cfg = ProfitTargetConfig()
    pos = PositionRecord(symbol="AAPL", entry_price=100.0, is_long=True)
    positions = {"AAPL": pos}
    prices = {"AAPL": 115.0}  # +15% → tier 1 triggers
    result = check_profit_targets(positions, prices, config=cfg)
    assert "AAPL" in result
    assert 0.0 < result["AAPL"] < 1.0
