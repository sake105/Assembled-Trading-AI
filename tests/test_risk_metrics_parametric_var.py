"""Tests for parametric & Cornish-Fisher VaR (Sprint 1 / C5a)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.risk_metrics import (
    compute_basic_risk_metrics,
    compute_cornish_fisher_var,
    compute_parametric_var,
)


def _gauss_returns(
    n: int = 2000, mu: float = 0.0, sigma: float = 0.01, seed: int = 42
) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def _fat_left_tail_returns(n: int = 2000, seed: int = 7) -> pd.Series:
    """Heavy left tail: Student-t(df=3) scaled, then shift negatively."""
    rng = np.random.default_rng(seed)
    t = rng.standard_t(df=3, size=n) * 0.01
    # inject a few large left-tail shocks to push skew negative
    t[:20] = -0.08
    return pd.Series(t)


def test_parametric_var_gauss_matches_historical_within_tolerance() -> None:
    r = _gauss_returns(n=5000)
    var_param = compute_parametric_var(r, alpha=0.95)
    var_hist = float(-np.percentile(r, 5))  # sign-flipped historical

    assert var_param is not None
    # For Gaussian returns, parametric ≈ historical (within 10 %)
    assert abs(var_param - var_hist) / max(var_hist, 1e-9) < 0.10


def test_parametric_var_99_greater_than_95() -> None:
    r = _gauss_returns(n=5000)
    v95 = compute_parametric_var(r, alpha=0.95)
    v99 = compute_parametric_var(r, alpha=0.99)
    assert v95 is not None and v99 is not None
    assert v99 > v95


def test_parametric_var_horizon_scales_sqrt() -> None:
    r = _gauss_returns(n=5000)
    v1 = compute_parametric_var(r, alpha=0.95, horizon=1)
    v10 = compute_parametric_var(r, alpha=0.95, horizon=10)
    assert v1 is not None and v10 is not None
    # √10 ≈ 3.162
    assert abs(v10 / v1 - np.sqrt(10)) < 0.05


def test_cornish_fisher_exceeds_parametric_on_fat_left_tail() -> None:
    r = _fat_left_tail_returns()
    v_param = compute_parametric_var(r, alpha=0.99)
    v_cf = compute_cornish_fisher_var(r, alpha=0.99)
    assert v_param is not None and v_cf is not None
    # Fat-tail + left skew → CF should report a materially larger loss
    assert v_cf > v_param


def test_cornish_fisher_close_to_parametric_on_gauss() -> None:
    r = _gauss_returns(n=5000)
    v_param = compute_parametric_var(r, alpha=0.95)
    v_cf = compute_cornish_fisher_var(r, alpha=0.95)
    assert v_param is not None and v_cf is not None
    # Near-Gaussian sample: CF should be within ~15 % of parametric
    assert abs(v_cf - v_param) / v_param < 0.15


def test_short_series_returns_none() -> None:
    r = pd.Series([0.01, -0.02, 0.0])  # len < 5
    assert compute_parametric_var(r) is None
    # 4 observations: Cornish-Fisher just barely allowed
    r4 = pd.Series([0.01, -0.02, 0.005, -0.01])
    assert compute_cornish_fisher_var(r4) is not None


def test_zero_variance_returns_none() -> None:
    r = pd.Series([0.0] * 100)
    assert compute_parametric_var(r) is None
    assert compute_cornish_fisher_var(r) is None


def test_basic_risk_metrics_exposes_parametric_keys() -> None:
    r = _gauss_returns(n=1000)
    m = compute_basic_risk_metrics(r, freq="1d")
    for key in (
        "var_95_parametric",
        "var_99_parametric",
        "var_95_cornish_fisher",
        "var_99_cornish_fisher",
    ):
        assert key in m, f"missing {key} in basic risk metrics output"
        assert m[key] is not None
