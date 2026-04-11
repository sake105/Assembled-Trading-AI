"""Tests for component VaR / marginal VaR (Sprint 2 / C5b)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.risk_metrics import compute_component_var


def _mk_returns(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 3-asset correlated returns
    L = np.array([
        [0.015, 0.000, 0.000],
        [0.008, 0.012, 0.000],
        [0.005, 0.004, 0.010],
    ])
    z = rng.standard_normal((n, 3))
    r = z @ L.T
    return pd.DataFrame(r, columns=["A", "B", "C"])


def test_additivity_equal_weights() -> None:
    df = _mk_returns()
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    out = compute_component_var(df, w, alpha=0.95)
    total = float(np.sum(out["component_var"]))
    # portfolio_var is rounded to 8dp by the helper
    assert abs(total - out["portfolio_var"]) < 1e-7


def test_additivity_mixed_weights() -> None:
    df = _mk_returns()
    w = np.array([0.5, 0.3, 0.2])
    out = compute_component_var(df, w, alpha=0.99)
    total = float(np.sum(out["component_var"]))
    assert abs(total - out["portfolio_var"]) < 1e-7


def test_pct_contribution_sums_to_one() -> None:
    df = _mk_returns()
    w = np.array([0.4, 0.4, 0.2])
    out = compute_component_var(df, w)
    assert abs(float(np.sum(out["pct_contribution"])) - 1.0) < 1e-10


def test_zero_portfolio_returns_zero() -> None:
    df = pd.DataFrame(np.zeros((100, 2)), columns=["A", "B"])
    w = np.array([0.5, 0.5])
    out = compute_component_var(df, w)
    assert out["portfolio_var"] == 0.0
    assert out["portfolio_vol"] == 0.0
    assert np.all(out["component_var"] == 0.0)


def test_insufficient_data_returns_zero() -> None:
    df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
    out = compute_component_var(df, np.array([0.5, 0.5]))
    assert out["portfolio_var"] == 0.0


def test_weights_length_mismatch_returns_zero() -> None:
    df = _mk_returns()
    out = compute_component_var(df, np.array([0.5, 0.5]))  # 3 assets, 2 weights
    assert out["portfolio_var"] == 0.0


def test_marginal_var_matches_euler_increment() -> None:
    """dVaR/dw_i ≈ marginal_var_i (looser numerical check, tolerant of rounding)."""
    df = _mk_returns()
    w = np.array([0.4, 0.3, 0.3])
    out = compute_component_var(df, w, alpha=0.95)
    base_var = out["portfolio_var"]

    eps = 1e-3  # large enough to dominate 1e-8 rounding noise
    for i in range(3):
        w_eps = w.copy()
        w_eps[i] += eps
        bumped = compute_component_var(df, w_eps, alpha=0.95)
        numerical = (bumped["portfolio_var"] - base_var) / eps
        analytic = float(out["marginal_var"][i])
        # Looser tolerance: second-order bias + rounding
        assert abs(numerical - analytic) < 5e-3


def test_single_asset_degenerate() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(200, 1)), columns=["A"])
    out = compute_component_var(df, np.array([1.0]))
    assert out["portfolio_var"] > 0
    assert abs(float(out["component_var"][0]) - out["portfolio_var"]) < 1e-7
