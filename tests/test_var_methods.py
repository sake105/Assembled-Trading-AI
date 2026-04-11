"""Tests for PortfolioVaR (C5a) — parametric, Cornish-Fisher, ES, historical.

These tests verify only the additive ``src/assembled_core/risk/var_methods.py``
module. They do not touch ``risk_metrics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.var_methods import PortfolioVaR, _z_from_alpha

pytestmark = pytest.mark.phase12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _gauss_panel(n: int = 10_000, sigma: float = 0.01, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, sigma, n)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    return pd.DataFrame({"AAPL": data}, index=idx)


def _two_symbol_panel(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.01, n)
    b = rng.normal(0.0, 0.01, n)  # independent
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    return pd.DataFrame({"AAPL": a, "MSFT": b}, index=idx)


# --------------------------------------------------------------------------- #
# 1. Gauss-returns agreement: historical ~ parametric
# --------------------------------------------------------------------------- #


def test_gauss_returns_historical_matches_parametric() -> None:
    df = _gauss_panel()
    pvar = PortfolioVaR(df, pd.Series({"AAPL": 1.0}))
    hist = pvar.historical_var(alpha=0.95)
    para = pvar.parametric_var(alpha=0.95)
    # Within 5 % relative tolerance under a clean Gaussian sample.
    assert abs(hist - para) / para < 0.05


# --------------------------------------------------------------------------- #
# 2. Cornish-Fisher expands on fat / skewed tails
# --------------------------------------------------------------------------- #


def test_cornish_fisher_greater_than_parametric_on_fat_tails() -> None:
    rng = np.random.default_rng(123)
    n = 20_000
    # Symmetric scale-mixture of normals: produces positive excess kurtosis
    # (fat tails) without extreme skewness. This is the regime where
    # Cornish-Fisher is well-behaved and monotone.
    regime = rng.random(n) < 0.10
    calm = rng.normal(0.0, 0.005, n)
    wild = rng.normal(0.0, 0.03, n)
    data = np.where(regime, wild, calm)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    df = pd.DataFrame({"AAPL": data}, index=idx)

    pvar = PortfolioVaR(df, pd.Series({"AAPL": 1.0}))
    # At alpha=0.99 the kurtosis term (z^3 - 3z)/24 is much larger than at
    # 0.95, so Cornish-Fisher clearly expands the Gaussian tail on fat-tailed
    # samples.
    para = pvar.parametric_var(alpha=0.99)
    cf = pvar.cornish_fisher_var(alpha=0.99)
    assert cf > para


# --------------------------------------------------------------------------- #
# 3. Horizon scaling sqrt(h)
# --------------------------------------------------------------------------- #


def test_horizon_scaling_sqrt_rule() -> None:
    df = _gauss_panel()
    pvar = PortfolioVaR(df, pd.Series({"AAPL": 1.0}))
    v1 = pvar.parametric_var(alpha=0.95, horizon=1)
    v4 = pvar.parametric_var(alpha=0.95, horizon=4)
    # sqrt(4) = 2, within 5 %
    assert abs(v4 - 2.0 * v1) / (2.0 * v1) < 0.05


# --------------------------------------------------------------------------- #
# 4. Expected shortfall >= historical VaR
# --------------------------------------------------------------------------- #


def test_expected_shortfall_ge_historical_var() -> None:
    df = _gauss_panel()
    pvar = PortfolioVaR(df, pd.Series({"AAPL": 1.0}))
    hist = pvar.historical_var(alpha=0.95)
    es = pvar.expected_shortfall(alpha=0.95)
    assert es >= hist


# --------------------------------------------------------------------------- #
# 5. Single-symbol, single-weight reduces to that symbol's VaR
# --------------------------------------------------------------------------- #


def test_single_symbol_single_weight_reduces_to_own_var() -> None:
    df = _two_symbol_panel()
    pvar_full = PortfolioVaR(df, pd.Series({"AAPL": 1.0, "MSFT": 0.0}))
    # Direct single-symbol VaR for comparison.
    pvar_only = PortfolioVaR(df[["AAPL"]], pd.Series({"AAPL": 1.0}))
    assert pvar_full.historical_var(0.95) == pytest.approx(
        pvar_only.historical_var(0.95), rel=1e-9
    )


# --------------------------------------------------------------------------- #
# 6. Two-symbol diversification benefit
# --------------------------------------------------------------------------- #


def test_two_symbol_diversification_benefit() -> None:
    df = _two_symbol_panel()
    w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    pvar = PortfolioVaR(df, w)
    v_port = pvar.historical_var(0.95)

    v_a = PortfolioVaR(df[["AAPL"]], pd.Series({"AAPL": 0.5})).historical_var(0.95)
    v_b = PortfolioVaR(df[["MSFT"]], pd.Series({"MSFT": 0.5})).historical_var(0.95)
    assert v_port < v_a + v_b


# --------------------------------------------------------------------------- #
# 7. Empty returns raises
# --------------------------------------------------------------------------- #


def test_empty_returns_raises() -> None:
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        PortfolioVaR(empty, pd.Series({"AAPL": 1.0}))


# --------------------------------------------------------------------------- #
# 8. Alpha bounds
# --------------------------------------------------------------------------- #


def test_alpha_bounds() -> None:
    df = _gauss_panel(n=1000)
    pvar = PortfolioVaR(df, pd.Series({"AAPL": 1.0}))
    # 0.5 allowed (median).
    _ = pvar.historical_var(alpha=0.5)
    with pytest.raises(ValueError):
        pvar.historical_var(alpha=0.0)
    with pytest.raises(ValueError):
        pvar.historical_var(alpha=1.0)
    with pytest.raises(ValueError):
        pvar.parametric_var(alpha=-0.1)
    with pytest.raises(ValueError):
        pvar.cornish_fisher_var(alpha=1.5)
    with pytest.raises(ValueError):
        pvar.expected_shortfall(alpha=0.0)


# --------------------------------------------------------------------------- #
# 9. Reproducibility / idempotence
# --------------------------------------------------------------------------- #


def test_reproducibility_idempotent() -> None:
    df = _gauss_panel()
    w = pd.Series({"AAPL": 1.0})
    p1 = PortfolioVaR(df, w)
    p2 = PortfolioVaR(df, w)
    assert p1.historical_var(0.95) == p2.historical_var(0.95)
    assert p1.parametric_var(0.95) == p2.parametric_var(0.95)
    assert p1.cornish_fisher_var(0.95) == p2.cornish_fisher_var(0.95)
    assert p1.expected_shortfall(0.95) == p2.expected_shortfall(0.95)
    # Same object called twice.
    assert p1.historical_var(0.95) == p1.historical_var(0.95)


# --------------------------------------------------------------------------- #
# 10. Negative weights (shorts) are allowed
# --------------------------------------------------------------------------- #


def test_negative_weights_allowed() -> None:
    df = _two_symbol_panel()
    w = pd.Series({"AAPL": 1.0, "MSFT": -0.5})
    pvar = PortfolioVaR(df, w)
    v = pvar.historical_var(0.95)
    assert np.isfinite(v)
    assert v > 0.0


# --------------------------------------------------------------------------- #
# Sanity: z_from_alpha helper
# --------------------------------------------------------------------------- #


def test_z_from_alpha_known_points() -> None:
    assert _z_from_alpha(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert _z_from_alpha(0.99) == pytest.approx(2.3263, abs=1e-4)
    assert _z_from_alpha(0.995) == pytest.approx(2.5758, abs=1e-4)
    with pytest.raises(ValueError):
        _z_from_alpha(0.0)
    with pytest.raises(ValueError):
        _z_from_alpha(1.0)
