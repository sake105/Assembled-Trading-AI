"""Tests for scripts/forensic/_helpers.py (F-S2-HOL-1 helper-extract)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.forensic._helpers import annualised_sharpe, max_drawdown


# ---------------------------------------------------------------------------
# annualised_sharpe
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAnnualisedSharpe:
    def test_positive_drift_positive_sharpe(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, size=500)
        s = annualised_sharpe(r)
        assert s > 0.5

    def test_negative_drift_negative_sharpe(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(-0.001, 0.01, size=500)
        s = annualised_sharpe(r)
        assert s < 0.0

    def test_zero_std_returns_nan(self) -> None:
        assert np.isnan(annualised_sharpe(np.zeros(50)))

    def test_short_series_returns_nan(self) -> None:
        assert np.isnan(annualised_sharpe(np.array([0.01])))
        assert np.isnan(annualised_sharpe(np.array([])))

    def test_annualisation_factor_scales(self) -> None:
        rng = np.random.default_rng(7)
        r = rng.normal(0.001, 0.01, size=100)
        s_daily = annualised_sharpe(r, periods_per_year=252)
        s_monthly = annualised_sharpe(r, periods_per_year=12)
        # Ratio = sqrt(252/12) ≈ 4.58
        assert abs(s_daily / s_monthly - np.sqrt(252 / 12)) < 1e-6

    def test_nan_mean_returns_nan(self) -> None:
        """If mean is NaN (e.g. all-NaN returns), result is NaN."""
        r = np.full(10, np.nan)
        assert np.isnan(annualised_sharpe(r))


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestMaxDrawdown:
    def test_no_drawdown_monotonic_up(self) -> None:
        eq = np.linspace(100, 200, 50)
        assert max_drawdown(eq) == 0.0

    def test_single_drawdown_episode(self) -> None:
        eq = np.array([100, 110, 90, 95, 120])
        # Peak 110 at idx 1, trough 90 at idx 2 → (90-110)/110 = -0.1818
        mdd = max_drawdown(eq)
        assert mdd == pytest.approx(-0.18181818, abs=1e-6)

    def test_empty_or_single_value(self) -> None:
        assert max_drawdown(np.array([])) == 0.0
        assert max_drawdown(np.array([100.0])) == 0.0

    def test_negative_only_path(self) -> None:
        """Monotonically decreasing equity → MDD approaches (final-first)/first."""
        eq = np.array([100, 80, 60, 40, 20])
        mdd = max_drawdown(eq)
        # Peak at start, trough at end: (20-100)/100 = -0.8
        assert mdd == pytest.approx(-0.8, abs=1e-9)

    def test_recovery_then_new_high(self) -> None:
        """Drawdown then recovery + new high: MDD captures the original dip."""
        eq = np.array([100, 110, 85, 115, 130])
        # Peak 110 idx 1, trough 85 idx 2 → -0.2272
        mdd = max_drawdown(eq)
        assert mdd == pytest.approx(-0.22727273, abs=1e-6)
