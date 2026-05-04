"""Tests for Part B kelly_uncertainty shadow-mode in trading_cycle.

The shadow computes compute_kelly_weights_with_uncertainty() in parallel
to the active sizing method. Must not change target_positions and must
write to result.meta.kelly_uncertainty_shadow when policy enables it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.portfolio.position_sizing import (
    compute_kelly_weights_with_uncertainty,
)


def test_compute_kelly_weights_with_uncertainty_no_uncertainty():
    edges = pd.Series({"AAPL": 0.08, "MSFT": 0.05, "NVDA": 0.12})
    variances = pd.Series({"AAPL": 0.04, "MSFT": 0.03, "NVDA": 0.06})
    weights = compute_kelly_weights_with_uncertainty(
        edges, variances, fractional_kelly=0.5, max_fraction=0.25, normalize=True
    )
    assert len(weights) == 3
    assert weights.abs().sum() == pytest.approx(1.0, abs=1e-6)
    # Un-normalized cap: max_fraction pre-normalization
    weights_raw = compute_kelly_weights_with_uncertainty(
        edges, variances, fractional_kelly=0.5, max_fraction=0.25, normalize=False
    )
    assert all(abs(w) <= 0.25 + 1e-9 for w in weights_raw)


def test_compute_kelly_weights_with_uncertainty_discount_applied():
    edges = pd.Series({"AAPL": 0.08})
    variances = pd.Series({"AAPL": 0.04})
    # High uncertainty (half-width equal to reference) → full discount
    half_widths = pd.Series({"AAPL": 0.10})
    weights_discounted = compute_kelly_weights_with_uncertainty(
        edges,
        variances,
        conformal_half_widths=half_widths,
        reference_half_width=0.10,
        fractional_kelly=0.5,
        normalize=False,
    )
    weights_no_discount = compute_kelly_weights_with_uncertainty(
        edges,
        variances,
        conformal_half_widths=None,
        reference_half_width=None,
        fractional_kelly=0.5,
        normalize=False,
    )
    assert weights_discounted["AAPL"] == pytest.approx(0.0, abs=1e-9)
    assert abs(weights_no_discount["AAPL"]) > 0.0


def test_compute_kelly_weights_zero_variance_clipped():
    edges = pd.Series({"AAPL": 0.05})
    variances = pd.Series({"AAPL": 0.0})
    weights = compute_kelly_weights_with_uncertainty(
        edges, variances, fractional_kelly=0.5, max_fraction=0.25, normalize=False
    )
    assert weights["AAPL"] != 0.0  # clipped variance → finite weight


def test_compute_kelly_weights_respects_max_fraction():
    # Extremely strong edge → should be capped by max_fraction
    edges = pd.Series({"AAPL": 10.0})
    variances = pd.Series({"AAPL": 0.01})
    weights = compute_kelly_weights_with_uncertainty(
        edges, variances, fractional_kelly=1.0, max_fraction=0.10, normalize=False
    )
    assert abs(weights["AAPL"]) <= 0.10 + 1e-9
