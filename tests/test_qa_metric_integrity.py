"""Regression tests for Diagnostik Batch 2 (qa metric integrity).

A3  qa_gates.check_max_drawdown crashed with `None < float` TypeError when
    max_drawdown_pct was None (it lacked the None-guard every sibling gate has).
    Now it degrades to WARNING.
A4  qa.risk_metrics.compute_portfolio_risk_metrics back-filled leading pre-inception
    NaNs via ffill().bfill(), fabricating a flat early segment from a *future* value
    that diluted volatility / inflated Sharpe. Now ffill().dropna() (PIT-safe).
--  qa.bootstrap_metrics._sharpe returned 0.0 on an exact zero-vol sample, masking the
    degenerate case as a real Sharpe of 0. Now returns NaN; the CI filters non-finite.

Each test fails on the pre-fix code and passes after the fix.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.bootstrap_metrics import _sharpe, compute_sharpe_with_ci
from src.assembled_core.qa.qa_gates import QAResult, check_max_drawdown
from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# --- A3: None-guard --------------------------------------------------------
def test_check_max_drawdown_none_degrades_to_warning_not_crash():
    """A None max_drawdown_pct must WARN, not raise `None < float` TypeError."""
    result = check_max_drawdown(SimpleNamespace(max_drawdown_pct=None))
    assert result.result == QAResult.WARNING
    assert result.details["max_drawdown_pct"] is None


def test_check_max_drawdown_real_value_still_flows_through():
    """Regression guard: a real (non-None) value behaves exactly as before."""
    breach = check_max_drawdown(
        SimpleNamespace(max_drawdown_pct=-50.0), max_dd_pct_limit=-20.0
    )
    assert breach.result == QAResult.BLOCK
    ok = check_max_drawdown(
        SimpleNamespace(max_drawdown_pct=-5.0), max_dd_pct_limit=-20.0
    )
    assert ok.result != QAResult.BLOCK


# --- A4: no back-fill of leading pre-inception NaNs ------------------------
def test_leading_nan_equity_not_backfilled_into_flat_segment():
    """Leading NaNs must be dropped, not back-filled from a later (future) value.

    Old ffill().bfill() turned the leading NaNs into the first valid equity, adding
    spurious flat (zero-return) points that diluted volatility. The cleaned metrics
    must match those of the same curve without the leading NaNs.
    """
    clean = pd.Series([100.0, 102.0, 99.0, 103.0, 101.0, 104.0])
    with_lead = pd.Series([np.nan, np.nan, 100.0, 102.0, 99.0, 103.0, 101.0, 104.0])
    m_clean = compute_portfolio_risk_metrics(clean)
    m_lead = compute_portfolio_risk_metrics(with_lead)
    assert m_clean["max_drawdown"] == pytest.approx(m_lead["max_drawdown"])
    assert m_clean["daily_vol"] == pytest.approx(m_lead["daily_vol"])
    assert m_clean["ann_vol"] == pytest.approx(m_lead["ann_vol"])


def test_internal_gap_still_forward_filled_pit_safe():
    """An internal gap is still forward-filled (last-known, PIT-safe) — not dropped."""
    gapped = pd.Series([100.0, np.nan, 102.0, 103.0])
    m = compute_portfolio_risk_metrics(gapped)
    # 4 points after ffill -> 3 returns -> vol computable (not None)
    assert m["daily_vol"] is not None


# --- bootstrap: zero-vol -> NaN, not 0.0 -----------------------------------
def test_sharpe_returns_nan_on_exact_zero_vol():
    """An exactly flat (zero-vol) sample has no Sharpe — must be NaN, not 0.0."""
    assert math.isnan(_sharpe(np.zeros(20)))


def test_compute_sharpe_with_ci_degenerate_is_nan_not_zero():
    """A fully degenerate (all-zero) input yields NaN point + NaN CIs, not 0.0."""
    result = compute_sharpe_with_ci(pd.Series(np.zeros(50)), n_bootstrap=50, seed=3)
    assert math.isnan(result["sharpe"])
    assert math.isnan(result["sharpe_ci_lower"])
    assert math.isnan(result["sharpe_ci_upper"])
    assert math.isnan(result["sharpe_p_value"])
