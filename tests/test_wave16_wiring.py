"""Tests for wave-16 module wiring into trading_cycle.py.

Covers:
  Step 2.12 — features.weekly_alignment (add_weekly_alignment)
  Step 5.6  — risk.tail_dependence (compute_empirical_tail_dependence, classify_tail_regime)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.weekly_alignment import (
    add_weekly_alignment,
    WeeklyAlignmentConfig,
)
from src.assembled_core.risk.tail_dependence import (
    compute_empirical_tail_dependence,
    compute_portfolio_tail_dependence_score,
    classify_tail_regime,
)


# ---------------------------------------------------------------------------
# add_weekly_alignment (Step 2.12)
# ---------------------------------------------------------------------------

def _make_daily_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "close": 100.0 + np.cumsum(rng.normal(0, 0.5, n)),
        "daily_trend": rng.standard_normal(n),
    }, index=idx)


def test_weekly_alignment_returns_df():
    df = _make_daily_df()
    result = add_weekly_alignment(df)
    assert isinstance(result, pd.DataFrame)


def test_weekly_alignment_adds_columns():
    df = _make_daily_df()
    result = add_weekly_alignment(df)
    assert "weekly_alignment_ok" in result.columns
    assert "weekly_ema_slope" in result.columns


def test_weekly_alignment_row_count_preserved():
    df = _make_daily_df()
    result = add_weekly_alignment(df)
    assert len(result) == len(df)


def test_weekly_alignment_ok_is_bool():
    df = _make_daily_df()
    result = add_weekly_alignment(df)
    assert result["weekly_alignment_ok"].dtype == bool


def test_weekly_alignment_missing_close_raises():
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({"daily_trend": np.ones(20)}, index=idx)
    with pytest.raises(ValueError, match="close"):
        add_weekly_alignment(df)


def test_weekly_alignment_missing_daily_trend_raises():
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({"close": np.ones(20)}, index=idx)
    with pytest.raises(ValueError):
        add_weekly_alignment(df)


def test_weekly_alignment_not_datetime_index_raises():
    df = pd.DataFrame({
        "close": [100.0] * 20,
        "daily_trend": [0.1] * 20,
    })
    with pytest.raises(ValueError, match="datetime"):
        add_weekly_alignment(df)


def test_weekly_alignment_with_symbol_column():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "close": 100.0 + np.cumsum(rng.normal(0, 0.5, 40)),
        "daily_trend": rng.standard_normal(40),
        "symbol": ["A"] * 40,
    }, index=idx)
    result = add_weekly_alignment(df)
    assert "weekly_alignment_ok" in result.columns


# ---------------------------------------------------------------------------
# tail_dependence (Step 5.6)
# ---------------------------------------------------------------------------

def _make_returns(n_symbols: int = 5, n_days: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days)
    return pd.DataFrame(rng.standard_normal((n_days, n_symbols)),
                        index=idx, columns=[f"S{i}" for i in range(n_symbols)])


def test_tail_dep_matrix_returns_df():
    returns = _make_returns()
    result = compute_empirical_tail_dependence(returns)
    assert isinstance(result, pd.DataFrame)


def test_tail_dep_matrix_is_square():
    returns = _make_returns(n_symbols=4)
    result = compute_empirical_tail_dependence(returns)
    assert result.shape == (4, 4)


def test_tail_dep_matrix_diagonal_is_1():
    returns = _make_returns()
    result = compute_empirical_tail_dependence(returns)
    for sym in result.columns:
        assert abs(result.loc[sym, sym] - 1.0) < 1e-9


def test_tail_dep_matrix_values_in_01():
    returns = _make_returns()
    result = compute_empirical_tail_dependence(returns)
    assert (result.values >= 0).all() and (result.values <= 1.0 + 1e-9).all()


def test_tail_dep_score_in_range():
    returns = _make_returns()
    matrix = compute_empirical_tail_dependence(returns)
    score = compute_portfolio_tail_dependence_score(matrix)
    assert 0.0 <= score <= 1.0


def test_tail_dep_too_few_rows_raises():
    returns = _make_returns(n_days=10)
    with pytest.raises(ValueError):
        compute_empirical_tail_dependence(returns)


def test_classify_tail_regime_low():
    assert classify_tail_regime(0.05) == "low"


def test_classify_tail_regime_medium():
    assert classify_tail_regime(0.25) == "medium"


def test_classify_tail_regime_high():
    assert classify_tail_regime(0.50) == "high"
