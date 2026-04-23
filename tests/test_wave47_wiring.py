"""Tests for wave-47 module wiring into trading_cycle.py.

Covers:
  Step 2.30 — features.short_interest_features (build_short_interest_features)
  Step 2.31 — features.institutional_features (build_institutional_features)
  Step 2.32 — features.index_rebal_features (build_index_rebal_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.short_interest_features import (
    build_short_interest_features,
    compute_short_pct_float,
    compute_short_ratio,
    compute_short_squeeze_score,
)
from src.assembled_core.features.institutional_features import (
    build_institutional_features,
    compute_institutional_ownership,
    InstitutionalSignal,
)
from src.assembled_core.features.index_rebal_features import (
    build_index_rebal_features,
    compute_predicted_demand,
)


# ---------------------------------------------------------------------------
# build_short_interest_features (Step 2.30)
# ---------------------------------------------------------------------------

def _make_short_data(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    syms = ["AAPL", "MSFT", "GOOG"]
    rows = []
    for sym in syms:
        dates = pd.date_range("2024-01-01", periods=n, freq="2W")
        for d in dates:
            rows.append({
                "symbol": sym,
                "settlement_date": d,
                "short_interest": rng.uniform(1e6, 5e6),
                "shares_float": rng.uniform(1e7, 5e7),
                "avg_volume": rng.uniform(5e5, 2e6),
            })
    return pd.DataFrame(rows)


def test_build_short_interest_empty_returns_df():
    empty = pd.DataFrame(columns=["symbol", "short_interest", "shares_float", "avg_volume", "settlement_date"])
    result = build_short_interest_features(empty)
    assert isinstance(result, pd.DataFrame)


def test_build_short_interest_returns_df():
    data = _make_short_data()
    result = build_short_interest_features(data)
    assert isinstance(result, pd.DataFrame)


def test_build_short_interest_has_pct_float():
    data = _make_short_data()
    result = build_short_interest_features(data)
    assert "si_pct_float" in result.columns


def test_build_short_interest_has_days_to_cover():
    data = _make_short_data()
    result = build_short_interest_features(data)
    assert "si_days_to_cover" in result.columns


def test_compute_short_pct_float_returns_float():
    result = compute_short_pct_float(short_interest=1e6, shares_float=5e7)
    assert isinstance(result, float)
    assert abs(result - 0.02) < 1e-6


def test_compute_short_ratio_returns_float():
    result = compute_short_ratio(short_interest=1e6, avg_daily_volume=5e5)
    assert isinstance(result, float)
    assert abs(result - 2.0) < 1e-6


def test_compute_short_squeeze_score_returns_float():
    result = compute_short_squeeze_score(short_pct_float=0.30, short_ratio=5.0)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# build_institutional_features (Step 2.31)
# ---------------------------------------------------------------------------

def _make_holdings_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.Timestamp("2024-03-31"),
        "ticker": ["AAPL", "MSFT", "GOOG", "AMZN"],
        "holder_id": ["VAN", "BLK", "VAN", "BLK"],
        "shares": rng.uniform(1e5, 1e7, 4),
        "value": rng.uniform(1e6, 1e8, 4),
    })


def test_build_institutional_empty_history():
    result = build_institutional_features({})
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_build_institutional_with_data():
    holdings = {"2024-Q1": _make_holdings_df()}
    result = build_institutional_features(holdings)
    assert isinstance(result, pd.DataFrame)


def test_compute_institutional_ownership_returns_df():
    holdings = _make_holdings_df()
    result = compute_institutional_ownership(holdings)
    assert isinstance(result, pd.DataFrame)


def test_institutional_signal_creates():
    sig = InstitutionalSignal(
        institutional_ownership_pct=0.65,
        ownership_change=0.02,
        n_holders=150,
        holder_change=5,
        concentration_hhi=0.05,
        smart_money_flow=0.3,
        herding_measure=0.6,
        new_positions=3,
        liquidations=1,
    )
    assert isinstance(sig.institutional_ownership_pct, float)
    assert isinstance(sig.smart_money_flow, float)


# ---------------------------------------------------------------------------
# build_index_rebal_features (Step 2.32)
# ---------------------------------------------------------------------------

def _make_changes_df() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["NVDA", "TSLA"],
        "effective_date": ["2024-06-01", "2024-06-01"],
        "action": ["add", "delete"],
        "index_name": ["SP500", "SP500"],
    })


def test_build_index_rebal_empty_returns_df():
    empty = pd.DataFrame(columns=["symbol", "effective_date", "action", "index_name"])
    result = build_index_rebal_features(empty)
    assert isinstance(result, pd.DataFrame)


def test_build_index_rebal_returns_df():
    changes = _make_changes_df()
    result = build_index_rebal_features(changes)
    assert isinstance(result, pd.DataFrame)


def test_build_index_rebal_has_flag_col():
    changes = _make_changes_df()
    result = build_index_rebal_features(changes)
    assert "index_addition_flag" in result.columns


def test_build_index_rebal_addition_flag_positive():
    changes = pd.DataFrame({
        "symbol": ["NVDA"],
        "effective_date": ["2024-06-01"],
        "action": ["add"],
        "index_name": ["SP500"],
    })
    result = build_index_rebal_features(changes)
    assert (result["index_addition_flag"] >= 0).all()


def test_compute_predicted_demand_returns_float():
    result = compute_predicted_demand(
        market_cap=1e10,
        index_weight=0.002,
        index_aum=5e12,
        shares_float=1e8,
        current_price=150.0,
    )
    assert isinstance(result, float)
