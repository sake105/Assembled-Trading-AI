"""Tests for wave-11 module wiring into trading_cycle.py.

Covers:
  Step 2.6 — features.seasonal_features (build_seasonal_features)
  Step 2.7 — features.correlation_features (compute_correlation_regime_features)
  Step 5.9 — risk.param_stability (compute_stability_report)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.seasonal_features import (
    build_seasonal_features,
    get_seasonal_feature_names,
)
from src.assembled_core.features.correlation_features import (
    compute_correlation_regime_features,
)
from src.assembled_core.risk.param_stability import (
    compute_stability_report,
    compute_rolling_vol_estimates,
)


# ---------------------------------------------------------------------------
# build_seasonal_features (Step 2.6)
# ---------------------------------------------------------------------------

def _make_date_index(n: int = 60) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_seasonal_features_returns_df():
    idx = _make_date_index()
    result = build_seasonal_features(idx)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(idx)


def test_seasonal_features_columns():
    idx = _make_date_index()
    result = build_seasonal_features(idx)
    expected = get_seasonal_feature_names()
    for col in expected:
        assert col in result.columns, f"Missing: {col}"


def test_seasonal_features_from_series():
    ts = pd.Series(pd.date_range("2024-01-01", periods=20, freq="B"))
    result = build_seasonal_features(ts)
    assert len(result) == 20


def test_seasonal_features_no_nans():
    idx = _make_date_range = pd.date_range("2024-01-01", periods=100, freq="B")
    result = build_seasonal_features(idx)
    assert not result.isnull().any().any()


def test_seasonal_turn_of_month_binary():
    idx = _make_date_index(120)
    result = build_seasonal_features(idx)
    assert result["seasonal_turn_of_month"].isin([0.0, 1.0]).all()


def test_seasonal_january_only_jan():
    idx = pd.date_range("2024-01-01", periods=10)
    result = build_seasonal_features(idx)
    assert result["seasonal_january"].iloc[0] == 1.0  # January


def test_seasonal_sell_in_may_in_range():
    idx = pd.date_range("2024-06-01", periods=5)
    result = build_seasonal_features(idx)
    # June = May-Oct sell period → signal is -1.0 (directional: -1 sell, +1 hold)
    assert result["seasonal_sell_in_may"].iloc[0] == -1.0


# ---------------------------------------------------------------------------
# compute_correlation_regime_features (Step 2.7)
# ---------------------------------------------------------------------------

def _make_returns_wide(n_days: int = 100, n_syms: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days)
    return pd.DataFrame(
        rng.standard_normal((n_days, n_syms)),
        index=idx,
        columns=[f"S{i}" for i in range(n_syms)],
    )


def test_corr_regime_returns_df():
    returns = _make_returns_wide()
    result = compute_correlation_regime_features(returns)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(returns)


def test_corr_regime_has_required_cols():
    returns = _make_returns_wide()
    result = compute_correlation_regime_features(returns)
    for col in ["avg_corr_short", "avg_corr_long", "corr_regime_zscore", "corr_momentum"]:
        assert col in result.columns


def test_corr_regime_avg_corr_between_neg1_and_1():
    returns = _make_returns_wide(n_days=150)
    result = compute_correlation_regime_features(returns)
    valid = result["avg_corr_short"].dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


def test_corr_regime_high_correlation_detected():
    n = 150
    rng = np.random.default_rng(5)
    # Use common factor + small noise to ensure high pairwise correlation in returns
    common = rng.standard_normal(n)
    returns_data = {f"S{i}": common + 0.05 * rng.standard_normal(n) for i in range(4)}
    returns = pd.DataFrame(returns_data, index=pd.date_range("2024-01-01", periods=n))
    result = compute_correlation_regime_features(returns)
    valid = result["avg_corr_short"].dropna()
    # Highly correlated assets → avg_corr_short should be positive and high
    assert valid.iloc[-1] > 0.5


# ---------------------------------------------------------------------------
# compute_stability_report (Step 5.9)
# ---------------------------------------------------------------------------

def _make_equity_curve(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n)
    equity = 100.0 * np.cumprod(1 + returns)
    return pd.Series(equity)


def test_stability_report_returns_dict():
    equity = _make_equity_curve()
    result = compute_stability_report(equity)
    assert isinstance(result, dict)


def test_stability_report_has_all_stable():
    equity = _make_equity_curve()
    result = compute_stability_report(equity)
    assert "all_stable" in result


def test_stability_report_stable_series():
    equity = _make_equity_curve(n=120, seed=42)
    result = compute_stability_report(equity)
    assert isinstance(result["all_stable"], bool)


def test_stability_report_checks_counts():
    equity = _make_equity_curve(n=100)
    result = compute_stability_report(equity)
    assert "checks_passed" in result
    assert "checks_total" in result
    assert result["checks_passed"] <= result["checks_total"]


def test_rolling_vol_estimates_returns_dict():
    equity = _make_equity_curve(n=80)
    result = compute_rolling_vol_estimates(equity, window_sizes=[10, 20, 40])
    assert isinstance(result, dict)
    assert set(result.keys()) == {10, 20, 40}


def test_rolling_vol_estimates_positive():
    equity = _make_equity_curve(n=80)
    result = compute_rolling_vol_estimates(equity, window_sizes=[20])
    val = result[20]
    assert pd.isna(val) or val > 0.0
