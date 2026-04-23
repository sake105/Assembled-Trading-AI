"""Tests for wave-48 module wiring into trading_cycle.py.

Covers:
  Step 2.33 — features.intraday_features (build_intraday_features)
  Step 2.34 — features.options_derived_signals (build_options_regime_factors)
  Step 2.35 — features.cross_asset_leads (build_cross_asset_signals)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.intraday_features import (
    build_intraday_features,
    compute_overnight_return,
    compute_vwap_deviation,
    IntradayFeatureResult,
)
from src.assembled_core.features.options_derived_signals import (
    build_options_regime_factors,
    compute_vix_term_structure,
    compute_implied_vs_realized_spread,
)
from src.assembled_core.features.cross_asset_leads import (
    build_cross_asset_signals,
    compute_bond_equity_signal,
    CrossAssetSignal,
)


# ---------------------------------------------------------------------------
# build_intraday_features (Step 2.33)
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 50) -> tuple:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = pd.Series(rng.uniform(1e5, 1e6, n), index=idx)
    return pd.Series(open_, index=idx), high, low, close, volume


def test_build_intraday_returns_result():
    open_, high, low, close, vol = _make_ohlcv()
    result = build_intraday_features(open_, high, low, close, vol)
    assert isinstance(result, IntradayFeatureResult)


def test_build_intraday_features_df():
    open_, high, low, close, vol = _make_ohlcv()
    result = build_intraday_features(open_, high, low, close, vol)
    assert isinstance(result.features, pd.DataFrame)


def test_build_intraday_has_vwap_deviation():
    open_, high, low, close, vol = _make_ohlcv()
    result = build_intraday_features(open_, high, low, close, vol)
    assert "vwap_deviation" in result.features.columns


def test_build_intraday_coverage_in_01():
    open_, high, low, close, vol = _make_ohlcv()
    result = build_intraday_features(open_, high, low, close, vol)
    assert 0.0 <= result.coverage <= 1.0


def test_compute_overnight_return_returns_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    open_ = pd.Series(100.0 + rng.normal(0, 0.5, 30), index=idx)
    close = pd.Series(100.0 + rng.normal(0, 0.5, 30), index=idx)
    overnight, intraday = compute_overnight_return(open_, close)
    assert isinstance(overnight, pd.Series)
    assert isinstance(intraday, pd.Series)


def test_compute_vwap_deviation_returns_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    close = pd.Series(100.0 + rng.normal(0, 0.5, 30), index=idx)
    volume = pd.Series(rng.uniform(1e5, 1e6, 30), index=idx)
    result = compute_vwap_deviation(close, volume)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# build_options_regime_factors (Step 2.34)
# ---------------------------------------------------------------------------

def _make_cboe_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "vix": 15.0 + rng.uniform(0, 20, n),
        "vix3m": 16.0 + rng.uniform(0, 20, n),
        "put_call_ratio": rng.uniform(0.7, 1.3, n),
    })


def test_build_options_regime_empty_returns_df():
    empty = pd.DataFrame(columns=["timestamp", "vix", "vix3m", "put_call_ratio"])
    result = build_options_regime_factors(empty)
    assert isinstance(result, pd.DataFrame)


def test_build_options_regime_returns_df():
    df = _make_cboe_df()
    result = build_options_regime_factors(df)
    assert isinstance(result, pd.DataFrame)


def test_build_options_regime_has_vix_level():
    df = _make_cboe_df()
    result = build_options_regime_factors(df)
    assert "vix_level" in result.columns


def test_build_options_regime_has_vix_regime():
    df = _make_cboe_df(200)
    result = build_options_regime_factors(df)
    assert "vix_regime" in result.columns


def test_compute_vix_term_structure_returns_float():
    result = compute_vix_term_structure(vix=15.0, vix3m=18.0)
    assert isinstance(result, float)


def test_compute_implied_vs_realized_spread():
    result = compute_implied_vs_realized_spread(vix=18.0, realized_vol_20d=14.0)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# build_cross_asset_signals (Step 2.35)
# ---------------------------------------------------------------------------

def _make_equity_returns(n: int = 60, n_stocks: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = [f"S{i}" for i in range(n_stocks)]
    return pd.DataFrame(rng.normal(0, 0.01, (n, n_stocks)), index=idx, columns=cols)


def test_build_cross_asset_returns_df():
    rets = _make_equity_returns()
    result = build_cross_asset_signals(rets)
    assert isinstance(result, pd.DataFrame)


def test_build_cross_asset_same_shape():
    rets = _make_equity_returns(40, 3)
    result = build_cross_asset_signals(rets)
    assert result.shape == rets.shape


def test_build_cross_asset_no_inputs():
    rets = _make_equity_returns()
    result = build_cross_asset_signals(rets, bond_data=None, commodity_returns=None)
    assert isinstance(result, pd.DataFrame)


def test_compute_bond_equity_signal_returns_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    credit_spread = pd.Series(rng.normal(0.02, 0.005, 50), index=idx)
    term_spread = pd.Series(rng.normal(0.01, 0.003, 50), index=idx)
    market_ret = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    result = compute_bond_equity_signal(credit_spread, term_spread, market_ret)
    assert isinstance(result, pd.Series)
    assert len(result) == 50


def test_cross_asset_signal_creates():
    sig = CrossAssetSignal(
        bond_equity_signal=0.2,
        commodity_sector_signal=0.1,
        fx_adr_signal=0.0,
        composite_signal=0.15,
        confidence=0.7,
    )
    assert isinstance(sig.composite_signal, float)
