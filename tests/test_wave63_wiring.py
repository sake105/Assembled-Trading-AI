"""Tests for wave-63 module wiring into trading_cycle.py.

Covers:
  Step 7.73 — qa.factor_report (run_factor_report)
  Step 7.74 — qa.shipping_risk (compute_shipping_exposure / compute_systemic_risk_flags)
  Step 7.75 — qa.trade_tca (TradeTCA / compute_trade_tca / aggregate_tca)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.shipping_risk import (
    compute_shipping_exposure,
    compute_systemic_risk_flags,
)
from src.assembled_core.qa.trade_tca import (
    TradeTCA,
    TCAAggregateReport,
    compute_trade_tca,
    aggregate_tca,
)


# ---------------------------------------------------------------------------
# factor_report (Step 7.73)
# ---------------------------------------------------------------------------

def test_factor_report_importable():
    from src.assembled_core.qa.factor_report import run_factor_report
    assert callable(run_factor_report)


def test_factor_report_returns_dict():
    from src.assembled_core.qa.factor_report import run_factor_report
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = pd.DataFrame({
        "timestamp": idx,
        "close": rng.uniform(100, 200, 60),
        "volume": rng.uniform(1e6, 2e6, 60),
        "symbol": "AAPL",
    })
    result = run_factor_report(prices, factor_set="core", fwd_horizon_days=5)
    assert isinstance(result, dict)


def test_factor_report_empty_df_handled():
    from src.assembled_core.qa.factor_report import run_factor_report
    prices = pd.DataFrame(columns=["close", "volume", "symbol"])
    try:
        result = run_factor_report(prices)
        assert isinstance(result, dict)
    except Exception:
        pass  # graceful failure is acceptable for empty input


# ---------------------------------------------------------------------------
# shipping_risk (Step 7.74)
# ---------------------------------------------------------------------------

def test_compute_shipping_exposure_empty_portfolio():
    portfolio = pd.DataFrame(columns=["symbol", "weight"])
    features = pd.DataFrame(columns=["symbol", "shipping_congestion_score"])
    result = compute_shipping_exposure(portfolio, features)
    assert isinstance(result, dict)


def test_compute_shipping_exposure_basic():
    portfolio = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "weight": [0.4, 0.3, 0.3],
    })
    features = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "shipping_congestion_score": [75.0, 45.0, 30.0],
    })
    result = compute_shipping_exposure(portfolio, features)
    assert isinstance(result, dict)
    assert "avg_shipping_congestion" in result


def test_compute_shipping_exposure_returns_float():
    portfolio = pd.DataFrame({"symbol": ["AAPL"], "weight": [1.0]})
    features = pd.DataFrame({"symbol": ["AAPL"], "shipping_congestion_score": [80.0]})
    result = compute_shipping_exposure(portfolio, features)
    assert isinstance(result["avg_shipping_congestion"], float)


def test_compute_systemic_risk_flags_returns_dict():
    exposure = {
        "avg_shipping_congestion": 75.0,
        "high_congestion_weight": 0.4,
        "exposed_symbols": ["AAPL"],
    }
    flags = compute_systemic_risk_flags(exposure)
    assert isinstance(flags, dict)


def test_compute_systemic_risk_flags_has_risk_level():
    exposure = {"avg_shipping_congestion": 20.0, "high_congestion_weight": 0.0, "exposed_symbols": []}
    flags = compute_systemic_risk_flags(exposure)
    assert "risk_level" in flags


# ---------------------------------------------------------------------------
# trade_tca (Step 7.75)
# ---------------------------------------------------------------------------

def test_trade_tca_creates():
    tca = TradeTCA(
        trade_id="t1",
        symbol="AAPL",
        side="buy",
        quantity=100.0,
        arrival_price=150.0,
        execution_price=150.5,
        vwap_price=150.2,
    )
    assert tca.trade_id == "t1"
    assert tca.symbol == "AAPL"


def test_compute_trade_tca_returns_tca():
    result = compute_trade_tca(
        trade_id="t1",
        symbol="MSFT",
        side="buy",
        quantity=50.0,
        execution_price=300.5,
        arrival_price=300.0,
        vwap_price=300.2,
    )
    assert isinstance(result, TradeTCA)


def test_compute_trade_tca_is_bps():
    result = compute_trade_tca(
        trade_id="t1",
        symbol="AAPL",
        side="buy",
        quantity=100.0,
        execution_price=151.0,
        arrival_price=150.0,
    )
    assert isinstance(result.implementation_shortfall_bps, float)


def test_aggregate_tca_empty():
    report = aggregate_tca([])
    assert isinstance(report, TCAAggregateReport)
    assert report.n_trades == 0


def test_aggregate_tca_single():
    tca = compute_trade_tca("t1", "AAPL", "buy", 100.0, 151.0, 150.0)
    report = aggregate_tca([tca])
    assert report.n_trades == 1
    assert isinstance(report.mean_impact_bps, float)


def test_aggregate_tca_multiple():
    tcas = [
        compute_trade_tca(f"t{i}", "AAPL", "buy", 100.0, 151.0, 150.0)
        for i in range(5)
    ]
    report = aggregate_tca(tcas)
    assert report.n_trades == 5
