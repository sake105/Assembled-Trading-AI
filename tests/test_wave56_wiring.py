"""Tests for wave-56 module wiring into trading_cycle.py.

Covers:
  Step 5.97 — risk.exposure_engine (compute_target_positions / compute_exposures)
  Step 5.98 — risk.intraday_monitor (IntradayRiskConfig / IntradayRiskMonitor)
  Step 5.99 — portfolio.market_neutral_optimizer (MarketNeutralConfig / optimize_market_neutral)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.exposure_engine import (
    compute_target_positions,
    compute_exposures,
    ExposureSummary,
)
from src.assembled_core.risk.intraday_monitor import (
    IntradayRiskConfig,
    IntradayRiskMonitor,
    PositionSnapshot,
    RiskAlert,
)
from src.assembled_core.portfolio.market_neutral_optimizer import (
    MarketNeutralConfig,
    MarketNeutralResult,
    optimize_market_neutral,
    CVXPY_AVAILABLE,
)


# ---------------------------------------------------------------------------
# exposure_engine (Step 5.97)
# ---------------------------------------------------------------------------

def _make_positions() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [100.0, -50.0]})


def _make_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAPL", "GOOG"],
        "side": ["BUY", "BUY"],
        "qty": [50.0, 100.0],
    })


def test_compute_target_positions_returns_df():
    positions = _make_positions()
    orders = _make_orders()
    result = compute_target_positions(positions, orders)
    assert isinstance(result, pd.DataFrame)


def test_compute_target_positions_has_symbol():
    result = compute_target_positions(_make_positions(), _make_orders())
    assert "symbol" in result.columns


def test_compute_target_positions_empty_inputs():
    positions = pd.DataFrame(columns=["symbol", "qty"])
    orders = pd.DataFrame(columns=["symbol", "side", "qty"])
    result = compute_target_positions(positions, orders)
    assert isinstance(result, pd.DataFrame)


def test_compute_exposures_returns_tuple():
    positions = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "target_qty": [100.0, -50.0]})
    prices = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "close": [150.0, 300.0]})
    exp_df, summary = compute_exposures(positions, prices, equity=1_000_000.0)
    assert isinstance(summary, ExposureSummary)
    assert isinstance(exp_df, pd.DataFrame)


def test_exposure_summary_fields():
    positions = pd.DataFrame({"symbol": ["AAPL"], "target_qty": [100.0]})
    prices = pd.DataFrame({"symbol": ["AAPL"], "close": [150.0]})
    _, summary = compute_exposures(positions, prices, equity=1_000_000.0)
    assert hasattr(summary, "gross_exposure")
    assert hasattr(summary, "net_exposure")
    assert summary.n_positions >= 0


# ---------------------------------------------------------------------------
# intraday_monitor (Step 5.98)
# ---------------------------------------------------------------------------

def test_intraday_risk_config_creates():
    cfg = IntradayRiskConfig()
    assert isinstance(cfg, IntradayRiskConfig)


def test_intraday_risk_config_defaults():
    cfg = IntradayRiskConfig()
    assert cfg.max_intraday_drawdown_pct > 0
    assert cfg.var_confidence > 0.5


def test_intraday_risk_monitor_creates():
    monitor = IntradayRiskMonitor()
    assert isinstance(monitor, IntradayRiskMonitor)


def test_position_snapshot_creates():
    snap = PositionSnapshot(symbol="AAPL", shares=100, entry_price=150.0)
    assert snap.symbol == "AAPL"
    assert snap.entry_price == 150.0


def test_position_snapshot_update_price():
    snap = PositionSnapshot(symbol="AAPL", shares=100, entry_price=150.0)
    snap.update_price(165.0)
    assert snap.current_price == 165.0
    assert snap.pnl == pytest.approx(100 * (165.0 - 150.0))


# ---------------------------------------------------------------------------
# market_neutral_optimizer (Step 5.99)
# ---------------------------------------------------------------------------

def test_market_neutral_config_creates():
    cfg = MarketNeutralConfig()
    assert isinstance(cfg, MarketNeutralConfig)


def test_market_neutral_config_defaults():
    cfg = MarketNeutralConfig()
    assert cfg.max_weight > 0
    assert cfg.max_gross_exposure > 0
    assert cfg.beta_neutral in (True, False)


def test_cvxpy_available_flag():
    assert isinstance(CVXPY_AVAILABLE, bool)


def _make_market_neutral_inputs(n_assets: int = 6) -> tuple:
    rng = np.random.default_rng(0)
    symbols = [f"S{i}" for i in range(n_assets)]
    scores = pd.Series(rng.normal(0, 1, n_assets), index=symbols)
    returns = pd.DataFrame(rng.normal(0, 0.01, (60, n_assets)), columns=symbols)
    betas = pd.Series(rng.normal(1, 0.3, n_assets), index=symbols)
    return symbols, scores, returns, betas


def test_optimize_market_neutral_returns_result():
    symbols, scores, returns, betas = _make_market_neutral_inputs()
    cfg = MarketNeutralConfig()
    result = optimize_market_neutral(scores, returns.cov(), betas=betas, config=cfg)
    assert isinstance(result, MarketNeutralResult)


def test_market_neutral_result_has_weights():
    symbols, scores, returns, betas = _make_market_neutral_inputs()
    cfg = MarketNeutralConfig()
    result = optimize_market_neutral(scores, returns.cov(), betas=betas, config=cfg)
    assert hasattr(result, "long_weights")
    assert hasattr(result, "short_weights")
    assert isinstance(result.long_weights, dict)
