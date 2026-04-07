"""M16 Welle 2 — Tests for HIGH-priority fixes.

Verifies:
- Market confirmation module
- ATR-based dynamic stop-losses
- Data quality checks
- Drawdown-based risk level
- Min-trade-value filtering in order generation
- CVaR pre-trade check

Marker: phase12
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Market Confirmation
# ---------------------------------------------------------------------------

class TestMarketConfirmation:
    """Test market_confirmation.py module basics (no yfinance calls)."""

    @pytest.mark.phase12
    def test_import(self):
        from src.assembled_core.intel.market_confirmation import compute_market_confirmation
        assert callable(compute_market_confirmation)

    @pytest.mark.phase12
    def test_returns_dict_structure(self):
        """Without yfinance or with import error, returns conservative zeros."""
        from src.assembled_core.intel.market_confirmation import compute_market_confirmation
        result = compute_market_confirmation(lookback_days=5)
        assert isinstance(result, dict)
        assert "oil_move" in result
        assert "gold_move" in result
        assert "vix_spike" in result
        assert "computed_utc" in result
        # Should be conservative (no confirmation) if yfinance fails
        assert isinstance(result["vix_spike"], bool)


# ---------------------------------------------------------------------------
# ATR-based Stop-Loss
# ---------------------------------------------------------------------------

class TestATRStop:
    """Test compute_atr_stop_pct function."""

    @pytest.mark.phase12
    def test_import(self):
        from src.assembled_core.risk.short_risk import compute_atr_stop_pct
        assert callable(compute_atr_stop_pct)

    @pytest.mark.phase12
    def test_basic_computation(self):
        from src.assembled_core.risk.short_risk import compute_atr_stop_pct

        np.random.seed(42)
        n = 30
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        df = pd.DataFrame({
            "symbol": ["TEST"] * n,
            "high": high,
            "low": low,
            "close": close,
        })

        stop = compute_atr_stop_pct(df, "TEST", atr_period=14, regime="sideways")
        assert stop is not None
        assert 0.0 < stop <= 0.50

    @pytest.mark.phase12
    def test_regime_affects_stop_width(self):
        from src.assembled_core.risk.short_risk import compute_atr_stop_pct

        np.random.seed(42)
        n = 30
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        df = pd.DataFrame({
            "symbol": ["TEST"] * n,
            "high": high,
            "low": low,
            "close": close,
        })

        stop_bull = compute_atr_stop_pct(df, "TEST", regime="bull")
        stop_crisis = compute_atr_stop_pct(df, "TEST", regime="crisis")
        assert stop_bull is not None
        assert stop_crisis is not None
        # Bull should have wider stop (3x ATR) than crisis (1.5x ATR)
        assert stop_bull > stop_crisis

    @pytest.mark.phase12
    def test_insufficient_data_returns_none(self):
        from src.assembled_core.risk.short_risk import compute_atr_stop_pct

        df = pd.DataFrame({
            "symbol": ["TEST"] * 5,
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
        })
        stop = compute_atr_stop_pct(df, "TEST", atr_period=14)
        assert stop is None

    @pytest.mark.phase12
    def test_mark_to_market_with_atr_stops(self):
        """Test that mark_to_market_check uses atr_stops when provided."""
        from src.assembled_core.risk.short_risk import ShortRiskManager

        mgr = ShortRiskManager()
        positions = pd.DataFrame({
            "symbol": ["AAPL", "TSLA"],
        })
        current_prices = pd.Series({"AAPL": 116.0, "TSLA": 108.0})
        entry_prices = pd.Series({"AAPL": 100.0, "TSLA": 100.0})

        # Without ATR stops: default 15% → AAPL (+16%) hits, TSLA (+8%) doesn't
        stops_default = mgr.mark_to_market_check(positions, current_prices, entry_prices)
        assert "AAPL" in stops_default
        assert "TSLA" not in stops_default

        # With ATR stops: AAPL gets 20% stop (wider), TSLA gets 5% stop (tighter)
        atr_stops = {"AAPL": 0.20, "TSLA": 0.05}
        stops_atr = mgr.mark_to_market_check(
            positions, current_prices, entry_prices, atr_stops=atr_stops
        )
        assert "AAPL" not in stops_atr  # 16% < 20% → no stop
        assert "TSLA" in stops_atr      # 8% > 5% → stop triggered


# ---------------------------------------------------------------------------
# Data Quality Checks
# ---------------------------------------------------------------------------

class TestDataQualityChecks:
    """Test quality_checks.py module."""

    @pytest.mark.phase12
    def test_import(self):
        from src.assembled_core.data.quality_checks import check_price_quality
        assert callable(check_price_quality)

    @pytest.mark.phase12
    def test_clean_data_passes(self):
        from src.assembled_core.data.quality_checks import check_price_quality

        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "close": np.random.uniform(90, 110, 100),
        })
        result = check_price_quality(df, "TEST")
        assert result.passed
        assert result.rows_checked == 100

    @pytest.mark.phase12
    def test_detects_null_prices(self):
        from src.assembled_core.data.quality_checks import check_price_quality

        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        close = np.random.uniform(90, 110, 20)
        close[5:15] = np.nan  # 50% NaN
        df = pd.DataFrame({"date": dates, "close": close})
        result = check_price_quality(df, "TEST")
        assert not result.passed
        assert any(i["type"] == "null_prices" for i in result.issues)

    @pytest.mark.phase12
    def test_detects_extreme_returns(self):
        from src.assembled_core.data.quality_checks import check_price_quality

        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        close = [100.0] * 20
        close[10] = 200.0  # +100% jump
        df = pd.DataFrame({"date": dates, "close": close})
        result = check_price_quality(df, "TEST")
        assert any(i["type"] == "extreme_returns" for i in result.issues)

    @pytest.mark.phase12
    def test_insufficient_data_fails(self):
        from src.assembled_core.data.quality_checks import check_price_quality

        df = pd.DataFrame({"close": [100.0, 101.0]})
        result = check_price_quality(df, "TEST", min_rows=10)
        assert not result.passed
        assert any(i["type"] == "insufficient_data" for i in result.issues)


# ---------------------------------------------------------------------------
# Drawdown Risk Level
# ---------------------------------------------------------------------------

class TestDrawdownRiskLevel:
    """Test compute_drawdown_risk_level function."""

    @pytest.mark.phase12
    def test_import(self):
        from src.assembled_core.risk.state_machine import compute_drawdown_risk_level
        assert callable(compute_drawdown_risk_level)

    @pytest.mark.phase12
    def test_normal(self):
        from src.assembled_core.risk.state_machine import compute_drawdown_risk_level
        level, cap = compute_drawdown_risk_level(-2.0)
        assert level == "NORMAL"
        assert cap == 1.0

    @pytest.mark.phase12
    def test_caution(self):
        from src.assembled_core.risk.state_machine import compute_drawdown_risk_level
        level, cap = compute_drawdown_risk_level(-7.0)
        assert level == "CAUTION"
        assert cap == 0.75

    @pytest.mark.phase12
    def test_reduce(self):
        from src.assembled_core.risk.state_machine import compute_drawdown_risk_level
        level, cap = compute_drawdown_risk_level(-12.0)
        assert level == "REDUCE"
        assert cap == 0.50

    @pytest.mark.phase12
    def test_minimum(self):
        from src.assembled_core.risk.state_machine import compute_drawdown_risk_level
        level, cap = compute_drawdown_risk_level(-20.0)
        assert level == "MINIMUM"
        assert cap == 0.25


# ---------------------------------------------------------------------------
# Min Trade Value in Order Generation
# ---------------------------------------------------------------------------

class TestMinTradeValue:
    """Test min_trade_value filtering in order generation."""

    @pytest.mark.phase12
    def test_min_trade_value_filters_small_orders(self):
        from src.assembled_core.execution.order_generation import (
            generate_orders_from_targets_fast,
        )

        targets = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "target_qty": [100.0, 1.0],  # AAPL large, MSFT tiny
        }).sort_values("symbol").reset_index(drop=True)

        prices = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "close": [150.0, 300.0],
        }).sort_values("symbol").reset_index(drop=True)

        # Without min_trade_value: both orders
        orders_all = generate_orders_from_targets_fast(
            targets, prices_latest=prices, min_trade_value=0.0
        )
        assert len(orders_all) == 2

        # With min_trade_value=500: MSFT order (1*300=300) filtered out
        orders_filtered = generate_orders_from_targets_fast(
            targets, prices_latest=prices, min_trade_value=500.0
        )
        assert len(orders_filtered) == 1
        assert orders_filtered.iloc[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# CVaR Pre-Trade Check
# ---------------------------------------------------------------------------

class TestCVaRPreTradeCheck:
    """Test CVaR integration in pre-trade checks."""

    @pytest.mark.phase12
    def test_cvar_field_exists(self):
        from src.assembled_core.execution.pre_trade_checks import PreTradeConfig
        cfg = PreTradeConfig(max_cvar_95=-0.05)
        assert cfg.max_cvar_95 == -0.05

    @pytest.mark.phase12
    def test_cvar_check_scales_orders(self):
        from src.assembled_core.execution.pre_trade_checks import (
            PreTradeConfig,
            run_pre_trade_checks,
        )

        orders = pd.DataFrame({
            "symbol": ["AAPL"],
            "side": ["BUY"],
            "qty": [100.0],
            "price": [150.0],
        })

        # CVaR is -10%, limit is -5% → should scale down
        config = PreTradeConfig(max_cvar_95=-0.05)
        risk_summary = {"cvar_95": -0.10}
        result, filtered = run_pre_trade_checks(
            orders, config=config, risk_summary=risk_summary
        )
        # Orders should be reduced
        if not filtered.empty:
            assert filtered.iloc[0]["qty"] <= 100.0
            assert result.reduced_orders  # should have reduction entries
