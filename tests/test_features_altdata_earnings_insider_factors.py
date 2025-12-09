"""Tests for Alt-Data Earnings & Insider Factors module (Phase B1).

Tests the build_earnings_surprise_factors() and build_insider_activity_factors() functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create a synthetic price panel with 2 symbols and 100 days of data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 150.0
        price_series = base_price + np.arange(100) * 0.1 + np.random.randn(100) * 1.0
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "close": price_series,
            "high": price_series + np.abs(np.random.randn(100)) * 0.5,
            "low": price_series - np.abs(np.random.randn(100)) * 0.5,
            "volume": np.random.randint(1000000, 10000000, size=100),
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def sample_earnings_events() -> pd.DataFrame:
    """Create synthetic earnings events."""
    events = [
        {
            "timestamp": pd.Timestamp("2020-01-15", tz="UTC"),
            "symbol": "AAPL",
            "event_type": "earnings",
            "event_id": "earnings_2020-01-15_AAPL",
            "eps_actual": 2.1,
            "eps_estimate": 2.0,
            "revenue_actual": 100.0,
            "revenue_estimate": 95.0,
        },
        {
            "timestamp": pd.Timestamp("2020-03-15", tz="UTC"),
            "symbol": "AAPL",
            "event_type": "earnings",
            "event_id": "earnings_2020-03-15_AAPL",
            "eps_actual": 1.8,
            "eps_estimate": 2.0,
            "revenue_actual": 90.0,
            "revenue_estimate": 95.0,
        },
        {
            "timestamp": pd.Timestamp("2020-02-10", tz="UTC"),
            "symbol": "MSFT",
            "event_type": "earnings",
            "event_id": "earnings_2020-02-10_MSFT",
            "eps_actual": 3.0,
            "eps_estimate": 3.0,
            "revenue_actual": 200.0,
            "revenue_estimate": 200.0,
        },
    ]
    return pd.DataFrame(events)


@pytest.fixture
def sample_insider_events() -> pd.DataFrame:
    """Create synthetic insider events."""
    events = [
        {
            "timestamp": pd.Timestamp("2020-01-20", tz="UTC"),
            "symbol": "AAPL",
            "event_type": "insider_buy",
            "event_id": "insider_2020-01-20_AAPL_1",
            "usd_notional": 1000000.0,
            "direction": "buy",
            "shares": 10000,
            "price": 100.0,
        },
        {
            "timestamp": pd.Timestamp("2020-01-25", tz="UTC"),
            "symbol": "AAPL",
            "event_type": "insider_sell",
            "event_id": "insider_2020-01-25_AAPL_1",
            "usd_notional": 500000.0,
            "direction": "sell",
            "shares": 5000,
            "price": 100.0,
        },
        {
            "timestamp": pd.Timestamp("2020-02-15", tz="UTC"),
            "symbol": "MSFT",
            "event_type": "insider_buy",
            "event_id": "insider_2020-02-15_MSFT_1",
            "usd_notional": 2000000.0,
            "direction": "buy",
            "shares": 13000,
            "price": 153.85,
        },
    ]
    return pd.DataFrame(events)


class TestBuildEarningsSurpriseFactors:
    """Tests for build_earnings_surprise_factors()."""
    
    def test_basic_functionality(self, sample_price_panel, sample_earnings_events):
        """Test basic earnings surprise factor computation."""
        result = build_earnings_surprise_factors(
            sample_earnings_events,
            sample_price_panel,
            window_days=20,
        )
        
        # Check that result has expected columns
        assert "earnings_eps_surprise_last" in result.columns
        assert "earnings_revenue_surprise_last" in result.columns
        assert "earnings_positive_surprise_flag" in result.columns
        assert "earnings_negative_surprise_flag" in result.columns
        assert "post_earnings_drift_return_20d" in result.columns
        
        # Check that all price rows are preserved
        assert len(result) == len(sample_price_panel)
        
        # Check that timestamps and symbols are preserved
        assert result["timestamp"].equals(sample_price_panel["timestamp"])
        assert result["symbol"].equals(sample_price_panel["symbol"])
    
    def test_positive_eps_surprise(self, sample_price_panel, sample_earnings_events):
        """Test that positive EPS surprise is calculated correctly."""
        # AAPL on 2020-01-15: eps_actual=2.1, eps_estimate=2.0
        # Surprise = (2.1 - 2.0) / |2.0| * 100 = 5.0%
        result = build_earnings_surprise_factors(
            sample_earnings_events,
            sample_price_panel,
            window_days=20,
        )
        
        # Find the row for AAPL on 2020-01-15
        aapl_jan15 = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] == pd.Timestamp("2020-01-15", tz="UTC"))
        ]
        
        if not aapl_jan15.empty:
            eps_surprise = aapl_jan15["earnings_eps_surprise_last"].iloc[0]
            assert not pd.isna(eps_surprise)
            assert abs(eps_surprise - 5.0) < 0.1  # Should be ~5%
            
            # Check flags
            assert aapl_jan15["earnings_positive_surprise_flag"].iloc[0] == 1.0
            assert aapl_jan15["earnings_negative_surprise_flag"].iloc[0] == 0.0
    
    def test_negative_eps_surprise(self, sample_price_panel, sample_earnings_events):
        """Test that negative EPS surprise is calculated correctly."""
        # AAPL on 2020-03-15: eps_actual=1.8, eps_estimate=2.0
        # Surprise = (1.8 - 2.0) / |2.0| * 100 = -10.0%
        result = build_earnings_surprise_factors(
            sample_earnings_events,
            sample_price_panel,
            window_days=20,
        )
        
        # Find the row for AAPL on 2020-03-15
        aapl_mar15 = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] == pd.Timestamp("2020-03-15", tz="UTC"))
        ]
        
        if not aapl_mar15.empty:
            eps_surprise = aapl_mar15["earnings_eps_surprise_last"].iloc[0]
            assert not pd.isna(eps_surprise)
            assert abs(eps_surprise - (-10.0)) < 0.1  # Should be ~-10%
            
            # Check flags
            assert aapl_mar15["earnings_positive_surprise_flag"].iloc[0] == 0.0
            assert aapl_mar15["earnings_negative_surprise_flag"].iloc[0] == 1.0
    
    def test_zero_surprise(self, sample_price_panel, sample_earnings_events):
        """Test that zero surprise (actual == estimate) is handled correctly."""
        # MSFT on 2020-02-10: eps_actual=3.0, eps_estimate=3.0
        result = build_earnings_surprise_factors(
            sample_earnings_events,
            sample_price_panel,
            window_days=20,
        )
        
        # Find the row for MSFT on 2020-02-10
        msft_feb10 = result[
            (result["symbol"] == "MSFT") &
            (result["timestamp"] == pd.Timestamp("2020-02-10", tz="UTC"))
        ]
        
        if not msft_feb10.empty:
            eps_surprise = msft_feb10["earnings_eps_surprise_last"].iloc[0]
            # Should be 0.0 or very close to 0
            assert abs(eps_surprise) < 0.01
            
            # Flags should both be 0
            assert msft_feb10["earnings_positive_surprise_flag"].iloc[0] == 0.0
            assert msft_feb10["earnings_negative_surprise_flag"].iloc[0] == 0.0
    
    def test_surprise_propagation(self, sample_price_panel, sample_earnings_events):
        """Test that surprise values propagate forward (last event up to next event)."""
        result = build_earnings_surprise_factors(
            sample_earnings_events,
            sample_price_panel,
            window_days=20,
        )
        
        # AAPL: First event on 2020-01-15 (positive surprise), second on 2020-03-15 (negative)
        # Dates between 2020-01-15 and 2020-03-15 should have the first event's surprise
        aapl_between = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-16", tz="UTC")) &
            (result["timestamp"] < pd.Timestamp("2020-03-15", tz="UTC"))
        ]
        
        if not aapl_between.empty:
            # All should have the first event's surprise (~5%)
            eps_surprises = aapl_between["earnings_eps_surprise_last"].dropna()
            if not eps_surprises.empty:
                assert all(abs(s - 5.0) < 0.1 for s in eps_surprises)
    
    def test_multiple_events_per_symbol(self, sample_price_panel):
        """Test handling of multiple events per symbol in short time."""
        # Create events with two events for AAPL close together
        events = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2020-01-15", tz="UTC"),
                "symbol": "AAPL",
                "event_type": "earnings",
                "event_id": "earnings_1",
                "eps_actual": 2.1,
                "eps_estimate": 2.0,
            },
            {
                "timestamp": pd.Timestamp("2020-01-20", tz="UTC"),
                "symbol": "AAPL",
                "event_type": "earnings",
                "event_id": "earnings_2",
                "eps_actual": 2.2,
                "eps_estimate": 2.1,
            },
        ])
        
        result = build_earnings_surprise_factors(
            events,
            sample_price_panel,
            window_days=20,
        )
        
        # After 2020-01-20, should have the second event's surprise
        aapl_after = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-20", tz="UTC"))
        ]
        
        if not aapl_after.empty:
            # Second surprise: (2.2 - 2.1) / |2.1| * 100 ≈ 4.76%
            eps_surprises = aapl_after["earnings_eps_surprise_last"].dropna()
            if not eps_surprises.empty:
                # Should be around 4.76% (second event)
                assert all(abs(s - 4.76) < 1.0 for s in eps_surprises)
    
    def test_empty_events(self, sample_price_panel):
        """Test behavior with empty events DataFrame."""
        empty_events = pd.DataFrame(columns=["timestamp", "symbol", "event_type"])
        
        result = build_earnings_surprise_factors(
            empty_events,
            sample_price_panel,
            window_days=20,
        )
        
        # Should still return price DataFrame with NaN factors
        assert len(result) == len(sample_price_panel)
        assert result["earnings_eps_surprise_last"].isna().all()
        assert (result["earnings_positive_surprise_flag"] == 0.0).all()
    
    def test_missing_required_columns(self, sample_price_panel):
        """Test that missing required columns raise KeyError."""
        events_missing_cols = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2020-01-15", tz="UTC"),
                "symbol": "AAPL",
                # Missing event_type
            }
        ])
        
        with pytest.raises(KeyError, match="event_type"):
            build_earnings_surprise_factors(
                events_missing_cols,
                sample_price_panel,
                window_days=20,
            )


class TestBuildInsiderActivityFactors:
    """Tests for build_insider_activity_factors()."""
    
    def test_basic_functionality(self, sample_price_panel, sample_insider_events):
        """Test basic insider activity factor computation."""
        result = build_insider_activity_factors(
            sample_insider_events,
            sample_price_panel,
            lookback_days=60,
        )
        
        # Check that result has expected columns
        assert "insider_net_notional_60d" in result.columns
        assert "insider_buy_count_60d" in result.columns
        assert "insider_sell_count_60d" in result.columns
        assert "insider_buy_sell_ratio_60d" in result.columns
        assert "insider_net_notional_normalized_60d" in result.columns
        
        # Check that all price rows are preserved
        assert len(result) == len(sample_price_panel)
    
    def test_net_notional_sign(self, sample_price_panel, sample_insider_events):
        """Test that net notional has correct sign (more buys → positive, more sells → negative)."""
        result = build_insider_activity_factors(
            sample_insider_events,
            sample_price_panel,
            lookback_days=60,
        )
        
        # AAPL: 1 buy (1M on 2020-01-20) + 1 sell (0.5M on 2020-01-25) = net +0.5M (positive)
        # After 2020-01-25, both events are in the 60-day window
        # After 60 days from first event (2020-03-20), first event falls out, leaving only sell (negative)
        aapl_after_events = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-25", tz="UTC")) &
            (result["timestamp"] < pd.Timestamp("2020-03-20", tz="UTC"))  # Before first event falls out
        ]
        
        if not aapl_after_events.empty:
            net_notional = aapl_after_events["insider_net_notional_60d"].dropna()
            # Filter out zeros (outside lookback window)
            net_notional_nonzero = net_notional[net_notional != 0.0]
            if not net_notional_nonzero.empty:
                # Should be positive (more buys than sells) in this window
                assert (net_notional_nonzero > 0).all()
        
        # MSFT: 1 buy (2M) = net +2M (positive)
        msft_after_events = result[
            (result["symbol"] == "MSFT") &
            (result["timestamp"] >= pd.Timestamp("2020-02-15", tz="UTC"))
        ]
        
        if not msft_after_events.empty:
            net_notional = msft_after_events["insider_net_notional_60d"].dropna()
            # Filter out zeros (outside lookback window)
            net_notional_nonzero = net_notional[net_notional != 0.0]
            if not net_notional_nonzero.empty:
                # Should be positive (only buys)
                assert (net_notional_nonzero > 0).all()
    
    def test_buy_sell_counts(self, sample_price_panel, sample_insider_events):
        """Test that buy and sell counts are correct."""
        result = build_insider_activity_factors(
            sample_insider_events,
            sample_price_panel,
            lookback_days=60,
        )
        
        # AAPL after both events: 1 buy, 1 sell
        aapl_after = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-25", tz="UTC"))
        ]
        
        if not aapl_after.empty:
            buy_count = aapl_after["insider_buy_count_60d"].dropna()
            sell_count = aapl_after["insider_sell_count_60d"].dropna()
            
            if not buy_count.empty and not sell_count.empty:
                assert buy_count.iloc[0] >= 1.0  # At least 1 buy
                assert sell_count.iloc[0] >= 1.0  # At least 1 sell
    
    def test_buy_sell_ratio(self, sample_price_panel, sample_insider_events):
        """Test that buy/sell ratio is calculated correctly."""
        result = build_insider_activity_factors(
            sample_insider_events,
            sample_price_panel,
            lookback_days=60,
        )
        
        # AAPL: 1 buy, 1 sell → ratio = 1.0
        aapl_after = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-25", tz="UTC"))
        ]
        
        if not aapl_after.empty:
            ratio = aapl_after["insider_buy_sell_ratio_60d"].dropna()
            if not ratio.empty:
                # Should be around 1.0 (equal buys and sells)
                assert abs(ratio.iloc[0] - 1.0) < 0.1
    
    def test_only_buys(self, sample_price_panel):
        """Test behavior with only buy events."""
        events_only_buys = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2020-01-20", tz="UTC"),
                "symbol": "AAPL",
                "event_type": "insider_buy",
                "event_id": "insider_1",
                "usd_notional": 1000000.0,
                "direction": "buy",
            },
            {
                "timestamp": pd.Timestamp("2020-01-25", tz="UTC"),
                "symbol": "AAPL",
                "event_type": "insider_buy",
                "event_id": "insider_2",
                "usd_notional": 500000.0,
                "direction": "buy",
            },
        ])
        
        result = build_insider_activity_factors(
            events_only_buys,
            sample_price_panel,
            lookback_days=60,
        )
        
        aapl_after = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-25", tz="UTC"))
        ]
        
        if not aapl_after.empty:
            net_notional = aapl_after["insider_net_notional_60d"].dropna()
            buy_count = aapl_after["insider_buy_count_60d"].dropna()
            sell_count = aapl_after["insider_sell_count_60d"].dropna()
            ratio = aapl_after["insider_buy_sell_ratio_60d"].dropna()
            
            if not net_notional.empty:
                # Filter out zeros (outside lookback window)
                net_notional_nonzero = net_notional[net_notional != 0.0]
                if not net_notional_nonzero.empty:
                    assert (net_notional_nonzero > 0).all()  # Positive (only buys)
            if not buy_count.empty:
                assert buy_count.iloc[0] >= 2.0  # At least 2 buys
            if not sell_count.empty:
                assert sell_count.iloc[0] == 0.0  # No sells
            if not ratio.empty:
                # Ratio should be inf (only buys, no sells)
                assert np.isinf(ratio.iloc[0])
    
    def test_only_sells(self, sample_price_panel):
        """Test behavior with only sell events."""
        events_only_sells = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2020-01-20", tz="UTC"),
                "symbol": "AAPL",
                "event_type": "insider_sell",
                "event_id": "insider_1",
                "usd_notional": 1000000.0,
                "direction": "sell",
            },
        ])
        
        result = build_insider_activity_factors(
            events_only_sells,
            sample_price_panel,
            lookback_days=60,
        )
        
        aapl_after = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] >= pd.Timestamp("2020-01-20", tz="UTC"))
        ]
        
        if not aapl_after.empty:
            net_notional = aapl_after["insider_net_notional_60d"].dropna()
            buy_count = aapl_after["insider_buy_count_60d"].dropna()
            sell_count = aapl_after["insider_sell_count_60d"].dropna()
            
            if not net_notional.empty:
                # Filter out zeros (outside lookback window)
                net_notional_nonzero = net_notional[net_notional != 0.0]
                if not net_notional_nonzero.empty:
                    assert (net_notional_nonzero < 0).all()  # Negative (only sells)
            if not buy_count.empty:
                assert buy_count.iloc[0] == 0.0  # No buys
            if not sell_count.empty:
                assert sell_count.iloc[0] >= 1.0  # At least 1 sell
    
    def test_empty_events(self, sample_price_panel):
        """Test behavior with empty events DataFrame."""
        empty_events = pd.DataFrame(columns=["timestamp", "symbol", "event_type"])
        
        result = build_insider_activity_factors(
            empty_events,
            sample_price_panel,
            lookback_days=60,
        )
        
        # Should still return price DataFrame with NaN/zero factors
        assert len(result) == len(sample_price_panel)
        assert result["insider_net_notional_60d"].isna().all() or (result["insider_net_notional_60d"] == 0.0).all()
        assert (result["insider_buy_count_60d"] == 0.0).all()
        assert (result["insider_sell_count_60d"] == 0.0).all()
    
    def test_lookback_window(self, sample_price_panel, sample_insider_events):
        """Test that lookback window correctly aggregates events."""
        result = build_insider_activity_factors(
            sample_insider_events,
            sample_price_panel,
            lookback_days=30,  # Shorter window
        )
        
        # Events outside the 30-day window should not be included
        # AAPL events: 2020-01-20 (buy) and 2020-01-25 (sell)
        # On 2020-02-20 (30 days after first event), first event should be out of window
        aapl_feb20 = result[
            (result["symbol"] == "AAPL") &
            (result["timestamp"] == pd.Timestamp("2020-02-20", tz="UTC"))
        ]
        
        if not aapl_feb20.empty:
            # First event (2020-01-20) should be out of window, only second event (2020-01-25) should remain
            # Net notional should be negative (only sell remains)
            net_notional = aapl_feb20["insider_net_notional_30d"].dropna()
            if not net_notional.empty:
                # Should be negative (only sell in window)
                assert net_notional.iloc[0] < 0

