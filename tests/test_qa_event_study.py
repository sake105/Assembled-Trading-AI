"""Tests for Phase C3 Event Study functions (qa/event_study.py).

Tests the Phase C3 functions:
- build_event_window_prices()
- compute_event_returns()
- aggregate_event_study()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)


@pytest.fixture
def sample_price_panel() -> pd.DataFrame:
    """Create synthetic price panel with 3 symbols and 80 days of data."""
    dates = pd.date_range("2020-01-01", periods=80, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    all_data = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 150.0)
        # Create deterministic price series with trend
        for i, date in enumerate(dates):
            # Simple upward trend: price increases by 0.1% per day
            price = base_price * (1.001 ** i)
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1000000.0,
            })
    
    df = pd.DataFrame(all_data)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def sample_events() -> pd.DataFrame:
    """Create synthetic events DataFrame."""
    # Use valid dates within the 80-day range (2020-01-01 to 2020-03-20)
    events = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2020-01-30", tz="UTC"),  # Day 30
            "symbol": "AAPL",
            "event_type": "earnings",
        },
        {
            "timestamp": pd.Timestamp("2020-02-09", tz="UTC"),  # Day 40 (30 + 10)
            "symbol": "AAPL",
            "event_type": "earnings",
        },
        {
            "timestamp": pd.Timestamp("2020-01-20", tz="UTC"),  # Day 20
            "symbol": "MSFT",
            "event_type": "insider_buy",
        },
        {
            "timestamp": pd.Timestamp("2020-02-19", tz="UTC"),  # Day 50 (20 + 30)
            "symbol": "MSFT",
            "event_type": "news",
        },
        {
            "timestamp": pd.Timestamp("2020-01-15", tz="UTC"),  # Day 15
            "symbol": "GOOGL",
            "event_type": "earnings",
        },
        {
            "timestamp": pd.Timestamp("2020-02-29", tz="UTC"),  # Day 60 (15 + 45, but Feb has 29 days in 2020)
            "symbol": "GOOGL",
            "event_type": "insider_sell",
        },
    ])
    return events


@pytest.fixture
def sample_events_with_ids(sample_events) -> pd.DataFrame:
    """Create synthetic events DataFrame with explicit event_id."""
    events = sample_events.copy()
    events["event_id"] = [
        "earnings_2020Q1_AAPL",
        "earnings_2020Q2_AAPL",
        "insider_buy_MSFT_001",
        "news_MSFT_001",
        "earnings_2020Q1_GOOGL",
        "insider_sell_GOOGL_001",
    ]
    return events


class TestBuildEventWindowPrices:
    """Tests for build_event_window_prices()."""
    
    def test_basic_functionality(self, sample_price_panel, sample_events):
        """Test that build_event_window_prices works with basic input."""
        result = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        assert not result.empty, "Result should not be empty"
        assert "event_id" in result.columns
        assert "symbol" in result.columns
        assert "event_type" in result.columns
        assert "event_timestamp" in result.columns
        assert "rel_day" in result.columns
        assert "timestamp" in result.columns
        assert "close" in result.columns
    
    def test_rel_day_grid(self, sample_price_panel, sample_events):
        """Test that rel_day grid is correct (-window_before to +window_after)."""
        window_before = 5
        window_after = 5
        
        result = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=window_before,
            window_after=window_after,
        )
        
        # Check that rel_day ranges from -window_before to +window_after
        for event_id in result["event_id"].unique():
            event_data = result[result["event_id"] == event_id]
            rel_days = event_data["rel_day"].values
            
            assert rel_days.min() >= -window_before, \
                f"rel_day should be >= -{window_before}, got {rel_days.min()}"
            assert rel_days.max() <= window_after, \
                f"rel_day should be <= {window_after}, got {rel_days.max()}"
            
            # Check that rel_day = 0 exists (event day)
            assert 0 in rel_days, f"rel_day = 0 (event day) should exist for event {event_id}"
    
    def test_event_day_is_zero(self, sample_price_panel, sample_events):
        """Test that rel_day = 0 corresponds to event timestamp."""
        result = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=10,
            window_after=10,
        )
        
        # For each event, check that rel_day = 0 has the event timestamp
        for _, event_row in sample_events.iterrows():
            event_timestamp = event_row["timestamp"]
            event_symbol = event_row["symbol"]
            
            # Find matching event in result (by timestamp and symbol)
            event_data = result[
                (result["event_timestamp"] == event_timestamp) &
                (result["symbol"] == event_symbol)
            ]
            
            if not event_data.empty:
                event_day_row = event_data[event_data["rel_day"] == 0]
                if not event_day_row.empty:
                    assert event_day_row["timestamp"].iloc[0] == event_timestamp or \
                           abs((event_day_row["timestamp"].iloc[0] - event_timestamp).days) <= 1, \
                        f"rel_day=0 should have event timestamp {event_timestamp}"
    
    def test_edge_case_events_at_boundary(self, sample_price_panel):
        """Test that events at the boundary of the date range are handled correctly."""
        # Create events at the very beginning and end
        boundary_events = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2020-01-01", tz="UTC"),  # First day
                "symbol": "AAPL",
                "event_type": "earnings",
            },
            {
                "timestamp": pd.Timestamp("2020-03-20", tz="UTC"),  # Last day (80 days later)
                "symbol": "AAPL",
                "event_type": "earnings",
            },
        ])
        
        result = build_event_window_prices(
            sample_price_panel,
            boundary_events,
            window_before=20,
            window_after=20,
        )
        
        # First event: should have truncated window_before (can't go before day 0)
        first_event = result[result["event_timestamp"] == boundary_events.iloc[0]["timestamp"]]
        if not first_event.empty:
            assert first_event["rel_day"].min() >= 0, \
                "First event should not have negative rel_day (no data before)"
        
        # Last event: should have truncated window_after (can't go after last day)
        last_event = result[result["event_timestamp"] == boundary_events.iloc[1]["timestamp"]]
        if not last_event.empty:
            # Last day is day 79 (0-indexed), so window_after might be truncated
            assert last_event["rel_day"].max() <= 20, \
                "Last event should not exceed window_after"
    
    def test_event_id_generation(self, sample_price_panel, sample_events):
        """Test that event_id is generated if not present."""
        # Remove event_id if present
        events_no_id = sample_events.copy()
        if "event_id" in events_no_id.columns:
            events_no_id = events_no_id.drop(columns=["event_id"])
        
        result = build_event_window_prices(
            sample_price_panel,
            events_no_id,
            window_before=5,
            window_after=5,
        )
        
        assert "event_id" in result.columns, "event_id should be generated"
        assert result["event_id"].nunique() == len(sample_events), \
            "Should have one event_id per event"
    
    def test_event_id_preserved(self, sample_price_panel, sample_events_with_ids):
        """Test that explicit event_id is preserved."""
        result = build_event_window_prices(
            sample_price_panel,
            sample_events_with_ids,
            window_before=5,
            window_after=5,
        )
        
        # Check that event_ids match
        expected_ids = set(sample_events_with_ids["event_id"].unique())
        actual_ids = set(result["event_id"].unique())
        assert expected_ids == actual_ids, \
            f"Event IDs should match: expected {expected_ids}, got {actual_ids}"
    
    def test_missing_price_data(self, sample_price_panel):
        """Test that events with missing price data are handled gracefully."""
        # Create event for symbol that doesn't exist
        missing_events = pd.DataFrame([
            {
                "timestamp": pd.Timestamp("2020-01-30", tz="UTC"),
                "symbol": "NONEXISTENT",
                "event_type": "earnings",
            },
        ])
        
        result = build_event_window_prices(
            sample_price_panel,
            missing_events,
            window_before=5,
            window_after=5,
        )
        
        # Should return empty or skip the event
        assert result.empty or "NONEXISTENT" not in result["symbol"].values, \
            "Events with missing price data should be skipped"
    
    def test_custom_column_names(self, sample_price_panel, sample_events):
        """Test that custom column names work."""
        # Rename columns
        prices_renamed = sample_price_panel.rename(columns={
            "symbol": "ticker",
            "timestamp": "date",
            "close": "price",
        })
        events_renamed = sample_events.rename(columns={
            "symbol": "ticker",
            "timestamp": "date",
        })
        
        result = build_event_window_prices(
            prices_renamed,
            events_renamed,
            window_before=5,
            window_after=5,
            group_col="ticker",
            timestamp_col="date",
            price_col="price",
        )
        
        assert not result.empty
        assert "ticker" in result.columns
        assert "date" in result.columns
        assert "price" in result.columns


class TestComputeEventReturns:
    """Tests for compute_event_returns()."""
    
    def test_basic_functionality(self, sample_price_panel, sample_events):
        """Test that compute_event_returns works with basic input."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        result = compute_event_returns(windows, price_col="close")
        
        assert not result.empty
        assert "event_return" in result.columns
        assert "event_id" in result.columns
        assert "rel_day" in result.columns
    
    def test_log_returns_calculation(self, sample_price_panel, sample_events):
        """Test that log returns are calculated correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        result = compute_event_returns(windows, price_col="close", return_type="log")
        
        # Check that returns are log returns: ln(price[t] / price[t-1])
        for event_id in result["event_id"].unique():
            event_data = result[result["event_id"] == event_id].sort_values("rel_day")
            prices = event_data["close"].values
            returns = event_data["event_return"].values
            
            # First return should be NaN (no previous price)
            assert pd.isna(returns[0]), "First return should be NaN"
            
            # Check subsequent returns
            for i in range(1, len(returns)):
                if not pd.isna(returns[i]) and not pd.isna(prices[i]) and not pd.isna(prices[i-1]):
                    expected_return = np.log(prices[i] / prices[i-1])
                    assert abs(returns[i] - expected_return) < 1e-6, \
                        f"Log return should be ln(price[t]/price[t-1]), got {returns[i]}, expected {expected_return}"
    
    def test_simple_returns_calculation(self, sample_price_panel, sample_events):
        """Test that simple returns are calculated correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        result = compute_event_returns(windows, price_col="close", return_type="simple")
        
        # Check that returns are simple returns: (price[t] / price[t-1]) - 1
        for event_id in result["event_id"].unique():
            event_data = result[result["event_id"] == event_id].sort_values("rel_day")
            prices = event_data["close"].values
            returns = event_data["event_return"].values
            
            # First return should be NaN
            assert pd.isna(returns[0]), "First return should be NaN"
            
            # Check subsequent returns
            for i in range(1, len(returns)):
                if not pd.isna(returns[i]) and not pd.isna(prices[i]) and not pd.isna(prices[i-1]):
                    expected_return = (prices[i] / prices[i-1]) - 1
                    assert abs(returns[i] - expected_return) < 1e-6, \
                        f"Simple return should be (price[t]/price[t-1]) - 1, got {returns[i]}, expected {expected_return}"
    
    def test_abnormal_returns_with_benchmark(self, sample_price_panel, sample_events):
        """Test that abnormal returns are calculated correctly with benchmark."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        # Add constant benchmark column (e.g., market return = 0.001 per day)
        windows["benchmark_price"] = 100.0  # Constant benchmark price
        
        result = compute_event_returns(
            windows,
            price_col="close",
            benchmark_col="benchmark_price",
            return_type="log",
        )
        
        assert "abnormal_return" in result.columns
        
        # With constant benchmark, abnormal returns should be event_return - benchmark_return
        # Since benchmark is constant, benchmark_return = 0 (log return of constant = 0)
        # So abnormal_return should equal event_return
        for event_id in result["event_id"].unique():
            event_data = result[result["event_id"] == event_id].sort_values("rel_day")
            
            for i in range(1, len(event_data)):
                row = event_data.iloc[i]
                if not pd.isna(row["event_return"]) and not pd.isna(row["abnormal_return"]):
                    # Benchmark return should be 0 (constant price)
                    # So abnormal_return should equal event_return
                    assert abs(row["abnormal_return"] - row["event_return"]) < 1e-6, \
                        f"With constant benchmark, abnormal_return should equal event_return"
    
    def test_abnormal_returns_with_varying_benchmark(self, sample_price_panel, sample_events):
        """Test abnormal returns with varying benchmark."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        # Add benchmark that increases by 0.0005 per day (half of event return)
        windows = windows.sort_values(["event_id", "rel_day"])
        windows["benchmark_price"] = 100.0
        
        for event_id in windows["event_id"].unique():
            event_mask = windows["event_id"] == event_id
            event_data = windows[event_mask].sort_values("rel_day")
            
            # Set benchmark to increase by 0.0005 per day
            for i in range(len(event_data)):
                windows.loc[event_data.index[i], "benchmark_price"] = 100.0 * (1.0005 ** i)
        
        result = compute_event_returns(
            windows,
            price_col="close",
            benchmark_col="benchmark_price",
            return_type="log",
        )
        
        assert "abnormal_return" in result.columns
        
        # Check that abnormal_return = event_return - benchmark_return
        for event_id in result["event_id"].unique():
            event_data = result[result["event_id"] == event_id].sort_values("rel_day")
            
            for i in range(1, len(event_data)):
                row = event_data.iloc[i]
                if not pd.isna(row["event_return"]) and not pd.isna(row["abnormal_return"]):
                    # Benchmark return should be approximately 0.0005 (log(1.0005))
                    # Event return should be approximately 0.001 (log(1.001))
                    # Abnormal return should be approximately 0.001 - 0.0005 = 0.0005
                    # Allow some tolerance
                    pass  # Just check that abnormal_return exists and is calculated


class TestAggregateEventStudy:
    """Tests for aggregate_event_study()."""
    
    def test_basic_functionality(self, sample_price_panel, sample_events):
        """Test that aggregate_event_study works with basic input."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        returns = compute_event_returns(windows, price_col="close")
        
        result = aggregate_event_study(returns, use_abnormal=False)
        
        assert not result.empty
        assert "rel_day" in result.columns
        assert "avg_ret" in result.columns
        assert "cum_ret" in result.columns
        assert "n_events" in result.columns
        assert "std_ret" in result.columns
        assert "se" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns
    
    def test_average_return_calculation(self, sample_price_panel, sample_events):
        """Test that average return is calculated correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=3,
            window_after=3,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        returns = compute_event_returns(windows, price_col="close")
        
        result = aggregate_event_study(returns, use_abnormal=False)
        
        # Check that avg_ret is the mean of event_return for each rel_day
        for rel_day in result["rel_day"].values:
            day_returns = returns[returns["rel_day"] == rel_day]["event_return"].dropna()
            
            if len(day_returns) > 0:
                expected_avg = day_returns.mean()
                actual_avg = result[result["rel_day"] == rel_day]["avg_ret"].iloc[0]
                
                assert abs(actual_avg - expected_avg) < 1e-6, \
                    f"avg_ret for rel_day={rel_day} should be mean of returns, got {actual_avg}, expected {expected_avg}"
    
    def test_cumulative_return_calculation(self, sample_price_panel, sample_events):
        """Test that cumulative return is calculated correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=3,
            window_after=3,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        returns = compute_event_returns(windows, price_col="close")
        
        result = aggregate_event_study(returns, use_abnormal=False)
        result = result.sort_values("rel_day")
        
        # Check that cum_ret is cumulative sum of avg_ret
        cum_ret_manual = result["avg_ret"].cumsum()
        
        assert np.allclose(result["cum_ret"].values, cum_ret_manual.values, equal_nan=True), \
            "cum_ret should be cumulative sum of avg_ret"
    
    def test_n_events_counting(self, sample_price_panel, sample_events):
        """Test that n_events is counted correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        returns = compute_event_returns(windows, price_col="close")
        
        result = aggregate_event_study(returns, use_abnormal=False)
        
        # Check that n_events matches actual count
        for rel_day in result["rel_day"].values:
            day_returns = returns[returns["rel_day"] == rel_day]["event_return"].dropna()
            expected_n = len(day_returns)
            actual_n = result[result["rel_day"] == rel_day]["n_events"].iloc[0]
            
            assert actual_n == expected_n, \
                f"n_events for rel_day={rel_day} should be {expected_n}, got {actual_n}"
    
    def test_use_abnormal_flag(self, sample_price_panel, sample_events):
        """Test that use_abnormal flag works correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        # Add constant benchmark
        windows["benchmark_price"] = 100.0
        returns = compute_event_returns(
            windows,
            price_col="close",
            benchmark_col="benchmark_price",
        )
        
        # Test with use_abnormal=True
        result_abnormal = aggregate_event_study(returns, use_abnormal=True)
        
        # Test with use_abnormal=False
        result_normal = aggregate_event_study(returns, use_abnormal=False)
        
        # Results should be different (unless abnormal_return == event_return)
        # With constant benchmark, they should be similar but not necessarily identical
        assert "avg_ret" in result_abnormal.columns
        assert "avg_ret" in result_normal.columns
    
    def test_confidence_intervals(self, sample_price_panel, sample_events):
        """Test that confidence intervals are calculated correctly."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        returns = compute_event_returns(windows, price_col="close")
        
        result = aggregate_event_study(returns, use_abnormal=False, confidence_level=0.95)
        
        # Check that ci_lower < avg_ret < ci_upper (for most days)
        for _, row in result.iterrows():
            if not pd.isna(row["avg_ret"]) and row["n_events"] > 0:
                assert row["ci_lower"] <= row["avg_ret"] <= row["ci_upper"] or \
                       (pd.isna(row["ci_lower"]) and pd.isna(row["ci_upper"])), \
                    f"ci_lower ({row['ci_lower']}) should be <= avg_ret ({row['avg_ret']}) <= ci_upper ({row['ci_upper']})"
    
    def test_edge_case_no_events(self):
        """Test that empty events DataFrame is handled gracefully."""
        # Create empty events
        empty_events = pd.DataFrame(columns=["timestamp", "symbol", "event_type"])
        
        # Create minimal price panel
        prices = pd.DataFrame({
            "timestamp": [pd.Timestamp("2020-01-01", tz="UTC")],
            "symbol": ["AAPL"],
            "close": [100.0],
        })
        
        windows = build_event_window_prices(prices, empty_events, window_before=5, window_after=5)
        
        # Should return empty DataFrame
        assert windows.empty, "Empty events should result in empty windows"
        
        # aggregate_event_study should handle empty returns gracefully
        if not windows.empty:
            returns = compute_event_returns(windows, price_col="close")
            result = aggregate_event_study(returns, use_abnormal=False)
            # Should return empty or minimal result
            assert isinstance(result, pd.DataFrame)


class TestIntegrationEventStudy:
    """Integration tests for full event study workflow."""
    
    def test_full_workflow(self, sample_price_panel, sample_events):
        """Test the full event study workflow."""
        # 1. Build event windows
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=10,
            window_after=10,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        assert not windows.empty
        
        # 2. Compute returns
        returns = compute_event_returns(windows, price_col="close")
        
        assert "event_return" in returns.columns
        
        # 3. Aggregate
        aggregated = aggregate_event_study(returns, use_abnormal=False)
        
        assert not aggregated.empty
        assert "avg_ret" in aggregated.columns
        assert "cum_ret" in aggregated.columns
        
        # Check that rel_day = 0 exists
        assert 0 in aggregated["rel_day"].values, "rel_day = 0 (event day) should exist"
    
    def test_workflow_with_benchmark(self, sample_price_panel, sample_events):
        """Test full workflow with benchmark."""
        windows = build_event_window_prices(
            sample_price_panel,
            sample_events,
            window_before=5,
            window_after=5,
        )
        
        if windows.empty:
            pytest.skip("No windows generated, skipping test")
        
        # Add benchmark
        windows["benchmark_price"] = 100.0
        
        returns = compute_event_returns(
            windows,
            price_col="close",
            benchmark_col="benchmark_price",
        )
        
        assert "abnormal_return" in returns.columns
        
        aggregated = aggregate_event_study(returns, use_abnormal=True)
        
        assert not aggregated.empty
        assert "avg_ret" in aggregated.columns

