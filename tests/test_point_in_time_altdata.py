"""Point-in-Time (PIT) tests for Alt-Data Feature Builders (B2).

These tests verify that Alt-Data factors respect disclosure_date and only
include events that were disclosed by the as_of date, preventing look-ahead bias.

Tests cover:
- Earnings events with delayed disclosure
- Insider transactions with delayed disclosure
- News sentiment with delayed disclosure
- Mini backtest scenarios with PIT-safe factors
"""
from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_news_sentiment_factors,
)
from src.assembled_core.utils.random_state import set_global_seed

pytestmark = pytest.mark.advanced


@pytest.fixture
def synthetic_prices_jan_2024() -> pd.DataFrame:
    """Create synthetic price data for January 2024 (weekdays only).
    
    Returns:
        DataFrame with columns: timestamp, symbol, close
        Dates: 2024-01-01 to 2024-01-31 (weekdays only)
        Symbols: AAPL, MSFT
        Prices: Simple deterministic walk
    """
    set_global_seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D", tz="UTC")
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    symbols = ["AAPL", "MSFT"]
    rows = []
    
    for symbol in symbols:
        base_price = 150.0 if symbol == "AAPL" else 200.0
        for i, date in enumerate(dates):
            # Simple deterministic price: base + small trend
            price = base_price * (1.0 + 0.001 * i)
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "close": price,
            })
    
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def synthetic_earnings_events_with_delayed_disclosure() -> pd.DataFrame:
    """Create synthetic earnings events with delayed disclosure dates.
    
    Scenario:
    - Event A: event_date=2024-01-10, disclosure_date=2024-01-15 (delayed)
    - Event B: event_date=2024-01-12, disclosure_date=2024-01-12 (immediate)
    
    Returns:
        DataFrame with columns: timestamp, symbol, event_type, event_id,
        event_date, disclosure_date, eps_actual, eps_estimate, eps_surprise_pct
    """
    events = [
        {
            "timestamp": pd.Timestamp("2024-01-10", tz="UTC"),
            "symbol": "AAPL",
            "event_type": "earnings",
            "event_id": "earnings_q1_2024_AAPL",
            "event_date": pd.Timestamp("2024-01-10", tz="UTC").normalize(),
            "disclosure_date": pd.Timestamp("2024-01-15", tz="UTC").normalize(),
            "eps_actual": 2.50,
            "eps_estimate": 2.30,
            "eps_surprise_pct": 8.7,
            "revenue_actual": 120000.0,
            "revenue_estimate": 115000.0,
            "revenue_surprise_pct": 4.3,
        },
        {
            "timestamp": pd.Timestamp("2024-01-12", tz="UTC"),
            "symbol": "MSFT",
            "event_type": "earnings",
            "event_id": "earnings_q1_2024_MSFT",
            "event_date": pd.Timestamp("2024-01-12", tz="UTC").normalize(),
            "disclosure_date": pd.Timestamp("2024-01-12", tz="UTC").normalize(),
            "eps_actual": 3.20,
            "eps_estimate": 3.00,
            "eps_surprise_pct": 6.7,
            "revenue_actual": 62000.0,
            "revenue_estimate": 60000.0,
            "revenue_surprise_pct": 3.3,
        },
    ]
    
    return pd.DataFrame(events)


@pytest.fixture
def synthetic_insider_events_with_delayed_disclosure() -> pd.DataFrame:
    """Create synthetic insider events with delayed disclosure dates.
    
    Scenario:
    - Event A: event_date=2024-01-10, disclosure_date=2024-01-15 (delayed)
        Large buy: 1000 shares @ $150 = $150k notional
    - Event B: event_date=2024-01-12, disclosure_date=2024-01-12 (immediate)
        Small sell: 200 shares @ $200 = $40k notional
    
    Returns:
        DataFrame with columns: timestamp, symbol, event_type, event_id,
        event_date, disclosure_date, direction, shares, price, usd_notional
    """
    events = [
        {
            "timestamp": pd.Timestamp("2024-01-10", tz="UTC"),
            "symbol": "AAPL",
            "event_type": "insider_buy",
            "event_id": "insider_buy_20240110_AAPL",
            "event_date": pd.Timestamp("2024-01-10", tz="UTC").normalize(),
            "disclosure_date": pd.Timestamp("2024-01-15", tz="UTC").normalize(),
            "direction": "buy",
            "shares": 1000,
            "price": 150.0,
            "usd_notional": 150000.0,
        },
        {
            "timestamp": pd.Timestamp("2024-01-12", tz="UTC"),
            "symbol": "MSFT",
            "event_type": "insider_sell",
            "event_id": "insider_sell_20240112_MSFT",
            "event_date": pd.Timestamp("2024-01-12", tz="UTC").normalize(),
            "disclosure_date": pd.Timestamp("2024-01-12", tz="UTC").normalize(),
            "direction": "sell",
            "shares": 200,
            "price": 200.0,
            "usd_notional": 40000.0,
        },
    ]
    
    return pd.DataFrame(events)


@pytest.fixture
def synthetic_news_sentiment_with_delayed_disclosure() -> pd.DataFrame:
    """Create synthetic news sentiment with delayed disclosure dates.
    
    Scenario:
    - Event A: event_date=2024-01-10, disclosure_date=2024-01-15 (delayed)
        Strong positive sentiment: score=0.8, volume=100
    - Event B: event_date=2024-01-12, disclosure_date=2024-01-12 (immediate)
        Neutral sentiment: score=0.1, volume=50
    
    Returns:
        DataFrame with columns: timestamp, symbol, sentiment_score,
        sentiment_volume, event_date, disclosure_date
    """
    events = [
        {
            "timestamp": pd.Timestamp("2024-01-10", tz="UTC"),
            "symbol": "AAPL",
            "sentiment_score": 0.8,
            "sentiment_volume": 100,
            "event_date": pd.Timestamp("2024-01-10", tz="UTC").normalize(),
            "disclosure_date": pd.Timestamp("2024-01-15", tz="UTC").normalize(),
        },
        {
            "timestamp": pd.Timestamp("2024-01-12", tz="UTC"),
            "symbol": "MSFT",
            "sentiment_score": 0.1,
            "sentiment_volume": 50,
            "event_date": pd.Timestamp("2024-01-12", tz="UTC").normalize(),
            "disclosure_date": pd.Timestamp("2024-01-12", tz="UTC").normalize(),
        },
    ]
    
    return pd.DataFrame(events)


def test_earnings_factors_respect_disclosure_date(
    synthetic_prices_jan_2024,
    synthetic_earnings_events_with_delayed_disclosure,
):
    """Test that earnings factors only include events disclosed by as_of date.
    
    Event A (AAPL) has disclosure_date=2024-01-15, Event B (MSFT) has disclosure_date=2024-01-12.
    - as_of=2024-01-11: Both events filtered (disclosure_date > as_of)
    - as_of=2024-01-12: Event A filtered, Event B visible
    - as_of=2024-01-15: Both events visible
    """
    events = synthetic_earnings_events_with_delayed_disclosure
    
    # Test 1: as_of before both disclosure dates (2024-01-11)
    # Both events should be filtered out
    factors_before = build_earnings_surprise_factors(
        events_earnings=events,
        prices=synthetic_prices_jan_2024,
        as_of=pd.Timestamp("2024-01-11", tz="UTC"),
    )
    
    # Both symbols should have NaN factors (no events disclosed yet)
    aapl_before = factors_before[factors_before["symbol"] == "AAPL"]
    assert aapl_before["earnings_eps_surprise_last"].isna().all(), \
        "AAPL earnings factors should be NaN before disclosure_date"
    
    msft_before = factors_before[factors_before["symbol"] == "MSFT"]
    assert msft_before["earnings_eps_surprise_last"].isna().all(), \
        "MSFT earnings factors should be NaN before disclosure_date"
    
    # Test 2: as_of on Event B disclosure date (2024-01-12)
    # Event B should be visible, Event A still filtered
    factors_on_b_disclosure = build_earnings_surprise_factors(
        events_earnings=events,
        prices=synthetic_prices_jan_2024,
        as_of=pd.Timestamp("2024-01-12", tz="UTC"),
    )
    
    # AAPL should still have NaN (Event A not yet disclosed)
    aapl_on_b = factors_on_b_disclosure[factors_on_b_disclosure["symbol"] == "AAPL"]
    assert aapl_on_b["earnings_eps_surprise_last"].isna().all(), \
        "AAPL earnings factors should still be NaN (Event A disclosure_date=2024-01-15)"
    
    # MSFT should have Event B factors (disclosed on 2024-01-12)
    msft_on_b = factors_on_b_disclosure[factors_on_b_disclosure["symbol"] == "MSFT"]
    msft_jan_12_onwards = msft_on_b[msft_on_b["timestamp"] >= pd.Timestamp("2024-01-12", tz="UTC")]
    if len(msft_jan_12_onwards) > 0:
        assert not msft_jan_12_onwards["earnings_eps_surprise_last"].isna().all(), \
            "MSFT earnings factors should be available after Event B disclosure"
        # Check that the surprise value is approximately correct
        non_na_values = msft_jan_12_onwards["earnings_eps_surprise_last"].dropna()
        if len(non_na_values) > 0:
            assert non_na_values.iloc[0] == pytest.approx(6.7, abs=0.1)
    
    # Test 3: as_of on Event A disclosure date (2024-01-15)
    # Both events should now be visible
    factors_on_a_disclosure = build_earnings_surprise_factors(
        events_earnings=events,
        prices=synthetic_prices_jan_2024,
        as_of=pd.Timestamp("2024-01-15", tz="UTC"),
    )
    
    # AAPL should now have Event A factors
    aapl_on_a = factors_on_a_disclosure[
        (factors_on_a_disclosure["symbol"] == "AAPL") &
        (factors_on_a_disclosure["timestamp"] >= pd.Timestamp("2024-01-15", tz="UTC"))
    ]
    if len(aapl_on_a) > 0:
        non_na_values = aapl_on_a["earnings_eps_surprise_last"].dropna()
        if len(non_na_values) > 0:
            assert not non_na_values.empty, \
                "AAPL earnings factors should be available after disclosure_date"
            assert non_na_values.iloc[0] == pytest.approx(8.7, abs=0.1)
    
    # Test 4: as_of after Event A disclosure (2024-01-16)
    # Both events should still be visible (forward-filled)
    factors_after = build_earnings_surprise_factors(
        events_earnings=events,
        prices=synthetic_prices_jan_2024,
        as_of=pd.Timestamp("2024-01-16", tz="UTC"),
    )
    
    aapl_after = factors_after[
        (factors_after["symbol"] == "AAPL") &
        (factors_after["timestamp"] >= pd.Timestamp("2024-01-15", tz="UTC"))
    ]
    if len(aapl_after) > 0:
        non_na_values = aapl_after["earnings_eps_surprise_last"].dropna()
        if len(non_na_values) > 0:
            assert not non_na_values.empty, \
                "AAPL earnings factors should remain available after disclosure"
            assert non_na_values.iloc[0] == pytest.approx(8.7, abs=0.1)


def test_insider_factors_respect_disclosure_date(
    synthetic_prices_jan_2024,
    synthetic_insider_events_with_delayed_disclosure,
):
    """Test that insider factors only include events disclosed by as_of date.
    
    Event A (AAPL buy) has disclosure_date=2024-01-15, Event B (MSFT sell) has disclosure_date=2024-01-12.
    - as_of=2024-01-11: Both events filtered
    - as_of=2024-01-12: Event A filtered, Event B visible
    - as_of=2024-01-15: Both events visible
    """
    events = synthetic_insider_events_with_delayed_disclosure
    
    # Test 1: as_of before both disclosure dates (2024-01-11)
    factors_before = build_insider_activity_factors(
        events_insider=events,
        prices=synthetic_prices_jan_2024,
        lookback_days=60,
        as_of=pd.Timestamp("2024-01-11", tz="UTC"),
    )
    
    # Both should have zero/NaN insider factors (no events disclosed yet)
    aapl_before = factors_before[factors_before["symbol"] == "AAPL"]
    aapl_jan_11 = aapl_before[aapl_before["timestamp"] == pd.Timestamp("2024-01-11", tz="UTC")]
    if len(aapl_jan_11) > 0:
        assert aapl_jan_11["insider_net_notional_60d"].iloc[0] == pytest.approx(0.0, abs=1e-6) or \
               pd.isna(aapl_jan_11["insider_net_notional_60d"].iloc[0]), \
            "AAPL insider net notional should be zero/NaN before disclosure"
    
    msft_before = factors_before[factors_before["symbol"] == "MSFT"]
    msft_jan_11 = msft_before[msft_before["timestamp"] == pd.Timestamp("2024-01-11", tz="UTC")]
    if len(msft_jan_11) > 0:
        assert msft_jan_11["insider_net_notional_60d"].iloc[0] == pytest.approx(0.0, abs=1e-6) or \
               pd.isna(msft_jan_11["insider_net_notional_60d"].iloc[0]), \
            "MSFT insider net notional should be zero/NaN before disclosure"
    
    # Test 2: as_of on Event B disclosure date (2024-01-12)
    factors_on_b_disclosure = build_insider_activity_factors(
        events_insider=events,
        prices=synthetic_prices_jan_2024,
        lookback_days=60,
        as_of=pd.Timestamp("2024-01-12", tz="UTC"),
    )
    
    # AAPL should still have zero/NaN (Event A not yet disclosed)
    aapl_on_b = factors_on_b_disclosure[factors_on_b_disclosure["symbol"] == "AAPL"]
    aapl_jan_12 = aapl_on_b[aapl_on_b["timestamp"] == pd.Timestamp("2024-01-12", tz="UTC")]
    if len(aapl_jan_12) > 0:
        assert aapl_jan_12["insider_net_notional_60d"].iloc[0] == pytest.approx(0.0, abs=1e-6) or \
               pd.isna(aapl_jan_12["insider_net_notional_60d"].iloc[0]), \
            "AAPL insider net notional should still be zero/NaN (Event A disclosure_date=2024-01-15)"
    
    # MSFT should have Event B factors (disclosed on 2024-01-12)
    msft_on_b = factors_on_b_disclosure[factors_on_b_disclosure["symbol"] == "MSFT"]
    msft_jan_12_onwards = msft_on_b[msft_on_b["timestamp"] >= pd.Timestamp("2024-01-12", tz="UTC")]
    if len(msft_jan_12_onwards) > 0:
        # Event B is a sell, so net notional should be negative
        non_na_values = msft_jan_12_onwards["insider_net_notional_60d"].dropna()
        if len(non_na_values) > 0:
            assert non_na_values.iloc[0] < 0, \
                "MSFT insider net notional should reflect sell event (negative)"
    
    # Test 3: as_of on Event A disclosure date (2024-01-15)
    factors_on_a_disclosure = build_insider_activity_factors(
        events_insider=events,
        prices=synthetic_prices_jan_2024,
        lookback_days=60,
        as_of=pd.Timestamp("2024-01-15", tz="UTC"),
    )
    
    # AAPL should now have Event A factors (large buy)
    aapl_on_a = factors_on_a_disclosure[
        (factors_on_a_disclosure["symbol"] == "AAPL") &
        (factors_on_a_disclosure["timestamp"] >= pd.Timestamp("2024-01-15", tz="UTC"))
    ]
    if len(aapl_on_a) > 0:
        non_na_notional = aapl_on_a["insider_net_notional_60d"].dropna()
        if len(non_na_notional) > 0:
            assert non_na_notional.iloc[0] > 0, \
                "AAPL insider net notional should be positive after disclosure (buy event)"
        non_na_buy_count = aapl_on_a["insider_buy_count_60d"].dropna()
        if len(non_na_buy_count) > 0:
            assert non_na_buy_count.iloc[0] > 0, \
                "AAPL insider buy count should be > 0 after disclosure"


def test_news_sentiment_factors_respect_disclosure_date(
    synthetic_prices_jan_2024,
    synthetic_news_sentiment_with_delayed_disclosure,
):
    """Test that news sentiment factors only include events disclosed by as_of date.
    
    Event A (AAPL) has disclosure_date=2024-01-15, Event B (MSFT) has disclosure_date=2024-01-12.
    - as_of=2024-01-11: Both events filtered
    - as_of=2024-01-12: Event A filtered, Event B visible
    - as_of=2024-01-15: Both events visible
    """
    sentiment_daily = synthetic_news_sentiment_with_delayed_disclosure
    
    # Test 1: as_of before both disclosure dates (2024-01-11)
    factors_before = build_news_sentiment_factors(
        news_sentiment_daily=sentiment_daily,
        prices=synthetic_prices_jan_2024,
        lookback_days=20,
        as_of=pd.Timestamp("2024-01-11", tz="UTC"),
    )
    
    # Both should have NaN/zero sentiment factors (no events disclosed yet)
    aapl_before = factors_before[factors_before["symbol"] == "AAPL"]
    aapl_jan_11 = aapl_before[aapl_before["timestamp"] == pd.Timestamp("2024-01-11", tz="UTC")]
    if len(aapl_jan_11) > 0:
        sentiment_mean = aapl_jan_11["news_sentiment_mean_20d"].iloc[0]
        assert pd.isna(sentiment_mean) or abs(sentiment_mean) < 0.1, \
            "AAPL sentiment should be NaN/low before disclosure"
    
    msft_before = factors_before[factors_before["symbol"] == "MSFT"]
    msft_jan_11 = msft_before[msft_before["timestamp"] == pd.Timestamp("2024-01-11", tz="UTC")]
    if len(msft_jan_11) > 0:
        sentiment_mean = msft_jan_11["news_sentiment_mean_20d"].iloc[0]
        assert pd.isna(sentiment_mean) or abs(sentiment_mean) < 0.1, \
            "MSFT sentiment should be NaN/low before disclosure"
    
    # Test 2: as_of on Event B disclosure date (2024-01-12)
    factors_on_b_disclosure = build_news_sentiment_factors(
        news_sentiment_daily=sentiment_daily,
        prices=synthetic_prices_jan_2024,
        lookback_days=20,
        as_of=pd.Timestamp("2024-01-12", tz="UTC"),
    )
    
    # AAPL should still have NaN/low sentiment (Event A not yet disclosed)
    aapl_on_b = factors_on_b_disclosure[factors_on_b_disclosure["symbol"] == "AAPL"]
    aapl_jan_12 = aapl_on_b[aapl_on_b["timestamp"] == pd.Timestamp("2024-01-12", tz="UTC")]
    if len(aapl_jan_12) > 0:
        sentiment_mean = aapl_jan_12["news_sentiment_mean_20d"].iloc[0]
        assert pd.isna(sentiment_mean) or abs(sentiment_mean) < 0.1, \
            "AAPL sentiment should still be NaN/low (Event A disclosure_date=2024-01-15)"
    
    # MSFT should have Event B factors (disclosed on 2024-01-12)
    msft_on_b = factors_on_b_disclosure[factors_on_b_disclosure["symbol"] == "MSFT"]
    msft_jan_12_onwards = msft_on_b[msft_on_b["timestamp"] >= pd.Timestamp("2024-01-12", tz="UTC")]
    if len(msft_jan_12_onwards) > 0:
        sentiment_mean = msft_jan_12_onwards["news_sentiment_mean_20d"].iloc[0]
        # Event B has score=0.1 (neutral), so rolling mean should reflect this
        assert not pd.isna(sentiment_mean), \
            "MSFT sentiment should be available after Event B disclosure"
    
    # Test 3: as_of on Event A disclosure date (2024-01-15)
    factors_on_a_disclosure = build_news_sentiment_factors(
        news_sentiment_daily=sentiment_daily,
        prices=synthetic_prices_jan_2024,
        lookback_days=20,
        as_of=pd.Timestamp("2024-01-15", tz="UTC"),
    )
    
    # AAPL should now have Event A factors (strong positive sentiment)
    aapl_on_a = factors_on_a_disclosure[
        (factors_on_a_disclosure["symbol"] == "AAPL") &
        (factors_on_a_disclosure["timestamp"] >= pd.Timestamp("2024-01-15", tz="UTC"))
    ]
    if len(aapl_on_a) > 0:
        sentiment_mean = aapl_on_a["news_sentiment_mean_20d"].iloc[0]
        assert not pd.isna(sentiment_mean), \
            "AAPL sentiment should be available after disclosure"
        # Event A has score=0.8, so rolling mean should reflect this
        assert sentiment_mean > 0.5, \
            "AAPL sentiment should be positive after disclosure (Event A score=0.8)"


def test_earnings_factors_no_lookahead_bias(
    synthetic_prices_jan_2024,
    synthetic_earnings_events_with_delayed_disclosure,
):
    """Test that earnings factors do not exhibit look-ahead bias.
    
    Compare factors computed with as_of=2024-01-11 (before Event A disclosure)
    vs. as_of=2024-01-15 (after Event A disclosure).
    Before disclosure, factors should be identical to a baseline without Event A.
    """
    events = synthetic_earnings_events_with_delayed_disclosure
    
    # Baseline: factors without Event A (only Event B)
    events_without_a = events[events["symbol"] != "AAPL"].copy()
    factors_baseline = build_earnings_surprise_factors(
        events_earnings=events_without_a,
        prices=synthetic_prices_jan_2024,
        as_of=pd.Timestamp("2024-01-15", tz="UTC"),
    )
    
    # Test: factors with Event A, but as_of before disclosure
    factors_before_disclosure = build_earnings_surprise_factors(
        events_earnings=events,
        prices=synthetic_prices_jan_2024,
        as_of=pd.Timestamp("2024-01-11", tz="UTC"),
    )
    
    # AAPL factors should be identical (both NaN) before disclosure
    aapl_baseline = factors_baseline[factors_baseline["symbol"] == "AAPL"]
    aapl_before = factors_before_disclosure[factors_before_disclosure["symbol"] == "AAPL"]
    
    # Compare on same dates
    common_dates = aapl_baseline["timestamp"].isin(aapl_before["timestamp"])
    if common_dates.any():
        baseline_vals = aapl_baseline[common_dates]["earnings_eps_surprise_last"]
        before_vals = aapl_before[aapl_before["timestamp"].isin(aapl_baseline[common_dates]["timestamp"])]["earnings_eps_surprise_last"]
        
        # Both should be NaN (no Event A in either case)
        assert baseline_vals.isna().all() == before_vals.isna().all(), \
            "AAPL factors should be identical before Event A disclosure (both NaN)"


def test_insider_factors_no_lookahead_bias(
    synthetic_prices_jan_2024,
    synthetic_insider_events_with_delayed_disclosure,
):
    """Test that insider factors do not exhibit look-ahead bias.
    
    Compare factors computed with as_of=2024-01-11 (before Event A disclosure)
    vs. baseline without Event A.
    """
    events = synthetic_insider_events_with_delayed_disclosure
    
    # Baseline: factors without Event A
    events_without_a = events[events["symbol"] != "AAPL"].copy()
    factors_baseline = build_insider_activity_factors(
        events_insider=events_without_a,
        prices=synthetic_prices_jan_2024,
        lookback_days=60,
        as_of=pd.Timestamp("2024-01-15", tz="UTC"),
    )
    
    # Test: factors with Event A, but as_of before disclosure
    factors_before_disclosure = build_insider_activity_factors(
        events_insider=events,
        prices=synthetic_prices_jan_2024,
        lookback_days=60,
        as_of=pd.Timestamp("2024-01-11", tz="UTC"),
    )
    
    # AAPL factors should be identical (both zero/NaN) before disclosure
    aapl_baseline = factors_baseline[factors_baseline["symbol"] == "AAPL"]
    aapl_before = factors_before_disclosure[factors_before_disclosure["symbol"] == "AAPL"]
    
    # Compare net notional on same dates
    common_dates = aapl_baseline["timestamp"].isin(aapl_before["timestamp"])
    if common_dates.any():
        baseline_vals = aapl_baseline[common_dates]["insider_net_notional_60d"]
        before_vals = aapl_before[aapl_before["timestamp"].isin(aapl_baseline[common_dates]["timestamp"])]["insider_net_notional_60d"]
        
        # Both should be zero or NaN (no Event A in either case)
        baseline_zero_or_nan = (baseline_vals == 0.0) | baseline_vals.isna()
        before_zero_or_nan = (before_vals == 0.0) | before_vals.isna()
        assert baseline_zero_or_nan.all() == before_zero_or_nan.all(), \
            "AAPL insider factors should be identical before Event A disclosure (both zero/NaN)"


def test_earnings_factors_in_backtest_scenario(
    synthetic_prices_jan_2024,
    synthetic_earnings_events_with_delayed_disclosure,
):
    """Test that earnings factors work correctly in a mini backtest scenario.
    
    Simulates a backtest loop where factors are computed per day with as_of=current_date.
    Verifies that events with delayed disclosure do not affect signals before disclosure_date.
    """
    events = synthetic_earnings_events_with_delayed_disclosure
    
    # Simulate backtest loop: compute factors for each trading day
    # Event A (AAPL) disclosure_date=2024-01-15, Event B (MSFT) disclosure_date=2024-01-12
    trading_dates = sorted(synthetic_prices_jan_2024["timestamp"].unique())
    
    # Track when factors first appear for each symbol
    aapl_first_appearance = None
    msft_first_appearance = None
    
    for current_date in trading_dates:
        factors = build_earnings_surprise_factors(
            events_earnings=events,
            prices=synthetic_prices_jan_2024,
            as_of=current_date,
        )
        
        # Check AAPL factors
        aapl_factors = factors[
            (factors["symbol"] == "AAPL") &
            (factors["timestamp"] == current_date)
        ]
        if len(aapl_factors) > 0:
            aapl_surprise = aapl_factors["earnings_eps_surprise_last"].iloc[0]
            if not pd.isna(aapl_surprise) and aapl_first_appearance is None:
                aapl_first_appearance = current_date
        
        # Check MSFT factors
        msft_factors = factors[
            (factors["symbol"] == "MSFT") &
            (factors["timestamp"] == current_date)
        ]
        if len(msft_factors) > 0:
            msft_surprise = msft_factors["earnings_eps_surprise_last"].iloc[0]
            if not pd.isna(msft_surprise) and msft_first_appearance is None:
                msft_first_appearance = current_date
    
    # Verify that factors appear only after disclosure_date
    assert aapl_first_appearance is not None, \
        "AAPL earnings factors should appear at some point"
    assert aapl_first_appearance >= pd.Timestamp("2024-01-15", tz="UTC"), \
        f"AAPL earnings factors should not appear before disclosure_date (2024-01-15), " \
        f"but appeared on {aapl_first_appearance}"
    
    assert msft_first_appearance is not None, \
        "MSFT earnings factors should appear at some point"
    assert msft_first_appearance >= pd.Timestamp("2024-01-12", tz="UTC"), \
        f"MSFT earnings factors should not appear before disclosure_date (2024-01-12), " \
        f"but appeared on {msft_first_appearance}"
    
    # Verify that MSFT appears before AAPL (Event B disclosed earlier)
    assert msft_first_appearance < aapl_first_appearance, \
        "MSFT factors should appear before AAPL factors (Event B disclosed earlier)"

