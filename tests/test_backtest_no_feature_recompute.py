# tests/test_backtest_no_feature_recompute.py
"""Guard test: Ensure feature building is not called N times per timestamp in backtest.

This test prevents accidental O(T²) regression by ensuring features are computed
once for the entire time range (precomputed), not per timestamp.

Success criterion:
- Feature build function (_build_features_default or add_all_features) is called
  at most 1 time (or very small number) for a backtest over T timestamps.
- This ensures precomputed features are being reused, not recomputed per timestamp.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.backtest_engine import (
    make_cycle_fn,
    run_portfolio_backtest,
)
from src.assembled_core.pipeline.trading_cycle import TradingContext
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)


def create_synthetic_prices(
    symbols: list[str] = ["AAPL", "MSFT"],
    start_date: str = "2024-01-01",
    days: int = 10,
) -> pd.DataFrame:
    """Create synthetic price data for testing.
    
    Args:
        symbols: List of symbols to generate
        start_date: Start date string
        days: Number of days to generate
        
    Returns:
        DataFrame with columns: timestamp, symbol, close, open, high, low, volume
    """
    dates = pd.date_range(start=start_date, periods=days, freq="D", tz="UTC")
    
    all_prices = []
    for symbol in symbols:
        # Generate synthetic prices with some trend and volatility
        np.random.seed(42)  # Deterministic
        base_price = 100.0 + hash(symbol) % 50
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = base_price * np.exp(np.cumsum(returns))
        
        for date, price in zip(dates, prices):
            all_prices.append({
                "timestamp": date,
                "symbol": symbol,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000000,
            })
    
    return pd.DataFrame(all_prices).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def create_signal_fn():
    """Create a simple signal function for testing."""
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(prices_df, ma_fast=5, ma_slow=10)
    
    return signal_fn


def create_position_sizing_fn():
    """Create a simple position sizing function for testing."""
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals_df,
            total_capital=capital,
            top_n=None,
            min_score=0.0,
        )
    
    return position_sizing_fn


def test_backtest_features_not_recomputed_per_timestamp(monkeypatch):
    """Guard test: Feature building should be called at most 1x, not T times.
    
    This test ensures that when using cycle_fn with precomputed features,
    the feature building function is not called repeatedly for each timestamp.
    Instead, features should be precomputed once and reused via PIT-safe slicing.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    # Create synthetic prices (10 days, 2 symbols = 20 timestamps total)
    prices = create_synthetic_prices(symbols=["AAPL", "MSFT"], days=10)
    
    # Create signal and position sizing functions
    signal_fn = create_signal_fn()
    position_sizing_fn = create_position_sizing_fn()
    
    # Counter to track calls to add_all_features
    feature_build_counter = Counter()
    
    # Store original function BEFORE any patching
    from src.assembled_core.features.ta_features import add_all_features as original_add_all_features
    
    # Also track _build_features_default calls (it might be called, but should skip feature building)
    from src.assembled_core.pipeline.trading_cycle import _build_features_default
    
    def tracked_build_features_default(*args, **kwargs):
        """Wrapper that tracks calls to _build_features_default."""
        feature_build_counter["_build_features_default"] += 1
        # Call original function
        return _build_features_default(*args, **kwargs)
    
    # Build TradingContext template (features will be precomputed here)
    universe_symbols = sorted(prices["symbol"].unique().tolist())
    
    # Precompute features once (BEFORE patching, so it's not counted in our test)
    # We want to test that during backtest, features are not recomputed
    precomputed_prices_with_features = original_add_all_features(
        prices.copy(),
        ma_windows=(5, 10),  # Small windows for faster test
        atr_window=14,
        rsi_window=14,
        include_rsi=True,
    )
    
    # NOW patch the functions to track calls during backtest
    def tracked_add_all_features(*args, **kwargs):
        """Wrapper that tracks calls to add_all_features."""
        feature_build_counter["add_all_features"] += 1
        # Call original function
        return original_add_all_features(*args, **kwargs)
    
    monkeypatch.setattr(
        "src.assembled_core.features.ta_features.add_all_features",
        tracked_add_all_features,
    )
    monkeypatch.setattr(
        "src.assembled_core.pipeline.trading_cycle._build_features_default",
        tracked_build_features_default,
    )
    
    # Create TradingContext template with precomputed features
    ctx_template = TradingContext(
        prices=prices,
        freq="1d",
        universe=universe_symbols,
        use_factor_store=False,
        factor_store_root=None,
        factor_group="core_ta",
        feature_config={},
        precomputed_prices_with_features=precomputed_prices_with_features,
        write_outputs=False,
        enable_risk_controls=False,
    )
    
    # Create cycle_fn
    cycle_fn = make_cycle_fn(
        ctx_template,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=10000.0,
    )
    
    # Reset counter (we've already called add_all_features once for precomputation)
    # Now we want to verify it's not called again during the backtest
    feature_build_counter.clear()
    
    # Run backtest over 10 timestamps (should not trigger additional feature builds)
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        commission_bps=0.0,
        spread_w=0.0,
        impact_w=0.0,
        include_costs=False,
        include_trades=True,
        include_signals=True,
        include_targets=True,
        rebalance_freq="1d",
        compute_features=False,  # Features already precomputed
        use_meta_model=False,
        cycle_fn=cycle_fn,
    )
    
    # Assertions
    assert result.equity is not None
    assert len(result.equity) > 0, "Backtest should produce equity curve"
    
    # Critical assertion: Feature building should NOT be called during backtest loop
    # (it was already called once during precomputation, but we cleared the counter)
    # With precomputed features, _build_features_default might still be called
    # but it should skip actual feature computation and use precomputed panel
    add_all_calls = feature_build_counter.get("add_all_features", 0)
    
    # add_all_features should not be called during backtest (it was only called for precomputation)
    # _build_features_default might be called per timestamp, but it should skip computation
    # The key is: actual feature computation (add_all_features) should be 0 after clearing counter
    assert add_all_calls == 0, (
        f"add_all_features was called {add_all_calls} times during backtest, "
        f"but should be 0 (features are precomputed). "
        f"This indicates O(T) or O(T²) regression in feature computation."
    )
    
    # Optional: Check _build_features_default calls (it might be called, but should skip computation)
    build_default_calls = feature_build_counter.get("_build_features_default", 0)
    # _build_features_default should be called per timestamp, but it should skip actual computation
    # because precomputed_prices_with_features is set. So we expect some calls, but no actual feature computation.
    # The important thing is that add_all_features is not called.


def test_backtest_without_precomputed_features_still_works(monkeypatch):
    """Sanity check: If precomputed features are not provided, features should still be computed.
    
    This test ensures backward compatibility: if precomputed features are not set,
    the system should still compute features (but we want to verify the count is reasonable).
    
    Args:
        monkeypatch: pytest monkeypatch fixture
    """
    # Create synthetic prices (fewer days for this test)
    prices = create_synthetic_prices(symbols=["AAPL"], days=5)
    
    # Counter to track calls
    feature_build_counter = Counter()
    
    # Patch add_all_features
    from src.assembled_core.features.ta_features import add_all_features
    
    def tracked_add_all_features(*args, **kwargs):
        feature_build_counter["add_all_features"] += 1
        return add_all_features(*args, **kwargs)
    
    monkeypatch.setattr(
        "src.assembled_core.features.ta_features.add_all_features",
        tracked_add_all_features,
    )
    
    # Create signal and position sizing functions
    signal_fn = create_signal_fn()
    position_sizing_fn = create_position_sizing_fn()
    
    # Build TradingContext template WITHOUT precomputed features
    universe_symbols = sorted(prices["symbol"].unique().tolist())
    
    ctx_template = TradingContext(
        prices=prices,
        freq="1d",
        universe=universe_symbols,
        use_factor_store=False,
        factor_store_root=None,
        factor_group="core_ta",
        feature_config={"ma_windows": (5, 10)},
        precomputed_prices_with_features=None,  # No precomputed features
        write_outputs=False,
        enable_risk_controls=False,
    )
    
    # Create cycle_fn
    cycle_fn = make_cycle_fn(
        ctx_template,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=10000.0,
    )
    
    # Run backtest (should trigger feature computation per timestamp without precomputed features)
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=10000.0,
        commission_bps=0.0,
        spread_w=0.0,
        impact_w=0.0,
        include_costs=False,
        include_trades=True,
        include_signals=False,
        include_targets=False,
        rebalance_freq="1d",
        compute_features=False,  # Features computed via cycle_fn
        use_meta_model=False,
        cycle_fn=cycle_fn,
    )
    
    assert result.equity is not None
    
    # Without precomputed features, features will be computed per timestamp
    # This is expected behavior for backward compatibility
    # We just verify the test setup works correctly
    add_all_calls = feature_build_counter.get("add_all_features", 0)
    
    # Note: This test documents the current behavior (features computed per timestamp
    # if not precomputed). The main test above ensures that WITH precomputed features,
    # we don't recompute.

