"""Portfolio-level backtest engine.

This module provides a unified backtest engine that orchestrates the complete backtest workflow:
1. Load price data (OHLCV)
2. Compute technical analysis features
3. Generate trading signals
4. Compute target positions (position sizing)
5. Generate orders
6. Simulate equity curve (with or without costs)
7. Compute performance metrics

The engine is designed to be flexible and composable:
- Accepts custom signal functions (callable)
- Accepts custom position sizing functions (callable)
- Supports both cost-free and cost-aware simulation
- Returns equity curve and optional trade list

Example usage:
    >>> from src.assembled_core.data.prices_ingest import load_eod_prices
    >>> from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
    >>> from src.assembled_core.portfolio.position_sizing import compute_target_positions
    >>>
    >>> # Load prices
    >>> prices = load_eod_prices(freq="1d")
    >>>
    >>> # Define signal function
    >>> def my_signal_fn(prices_df):
    ...     return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)
    >>>
    >>> # Define position sizing function
    >>> def my_sizing_fn(signals_df, capital):
    ...     return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)
    >>>
    >>> # Run backtest
    >>> result = run_portfolio_backtest(
    ...     prices=prices,
    ...     signal_fn=my_signal_fn,
    ...     position_sizing_fn=my_sizing_fn,
    ...     start_capital=10000.0,
    ...     commission_bps=0.0,
    ...     spread_w=0.25,
    ...     impact_w=0.5,
    ...     include_trades=True
    ... )
    >>>
    >>> equity = result["equity"]
    >>> metrics = result["metrics"]
    >>> trades = result["trades"]  # Optional

Zukünftige Integration:
- Nutzt pipeline.backtest.simulate_equity für kostenfreie Simulation
- Nutzt pipeline.portfolio.simulate_with_costs für kostenbewusste Simulation
- Nutzt execution.order_generation für Order-Generierung
- Erweitert um Walk-Forward-Analyse, Monte-Carlo-Simulation, etc.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.features.ta_features import (
    add_all_features,
    add_log_returns,
    add_moving_averages,
)
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.data.factor_store import compute_universe_key
from src.assembled_core.pipeline.backtest import compute_metrics, simulate_equity
from src.assembled_core.pipeline.portfolio import simulate_with_costs
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    TradingCycleResult,
    run_trading_cycle,
)
# Timing utilities are available via src.assembled_core.utils.timing if needed
# Currently not used in backtest_engine (timing is handled at script level)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a portfolio-level backtest.

    Attributes:
        equity: DataFrame with columns: date, timestamp, equity, daily_return
            Equity curve over time with daily returns
            - date: Date (date object)
            - timestamp: Timestamp (pd.Timestamp, UTC)
            - equity: Portfolio equity value
            - daily_return: Daily return (pct_change of equity)
        metrics: Dictionary with performance metrics:
            - final_pf: Final performance factor (equity[-1] / equity[0])
            - sharpe: Sharpe ratio
            - trades: Number of trades executed
            - Additional metrics may be present
        trades: Optional DataFrame with columns: timestamp, symbol, side, qty, price
            List of all trades executed during backtest
            Only present if include_trades=True in run_portfolio_backtest
        signals: Optional DataFrame with columns: timestamp, symbol, direction, score
            All signals generated during backtest
            Only present if include_signals=True in run_portfolio_backtest
        target_positions: Optional DataFrame with columns: symbol, target_weight, target_qty
            Target positions computed at each rebalancing point
            Only present if include_targets=True in run_portfolio_backtest
    """

    equity: pd.DataFrame
    metrics: dict[str, float | int]
    trades: pd.DataFrame | None = None
    signals: pd.DataFrame | None = None
    target_positions: pd.DataFrame | None = None


def _update_positions_vectorized(
    orders: pd.DataFrame,
    current_positions: pd.DataFrame,
    use_numba: bool = True,
) -> pd.DataFrame:
    """Update positions DataFrame from orders using vectorized operations.

    This function replaces the iterative order execution logic with vectorized
    pandas operations (optionally accelerated with Numba) for better performance.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
            Orders to execute (side is "BUY" or "SELL", qty is always positive)
        current_positions: DataFrame with columns: symbol, qty
            Current portfolio positions
        use_numba: If True, attempt to use Numba-accelerated path (default: True)
            Falls back to pure pandas if numba is not available

    Returns:
        Updated DataFrame with columns: symbol, qty
        Positions after executing all orders, with zero positions removed

    Note:
        This function preserves exact numerical behavior of the original
        iterative implementation by using the same logic (BUY adds qty,
        SELL subtracts qty), just with vectorized operations.
    """
    if orders.empty:
        return current_positions.copy()

    # Try Numba-accelerated path if available and requested
    if use_numba:
        try:
            from src.assembled_core.qa.backtest_engine_numba import (
                NUMBA_AVAILABLE,
                compute_position_deltas_numba,
                aggregate_position_deltas_numba,
            )

            if NUMBA_AVAILABLE:
                # Convert to numpy arrays for Numba
                symbols_list = orders["symbol"].unique().tolist()
                symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols_list)}

                # Map sides to integers (0=BUY, 1=SELL)
                side_map = {"BUY": 0, "SELL": 1}
                sides = orders["side"].map(side_map).values.astype(np.int32)
                qtys = orders["qty"].values.astype(np.float64)
                symbol_indices = (
                    orders["symbol"].map(symbol_to_idx).values.astype(np.int32)
                )

                # Compute deltas with Numba
                deltas = compute_position_deltas_numba(sides, qtys)

                # Aggregate by symbol with Numba
                unique_indices, aggregated_deltas = aggregate_position_deltas_numba(
                    symbol_indices, deltas
                )

                # Convert back to DataFrame
                unique_symbols = [symbols_list[i] for i in unique_indices]
                position_deltas = pd.DataFrame(
                    {"symbol": unique_symbols, "qty_delta": aggregated_deltas}
                )

                # Merge with current positions (pandas merge is still efficient)
                if current_positions.empty:
                    updated_positions = position_deltas.rename(
                        columns={"qty_delta": "qty"}
                    )
                else:
                    merged = current_positions.merge(
                        position_deltas, on="symbol", how="outer"
                    )
                    merged["qty"] = merged["qty"].fillna(0.0).astype(float)
                    merged["qty_delta"] = merged["qty_delta"].fillna(0.0).astype(float)
                    merged["qty"] = merged["qty"] + merged["qty_delta"]
                    updated_positions = merged[["symbol", "qty"]].copy()

                # Remove zero positions
                updated_positions = updated_positions[
                    updated_positions["qty"].abs() > 1e-6
                ].reset_index(drop=True)

                return updated_positions
        except (ImportError, AttributeError):
            # Fall through to pandas implementation
            pass

    # Pure pandas implementation (fallback or if use_numba=False)
    # Use vectorized numpy operations instead of apply
    # Note: np is already imported at module level
    position_delta_sign = np.where(orders["side"] == "BUY", 1.0, -1.0)
    orders_copy = orders.copy()
    orders_copy["position_delta"] = orders_copy["qty"].values * position_delta_sign

    # Aggregate deltas by symbol (multiple orders for same symbol are summed)
    position_deltas = (
        orders_copy.groupby("symbol")["position_delta"]
        .sum()
        .reset_index()
        .rename(columns={"position_delta": "qty_delta"})
    )

    # Merge with current positions
    if current_positions.empty:
        updated_positions = position_deltas.rename(columns={"qty_delta": "qty"})
    else:
        merged = current_positions.merge(position_deltas, on="symbol", how="outer")
        merged["qty"] = merged["qty"].fillna(0.0).astype(float)
        merged["qty_delta"] = merged["qty_delta"].fillna(0.0).astype(float)
        merged["qty"] = merged["qty"] + merged["qty_delta"]
        updated_positions = merged[["symbol", "qty"]].copy()

    # Remove zero positions (same threshold as original: 1e-6)
    updated_positions = updated_positions[
        updated_positions["qty"].abs() > 1e-6
    ].reset_index(drop=True)

    return updated_positions


def make_cycle_fn(
    ctx_template: TradingContext,
    *,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    capital: float,
) -> Callable[[pd.Timestamp, pd.DataFrame], TradingCycleResult]:
    """Create a callable that runs trading cycle for a given timestamp and positions.
    
    This is an adapter function that bridges the backtest engine's per-timestamp
    loop with the unified trading cycle orchestrator.
    
    Args:
        ctx_template: Template TradingContext with shared configuration
            (prices, universe, feature_config, factor_store settings, etc.)
        signal_fn: Signal function (same as in run_portfolio_backtest)
        position_sizing_fn: Position sizing function (same as in run_portfolio_backtest)
        capital: Capital for position sizing (updated per timestamp based on equity)
        
    Returns:
        Callable that takes (timestamp: pd.Timestamp, current_positions: pd.DataFrame)
        and returns TradingCycleResult
        
    Example:
        >>> ctx_template = TradingContext(
        ...     prices=prices,
        ...     freq="1d",
        ...     use_factor_store=True,
        ...     factor_group="core_ta",
        ... )
        >>> cycle_fn = make_cycle_fn(
        ...     ctx_template,
        ...     signal_fn=signal_fn,
        ...     position_sizing_fn=position_sizing_fn,
        ...     capital=10000.0,
        ... )
        >>> result = cycle_fn(timestamp, current_positions)
        >>> orders = result.orders_filtered  # or result.orders as fallback
    """
    def cycle_fn(timestamp: pd.Timestamp, current_positions: pd.DataFrame) -> TradingCycleResult:
        """Run trading cycle for a specific timestamp.
        
        Args:
            timestamp: Current timestamp for rebalancing
            current_positions: Current portfolio positions (columns: symbol, qty)
            
        Returns:
            TradingCycleResult with orders, signals, target_positions, etc.
        """
        # Build context from template, updating timestamp-specific fields
        ctx = replace(
            ctx_template,
            as_of=timestamp,
            mode="backtest",  # Use backtest mode (full history slice)
            current_positions=current_positions,
            order_timestamp=timestamp,
            capital=capital,  # Capital might be updated per timestamp in the future
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            write_outputs=False,  # Backtest engine handles outputs
            enable_risk_controls=False,  # Backtest engine doesn't use risk controls (can be added later)
        )
        
        # Run trading cycle
        return run_trading_cycle(ctx)
    
    return cycle_fn


def _process_rebalancing_timestamp(
    timestamp: pd.Timestamp,
    signal_group: pd.DataFrame,
    current_positions: pd.DataFrame,
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    start_capital: float,
    prices: pd.DataFrame,
    include_targets: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process rebalancing for a single timestamp (per-timestamp loop).

    This function handles:
    1. Computing target positions from signals
    2. Generating orders to transition from current to target positions
    3. Updating current positions after order execution

    Args:
        timestamp: Current timestamp for rebalancing
        signal_group: DataFrame with signals for this timestamp (columns: timestamp, symbol, direction, score)
        current_positions: DataFrame with current positions (columns: symbol, qty)
        position_sizing_fn: Function to compute target positions from signals
        start_capital: Starting capital (for position sizing)
        prices: Full prices DataFrame (for order generation)
        include_targets: If True, return targets DataFrame

    Returns:
        Tuple of (orders, updated_positions, targets)
        - orders: DataFrame with orders generated for this timestamp
        - updated_positions: DataFrame with positions after executing orders
        - targets: DataFrame with target positions (empty if include_targets=False)

    Note:
        This function is designed to be easily vectorized or parallelized later.
        The per-timestamp loop is explicit here for clarity and future optimization.
    """
    # Compute target positions for this timestamp
    targets = position_sizing_fn(signal_group, start_capital)

    # Generate orders to transition from current to target positions
    prices_at_timestamp = (
        prices[prices["timestamp"] == timestamp]
        if len(prices[prices["timestamp"] == timestamp]) > 0
        else None
    )
    orders = generate_orders_from_targets(
        target_positions=targets,
        current_positions=current_positions,
        timestamp=timestamp,
        prices=prices_at_timestamp,
    )

    # Update current positions using vectorized operations
    updated_positions = current_positions
    if not orders.empty:
        updated_positions = _update_positions_vectorized(orders, current_positions)

    # Prepare targets DataFrame if requested
    targets_df = pd.DataFrame()
    if include_targets and not targets.empty:
        targets_df = targets.copy()
        targets_df["timestamp"] = timestamp

    return orders, updated_positions, targets_df


def run_portfolio_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame] | None = None,
    start_capital: float = 10000.0,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    cost_model: CostModel | None = None,
    include_costs: bool = True,
    include_trades: bool = False,
    include_signals: bool = False,
    include_targets: bool = False,
    rebalance_freq: str = "1d",
    compute_features: bool = True,
    feature_config: dict[str, Any] | None = None,
    # Factor store parameters
    use_factor_store: bool = False,
    factor_store_root: Path | None = None,
    factor_group: str = "core_ta",
    # Meta-model ensemble parameters
    use_meta_model: bool = False,
    meta_model: Any | None = None,
    meta_model_path: str | None = None,
    meta_min_confidence: float = 0.5,
    meta_ensemble_mode: str = "filter",  # "filter" or "scaling"
    # Trading cycle integration (B1)
    cycle_fn: Callable[[pd.Timestamp, pd.DataFrame], TradingCycleResult] | None = None,
) -> BacktestResult:
    """Run a portfolio-level backtest with configurable signal and position sizing functions.

    This is the main entry point for portfolio-level backtesting. It orchestrates:
    1. Feature computation (optional)
    2. Signal generation (via signal_fn)
    3. Position sizing (via position_sizing_fn)
    4. Order generation
    5. Equity simulation (with or without costs)
    6. Performance metrics computation

    Args:
        prices: DataFrame with columns: timestamp, symbol, close (and optionally open, high, low, volume)
            Price data for backtesting. Must be sorted by symbol, then timestamp.
        signal_fn: Callable that takes prices DataFrame and returns signals DataFrame
            Input: DataFrame with columns: timestamp, symbol, close, ... (features if compute_features=True)
            Output: DataFrame with columns: timestamp, symbol, direction, score
            Example: signals.rules_trend.generate_trend_signals_from_prices
        position_sizing_fn: Callable that takes signals DataFrame and capital, returns target positions
            Input: (signals_df: pd.DataFrame, total_capital: float)
            Output: DataFrame with columns: symbol, target_weight, target_qty
            Example: portfolio.position_sizing.compute_target_positions
        start_capital: Starting capital (default: 10000.0)
        commission_bps: Commission in basis points (default: from cost_model or get_default_cost_model)
        spread_w: Spread weight (default: from cost_model or get_default_cost_model)
        impact_w: Market impact weight (default: from cost_model or get_default_cost_model)
        cost_model: Optional CostModel instance. If provided, overrides individual cost parameters.
        include_costs: If True, use cost-aware simulation (pipeline.portfolio.simulate_with_costs)
            If False, use cost-free simulation (pipeline.backtest.simulate_equity)
        include_trades: If True, include trades DataFrame in result (default: False)
        include_signals: If True, include signals DataFrame in result (default: False)
        include_targets: If True, include target_positions DataFrame in result (default: False)
        rebalance_freq: Rebalancing frequency string ("1d" or "5min") for order generation (default: "1d")
        compute_features: If True, compute TA features before signal generation (default: True)
        feature_config: Optional dict with feature configuration:
            - ma_windows: tuple[int, ...] = (20, 50, 200)
            - atr_window: int = 14
            - rsi_window: int = 14
            - include_rsi: bool = True

    Returns:
        BacktestResult with:
        - equity: DataFrame with columns: date, timestamp, equity, daily_return
        - metrics: Dictionary with performance metrics (final_pf, sharpe, trades, ...)
        - trades: Optional DataFrame with all trades (if include_trades=True)
        - signals: Optional DataFrame with all signals (if include_signals=True)
        - target_positions: Optional DataFrame with target positions (if include_targets=True)

    Raises:
        ValueError: If required columns are missing in prices
        KeyError: If signal_fn or position_sizing_fn return invalid DataFrames

    Example:
        >>> # Simple trend-following backtest
        >>> from src.assembled_core.data.prices_ingest import load_eod_prices
        >>> from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
        >>> from src.assembled_core.portfolio.position_sizing import compute_target_positions
        >>>
        >>> prices = load_eod_prices(freq="1d")
        >>>
        >>> def signal_fn(prices_df):
        ...     return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)
        >>>
        >>> def sizing_fn(signals_df, capital):
        ...     return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)
        >>>
        >>> result = run_portfolio_backtest(
        ...     prices=prices,
        ...     signal_fn=signal_fn,
        ...     position_sizing_fn=sizing_fn,
        ...     start_capital=10000.0,
        ...     include_costs=True,
        ...     include_trades=True
        ... )
        >>>
        >>> print(f"Final PF: {result.metrics['final_pf']:.4f}")
        >>> print(f"Sharpe: {result.metrics['sharpe']:.4f}")
        >>> print(f"Trades: {result.metrics['trades']}")
    """
    # Validate input
    if prices is None or prices.empty:
        raise ValueError("Missing required columns: prices DataFrame is None or empty")

    required_cols = ["timestamp", "symbol", "close"]
    missing = [c for c in required_cols if c not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Ensure prices are sorted
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()

    # Step 1: Compute features (optional) - skip if cycle_fn is provided (features precomputed and passed via ctx_template)
    # Note: timed_block is not defined, using nullcontext for now
    # TODO: Add proper timing support if needed
    with nullcontext():
        # Only compute features if prices is not empty (features require data)
        # Skip feature computation if cycle_fn is provided (features are precomputed once and passed via ctx_template.precomputed_prices_with_features)
        if compute_features and len(prices) > 0 and cycle_fn is None:
            config = feature_config or {}
            # Check if we have required columns for features (ATR needs high/low)
            has_ohlc = all(col in prices.columns for col in ["high", "low", "open"])
            
            if use_factor_store:
                # Use factor store (build_or_load_factors)
                logger.info(f"Using factor store: group={factor_group}, root={factor_store_root}")
                
                # Compute universe key
                universe_symbols = sorted(prices["symbol"].unique().tolist())
                universe_key = compute_universe_key(symbols=universe_symbols)
                
                # Determine date range for PIT-safe loading
                start_date = prices["timestamp"].min()
                end_date = prices["timestamp"].max()
                
                # Build or load factors
                prices_with_features = build_or_load_factors(
                    prices=prices,
                    factor_group=factor_group,
                    freq=rebalance_freq,
                    universe_key=universe_key,
                    start_date=start_date,
                    end_date=end_date,
                    as_of=None,  # Backtest uses full range
                    force_rebuild=False,
                    builder_fn=add_all_features if has_ohlc else None,
                    builder_kwargs={
                        "ma_windows": config.get("ma_windows", (20, 50, 200)),
                        "atr_window": config.get("atr_window", 14),
                        "rsi_window": config.get("rsi_window", 14),
                        "include_rsi": config.get("include_rsi", True),
                    } if has_ohlc else {
                        "windows": config.get("ma_windows", (20, 50, 200)),
                    },
                    factors_root=factor_store_root,
                )
            else:
                # Default: direct computation (backward compatible)
                if has_ohlc:
                    prices_with_features = add_all_features(
                        prices,
                        ma_windows=config.get("ma_windows", (20, 50, 200)),
                        atr_window=config.get("atr_window", 14),
                        rsi_window=config.get("rsi_window", 14),
                        include_rsi=config.get("include_rsi", True),
                    )
                else:
                    # If OHLC not available, only compute features that don't need them
                    prices_with_features = add_log_returns(prices.copy())
                    prices_with_features = add_moving_averages(
                        prices_with_features,
                        windows=config.get("ma_windows", (20, 50, 200)),
                    )
        else:
            prices_with_features = prices.copy()
        # If cycle_fn is provided, features are precomputed once and passed via ctx_template.precomputed_prices_with_features
        # Set prices_with_features to prices for now (will not be used, as cycle_fn uses precomputed features)
        if cycle_fn is not None:
            prices_with_features = prices.copy()

    # Step 2: Generate signals (skip if cycle_fn is provided, signals generated per timestamp)
    with nullcontext():
        if cycle_fn is None:
            if signal_fn is None:
                raise ValueError("signal_fn is required when cycle_fn is not provided")
            signals = signal_fn(prices_with_features)
        else:
            # Signals will be generated per timestamp via cycle_fn
            # Create empty signals DataFrame for compatibility
            signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    # Validate signals (skip validation if cycle_fn is provided)
    if cycle_fn is None:
        required_signal_cols = ["timestamp", "symbol", "direction"]
        missing_signal = [c for c in required_signal_cols if c not in signals.columns]
        if missing_signal:
            raise KeyError(
                f"signal_fn must return DataFrame with columns: {required_signal_cols}. Missing: {missing_signal}"
            )

        signals = signals.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Step 2.5: Apply meta-model ensemble (if enabled) - skip if cycle_fn is provided
    if use_meta_model and cycle_fn is None:
        logger.info("Applying meta-model ensemble...")

        # Load meta-model if path provided
        if meta_model is None and meta_model_path is not None:
            try:
                from src.assembled_core.signals.meta_model import load_meta_model

                meta_model = load_meta_model(meta_model_path)
                logger.info(f"Loaded meta-model from {meta_model_path}")
            except Exception as e:
                logger.error(f"Failed to load meta-model from {meta_model_path}: {e}")
                raise ValueError(f"Failed to load meta-model: {e}") from e

        if meta_model is None:
            raise ValueError(
                "use_meta_model=True but no meta_model or meta_model_path provided"
            )

        # Extract features for meta-model
        # Features must match the feature_names used during training
        feature_cols = meta_model.feature_names

        # Check which features are available in prices_with_features
        available_features = [
            f for f in feature_cols if f in prices_with_features.columns
        ]
        missing_features = [
            f for f in feature_cols if f not in prices_with_features.columns
        ]

        if missing_features:
            logger.warning(
                f"Missing {len(missing_features)} features for meta-model: {missing_features[:5]}..."
            )
            logger.warning(
                "Meta-model ensemble may not work correctly. Continuing anyway..."
            )

        if not available_features:
            logger.error("No features available for meta-model. Disabling ensemble.")
            use_meta_model = False
        else:
            # Join signals with prices_with_features to get features
            # Use timestamp and symbol as join keys
            signals_with_features = signals.merge(
                prices_with_features[["timestamp", "symbol"] + available_features],
                on=["timestamp", "symbol"],
                how="inner",
            )

            if signals_with_features.empty:
                logger.warning(
                    "No signals matched with features. Disabling meta-model ensemble."
                )
                use_meta_model = False
            else:
                # Extract features DataFrame (only available features)
                features_subset = signals_with_features[available_features].copy()

                # Fill missing features with 0 (for features not in prices_with_features)
                if missing_features:
                    for feat in missing_features:
                        features_subset[feat] = 0.0
                    # Reorder to match meta_model.feature_names
                    features_subset = features_subset[meta_model.feature_names]

                # Apply ensemble layer
                from src.assembled_core.signals.ensemble import (
                    apply_meta_filter,
                    apply_meta_scaling,
                )

                original_signal_count = len(signals_with_features)
                original_long_count = (
                    signals_with_features["direction"] == "LONG"
                ).sum()

                if meta_ensemble_mode == "filter":
                    signals_with_features = apply_meta_filter(
                        signals=signals_with_features,
                        meta_model=meta_model,
                        features=features_subset,
                        min_confidence=meta_min_confidence,
                        join_keys=["timestamp", "symbol"],
                    )
                elif meta_ensemble_mode == "scaling":
                    signals_with_features = apply_meta_scaling(
                        signals=signals_with_features,
                        meta_model=meta_model,
                        features=features_subset,
                        min_confidence=meta_min_confidence,
                        max_scaling=1.0,
                        join_keys=["timestamp", "symbol"],
                        scale_score=True,
                    )
                else:
                    raise ValueError(
                        f"Unsupported meta_ensemble_mode: {meta_ensemble_mode}. Supported: 'filter', 'scaling'"
                    )

                # Update signals with filtered/scaled results
                # Keep original signals structure but update direction and add meta_confidence
                meta_cols = ["timestamp", "symbol", "direction", "meta_confidence"]
                if "final_score" in signals_with_features.columns:
                    meta_cols.append("final_score")

                signals = signals.merge(
                    signals_with_features[meta_cols],
                    on=["timestamp", "symbol"],
                    how="left",
                    suffixes=("", "_meta"),
                )

                # Update direction from meta-filtered signals
                if "direction_meta" in signals.columns:
                    signals["direction"] = signals["direction_meta"].fillna(
                        signals["direction"]
                    )
                    signals = signals.drop(columns=["direction_meta"])

                # Update score if final_score is available (from scaling mode)
                if "final_score" in signals.columns:
                    if "score" not in signals.columns:
                        signals["score"] = 0.0
                    signals["score"] = signals["final_score"].fillna(signals["score"])
                    signals = signals.drop(columns=["final_score"])

                # Log results
                filtered_signal_count = len(signals_with_features)
                filtered_long_count = (
                    signals_with_features["direction"] == "LONG"
                ).sum()
                dropped_count = original_long_count - filtered_long_count

                logger.info("Meta-model ensemble applied:")
                logger.info(
                    f"  Original signals: {original_signal_count} (LONG: {original_long_count})"
                )
                logger.info(
                    f"  After filtering: {filtered_signal_count} (LONG: {filtered_long_count})"
                )
                logger.info(f"  Dropped signals: {dropped_count}")
                logger.info(
                    f"  Mode: {meta_ensemble_mode}, Min confidence: {meta_min_confidence}"
                )

    # Step 3: Compute target positions (group by timestamp for rebalancing)
    with nullcontext():
        all_targets = []
        all_orders = []
        all_signals_list = []  # For collecting signals if include_signals=True
        current_positions = pd.DataFrame(columns=["symbol", "qty"])

        # Determine timeline (unique timestamps from prices, sorted)
        timeline = sorted(prices["timestamp"].unique())

        # Use cycle_fn if provided (TradingCycle integration), otherwise use legacy path
        if cycle_fn is not None:
            # TradingCycle path: use cycle_fn for each timestamp
            for timestamp in timeline:
                # Run trading cycle for this timestamp
                cycle_result = cycle_fn(timestamp, current_positions)
                
                if cycle_result.status != "success":
                    logger.warning(
                        f"Trading cycle failed for timestamp {timestamp}: {cycle_result.error_message}"
                    )
                    continue
                
                # Extract orders (prefer orders_filtered, fallback to orders)
                orders = (
                    cycle_result.orders_filtered
                    if not cycle_result.orders_filtered.empty
                    else cycle_result.orders
                )
                
                # Update positions using vectorized operations (Fill/Fees/Equity-Update stays in Engine)
                if not orders.empty:
                    current_positions = _update_positions_vectorized(orders, current_positions)
                    all_orders.append(orders)
                
                # Store targets if requested
                if include_targets and not cycle_result.target_positions.empty:
                    targets_with_timestamp = cycle_result.target_positions.copy()
                    targets_with_timestamp["timestamp"] = timestamp
                    all_targets.append(targets_with_timestamp)
                
                # Store signals if requested
                if include_signals and not cycle_result.signals.empty:
                    all_signals_list.append(cycle_result.signals)
        else:
            # Legacy path: group signals by timestamp and process
            if signal_fn is None or position_sizing_fn is None:
                raise ValueError(
                    "signal_fn and position_sizing_fn are required when cycle_fn is not provided"
                )
            
            for timestamp, signal_group in signals.groupby("timestamp"):
                orders, current_positions, targets = _process_rebalancing_timestamp(
                    timestamp=timestamp,
                    signal_group=signal_group,
                    current_positions=current_positions,
                    position_sizing_fn=position_sizing_fn,
                    start_capital=start_capital,
                    prices=prices,
                    include_targets=include_targets,
                )

                # Store targets and orders
                if include_targets and not targets.empty:
                    all_targets.append(targets)

                if not orders.empty:
                    all_orders.append(orders)

        # Combine all orders
        if all_orders:
            orders_df = pd.concat(all_orders, ignore_index=True)
            orders_df = orders_df.sort_values("timestamp").reset_index(drop=True)
        else:
            orders_df = pd.DataFrame(
                columns=["timestamp", "symbol", "side", "qty", "price"]
            )

    # Step 4: Simulate equity
    with nullcontext():
        # Get cost parameters
        if cost_model is not None:
            commission_bps = (
                commission_bps
                if commission_bps is not None
                else cost_model.commission_bps
            )
            spread_w = spread_w if spread_w is not None else cost_model.spread_w
            impact_w = impact_w if impact_w is not None else cost_model.impact_w
        else:
            default_costs = get_default_cost_model()
            commission_bps = (
                commission_bps
                if commission_bps is not None
                else default_costs.commission_bps
            )
            spread_w = spread_w if spread_w is not None else default_costs.spread_w
            impact_w = impact_w if impact_w is not None else default_costs.impact_w

        if include_costs:
            equity, metrics = simulate_with_costs(
                orders=orders_df,
                start_capital=start_capital,
                commission_bps=commission_bps,
                spread_w=spread_w,
                impact_w=impact_w,
                freq=rebalance_freq,
            )
            # Add trades count to metrics
            metrics["trades"] = len(orders_df)
        else:
            equity = simulate_equity(prices, orders_df, start_capital)
            metrics = compute_metrics(equity)
            metrics["trades"] = len(orders_df)

    # Step 5: Enhance equity DataFrame with daily_return
    with nullcontext():
        # Ensure equity has timestamp column (rename if needed)
        if "timestamp" in equity.columns:
            equity = equity.copy()
            # Add date column (date part of timestamp)
            equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
            # Compute daily return
            equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
            # Ensure columns are in correct order: date, timestamp, equity, daily_return
            equity = equity[["date", "timestamp", "equity", "daily_return"]].copy()
        elif "date" in equity.columns:
            # If already has date, add daily_return
            equity = equity.copy()
            equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
            # Ensure columns are in correct order: date, equity, daily_return
            if "timestamp" not in equity.columns:
                equity = equity[["date", "equity", "daily_return"]].copy()
            else:
                equity = equity[["date", "timestamp", "equity", "daily_return"]].copy()
        else:
            # Fallback: create date from index or use timestamp
            equity = equity.copy()
            if equity.index.dtype == "datetime64[ns]":
                equity["date"] = equity.index.date
                equity["timestamp"] = equity.index
            else:
                # Try to infer from timestamp column
                if "timestamp" in equity.columns:
                    equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
                else:
                    # Last resort: use row number as date surrogate
                    equity["date"] = pd.date_range(
                        start="2000-01-01", periods=len(equity), freq="D"
                    ).date
                    equity["timestamp"] = pd.to_datetime(equity["date"])
            equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
            equity = equity[["date", "timestamp", "equity", "daily_return"]].copy()

    # Step 6: Build result
    # Combine signals if collected from cycle_fn
    signals_result = None
    if include_signals:
        if cycle_fn is not None and all_signals_list:
            signals_result = pd.concat(all_signals_list, ignore_index=True)
            signals_result = signals_result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        elif cycle_fn is None:
            signals_result = signals
    
    result = BacktestResult(
        equity=equity,
        metrics=metrics,
        trades=orders_df if include_trades else None,
        signals=signals_result,
        target_positions=pd.concat(all_targets, ignore_index=True)
        if include_targets and all_targets
        else None,
    )

    return result
