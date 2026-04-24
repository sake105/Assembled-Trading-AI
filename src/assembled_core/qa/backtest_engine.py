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

logger = logging.getLogger(__name__)

from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.execution.transaction_costs import (
    SlippageModel,
    SpreadModel,
    add_cost_columns_to_trades,
    commission_model_from_cost_params,
)
from src.assembled_core.features.ta_features import (
    add_all_features,
    add_log_returns,
    add_moving_averages,
)
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.data.factor_store import compute_universe_key
from src.assembled_core.pipeline.backtest import compute_metrics, simulate_equity
from src.assembled_core.pipeline.portfolio import simulate_with_costs

# Type-only imports to avoid circular dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.assembled_core.pipeline.trading_cycle import (
        TradingContext,
        TradingCycleResult,
    )
from src.assembled_core.utils.timing import timed_step
from src.assembled_core.config.settings import get_settings

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
        trades: Optional DataFrame with columns: timestamp, symbol, side, qty, price,
            fill_qty, fill_price, status, remaining_qty, commission_cash, spread_cash,
            slippage_cash, total_cost_cash
            List of all trades/fills executed during backtest (conforms to fill model schema)
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
    meta: dict[str, Any] | None = None
    """Optional metadata dictionary (e.g., timings, configuration)"""


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
        except (ImportError, AttributeError) as exc:
            # Fall through to pandas implementation
            logger.warning("[BacktestEngine] numba fill simulation failed, falling back to pandas: %s", exc)

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
    ctx_template: "TradingContext",
    *,
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    capital: float,
    run_trading_cycle_fn: Callable | None = None,
    enable_risk_controls: bool | None = None,
) -> Callable[[pd.Timestamp, pd.DataFrame], "TradingCycleResult"]:
    """Create a callable that runs trading cycle for a given timestamp and positions.

    This is an adapter function that bridges the backtest engine's per-timestamp
    loop with the unified trading cycle orchestrator.

    Args:
        ctx_template: Template TradingContext with shared configuration
            (prices, universe, feature_config, factor_store settings, etc.)
        signal_fn: Signal function (same as in run_portfolio_backtest)
        position_sizing_fn: Position sizing function (same as in run_portfolio_backtest)
        capital: Capital for position sizing (updated per timestamp based on equity)
        run_trading_cycle_fn: Optional callable to run trading cycle (default: imports at runtime)
            If None, imports run_trading_cycle from pipeline.trading_cycle at runtime

    Returns:
        Callable that takes (timestamp: pd.Timestamp, current_positions: pd.DataFrame)
        and returns TradingCycleResult

    Example:
        >>> from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
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
        ...     run_trading_cycle_fn=run_trading_cycle,
        ... )
        >>> result = cycle_fn(timestamp, current_positions)
        >>> orders = result.orders_filtered  # or result.orders as fallback
    """
    # Import at runtime to avoid circular dependency
    if run_trading_cycle_fn is None:
        from src.assembled_core.pipeline.trading_cycle import (
            run_trading_cycle as run_trading_cycle_fn,
        )

    # When not explicitly set, respect ctx_template's enable_risk_controls value.
    _enable_risk_controls: bool = (
        enable_risk_controls
        if enable_risk_controls is not None
        else getattr(ctx_template, "enable_risk_controls", True)
    )

    def cycle_fn(
        timestamp: pd.Timestamp,
        current_positions: pd.DataFrame,
        *,
        equity_curve: pd.Series | None = None,
        equity_curve_index: int | None = None,
        profit_lock_state: dict[str, Any] | None = None,
    ) -> "TradingCycleResult":
        """Run trading cycle for a specific timestamp.

        Args:
            timestamp: Current timestamp for rebalancing
            current_positions: Current portfolio positions (columns: symbol, qty)
            equity_curve: Optional equity series for profit_lock overlay (backtest sets per step)
            equity_curve_index: Current bar index into equity_curve
            profit_lock_state: Optional state dict for profit_lock cooldown roundtrip

        Returns:
            TradingCycleResult with orders, signals, target_positions, etc.
        """
        # Build context from template, updating timestamp-specific fields
        # Pass through backtest_use_snapshot so history-slice strategies (e.g. EMA trend) get full history
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
            # E0.1 parity: risk controls now honored in backtest path. Default
            # True so backtest and paper share the same decision logic.
            # Explicit opt-out (enable_risk_controls=False) is still possible
            # for speed-focused research runs but must be documented at the
            # call site.
            enable_risk_controls=_enable_risk_controls,
            security_meta_df=ctx_template.security_meta_df,  # Pass through security metadata
            backtest_use_snapshot=getattr(ctx_template, "backtest_use_snapshot", False),
            equity_curve=equity_curve,
            equity_curve_index=equity_curve_index,
            profit_lock_state=profit_lock_state,
        )

        # Run trading cycle
        return run_trading_cycle_fn(ctx)

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

    # Generate orders to transition from current to target positions.
    # Use prices at the exact timestamp first; if a symbol being sold is
    # missing (sparse data / gap day), fall back to its last known close
    # so SELL orders don't get price=0.
    prices_at_timestamp = prices[prices["timestamp"] == timestamp]
    if prices_at_timestamp.empty:
        prices_at_timestamp = None
    elif not current_positions.empty:
        # Ensure SELL-side symbols have prices even on gap days
        held_symbols = set(current_positions["symbol"].unique())
        available_symbols = set(prices_at_timestamp["symbol"].unique())
        missing_symbols = held_symbols - available_symbols
        if missing_symbols:
            ts_utc = pd.to_datetime(timestamp, utc=True)
            hist = prices[
                (pd.to_datetime(prices["timestamp"], utc=True) <= ts_utc)
                & (prices["symbol"].isin(missing_symbols))
            ]
            if not hist.empty:
                last_known = hist.groupby("symbol").tail(1)
                # Override timestamp so it merges correctly
                last_known = last_known.copy()
                last_known["timestamp"] = timestamp
                prices_at_timestamp = pd.concat(
                    [prices_at_timestamp, last_known], ignore_index=True
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


def _validate_order_notional_guard(
    orders_df: pd.DataFrame,
    start_capital: float,
    *,
    strict: bool | None = None,
) -> None:
    """Warn or raise if any order notional exceeds 2x start capital (qty unit mismatch).

    When strict is True (or AS_CORE_STRICT_QTY=1), raises ValueError. Otherwise logs warning.
    """
    if orders_df.empty or start_capital <= 0:
        return
    order_notional = orders_df["qty"].abs() * orders_df["price"].abs()
    over = (order_notional > 2.0 * start_capital).any()
    if not over:
        return
    if strict is None:
        import os

        strict = os.environ.get("AS_CORE_STRICT_QTY") == "1"
    if strict:
        raise ValueError(
            "At least one order has notional > 2x start capital; "
            "possible qty unit mismatch (notional vs shares). Set qty in shares."
        )
    logger.warning(
        "At least one order has notional > 2x start capital; "
        "possible qty unit mismatch (notional vs shares)."
    )


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
    cycle_fn: (
        Callable[[pd.Timestamp, pd.DataFrame], "TradingCycleResult"] | None
    ) = None,
    # Performance optimization
    use_numba: bool | None = None,
    # Ledger/Reconciliation integration (Sprint 13)
    include_ledger: bool = True,
    run_id: str | None = None,
    output_dir: Path | None = None,
    # Broker snapshot controls (Sprint 13 extension)
    broker_snapshot_policy: str = "prefer",
    write_broker_snapshot: bool = False,
    broker_snapshot_run_id: str | None = None,
    broker_snapshot_file: str | Path | None = None,
    broker_snapshot_date: str | None = None,
    # Evidence pack controls
    write_evidence_pack: bool = False,
    # Fill pipeline: strict_session_gate=False allows tests to run without exchange_calendars
    strict_session_gate: bool = True,
    # Rebalance schedule: "daily" = every bar; "weekly" = every 5th bar (1d) to reduce turnover
    rebalance_schedule: str = "daily",
    # Optional: restrict rebalance to these timestamps only (e.g. for EOD parity tests)
    rebalance_timestamps: list[pd.Timestamp] | None = None,
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
    # Get use_numba from parameter or settings (default: False)
    if use_numba is None:
        settings = get_settings()
        use_numba = settings.use_numba

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
                logger.info(
                    f"Using factor store: group={factor_group}, root={factor_store_root}"
                )

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
                    builder_kwargs=(
                        {
                            "ma_windows": config.get("ma_windows", (20, 50, 200)),
                            "atr_window": config.get("atr_window", 14),
                            "rsi_window": config.get("rsi_window", 14),
                            "include_rsi": config.get("include_rsi", True),
                        }
                        if has_ohlc
                        else {
                            "windows": config.get("ma_windows", (20, 50, 200)),
                        }
                    ),
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
            signals = pd.DataFrame(
                columns=["timestamp", "symbol", "direction", "score"]
            )

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
    # Initialize timings dictionary for step-level profiling
    timings: dict[str, Any] = {}

    with nullcontext():
        all_targets = []
        all_orders = []
        all_signals_list = []  # For collecting signals if include_signals=True
        current_positions = pd.DataFrame(columns=["symbol", "qty"])

        # Determine timeline (unique timestamps from prices, sorted)
        # Normalize to UTC-aware timestamps for stable membership checks
        raw_timeline = sorted(prices["timestamp"].unique())
        timeline: list[pd.Timestamp] = []
        for ts in raw_timeline:
            ts_pd = pd.to_datetime(ts)
            if ts_pd.tzinfo is None or ts_pd.tz is None:
                ts_pd = ts_pd.tz_localize("UTC")
            else:
                ts_pd = ts_pd.tz_convert("UTC")
            timeline.append(ts_pd)

        # Normalize rebalance_timestamps (if provided) to UTC-aware as well
        if rebalance_timestamps is not None:
            normalized_rebalance: list[pd.Timestamp] = []
            for ts in rebalance_timestamps:
                ts_pd = pd.to_datetime(ts)
                if ts_pd.tzinfo is None or ts_pd.tz is None:
                    ts_pd = ts_pd.tz_localize("UTC")
                else:
                    ts_pd = ts_pd.tz_convert("UTC")
                normalized_rebalance.append(ts_pd)
            rebalance_timestamps_set = set(normalized_rebalance)
        elif rebalance_schedule == "weekly":
            rebalance_timestamps_set = set(
                timeline[i] for i in range(0, len(timeline), 5)
            )
        else:
            rebalance_timestamps_set = set(timeline)

        # Use cycle_fn if provided (TradingCycle integration), otherwise use legacy path
        if cycle_fn is not None:
            # TradingCycle path: use cycle_fn for each timestamp
            decision_timings = []
            position_update_timings = []
            # Running equity and state for profit_lock (INT-6.2)
            equity_values: list[float] = [start_capital]
            cash = start_capital
            profit_lock_state: dict[str, Any] | None = None

            for timestamp in timeline:
                if timestamp not in rebalance_timestamps_set:
                    continue
                # Equity curve up to (and including) previous step; index = len-1
                equity_curve_series = pd.Series(equity_values)
                equity_curve_index = len(equity_values) - 1
                # Decision (cycle_fn) - pass equity and profit_lock state
                with timed_step(f"decision_{timestamp}", timings, logger):
                    cycle_result = cycle_fn(
                        timestamp,
                        current_positions,
                        equity_curve=equity_curve_series,
                        equity_curve_index=equity_curve_index,
                        profit_lock_state=profit_lock_state,
                    )

                # Track per-timestamp decision timing for aggregation
                if f"decision_{timestamp}" in timings:
                    decision_timings.append(
                        timings[f"decision_{timestamp}"]["duration_ms"]
                    )

                if cycle_result.status != "success":
                    logger.warning(
                        f"Trading cycle failed for timestamp {timestamp}: {cycle_result.error_message}"
                    )
                    continue

                # Persist profit_lock state for next step
                profit_lock_state = cycle_result.meta.get("profit_lock_state")

                # Use orders_filtered (risk-checked). If empty (all blocked), use no orders.
                # Do NOT fall back to unfiltered cycle_result.orders — that bypasses
                # pre-trade risk controls and causes unbounded position accumulation.
                orders = cycle_result.orders_filtered

                # Defensive guard: qty must be >= 0 (side encodes direction)
                if not orders.empty and "qty" in orders.columns and (orders["qty"] < 0).any():
                    orders = orders.copy()
                    orders["qty"] = orders["qty"].abs()

                # Update cash from orders (buy = outflow, sell = inflow)
                if not orders.empty:
                    for _, row in orders.iterrows():
                        side = row.get("side", "BUY")
                        qty = float(row.get("qty", 0) or 0)
                        price = float(row.get("price", 0) or 0)
                        notional = qty * price
                        if side == "BUY":
                            cash -= notional
                        else:
                            cash += notional

                # Update positions using vectorized operations (Fill/Fees/Equity-Update stays in Engine)
                if not orders.empty:
                    with timed_step(f"position_update_{timestamp}", timings, logger):
                        current_positions = _update_positions_vectorized(
                            orders, current_positions, use_numba=use_numba
                        )
                    # Track per-timestamp position update timing
                    if f"position_update_{timestamp}" in timings:
                        position_update_timings.append(
                            timings[f"position_update_{timestamp}"]["duration_ms"]
                        )
                    all_orders.append(orders)

                # Running equity for next step (cash + MTM of current positions)
                prices_at_ts = prices[prices["timestamp"] == timestamp]
                if not prices_at_ts.empty and not current_positions.empty:
                    px = prices_at_ts.set_index("symbol")["close"]
                    qty_series = current_positions.set_index("symbol")["qty"]
                    mtm = (qty_series * px.reindex(qty_series.index).fillna(0)).sum()
                else:
                    mtm = 0.0
                equity_values.append(cash + float(mtm))

                # Store targets if requested
                if include_targets and not cycle_result.target_positions.empty:
                    targets_with_timestamp = cycle_result.target_positions.copy()
                    targets_with_timestamp["timestamp"] = timestamp
                    all_targets.append(targets_with_timestamp)

                # Store signals if requested
                if include_signals and not cycle_result.signals.empty:
                    all_signals_list.append(cycle_result.signals)

            # Aggregate per-timestamp timings into summary
            if decision_timings:
                timings["decision"] = {
                    "total_duration_ms": sum(decision_timings),
                    "avg_duration_ms": sum(decision_timings) / len(decision_timings),
                    "count": len(decision_timings),
                }
            if position_update_timings:
                timings["position_update"] = {
                    "total_duration_ms": sum(position_update_timings),
                    "avg_duration_ms": sum(position_update_timings)
                    / len(position_update_timings),
                    "count": len(position_update_timings),
                }
        else:
            # Legacy path: group signals by timestamp and process
            if signal_fn is None or position_sizing_fn is None:
                raise ValueError(
                    "signal_fn and position_sizing_fn are required when cycle_fn is not provided"
                )

            order_generation_timings = []
            # Fix 18: track running equity (cash + MTM) so position sizing uses
            # current equity rather than start_capital throughout the backtest.
            _legacy_cash: float = start_capital
            # Rebalance only on selected bars (weekly = every 5th for 1d)
            sig_timeline = sorted(signals["timestamp"].unique())
            if rebalance_schedule == "weekly":
                rebalance_timestamps_legacy = set(
                    sig_timeline[i] for i in range(0, len(sig_timeline), 5)
                )
            else:
                rebalance_timestamps_legacy = set(sig_timeline)

            for timestamp, signal_group in signals.groupby("timestamp"):
                if timestamp not in rebalance_timestamps_legacy:
                    continue

                # Fix 18: compute current equity = cash + mark-to-market of positions
                _current_equity: float = _legacy_cash
                try:
                    prices_at_ts = prices[prices["timestamp"] == timestamp]
                    if not prices_at_ts.empty and not current_positions.empty:
                        px = prices_at_ts.set_index("symbol")["close"]
                        qty_series = current_positions.set_index("symbol")["qty"]
                        mtm = float(
                            (qty_series * px.reindex(qty_series.index).fillna(0.0)).sum()
                        )
                        _current_equity = _legacy_cash + mtm
                except Exception:
                    pass

                # Order generation (includes position sizing and order generation)
                with timed_step(f"order_generation_{timestamp}", timings, logger):
                    orders, updated_positions, targets = _process_rebalancing_timestamp(
                        timestamp=timestamp,
                        signal_group=signal_group,
                        current_positions=current_positions,
                        position_sizing_fn=position_sizing_fn,
                        start_capital=_current_equity,
                        prices=prices,
                        include_targets=include_targets,
                    )
                # Track per-timestamp order generation timing
                if f"order_generation_{timestamp}" in timings:
                    order_generation_timings.append(
                        timings[f"order_generation_{timestamp}"]["duration_ms"]
                    )

                # Fix 18: update running cash from executed orders
                try:
                    if not orders.empty:
                        for _, _row in orders.iterrows():
                            _side = _row.get("side", "BUY")
                            _qty = float(_row.get("qty", 0) or 0)
                            _price = float(_row.get("price", 0) or 0)
                            _notional = _qty * _price
                            if _side == "BUY":
                                _legacy_cash -= _notional
                            else:
                                _legacy_cash += _notional
                except Exception:
                    pass

                # Position update is done inside _process_rebalancing_timestamp for legacy path
                current_positions = updated_positions

                # Store targets and orders
                if include_targets and not targets.empty:
                    all_targets.append(targets)

                if not orders.empty:
                    all_orders.append(orders)

            # Aggregate per-timestamp timings into summary
            if order_generation_timings:
                timings["order_generation"] = {
                    "total_duration_ms": sum(order_generation_timings),
                    "avg_duration_ms": sum(order_generation_timings)
                    / len(order_generation_timings),
                    "count": len(order_generation_timings),
                }

        # Combine all orders
        if all_orders:
            orders_df = pd.concat(all_orders, ignore_index=True)
            orders_df = orders_df.sort_values("timestamp").reset_index(drop=True)
        else:
            orders_df = pd.DataFrame(
                columns=["timestamp", "symbol", "side", "qty", "price"]
            )

    _validate_order_notional_guard(orders_df, start_capital)
    # Meta: mark unit for debugging (orders from order_generation are in shares)
    if not orders_df.empty:
        orders_df.attrs["qty_unit"] = "shares"

    # Step 4: Simulate equity (fill_sim + equity_update)
    with timed_step("fill_sim", timings, logger):
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
            equity, metrics, trades_df = simulate_with_costs(
                orders=orders_df,
                start_capital=start_capital,
                commission_bps=commission_bps,
                spread_w=spread_w,
                impact_w=impact_w,
                freq=rebalance_freq,
                prices=prices,  # Pass prices for fill model pipeline
                strict_session_gate=strict_session_gate,
            )
            # Add trades count to metrics
            metrics["trades"] = len(orders_df)
        else:
            equity = simulate_equity(prices, orders_df, start_capital)
            metrics = compute_metrics(equity)
            metrics["trades"] = len(orders_df)
            # For ledger integration, use orders_df as trades_df when costs are disabled
            trades_df = pd.DataFrame()

    # Step 4.5: Apply fill model pipeline (session gate -> limit -> partial)
    # This must happen BEFORE cost calculation, as costs are based on fill_qty
    if not orders_df.empty:
        from src.assembled_core.execution.fill_model_pipeline import (
            apply_fill_model_pipeline,
        )

        # Apply fill model pipeline
        # For now, use default partial fill model (can be made configurable later)
        partial_fill_model = None  # Default: full fills (no ADV cap)
        # TODO: Make partial_fill_model configurable via cost_model or separate parameter

        orders_df = apply_fill_model_pipeline(
            orders_df,
            prices=prices,
            freq=rebalance_freq,
            partial_fill_model=partial_fill_model,
            strict_session_gate=strict_session_gate,
        )

    # Step 4.6: Add cost columns to orders (if include_trades=True)
    # Costs are now computed based on fill_qty (for partial/rejected fills)
    if include_trades and not orders_df.empty:
        # Only compute costs if include_costs=True
        if include_costs:
            # Create commission model from cost parameters
            if cost_model is not None:
                commission_model = commission_model_from_cost_params(
                    commission_bps=cost_model.commission_bps
                )
            else:
                commission_model = commission_model_from_cost_params(
                    commission_bps=commission_bps if commission_bps is not None else 0.0
                )
            # Create spread model from legacy parameters (if spread_w > 0)
            spread_model = None
            if spread_w is not None and spread_w > 0.0:
                # Default spread model: simple buckets based on ADV
                # For now, use fallback spread_bps = spread_w (legacy compatibility)
                spread_model = SpreadModel(
                    adv_window=20,
                    buckets=None,  # No buckets: use fallback for all
                    fallback_spread_bps=spread_w
                    * 100.0,  # Convert spread_w (0.25 = 25 bps) to bps
                )

            # Create slippage model from legacy parameters (if impact_w > 0)
            slippage_model = None
            if impact_w is not None and impact_w > 0.0:
                # Default slippage model: volatility-based
                # Map impact_w to slippage model (impact_w is a weight, convert to reasonable slippage)
                slippage_model = SlippageModel(
                    vol_window=20,
                    k=impact_w,  # Use impact_w as scaling factor
                    min_bps=0.0,
                    max_bps=50.0,
                    fallback_slippage_bps=impact_w * 100.0,  # Convert to bps
                )

            orders_df = add_cost_columns_to_trades(
                orders_df,
                commission_model=commission_model,
                spread_model=spread_model,
                slippage_model=slippage_model,
                prices=prices if include_trades else None,
            )
        else:
            # Costs disabled: add cost columns with 0.0 values (for schema stability)
            orders_df["commission_cash"] = 0.0
            orders_df["spread_cash"] = 0.0
            orders_df["slippage_cash"] = 0.0
            orders_df["total_cost_cash"] = 0.0

    # Step 5: Enhance equity DataFrame with daily_return
    with nullcontext():
        # Ensure equity has timestamp column (rename if needed)
        if "timestamp" in equity.columns:
            equity = equity.copy()
            # Add date column (date part of timestamp)
            equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
            # Compute daily return
            equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
            # Ensure columns: date, timestamp, equity, daily_return; keep cash if present (for cash_curve CSV)
            base_cols = ["date", "timestamp", "equity", "daily_return"]
            extra = [c for c in ["cash"] if c in equity.columns]
            equity = equity[base_cols + extra].copy()
        elif "date" in equity.columns:
            # If already has date, add daily_return
            equity = equity.copy()
            equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
            # Ensure columns are in correct order: date, equity, daily_return
            base = ["date", "equity", "daily_return"]
            if "timestamp" in equity.columns:
                base = ["date", "timestamp", "equity", "daily_return"]
            extra = [c for c in ["cash"] if c in equity.columns]
            equity = equity[base + extra].copy()
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
            base_cols = ["date", "timestamp", "equity", "daily_return"]
            extra = [c for c in ["cash"] if c in equity.columns]
            equity = equity[base_cols + extra].copy()

    # Step 6: Ledger/Reconciliation integration (optional, default-on)
    ledger_result = None
    if include_ledger and run_id and output_dir:
        try:
            from src.assembled_core.accounting.ledger_integration import (
                build_ledger_from_trades,
            )

            # Get trades_df (from simulate_with_costs if include_costs=True, otherwise from orders_df)
            trades_for_ledger = orders_df.copy()
            if include_costs and not trades_df.empty:
                # Use trades_df from simulate_with_costs (has fill_qty, fill_price, status, costs)
                trades_for_ledger = trades_df.copy()

            # Determine snapshot run_id (for import and lookup)
            snapshot_run_id = (
                broker_snapshot_run_id if broker_snapshot_run_id is not None else run_id
            )

            # Step 6.1: Import external broker snapshot if provided
            if broker_snapshot_file:
                try:
                    logger.info(
                        f"Importing external broker snapshot from: {broker_snapshot_file}"
                    )
                    from pathlib import Path
                    from src.assembled_core.accounting.broker_snapshot_importer import (
                        import_broker_snapshot,
                    )

                    # Determine snapshot date (use provided date, or last trade date, or today)
                    snapshot_date = broker_snapshot_date
                    if snapshot_date is None:
                        if (
                            not trades_for_ledger.empty
                            and "timestamp" in trades_for_ledger.columns
                        ):
                            snapshot_date = pd.to_datetime(
                                trades_for_ledger["timestamp"].max(), utc=True
                            )
                        else:
                            snapshot_date = pd.Timestamp.now("UTC")
                    else:
                        snapshot_date = pd.to_datetime(snapshot_date, utc=True)

                    # Import snapshot
                    import_result = import_broker_snapshot(
                        snapshot_path=Path(broker_snapshot_file),
                        run_id=snapshot_run_id,
                        snapshot_date=snapshot_date,
                        output_dir=output_dir,
                        qty_tol=1e-8,
                        store_parquet=True,
                    )
                    logger.info(
                        f"Imported broker snapshot: {import_result['broker_snapshot_path']}, "
                        f"cash={import_result['cash']}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to import broker snapshot: {e}", exc_info=True
                    )
                    # If policy is require, we should fail here
                    if broker_snapshot_policy == "require":
                        raise ValueError(
                            f"Broker snapshot import failed (policy=require): {e}"
                        ) from e
                    # Otherwise, log and continue (snapshot might still exist from previous import)

            # Build ledger from trades
            ledger_result = build_ledger_from_trades(
                orders_df=orders_df,
                trades_df=trades_for_ledger,
                run_id=run_id,
                output_dir=output_dir,
                as_of_date=None,  # Use last timestamp from trades
                prices_df=prices,
                start_cash=start_capital,
                broker_snapshot_policy=broker_snapshot_policy,
                write_paper_broker_snapshot=write_broker_snapshot,
                broker_snapshot_run_id=snapshot_run_id,
                write_evidence_pack=write_evidence_pack,
            )
            logger.info(
                f"Ledger integration completed: ledger_pack_path={ledger_result.get('ledger_pack_path')}, reconciliation_ok={ledger_result.get('reconciliation_ok')}"
            )
        except Exception as e:
            logger.warning(f"Ledger integration failed: {e}", exc_info=True)
            # Distinguish "ledger not attempted" (None) from "attempted but
            # failed". A silent None after an exception has masked
            # reconciliation gaps in prior incidents — keep the failure
            # explicit so downstream meta consumers (QA, gating) see it.
            ledger_result = {
                "reconciliation_ok": False,
                "ledger_error": str(e),
            }

    # Step 7: Build result
    # Combine signals if collected from cycle_fn
    signals_result = None
    if include_signals:
        if cycle_fn is not None and all_signals_list:
            signals_result = pd.concat(all_signals_list, ignore_index=True)
            signals_result = signals_result.sort_values(
                ["symbol", "timestamp"]
            ).reset_index(drop=True)
        elif cycle_fn is None:
            signals_result = signals

    # Build meta dict with timings and ledger info
    meta_dict = {}
    if timings:
        meta_dict["timings"] = timings
    if ledger_result:
        # Core ledger / reconciliation paths
        meta_dict["ledger_pack_path"] = ledger_result.get("ledger_pack_path")
        meta_dict["reconcile_report_path"] = ledger_result.get("reconcile_report_path")
        meta_dict["reconciliation_ok"] = ledger_result.get("reconciliation_ok")
        meta_dict["broker_snapshot_path"] = ledger_result.get("broker_snapshot_path")
        # Surface ledger failures so downstream gates can see them instead of
        # misreading a missing key as "ledger not attempted".
        if ledger_result.get("ledger_error"):
            meta_dict["ledger_error"] = ledger_result.get("ledger_error")
        # Evidence pack fields (if written)
        meta_dict["evidence_index_path"] = ledger_result.get("evidence_index_path")
        meta_dict["evidence_pack_path"] = ledger_result.get("evidence_pack_path")
        meta_dict["evidence_pack_manifest_path"] = ledger_result.get(
            "evidence_pack_manifest_path"
        )

    # Use trades_df (with fill_qty, status, reject_reason) when from simulate_with_costs
    trades_for_result = None
    if include_trades:
        trades_for_result = (
            trades_df
            if (include_costs and not trades_df.empty and "status" in trades_df.columns)
            else orders_df
        )

    result = BacktestResult(
        equity=equity,
        metrics=metrics,
        trades=trades_for_result,
        signals=signals_result,
        target_positions=(
            pd.concat(all_targets, ignore_index=True)
            if include_targets and all_targets
            else None
        ),
        meta=meta_dict if meta_dict else None,
    )

    return result
