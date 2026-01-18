"""Unified Trading Cycle Orchestrator (B1).

This module provides a unified orchestrator interface for the common trading cycle steps:
1. Prices Loading (data ingest)
2. Features Building (TA features, factor store integration)
3. Signals Generation (trend, event, multi-factor)
4. Position Sizing (target positions computation)
5. Order Generation (orders from targets)
6. Risk Controls (pre-trade checks, kill switch)
7. Outputs (SAFE-CSV, equity curves, reports)

The orchestrator uses hook points for each step, allowing callers to override
default behavior or integrate with existing workflows.

Example usage:
    >>> from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
    >>> 
    >>> ctx = TradingContext(
    ...     prices=prices_df,
    ...     as_of=target_date,
    ...     signal_fn=lambda df: generate_trend_signals_from_prices(df, ma_fast=20, ma_slow=50),
    ...     position_sizing_fn=lambda sig, cap: compute_target_positions(sig, total_capital=cap),
    ...     capital=10000.0,
    ... )
    >>> 
    >>> result = run_trading_cycle(ctx)
    >>> print(f"Generated {len(result.orders)} orders")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

# Import existing modules (no duplication)
from src.assembled_core.config.models import (
    FeatureConfig,
    ensure_feature_config,
)

if TYPE_CHECKING:
    from src.assembled_core.config.models import RiskConfig, SignalConfig
from src.assembled_core.data.factor_store import compute_universe_key
from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.execution.position_alignment import align_current_and_target
from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.features.ta_features import (
    add_all_features,
)

logger = logging.getLogger(__name__)


@dataclass
class TradingContext:
    """Unified context for trading cycle execution.
    
    This context contains all configuration and data needed for executing
    a single trading cycle iteration (one day/timestamp in EOD, one rebalance
    in backtest, one day in paper track).
    
    Attributes:
        prices: DataFrame with columns: timestamp, symbol, close, ... (OHLCV)
            Input price data. Must be sorted by symbol, then timestamp.
        as_of: pd.Timestamp | None
            Point-in-time cutoff (PIT-safe filtering). If None, no filtering is applied.
        freq: str
            Trading frequency ("1d" or "5min") for context (default: "1d")
        universe: list[str] | None
            Universe symbols for validation (optional). If provided, prices will
            be filtered to only include symbols in universe.
            
        # Feature building
        use_factor_store: bool
            Enable factor store caching (default: False)
        factor_store_root: Path | None
            Factor store root directory (default: None)
        factor_group: str
            Factor group name for factor store (default: "core_ta")
        feature_config: dict[str, Any] | None
            Feature building configuration (e.g., ma_windows, atr_window, rsi_window)
            (default: None, uses defaults from add_all_features)
            
        # Signal generation
        signal_fn: Callable[[pd.DataFrame], pd.DataFrame]
            Signal function that takes prices DataFrame and returns signals DataFrame.
            Input: DataFrame with columns: timestamp, symbol, close, ... (features if built)
            Output: DataFrame with columns: timestamp, symbol, direction, score
        signal_config: dict[str, Any] | SignalConfig
            Signal-specific configuration (e.g., ma_fast, ma_slow) (default: {})
            
        # Position sizing
        position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame]
            Position sizing function that takes signals DataFrame and capital,
            returns target positions DataFrame.
            Input: (signals_df: pd.DataFrame, total_capital: float)
            Output: DataFrame with columns: symbol, target_weight, target_qty
        capital: float
            Total capital for position sizing (default: 10000.0)
            
        # Order generation
        current_positions: pd.DataFrame | None
            Current portfolio positions (columns: symbol, qty) (default: None)
            If None, assumes empty portfolio (all positions are new)
        order_timestamp: pd.Timestamp
            Timestamp for generated orders (default: current UTC timestamp)
            
    # Risk controls
    enable_risk_controls: bool
        Enable risk controls (pre-trade checks, kill switch) (default: True)
    risk_config: dict[str, Any] | RiskConfig
        Risk control configuration (default: {})
            
        # Outputs
        output_dir: Path
            Output directory for writing outputs (default: Path("output"))
        output_format: Literal["safe_csv", "equity_curve", "state", "none"]
            Output format type (default: "safe_csv")
        write_outputs: bool
            Whether to write output files (default: True)
            
        # Metadata
        run_id: str | None
            Run identifier for logging/tracking (default: None)
        strategy_name: str | None
            Strategy name for metadata (default: None)
        logger: logging.Logger | None
            Logger instance (default: None, uses module logger)
        timings: dict[str, Any] | None
            Timing dictionary for step timing (default: None)
    """
    
    # Input data
    prices: pd.DataFrame
    as_of: pd.Timestamp | None = None
    freq: str = "1d"
    universe: list[str] | None = None
    mode: Literal["eod", "backtest", "paper", "live"] = "eod"
    """Trading cycle mode.
    
    - "eod": EOD mode - filters to last row per symbol <= as_of (default, backward compatible)
    - "backtest": Backtest mode - keeps full history slice <= as_of for MAs/returns, plus latest row for orders
    - "paper": Paper trading mode - same as eod
    - "live": Live trading mode - same as eod
    """
    
    # Feature building
    use_factor_store: bool = False
    factor_store_root: Path | None = None
    factor_group: str = "core_ta"
    feature_config: dict[str, Any] | FeatureConfig | None = None
    precomputed_prices_with_features: pd.DataFrame | None = None
    """Precomputed prices with features (optional).
    
    If provided and mode=="backtest", this panel will be used instead of
    computing features per timestamp. The panel will be sliced PIT-safely
    (<= as_of) for each timestamp.
    
    This enables performance optimization in backtest mode where features
    are computed once upfront instead of per timestamp.
    """
    precomputed_panel_index: Any | None = None
    """Precomputed panel index for efficient snapshot extraction (optional).
    
    If provided and mode=="backtest" and precomputed_prices_with_features is set,
    this index will be used for O(S log N) snapshot extraction instead of
    O(N log N) groupby operations.
    
    Type: PrecomputedPanelIndex from src.assembled_core.pipeline.precomputed_index
    """
    backtest_use_snapshot: bool = True
    """Backtest snapshot mode (performance optimization).
    
    If True and mode=="backtest" and precomputed_prices_with_features is set,
    uses only a snapshot (latest row per symbol <= as_of) instead of full
    history slice. This avoids expensive slicing operations in long backtests.
    
    If False, uses full history slice (original behavior for strategies that
    need history for MAs/returns computation).
    """
    
    # Signal generation
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    signal_config: dict[str, Any] | SignalConfig = field(default_factory=dict)
    
    # Position sizing
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame] | None = None
    capital: float = 10000.0
    
    # Order generation
    current_positions: pd.DataFrame | None = None
    order_timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    
    # Risk controls
    enable_risk_controls: bool = True
    risk_config: dict[str, Any] | RiskConfig = field(default_factory=dict)
    security_meta_df: pd.DataFrame | None = None
    """Security metadata DataFrame (symbol -> sector/region/currency/asset_type).
    
    Required for sector/region/FX exposure limits. If None and limits are enabled,
    risk controls will skip group exposure checks.
    """
    
    # QA Gate (Sprint 3 / D2)
    qa_block_trading: bool = False
    qa_block_reason: str | None = None
    
    # Outputs
    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_format: Literal["safe_csv", "equity_curve", "state", "none"] = "safe_csv"
    write_outputs: bool = True
    
    # Metadata
    run_id: str | None = None
    strategy_name: str | None = None
    logger: logging.Logger | None = None
    timings: dict[str, Any] | None = None


@dataclass
class TradingCycleResult:
    """Result of unified trading cycle execution.
    
    This result contains all intermediate outputs from the trading cycle
    execution, allowing callers to inspect or use intermediate results.
    
    Attributes:
        prices_filtered: pd.DataFrame
            Prices after filtering (as_of, universe). Same schema as input prices.
            In "eod" mode: last row per symbol <= as_of.
            In "backtest" mode: full history slice <= as_of (for MAs/returns).
        prices_latest: pd.DataFrame | None
            Latest prices per symbol (one row per symbol) extracted from prices_filtered.
            Only populated in "backtest" mode (for order generation with latest prices).
            Columns: same as prices_filtered, but only latest timestamp per symbol.
        prices_with_features: pd.DataFrame
            Prices with computed features added. Contains all input columns
            plus feature columns (e.g., ma_20, ma_50, atr_14, rsi_14, etc.)
        signals: pd.DataFrame
            Generated signals (columns: timestamp, symbol, direction, score)
        target_positions: pd.DataFrame
            Target positions (columns: symbol, target_weight, target_qty)
        orders: pd.DataFrame
            Generated orders (columns: timestamp, symbol, side, qty, price)
        orders_filtered: pd.DataFrame
            Orders after risk controls applied (same schema as orders)
            
        # Metadata
        run_id: str | None
            Run identifier (from context)
        timestamp: pd.Timestamp
            Execution timestamp
        status: Literal["success", "error"]
            Execution status
        error_message: str | None
            Error message if status == "error" (None otherwise)
        meta: dict[str, Any]
            Additional metadata (e.g., feature cache status, risk control results)
        output_paths: dict[str, Path]
            Dictionary of output file paths (e.g., {"safe_csv": Path(...)})
            Keys depend on output_format
    """
    
    # Intermediate results
    prices_filtered: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    prices_latest: pd.DataFrame | None = None
    prices_with_features: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    signals: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    target_positions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    orders: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    orders_filtered: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    
    # Metadata
    run_id: str | None = None
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    status: Literal["success", "error"] = "success"
    error_message: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, Path] = field(default_factory=dict)


def _filter_prices_for_as_of(
    prices: pd.DataFrame,
    as_of: pd.Timestamp | None,
    universe: list[str] | None = None,
    mode: Literal["eod", "backtest", "paper", "live"] = "eod",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Filter prices based on mode: last row (eod) or full history slice (backtest).
    
    This is a PIT-safe filtering function that ensures no future data leaks into the cycle.
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, close, ...
        as_of: Maximum allowed timestamp (pd.Timestamp, UTC). If None, no time filtering.
        universe: Optional list of symbols to filter by. If None, all symbols are included.
        mode: Trading cycle mode:
            - "eod": Returns last row per symbol <= as_of (default, backward compatible)
            - "backtest": Returns full history slice <= as_of (for MAs/returns)
            - "paper": Same as "eod"
            - "live": Same as "eod"
        
    Returns:
        Tuple of (prices_filtered, prices_latest):
        - prices_filtered: Filtered DataFrame
          - In "eod" mode: one row per symbol (last available <= as_of)
          - In "backtest" mode: full history slice <= as_of (multiple rows per symbol)
        - prices_latest: Latest prices per symbol (one row per symbol)
          - In "eod" mode: None (same as prices_filtered)
          - In "backtest" mode: last row per symbol from prices_filtered
    """
    if prices.empty:
        return prices, None
    
    # Ensure timestamp is timezone-aware UTC
    if prices["timestamp"].dt.tz is None:
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    
    # Filter to dates <= as_of if as_of is provided
    if as_of is not None:
        filtered = prices[prices["timestamp"] <= as_of].copy()
    else:
        filtered = prices.copy()
    
    if filtered.empty:
        return pd.DataFrame(columns=prices.columns), None
    
    # Filter by universe if provided
    if universe is not None:
        universe_upper = [s.upper().strip() for s in universe]
        filtered = filtered[
            filtered["symbol"].str.upper().isin(universe_upper)
        ].copy()
    
    # Determine if we need history slice or just latest
    if mode == "backtest":
        # Backtest mode: keep full history slice for MAs/returns
        # Also extract latest prices per symbol for order generation
        prices_latest = (
            filtered.groupby("symbol", group_keys=False, dropna=False)
            .last()
            .reset_index()
        )
        # Ensure deterministic sorting (timestamp, symbol)
        filtered = filtered.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        prices_latest = prices_latest.sort_values("symbol").reset_index(drop=True)
        return filtered, prices_latest
    else:
        # EOD/Paper/Live mode: return last row per symbol (backward compatible)
        filtered = filtered.groupby("symbol", group_keys=False, dropna=False).last()
        filtered = filtered.reset_index()  # Keep 'symbol' as column
        # Ensure deterministic sorting
        filtered = filtered.sort_values("symbol").reset_index(drop=True)
        return filtered, None


def _build_features_default(
    ctx: TradingContext,
    prices_filtered: pd.DataFrame,
) -> pd.DataFrame:
    """Default feature building implementation using existing modules.
    
    Args:
        ctx: TradingContext with feature configuration
        prices_filtered: Filtered prices DataFrame
        
    Returns:
        DataFrame with features added
    """
    if ctx.use_factor_store:
        # Use factor store (build_or_load_factors)
        log = ctx.logger if ctx.logger is not None else logger
        log.debug(f"Using factor store: group={ctx.factor_group}, root={ctx.factor_store_root}")
        
        # Compute universe key for metadata
        universe_symbols = sorted(prices_filtered["symbol"].unique().tolist())
        universe_key = compute_universe_key(symbols=universe_symbols)
        
        # Determine date range for PIT-safe loading
        start_date = prices_filtered["timestamp"].min() if not prices_filtered.empty else None
        end_date = prices_filtered["timestamp"].max() if not prices_filtered.empty else None
        
        # Get feature config (validate and convert to dict for backward compatibility)
        feature_cfg = ensure_feature_config(ctx.feature_config)
        config: dict[str, Any] = {}
        if feature_cfg is not None:
            config = {
                "ma_windows": feature_cfg.ma_windows,
                "atr_window": feature_cfg.atr_window,
                "rsi_window": feature_cfg.rsi_window,
                "include_rsi": feature_cfg.include_rsi,
            }
        has_ohlc = all(col in prices_filtered.columns for col in ["high", "low", "open"])
        
        # Build or load factors
        prices_with_features = build_or_load_factors(
            prices=prices_filtered,
            factor_group=ctx.factor_group,
            freq=ctx.freq,
            universe_key=universe_key,
            start_date=start_date,
            end_date=end_date,
            as_of=ctx.as_of,  # PIT-safe: use as_of as cutoff
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
            factors_root=ctx.factor_store_root,
        )
    else:
        # Default: direct computation (backward compatible)
        feature_cfg = ensure_feature_config(ctx.feature_config)
        config: dict[str, Any] = {}
        if feature_cfg is not None:
            config = {
                "ma_windows": feature_cfg.ma_windows,
                "atr_window": feature_cfg.atr_window,
                "rsi_window": feature_cfg.rsi_window,
                "include_rsi": feature_cfg.include_rsi,
            }
        has_ohlc = all(col in prices_filtered.columns for col in ["high", "low", "open"])
        
        if has_ohlc:
            prices_with_features = add_all_features(
                prices_filtered,
                ma_windows=config.get("ma_windows", (20, 50, 200)),
                atr_window=config.get("atr_window", 14),
                rsi_window=config.get("rsi_window", 14),
                include_rsi=config.get("include_rsi", True),
            )
        else:
            # If OHLC not available, only compute features that don't need them
            from src.assembled_core.features.ta_features import add_log_returns, add_moving_averages
            prices_with_features = add_log_returns(prices_filtered.copy())
            prices_with_features = add_moving_averages(
                prices_with_features,
                windows=config.get("ma_windows", (20, 50, 200)),
            )
    
    return prices_with_features


def _generate_orders_default(
    ctx: TradingContext,
    target_positions: pd.DataFrame,
) -> pd.DataFrame:
    """Default order generation implementation using existing module.
    
    This function aligns current and target positions to ensure deterministic
    symbol ordering, enabling the fast-path order generation to trigger more often.
    
    Args:
        ctx: TradingContext with order generation configuration
        target_positions: Target positions DataFrame
        
    Returns:
        Orders DataFrame
    """
    if target_positions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    # Prepare target positions for alignment
    # Extract symbol and target_qty columns (handle both "target_qty" and "qty" column names)
    if "target_qty" in target_positions.columns:
        target_for_alignment = target_positions[["symbol", "target_qty"]].copy()
        target_for_alignment = target_for_alignment.rename(columns={"target_qty": "qty"})  # Rename to "qty" for alignment
    elif "qty" in target_positions.columns:
        target_for_alignment = target_positions[["symbol", "qty"]].copy()
    else:
        # Fallback: create empty target_for_alignment
        target_for_alignment = pd.DataFrame(columns=["symbol", "qty"])
    
    # Prepare current positions for alignment
    if ctx.current_positions is not None and not ctx.current_positions.empty:
        if "qty" not in ctx.current_positions.columns:
            # If qty column missing, create it with 0
            current_for_alignment = ctx.current_positions[["symbol"]].copy()
            current_for_alignment["qty"] = 0.0
        else:
            current_for_alignment = ctx.current_positions[["symbol", "qty"]].copy()
    else:
        current_for_alignment = pd.DataFrame(columns=["symbol", "qty"])
    
    # Align positions (same symbol set, same order, missing = 0)
    current_aligned, target_aligned = align_current_and_target(
        current_positions=current_for_alignment,
        target_positions=target_for_alignment,
        symbol_col="symbol",
        qty_col="qty",
    )
    
    # Rename target qty column back to "target_qty" (alignment function uses "qty")
    target_aligned = target_aligned.rename(columns={"qty": "target_qty"})
    
    # Now generate orders with aligned positions (fast-path should trigger)
    orders = generate_orders_from_targets(
        target_positions=target_aligned,
        current_positions=current_aligned,
        timestamp=ctx.order_timestamp,
        prices=None,  # Prices will be added later if needed (via hook or post-processing)
    )
    
    return orders


def _apply_risk_controls_default(
    ctx: TradingContext,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Default risk controls implementation using existing module.
    
    Args:
        ctx: TradingContext with risk control configuration
        orders: Orders DataFrame
        
    Returns:
        Filtered orders DataFrame
    """
    if orders.empty or not ctx.enable_risk_controls:
        return orders.copy()
    
    try:
        # Prepare current positions for risk controls
        # Convert current_positions to expected format (symbol, qty)
        current_positions_df = None
        if ctx.current_positions is not None and not ctx.current_positions.empty:
            if "qty" in ctx.current_positions.columns:
                current_positions_df = ctx.current_positions[["symbol", "qty"]].copy()
            elif "target_qty" in ctx.current_positions.columns:
                current_positions_df = ctx.current_positions[["symbol", "target_qty"]].rename(
                    columns={"target_qty": "qty"}
                )

        # Prepare prices_latest (latest price per symbol)
        prices_latest_df = None
        if ctx.prices is not None and not ctx.prices.empty:
            # Get latest price per symbol (for exposure calculation)
            if "close" in ctx.prices.columns:
                prices_latest_df = (
                    ctx.prices.groupby("symbol")["close"]
                    .last()
                    .reset_index()
                    .rename(columns={"close": "price"})
                )
            elif "price" in ctx.prices.columns:
                prices_latest_df = (
                    ctx.prices.groupby("symbol")["price"]
                    .last()
                    .reset_index()
                )

        # Compute equity (cash + mark-to-market positions)
        equity = ctx.capital  # Use capital as equity proxy (can be refined later)
        
        # Get current_equity and peak_equity if available (for drawdown de-risking)
        current_equity = getattr(ctx, "current_equity", None)
        peak_equity = getattr(ctx, "peak_equity", None)
        
        # Get security_meta_df from context (for sector/region/FX limits)
        security_meta_df = ctx.security_meta_df

        # Convert risk_config dict to PreTradeConfig
        from src.assembled_core.execution.pre_trade_checks import PreTradeConfig
        
        pre_trade_config = None
        if ctx.risk_config:
            # Extract PreTradeConfig fields from risk_config dict
            if isinstance(ctx.risk_config, dict):
                pre_trade_config = PreTradeConfig(
                    max_notional_per_symbol=ctx.risk_config.get("max_notional_per_symbol"),
                    max_weight_per_symbol=ctx.risk_config.get("max_weight_per_symbol"),
                    turnover_cap=ctx.risk_config.get("turnover_cap"),
                    drawdown_threshold=ctx.risk_config.get("drawdown_threshold"),
                    de_risk_scale=ctx.risk_config.get("de_risk_scale", 0.0),
                    max_gross_exposure=ctx.risk_config.get("max_gross_exposure"),
                    max_sector_exposure=ctx.risk_config.get("max_sector_exposure"),
                    max_region_exposure=ctx.risk_config.get("max_region_exposure"),
                    max_fx_exposure=ctx.risk_config.get("max_fx_exposure"),
                    base_currency=ctx.risk_config.get("base_currency", "USD"),
                    missing_security_meta=ctx.risk_config.get("missing_security_meta", "raise"),
                )
            elif hasattr(ctx.risk_config, "__dict__"):
                # If it's already a PreTradeConfig or similar object, try to extract fields
                pre_trade_config = PreTradeConfig(
                    max_notional_per_symbol=getattr(ctx.risk_config, "max_notional_per_symbol", None),
                    max_weight_per_symbol=getattr(ctx.risk_config, "max_weight_per_symbol", None),
                    turnover_cap=getattr(ctx.risk_config, "turnover_cap", None),
                    drawdown_threshold=getattr(ctx.risk_config, "drawdown_threshold", None),
                    de_risk_scale=getattr(ctx.risk_config, "de_risk_scale", 0.0),
                    max_gross_exposure=getattr(ctx.risk_config, "max_gross_exposure", None),
                    max_sector_exposure=getattr(ctx.risk_config, "max_sector_exposure", None),
                    max_region_exposure=getattr(ctx.risk_config, "max_region_exposure", None),
                    max_fx_exposure=getattr(ctx.risk_config, "max_fx_exposure", None),
                    base_currency=getattr(ctx.risk_config, "base_currency", "USD"),
                    missing_security_meta=getattr(ctx.risk_config, "missing_security_meta", "raise"),
                )

        # Use existing risk controls module with exposure data
        filtered_orders, risk_result = filter_orders_with_risk_controls(
            orders=orders,
            portfolio=None,  # Portfolio snapshot not available in cycle context
            qa_status=None,  # QA status not available in cycle context
            pre_trade_config=pre_trade_config,
            enable_pre_trade_checks=ctx.enable_risk_controls,
            enable_kill_switch=ctx.enable_risk_controls,
            current_positions=current_positions_df,
            prices_latest=prices_latest_df,
            equity=equity,
            current_equity=current_equity,
            peak_equity=peak_equity,
            security_meta_df=security_meta_df,
        )
        
        return filtered_orders
    except Exception as e:
        # If risk controls fail, log warning and pass through orders
        log = ctx.logger if ctx.logger is not None else logger
        log.warning(f"Risk controls failed: {e}. Passing through orders without filtering.")
        return orders.copy()


def run_trading_cycle(
    ctx: TradingContext,
    *,
    hooks: dict[str, Callable] | None = None,
) -> TradingCycleResult:
    """Execute unified trading cycle.
    
    This function orchestrates the common trading cycle steps using hook points
    for each step. The default implementation is a skeleton that validates inputs
    and provides clear hook points for integration.
    
    Steps (hook points):
    1. `load_prices`: Filter prices (as_of, universe validation)
    2. `build_features`: Build features (TA features, factor store integration)
    3. `generate_signals`: Generate signals (via signal_fn)
    4. `size_positions`: Compute target positions (via position_sizing_fn)
    5. `generate_orders`: Generate orders (current_positions vs. target_positions)
    6. `risk_controls`: Apply risk controls (pre-trade checks, kill switch)
    7. `write_outputs`: Write outputs (SAFE-CSV, equity curve, state, etc.)
    
    Args:
        ctx: TradingContext with all configuration and data
        hooks: Optional dictionary of hook functions to override default behavior.
               Keys: "load_prices", "build_features", "generate_signals",
                     "size_positions", "generate_orders", "risk_controls", "write_outputs"
               Hook function signatures:
               - load_prices(ctx) -> pd.DataFrame
               - build_features(ctx, prices_filtered) -> pd.DataFrame
               - generate_signals(ctx, prices_with_features) -> pd.DataFrame
               - size_positions(ctx, signals) -> pd.DataFrame
               - generate_orders(ctx, target_positions) -> pd.DataFrame
               - risk_controls(ctx, orders) -> pd.DataFrame
               - write_outputs(ctx, orders_filtered) -> dict[str, Path]
               
    Returns:
        TradingCycleResult with intermediate results and outputs
        
    Raises:
        ValueError: If required context fields are missing or invalid
        
    Note:
        This implementation uses existing modules for all steps (no duplication):
        - Price filtering: PIT-safe filtering via as_of and universe
        - Feature building: add_all_features or build_or_load_factors
        - Signal generation: via signal_fn (caller provides)
        - Position sizing: via position_sizing_fn (caller provides)
        - Order generation: generate_orders_from_targets
        - Risk controls: filter_orders_with_risk_controls
        - Outputs: No default implementation (pure function, no file writes)
        
        Hook points allow callers to override default behavior or integrate
        with existing workflows. Default implementations ensure deterministic
        behavior while maintaining flexibility.
    """
    # Use context logger or module logger
    log = ctx.logger if ctx.logger is not None else logger
    
    # Initialize result
    result = TradingCycleResult(
        run_id=ctx.run_id,
        timestamp=pd.Timestamp.utcnow(),
        status="success",
    )
    
    # Validate required fields
    if ctx.prices is None or ctx.prices.empty:
        result.status = "error"
        result.error_message = "prices DataFrame is None or empty"
        return result
    
    required_price_cols = ["timestamp", "symbol", "close"]
    missing_cols = [c for c in required_price_cols if c not in ctx.prices.columns]
    if missing_cols:
        result.status = "error"
        result.error_message = f"Missing required price columns: {', '.join(missing_cols)}"
        return result
    
    if ctx.signal_fn is None:
        result.status = "error"
        result.error_message = "signal_fn is required but not provided"
        return result
    
    if ctx.position_sizing_fn is None:
        result.status = "error"
        result.error_message = "position_sizing_fn is required but not provided"
        return result
    
    # Initialize hooks dict if not provided
    hooks = hooks or {}
    
    # Step 1: Load/Filter prices (hook point: load_prices)
    try:
        if "load_prices" in hooks:
            load_result = hooks["load_prices"](ctx)
            # Handle both tuple (filtered, latest) and single DataFrame (backward compat)
            if isinstance(load_result, tuple):
                result.prices_filtered, result.prices_latest = load_result
            else:
                result.prices_filtered = load_result
                result.prices_latest = None
        else:
            # Default: filter prices by as_of and universe (PIT-safe)
            result.prices_filtered, result.prices_latest = _filter_prices_for_as_of(
                prices=ctx.prices,
                as_of=ctx.as_of,
                universe=ctx.universe,
                mode=ctx.mode,
            )
        
        if result.prices_filtered.empty:
            result.status = "error"
            result.error_message = "No prices remaining after filtering (as_of or universe)"
            return result
        
        log.debug(
            f"Prices filtered: {len(result.prices_filtered)} rows, "
            f"{result.prices_filtered['symbol'].nunique()} symbols "
            f"(mode={ctx.mode}, latest={'yes' if result.prices_latest is not None else 'no'})"
        )
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in load_prices: {e}"
        return result
    
    # Step 2: Build features (hook point: build_features)
    try:
        if "build_features" in hooks:
            result.prices_with_features = hooks["build_features"](ctx, result.prices_filtered)
        elif ctx.mode == "backtest" and ctx.precomputed_prices_with_features is not None and not ctx.precomputed_prices_with_features.empty:
            # Backtest mode: use precomputed feature panel (PIT-safe slice)
            precomputed = ctx.precomputed_prices_with_features.copy()
            
            # Ensure timestamp column is UTC-aware for comparison
            if precomputed["timestamp"].dtype.tz is None:
                precomputed["timestamp"] = pd.to_datetime(precomputed["timestamp"], utc=True)
            elif precomputed["timestamp"].dtype.tz != pd.Timestamp.utcnow().tz:
                # Ensure UTC timezone
                precomputed["timestamp"] = precomputed["timestamp"].dt.tz_convert("UTC")
            
            if ctx.backtest_use_snapshot:
                # Snapshot mode: only use latest row per symbol <= as_of (performance optimization)
                # Use precomputed index if available (O(S log N) instead of O(N log N))
                if ctx.precomputed_panel_index is not None and ctx.as_of is not None:
                    # Use optimized index-based snapshot extraction
                    from src.assembled_core.pipeline.precomputed_index import snapshot_as_of
                    result.prices_latest = snapshot_as_of(
                        df=precomputed,
                        index=ctx.precomputed_panel_index,
                        as_of=ctx.as_of,
                        use_monotonic_optimization=True,
                    )
                else:
                    # Fallback to groupby-based extraction (if index not available)
                    if ctx.as_of is not None:
                        # PIT-safe filter: only rows <= as_of
                        precomputed_filtered = precomputed[
                            precomputed["timestamp"] <= ctx.as_of
                        ].copy()
                    else:
                        precomputed_filtered = precomputed.copy()
                    
                    # Extract snapshot (latest row per symbol)
                    result.prices_latest = (
                        precomputed_filtered.groupby("symbol", group_keys=False, dropna=False)
                        .last()
                        .reset_index()
                        .sort_values("symbol")
                        .reset_index(drop=True)
                    )
                
                # Set prices_with_features to snapshot (not full history)
                result.prices_with_features = result.prices_latest.copy()
                
                # Set prices_filtered to minimal (just snapshot) to avoid downstream confusion
                result.prices_filtered = result.prices_latest.copy()
                
                log.debug(
                    f"Using precomputed features (snapshot mode): {len(result.prices_with_features)} rows "
                    f"(latest per symbol <= {ctx.as_of if ctx.as_of else 'no cutoff'}, "
                    f"index={'yes' if ctx.precomputed_panel_index is not None else 'no'})"
                )
            else:
                # History-slice mode: use full history slice (original behavior)
                if ctx.as_of is not None:
                    # Slice to only rows <= as_of (PIT-safe)
                    result.prices_with_features = precomputed[
                        precomputed["timestamp"] <= ctx.as_of
                    ].copy()
                else:
                    # No as_of: use all precomputed data
                    result.prices_with_features = precomputed.copy()
                
                # Extract prices_latest from the sliced panel (for order generation)
                if not result.prices_with_features.empty and result.prices_latest is None:
                    result.prices_latest = (
                        result.prices_with_features.groupby("symbol", group_keys=False, dropna=False)
                        .last()
                        .reset_index()
                        .sort_values("symbol")
                        .reset_index(drop=True)
                    )
                
                log.debug(
                    f"Using precomputed features (history-slice mode): {len(result.prices_with_features)} rows "
                    f"(sliced to <= {ctx.as_of if ctx.as_of else 'no cutoff'})"
                )
        else:
            # Default: use existing feature building modules
            result.prices_with_features = _build_features_default(ctx, result.prices_filtered)
        
        log.debug(f"Features: {len(result.prices_with_features.columns)} columns (was {len(result.prices_filtered.columns)})")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in build_features: {e}"
        return result
    
    # Step 3: Generate signals (hook point: generate_signals)
    try:
        if "generate_signals" in hooks:
            result.signals = hooks["generate_signals"](ctx, result.prices_with_features)
        else:
            # Default: call signal_fn
            result.signals = ctx.signal_fn(result.prices_with_features)
        
        # Validate signals format
        required_signal_cols = ["timestamp", "symbol", "direction"]
        missing_signal_cols = [c for c in required_signal_cols if c not in result.signals.columns]
        if missing_signal_cols:
            result.status = "error"
            result.error_message = f"signals missing required columns: {', '.join(missing_signal_cols)}"
            return result
        
        log.debug(f"Signals generated: {len(result.signals)} rows")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in generate_signals hook: {e}"
        return result
    
    # Step 4: Size positions (hook point: size_positions)
    try:
        if "size_positions" in hooks:
            result.target_positions = hooks["size_positions"](ctx, result.signals)
        else:
            # Default: call position_sizing_fn
            result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
        
        # Validate target_positions format
        required_target_cols = ["symbol", "target_weight", "target_qty"]
        missing_target_cols = [c for c in required_target_cols if c not in result.target_positions.columns]
        if missing_target_cols:
            # Allow missing target_weight or target_qty (at least one should be present)
            if not any(c in result.target_positions.columns for c in ["target_weight", "target_qty"]):
                result.status = "error"
                result.error_message = "target_positions missing required columns: symbol and (target_weight or target_qty)"
                return result
        
        log.debug(f"Target positions computed: {len(result.target_positions)} symbols")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in size_positions hook: {e}"
        return result
    
    # Step 5: Generate orders (hook point: generate_orders)
    try:
        if "generate_orders" in hooks:
            result.orders = hooks["generate_orders"](ctx, result.target_positions)
        else:
            # Default: use existing order generation module
            # Note: We need to add prices to orders after generation
            # For now, generate_orders_from_targets will use 0.0 if prices not provided
            # Prices can be added via hook or post-processing
            result.orders = _generate_orders_default(ctx, result.target_positions)
            
            # Add prices from prices_with_features if available (for symbols in orders)
            if not result.orders.empty and not result.prices_with_features.empty:
                # Use prices_latest if available (backtest mode), otherwise extract from prices_with_features
                if result.prices_latest is not None and "close" in result.prices_latest.columns:
                    # Backtest mode: use pre-extracted latest prices
                    latest_prices = result.prices_latest[["symbol", "close"]].rename(columns={"close": "price"})
                elif "close" in result.prices_with_features.columns:
                    # EOD mode: extract latest prices from prices_with_features
                    latest_prices = (
                        result.prices_with_features
                        .groupby("symbol", group_keys=False)["close"]
                        .last()
                        .reset_index()
                        .rename(columns={"close": "price"})
                    )
                else:
                    latest_prices = None
                
                if latest_prices is not None:
                    # Merge prices into orders
                    result.orders = result.orders.merge(
                        latest_prices,
                        on="symbol",
                        how="left",
                        suffixes=("", "_latest"),
                    )
                    
                    # Use latest price if order price is 0.0 or missing
                    if "price_latest" in result.orders.columns:
                        result.orders["price"] = result.orders["price_latest"].fillna(result.orders["price"])
                        result.orders = result.orders.drop(columns=["price_latest"])
        
        log.debug(f"Orders generated: {len(result.orders)} orders")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in generate_orders: {e}"
        return result
    
    # QA Gate: Block orders if qa_block_trading is True (Sprint 3 / D2)
    if ctx.qa_block_trading:
        log.warning(f"QA Gate: Trading blocked - {ctx.qa_block_reason or 'No reason provided'}")
        # Set orders to empty DataFrame with correct schema
        result.orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        result.meta["qa_block_reason"] = ctx.qa_block_reason
        result.meta["qa_block_trading"] = True
        log.info("QA Gate: Orders set to empty (trading blocked)")
    
    # Step 6: Apply risk controls (hook point: risk_controls)
    try:
        if "risk_controls" in hooks:
            result.orders_filtered = hooks["risk_controls"](ctx, result.orders)
        else:
            # Default: use existing risk controls module
            result.orders_filtered = _apply_risk_controls_default(ctx, result.orders)
        
        if len(result.orders_filtered) < len(result.orders):
            log.info(f"Risk controls filtered orders: {len(result.orders)} -> {len(result.orders_filtered)} ({len(result.orders) - len(result.orders_filtered)} blocked)")
        
        log.debug(f"Orders after risk controls: {len(result.orders_filtered)} orders")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in risk_controls: {e}"
        return result
    
    # Step 7: Write outputs (hook point: write_outputs)
    try:
        if ctx.write_outputs:
            if "write_outputs" in hooks:
                result.output_paths = hooks["write_outputs"](ctx, result.orders_filtered)
            else:
                # Default: no outputs written
                # TODO: Implement default output writing logic in B1.2
                result.output_paths = {}
        
        log.debug(f"Outputs written: {len(result.output_paths)} files")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in write_outputs hook: {e}"
        return result
    
    log.info(f"Trading cycle completed successfully: {len(result.orders_filtered)} orders")
    
    return result

