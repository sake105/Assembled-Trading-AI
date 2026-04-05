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
import os
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
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.config.config import get_base_dir
from src.assembled_core.config.settings import get_settings

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
from src.assembled_core.risk.correlation_guard import apply_correlation_guard
from src.assembled_core.risk.georisk_overlay import (
    apply_exposure_multiplier_to_targets,
    compute_exposure_multiplier,
)
from src.assembled_core.risk.zombie_killer import get_zombie_positions
from src.assembled_core.risk.state_machine import (
    compute_next_state,
    load_risk_state,
    save_risk_state,
)
from src.assembled_core.risk.market_stress import compute_market_stress
from src.assembled_core.risk.profit_lock import compute_profit_lock_multiplier
from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result
from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
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
    order_timestamp: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp.now("UTC")
    )

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

    # Risk state machine (INT-4): WATCH / ACTIVE / COOLDOWN / PAUSE (read-only after run start)
    risk_state: dict[str, Any] | None = None

    # Market stress (INT-5): price-based stress_ok / stress_score for state machine
    market_stress: dict[str, Any] | None = None

    # Profit lock (INT-6.2): optional equity curve and current index for overlay
    equity_curve: pd.Series | None = None
    equity_curve_index: int | None = None
    profit_lock_state: dict[str, Any] | None = None

    # Intel (read-only): disclosures triggers snapshot; QC flags for degraded/missing intel
    disclosures_triggers: Any | None = None  # DisclosuresTriggerSnapshot | None
    intel_health_flags: dict[str, str] = field(default_factory=dict)

    # GeoRisk intel (read from data/intel/crisis_state.json + triggers_latest.json)
    news_geo: dict[str, Any] | None = None  # {"geo_score": int, "geo_confidence": float, "state_hint": str, ...}
    crisis_state_intel: dict[str, Any] | None = None  # full crisis state from intel cycle
    intel_sim_applied: bool = (
        False  # BENCH-1: when True, skip intel loading (paper_runner sets simulated intel)
    )

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
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now("UTC"))
    status: Literal["success", "error"] = "success"
    error_message: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, Path] = field(default_factory=dict)


def _estimate_symbol_volatilities(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> dict[str, float]:
    """Estimate annualized volatility per symbol from price data.

    Uses daily log returns over the lookback window.
    Returns dict of symbol -> annualized vol. Defaults to 0.20 for missing symbols.
    """
    import numpy as np

    if prices is None or prices.empty:
        return {}
    if "close" not in prices.columns or "symbol" not in prices.columns:
        return {}

    vols: dict[str, float] = {}
    for sym, grp in prices.groupby("symbol"):
        close = grp["close"].dropna()
        if len(close) < 5:
            vols[str(sym)] = 0.20
            continue
        close = close.iloc[-lookback:] if len(close) > lookback else close
        log_rets = np.log(close / close.shift(1)).dropna()
        if len(log_rets) < 3:
            vols[str(sym)] = 0.20
            continue
        daily_vol = float(log_rets.std())
        ann_vol = daily_vol * np.sqrt(252)
        vols[str(sym)] = max(ann_vol, 0.01)  # floor at 1%
    return vols


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
        filtered = filtered[filtered["symbol"].str.upper().isin(universe_upper)].copy()

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
        log.debug(
            f"Using factor store: group={ctx.factor_group}, root={ctx.factor_store_root}"
        )

        # Compute universe key for metadata
        universe_symbols = sorted(prices_filtered["symbol"].unique().tolist())
        universe_key = compute_universe_key(symbols=universe_symbols)

        # Determine date range for PIT-safe loading
        start_date = (
            prices_filtered["timestamp"].min() if not prices_filtered.empty else None
        )
        end_date = (
            prices_filtered["timestamp"].max() if not prices_filtered.empty else None
        )

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
        has_ohlc = all(
            col in prices_filtered.columns for col in ["high", "low", "open"]
        )

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
        has_ohlc = all(
            col in prices_filtered.columns for col in ["high", "low", "open"]
        )

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
            from src.assembled_core.features.ta_features import (
                add_log_returns,
                add_moving_averages,
            )

            prices_with_features = add_log_returns(prices_filtered.copy())
            prices_with_features = add_moving_averages(
                prices_with_features,
                windows=config.get("ma_windows", (20, 50, 200)),
            )

    # ---------------------------------------------------------------
    # D5: Intermarket factors (optional)
    # ---------------------------------------------------------------
    feature_cfg_obj = ensure_feature_config(ctx.feature_config)
    if feature_cfg_obj is not None and getattr(feature_cfg_obj, "include_intermarket", False):
        try:
            from src.assembled_core.features.intermarket_factors import (
                align_intermarket_factors_to_panel,
                build_intermarket_factors,
            )
            ts_min = prices_with_features["timestamp"].min()
            ts_max = prices_with_features["timestamp"].max()
            start_str = pd.Timestamp(ts_min).strftime("%Y-%m-%d")
            end_str = pd.Timestamp(ts_max).strftime("%Y-%m-%d")
            im_factors = build_intermarket_factors(start_date=start_str, end_date=end_str)
            if not im_factors.empty:
                prices_with_features = align_intermarket_factors_to_panel(
                    prices_with_features, im_factors
                )
                logger.debug("[Features] Intermarket factors merged: %d cols", len(im_factors.columns) - 1)
        except Exception as e:
            logger.debug("[Features] Intermarket factors skipped: %s", e)

    # ---------------------------------------------------------------
    # D6: Candlestick pattern features (optional, requires OHLC)
    # ---------------------------------------------------------------
    if feature_cfg_obj is not None and getattr(feature_cfg_obj, "include_candlestick", False):
        try:
            has_ohlc_for_candles = all(
                c in prices_with_features.columns for c in ["open", "high", "low", "close"]
            )
            if has_ohlc_for_candles:
                from src.assembled_core.features.ta_candlestick import build_candlestick_features
                prices_with_features = build_candlestick_features(prices_with_features)
                logger.debug("[Features] Candlestick patterns merged")
        except Exception as e:
            logger.debug("[Features] Candlestick features skipped: %s", e)

    # ---------------------------------------------------------------
    # D9: Earnings calendar timing factors (optional)
    # ---------------------------------------------------------------
    if feature_cfg_obj is not None and getattr(feature_cfg_obj, "include_earnings", False):
        try:
            from src.assembled_core.data.sources.earnings_calendar_source import (
                EarningsCalendarSource,
            )
            symbols = prices_with_features["symbol"].unique().tolist()
            ts_min = prices_with_features["timestamp"].min()
            ts_max = prices_with_features["timestamp"].max()
            cal_src = EarningsCalendarSource()
            cal_df = cal_src.fetch_calendar(
                symbols=symbols,
                start_date=pd.Timestamp(ts_min).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(ts_max).strftime("%Y-%m-%d"),
            )
            prices_with_features = cal_src.build_earnings_factors(
                calendar_df=cal_df,
                prices_df=prices_with_features,
            )
            logger.debug("[Features] Earnings calendar factors merged")
        except Exception as e:
            logger.debug("[Features] Earnings calendar features skipped: %s", e)

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
        target_for_alignment = target_for_alignment.rename(
            columns={"target_qty": "qty"}
        )  # Rename to "qty" for alignment
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

    # Prices are required to convert target notional to shares; use ctx.prices PIT-filtered if available
    prices_for_orders = None
    if (
        ctx.prices is not None
        and not ctx.prices.empty
        and "close" in ctx.prices.columns
        and "symbol" in ctx.prices.columns
    ):
        if ctx.as_of is not None:
            as_of_utc = pd.to_datetime(ctx.as_of, utc=True)
            p_ts = pd.to_datetime(ctx.prices["timestamp"], utc=True)
            p = ctx.prices.loc[p_ts <= as_of_utc]
        else:
            p = ctx.prices
        if not p.empty:
            prices_for_orders = (
                p.groupby("symbol", group_keys=False)["close"].last().reset_index()
            )

    # Now generate orders with aligned positions (prices needed for notional -> shares)
    orders = generate_orders_from_targets(
        target_positions=target_aligned,
        current_positions=current_aligned,
        timestamp=ctx.order_timestamp,
        prices=prices_for_orders,
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
                current_positions_df = ctx.current_positions[
                    ["symbol", "target_qty"]
                ].rename(columns={"target_qty": "qty"})

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
                    ctx.prices.groupby("symbol")["price"].last().reset_index()
                )

        # Compute equity (cash + mark-to-market positions)
        equity = ctx.capital  # Use capital as equity proxy (can be refined later)

        # Get current_equity and peak_equity if available (for drawdown de-risking)
        current_equity = getattr(ctx, "current_equity", None)
        peak_equity = getattr(ctx, "peak_equity", None)

        # Get security_meta_df from context (for sector/region/FX limits)
        security_meta_df = ctx.security_meta_df

        # Convert risk_config dict to PreTradeConfig
        # HIGH-2.3: fall back to policy.yaml risk_limits when ctx.risk_config is empty
        from src.assembled_core.execution.pre_trade_checks import PreTradeConfig

        _policy_defaults: dict[str, Any] = {}
        if not ctx.risk_config:
            try:
                _pol = load_policy()
                _rl = _pol.get("risk_limits") or {}
                _dd = _rl.get("max_drawdown") or {}
                _tv = _rl.get("turnover") or {}
                _policy_defaults = {
                    "max_weight_per_symbol": _rl.get("max_position_weight"),
                    "drawdown_threshold": _dd.get("kill"),
                    "turnover_cap": _tv.get("daily_cap"),
                }
                logger.debug(
                    "PRE_TRADE: using policy.yaml risk_limits as PreTradeConfig fallback: %s",
                    _policy_defaults,
                )
            except Exception as e:
                logger.warning(
                    "PRE_TRADE: could not load policy for risk defaults: %s", e
                )

        pre_trade_config = None
        if ctx.risk_config or _policy_defaults:
            # Extract PreTradeConfig fields from risk_config dict
            if isinstance(ctx.risk_config, dict):
                pre_trade_config = PreTradeConfig(
                    max_notional_per_symbol=ctx.risk_config.get(
                        "max_notional_per_symbol"
                    ),
                    max_weight_per_symbol=(
                        ctx.risk_config.get("max_weight_per_symbol")
                        or _policy_defaults.get("max_weight_per_symbol")
                    ),
                    turnover_cap=(
                        ctx.risk_config.get("turnover_cap")
                        or _policy_defaults.get("turnover_cap")
                    ),
                    drawdown_threshold=(
                        ctx.risk_config.get("drawdown_threshold")
                        or _policy_defaults.get("drawdown_threshold")
                    ),
                    de_risk_scale=ctx.risk_config.get("de_risk_scale", 0.0),
                    max_gross_exposure=ctx.risk_config.get("max_gross_exposure"),
                    max_sector_exposure=ctx.risk_config.get("max_sector_exposure"),
                    max_region_exposure=ctx.risk_config.get("max_region_exposure"),
                    max_fx_exposure=ctx.risk_config.get("max_fx_exposure"),
                    base_currency=ctx.risk_config.get("base_currency", "USD"),
                    missing_security_meta=ctx.risk_config.get(
                        "missing_security_meta", "raise"
                    ),
                )
            elif hasattr(ctx.risk_config, "__dict__"):
                # If it's already a PreTradeConfig or similar object, try to extract fields
                pre_trade_config = PreTradeConfig(
                    max_notional_per_symbol=getattr(
                        ctx.risk_config, "max_notional_per_symbol", None
                    ),
                    max_weight_per_symbol=getattr(
                        ctx.risk_config, "max_weight_per_symbol", None
                    ),
                    turnover_cap=getattr(ctx.risk_config, "turnover_cap", None),
                    drawdown_threshold=getattr(
                        ctx.risk_config, "drawdown_threshold", None
                    ),
                    de_risk_scale=getattr(ctx.risk_config, "de_risk_scale", 0.0),
                    max_gross_exposure=getattr(
                        ctx.risk_config, "max_gross_exposure", None
                    ),
                    max_sector_exposure=getattr(
                        ctx.risk_config, "max_sector_exposure", None
                    ),
                    max_region_exposure=getattr(
                        ctx.risk_config, "max_region_exposure", None
                    ),
                    max_fx_exposure=getattr(ctx.risk_config, "max_fx_exposure", None),
                    base_currency=getattr(ctx.risk_config, "base_currency", "USD"),
                    missing_security_meta=getattr(
                        ctx.risk_config, "missing_security_meta", "raise"
                    ),
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
        log.warning(
            f"Risk controls failed: {e}. Passing through orders without filtering."
        )
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
        timestamp=pd.Timestamp.now("UTC"),
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
        result.error_message = (
            f"Missing required price columns: {', '.join(missing_cols)}"
        )
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

    # Risk state machine (INT-4): load persisted state, compute next, save (if not ephemeral), fill ctx.risk_state
    try:
        policy = load_policy()
    except Exception as e:
        logger.warning("load_policy failed, using empty policy: %s", e)
        policy = {}
    rsm = policy.get("risk_state_machine") or {}
    base_dir = get_base_dir()
    persistence = rsm.get("persistence") or {}
    mode = os.environ.get("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE") or persistence.get(
        "mode", "live"
    )
    # Use simulation time (ctx.as_of) in paper/backtest so cooldown is in "simulation hours", not wall-clock
    if getattr(ctx, "as_of", None) is not None:
        as_of_utc = pd.to_datetime(ctx.as_of, utc=True)
        now_utc = as_of_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        now_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    if mode == "ephemeral":
        import tempfile

        _ephemeral_path = (
            Path(tempfile.gettempdir())
            / f"assembled_risk_state_ephemeral_{os.getpid()}.json"
        )
        prev = load_risk_state(_ephemeral_path)
        next_rec = compute_next_state(ctx, policy, now_utc, prev)
        ctx.risk_state = next_rec.to_dict()
    else:
        if mode == "per_run":
            run_id = (
                getattr(ctx, "run_id", None)
                or os.environ.get("ASSEMBLED_RUN_ID")
                or f"pid{os.getpid()}"
            )
            per_run_dir = base_dir / str(
                persistence.get("per_run_dir", "output/state/runs")
            )
            state_path = per_run_dir / str(run_id) / "risk_state.json"
        else:
            state_path = base_dir / str(
                rsm.get("state_path", "output/state/risk_state.json")
            )
        prev = load_risk_state(state_path)
        next_rec = compute_next_state(ctx, policy, now_utc, prev)
        if rsm.get("enabled", True):
            save_risk_state(next_rec, state_path, policy)
        ctx.risk_state = next_rec.to_dict()

    # Intel: disclosures triggers (read-only snapshot; missing/invalid -> DEGRADED flag)
    # Skip when intel_sim_applied (BENCH-1: paper_runner injected simulated intel)
    if not getattr(ctx, "intel_sim_applied", False):
        try:
            intel_cfg = policy.get("intel") or {}
            disc_tr_cfg = intel_cfg.get("disclosures_triggers") or {}
            if disc_tr_cfg.get("enabled", False):
                from src.assembled_core.intel.disclosures_triggers_loader import (
                    load_disclosures_triggers,
                )

                path_raw = disc_tr_cfg.get(
                    "path", "output/intel/disclosures/triggers_latest.json"
                )
                path_resolved = (
                    (base_dir / path_raw)
                    if not Path(path_raw).is_absolute()
                    else Path(path_raw)
                )
                snap = load_disclosures_triggers(path_resolved)
                if snap.generated_utc:
                    ctx.disclosures_triggers = snap
                else:
                    ctx.disclosures_triggers = None
                    ctx.intel_health_flags["intel_disclosures_triggers"] = "DEGRADED"
        except Exception as e:
            logger.warning("intel disclosures_triggers load failed: %s", e)
            ctx.disclosures_triggers = None
            if "intel_disclosures_triggers" not in (ctx.intel_health_flags or {}):
                ctx.intel_health_flags = ctx.intel_health_flags or {}
                ctx.intel_health_flags["intel_disclosures_triggers"] = "DEGRADED"

        # Crisis Alpha intel: load crisis_state.json + triggers_latest.json from GDELT cycle
        try:
            intel_cfg = policy.get("intel") or {}
            crisis_cfg = intel_cfg.get("crisis_alpha") or {}
            if crisis_cfg.get("enabled", False):
                import json as _json

                # Load crisis state
                cs_path_raw = crisis_cfg.get(
                    "crisis_state_path", "data/intel/crisis_state.json"
                )
                cs_path = (
                    (base_dir / cs_path_raw)
                    if not Path(cs_path_raw).is_absolute()
                    else Path(cs_path_raw)
                )
                if cs_path.exists():
                    cs_data = _json.loads(cs_path.read_text(encoding="utf-8"))
                    ctx.crisis_state_intel = cs_data
                    geo_score = int(cs_data.get("geo_score", 0))
                    mode_str = str(cs_data.get("mode", "NORMAL"))
                    ctx.news_geo = {
                        "geo_score": geo_score,
                        "geo_confidence": float(cs_data.get("confidence", 0.0)),
                        "state_hint": mode_str,
                        "crisis_mode": mode_str,
                        "active_triggers": cs_data.get("active_triggers", []),
                        "basket_overrides": cs_data.get("basket_overrides", {}),
                    }
                    logger.info(
                        "CRISIS_ALPHA: mode=%s, geo_score=%d, triggers=%d",
                        mode_str, geo_score,
                        len(cs_data.get("active_triggers", [])),
                    )
                else:
                    logger.debug("crisis_state.json not found at %s", cs_path)

                # Load geo triggers (for news_triggers_loader compatibility)
                tr_path_raw = crisis_cfg.get(
                    "triggers_path", "data/intel/triggers_latest.json"
                )
                tr_path = (
                    (base_dir / tr_path_raw)
                    if not Path(tr_path_raw).is_absolute()
                    else Path(tr_path_raw)
                )
                if tr_path.exists():
                    from src.assembled_core.intel.news_triggers_loader import (
                        load_news_triggers,
                    )
                    news_snap = load_news_triggers(tr_path)
                    if news_snap.generated_utc:
                        result.meta["intel_geo_triggers"] = {
                            "max_severity": news_snap.summary.get("max_severity", 0),
                            "watch_count": news_snap.summary.get("watch_count_sev1plus", 0),
                            "active_count": news_snap.summary.get("active_count_sev2plus", 0),
                        }
        except Exception as e:
            logger.warning("crisis_alpha intel load failed: %s", e)
            ctx.intel_health_flags["intel_crisis_alpha"] = "DEGRADED"

        # Market stress (INT-5): price-based stress signal for state machine
        ms_cfg = policy.get("market_stress") or {}
        if ms_cfg.get("enabled", False):
            ctx.market_stress = compute_market_stress(ctx.prices, policy)
        else:
            ctx.market_stress = None

        # Disclosures confirm (DISCL-4.2): boost news_geo.geo_confidence when disclosures triggers sev >= 1
        try:
            from src.assembled_core.risk.disclosures_confirm import (
                apply_disclosures_confirm,
            )

            apply_disclosures_confirm(ctx, policy)
        except Exception as e:
            logger.warning("disclosures_confirm apply failed: %s", e)

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
            result.error_message = (
                "No prices remaining after filtering (as_of or universe)"
            )
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
            result.prices_with_features = hooks["build_features"](
                ctx, result.prices_filtered
            )
        elif (
            ctx.mode == "backtest"
            and ctx.precomputed_prices_with_features is not None
            and not ctx.precomputed_prices_with_features.empty
        ):
            # Backtest mode: use precomputed feature panel (PIT-safe slice)
            precomputed = ctx.precomputed_prices_with_features.copy()

            # Ensure timestamp column is UTC-aware for comparison
            if precomputed["timestamp"].dtype.tz is None:
                precomputed["timestamp"] = pd.to_datetime(
                    precomputed["timestamp"], utc=True
                )
            elif precomputed["timestamp"].dtype.tz != pd.Timestamp.now("UTC").tz:
                # Ensure UTC timezone
                precomputed["timestamp"] = precomputed["timestamp"].dt.tz_convert("UTC")

            if ctx.backtest_use_snapshot:
                # Snapshot mode: only use latest row per symbol <= as_of (performance optimization)
                # Use precomputed index if available (O(S log N) instead of O(N log N))
                if ctx.precomputed_panel_index is not None and ctx.as_of is not None:
                    # Use optimized index-based snapshot extraction
                    from src.assembled_core.pipeline.precomputed_index import (
                        snapshot_as_of,
                    )

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
                        precomputed_filtered.groupby(
                            "symbol", group_keys=False, dropna=False
                        )
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
                if (
                    not result.prices_with_features.empty
                    and result.prices_latest is None
                ):
                    result.prices_latest = (
                        result.prices_with_features.groupby(
                            "symbol", group_keys=False, dropna=False
                        )
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
            # EOD/Paper/Live: use full history <= as_of for feature building (MAs etc. need history)
            if ctx.mode in ("eod", "paper", "live") and ctx.as_of is not None:
                prices_for_features = ctx.prices[
                    ctx.prices["timestamp"] <= ctx.as_of
                ].copy()
                if ctx.universe is not None:
                    universe_upper = [s.upper().strip() for s in ctx.universe]
                    prices_for_features = prices_for_features[
                        prices_for_features["symbol"].str.upper().isin(universe_upper)
                    ].copy()
                result.prices_with_features = _build_features_default(
                    ctx, prices_for_features
                )
                # Keep latest per symbol for order generation
                if not result.prices_with_features.empty:
                    result.prices_latest = (
                        result.prices_with_features.groupby(
                            "symbol", group_keys=False, dropna=False
                        )
                        .last()
                        .reset_index()
                        .sort_values("symbol")
                        .reset_index(drop=True)
                    )
            else:
                result.prices_with_features = _build_features_default(
                    ctx, result.prices_filtered
                )

        log.debug(
            f"Features: {len(result.prices_with_features.columns)} columns (was {len(result.prices_filtered.columns)})"
        )
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in build_features: {e}"
        return result

    # Step 2.5: D3 — Optional HMM regime detection (replaces/supplements ctx.regime_state)
    try:
        regime_detection_cfg = policy.get("regime_detection", {})
        if regime_detection_cfg.get("method") == "hmm" and ctx.regime_state is None:
            from src.assembled_core.risk.regime_models import build_regime_state_hmm
            prices_for_hmm = result.prices_filtered if result.prices_filtered is not None else ctx.prices
            if prices_for_hmm is not None and not prices_for_hmm.empty:
                hmm_df = build_regime_state_hmm(
                    prices=prices_for_hmm,
                    n_regimes=int(regime_detection_cfg.get("n_regimes", 3)),
                    benchmark_symbol=regime_detection_cfg.get("benchmark_symbol"),
                )
                if not hmm_df.empty:
                    latest = hmm_df.iloc[-1]
                    ctx.regime_state = latest.get("regime_label", "sideways")
                    result.meta["regime_hmm"] = {
                        "label": ctx.regime_state,
                        "confidence": round(float(latest.get("regime_confidence", 0)), 3),
                    }
                    log.info("REGIME_HMM: detected regime='%s'", ctx.regime_state)
    except Exception as e:
        log.debug("HMM regime detection skipped: %s", e)

    # Step 3: Generate signals (hook point: generate_signals)
    try:
        if "generate_signals" in hooks:
            result.signals = hooks["generate_signals"](ctx, result.prices_with_features)
        else:
            # Default: call signal_fn
            result.signals = ctx.signal_fn(result.prices_with_features)

        # In backtest/eod mode with history slice, signal_fn may return multiple rows per symbol.
        # Optionally keep only the latest signal per symbol (PIT: state at rebalance = last row per symbol in slice).
        settings = get_settings()
        if (
            settings.reduce_signals_to_latest_bar
            and ctx.mode in ("backtest", "eod", "paper", "live")
            and "timestamp" in result.signals.columns
            and not result.signals.empty
        ):
            result.signals["_ts"] = pd.to_datetime(
                result.signals["timestamp"], utc=True
            )
            result.signals = (
                result.signals.sort_values("_ts", ascending=True)
                .groupby("symbol", group_keys=False)
                .last()
                .reset_index()
                .drop(columns=["_ts"])
            )

        # Validate signals format
        required_signal_cols = ["timestamp", "symbol", "direction"]
        missing_signal_cols = [
            c for c in required_signal_cols if c not in result.signals.columns
        ]
        if missing_signal_cols:
            result.status = "error"
            result.error_message = (
                f"signals missing required columns: {', '.join(missing_signal_cols)}"
            )
            return result

        log.debug(f"Signals generated: {len(result.signals)} rows")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in generate_signals hook: {e}"
        return result

    # Zombie killer (M6-T05): force-exit positions held too long with no gain
    try:
        if (
            ctx.current_positions is not None
            and not ctx.current_positions.empty
            and not result.signals.empty
        ):
            now_utc = pd.Timestamp.now("UTC").to_pydatetime()
            # Convert positions to list of dicts for zombie_killer
            pos_dicts = ctx.current_positions.to_dict("records")
            zombies = get_zombie_positions(pos_dicts, now_utc, policy)
            if zombies:
                zombie_symbols = {pos["symbol"] for pos, _reason in zombies}
                for _pos, reason in zombies:
                    log.warning(reason)
                # Force zombie signals to FLAT so they get exit orders
                mask = result.signals["symbol"].isin(zombie_symbols)
                result.signals.loc[mask, "direction"] = "FLAT"
                # Also add FLAT signals for zombies not already in signals
                existing_syms = set(result.signals["symbol"].values)
                missing_zombies = zombie_symbols - existing_syms
                if missing_zombies:
                    zombie_rows = pd.DataFrame(
                        {
                            "timestamp": [ctx.as_of or pd.Timestamp.now("UTC")] * len(missing_zombies),
                            "symbol": list(missing_zombies),
                            "direction": ["FLAT"] * len(missing_zombies),
                            "score": [0.0] * len(missing_zombies),
                        }
                    )
                    result.signals = pd.concat(
                        [result.signals, zombie_rows], ignore_index=True
                    )
                result.meta["zombie_killer"] = {
                    "zombies_found": len(zombies),
                    "symbols": sorted(zombie_symbols),
                }
                log.info("ZOMBIE_KILLER: %d positions flagged for exit: %s",
                         len(zombies), sorted(zombie_symbols))
    except Exception as e:
        log.debug("zombie_killer check skipped: %s", e)

    # Step 3.5: Crash prediction + short signal generation
    try:
        shorts_policy = policy.get("shorts", {})
        if shorts_policy.get("enabled", False):
            from src.assembled_core.signals.crash_prediction import CrashPredictionEngine
            from src.assembled_core.signals.short_signals import ShortSignalGenerator
            from src.assembled_core.risk.short_risk import ShortRiskManager

            regime_state = getattr(ctx, "regime_state", None)
            intel_state = getattr(ctx, "crisis_state_intel", None)

            # Extract macro data from features/context if available
            macro_data: dict = {}
            if ctx.prices is not None and not ctx.prices.empty:
                # Use VIX if available in prices
                if "VIX" in ctx.prices.columns:
                    macro_data["vix"] = float(ctx.prices["VIX"].iloc[-1])

            crash_engine = CrashPredictionEngine()
            crash_signal = crash_engine.predict(
                market_data=ctx.prices,
                regime=regime_state,
                intel_state=intel_state,
                macro_data=macro_data if macro_data else None,
            )

            min_crash_prob = shorts_policy.get("min_crash_probability", 0.60)
            if crash_signal.crash_probability >= min_crash_prob:
                short_gen = ShortSignalGenerator(policy=shorts_policy)
                short_df = short_gen.generate_short_targets(
                    crash_signal=crash_signal,
                    universe=ctx.universe if hasattr(ctx, "universe") and ctx.universe is not None else pd.DataFrame(),
                    prices=ctx.prices,
                    regime=regime_state,
                )

                # Validate short targets via risk manager
                risk_mgr = ShortRiskManager(policy=policy)
                risk_check = risk_mgr.validate_short_targets(short_df, regime=regime_state)

                if risk_check.passed and not short_df.empty:
                    # Merge short signals with long signals
                    # Short signals have negative target_weight; convert to signals format
                    short_signals_rows = []
                    for _, row in short_df.iterrows():
                        short_signals_rows.append({
                            "timestamp": ctx.as_of or pd.Timestamp.now("UTC"),
                            "symbol": row["symbol"],
                            "direction": row.get("direction", "SHORT"),
                            "score": -abs(row["confidence"]),  # Negative score = short signal
                        })

                    if short_signals_rows:
                        short_signal_df = pd.DataFrame(short_signals_rows)
                        # Only add symbols not already in signals
                        existing_syms = set(result.signals["symbol"].values) if not result.signals.empty else set()
                        new_shorts = short_signal_df[~short_signal_df["symbol"].isin(existing_syms)]
                        if not new_shorts.empty:
                            result.signals = pd.concat(
                                [result.signals, new_shorts], ignore_index=True
                            )

                    log.info(
                        "CRASH_PREDICTION: prob=%.3f severity=%.3f → %d short signals added",
                        crash_signal.crash_probability,
                        crash_signal.expected_severity,
                        len(short_signals_rows),
                    )
                elif not risk_check.passed:
                    log.warning(
                        "CRASH_PREDICTION: short signals blocked by risk check: %s",
                        risk_check.violations,
                    )

            result.meta["crash_prediction"] = {
                "crash_probability": crash_signal.crash_probability,
                "severity": crash_signal.expected_severity,
                "horizon_days": crash_signal.time_horizon_days,
                "active": crash_signal.active,
                "contributing_signals": crash_signal.contributing_signals,
                "recommended_instruments": crash_signal.recommended_instruments,
            }
    except Exception as e:
        log.debug("crash_prediction step skipped: %s", e)

    # Step 4: Size positions (hook point: size_positions)
    try:
        if "size_positions" in hooks:
            result.target_positions = hooks["size_positions"](ctx, result.signals)
        else:
            # Policy-driven sizing method dispatch
            sizing_cfg = policy.get("position_sizing") or {}
            sizing_method = sizing_cfg.get("method", "default")
            if sizing_method != "default" and ctx.position_sizing_fn is not None:
                # Use the caller-provided function (backward compatible)
                sizing_method = "default"

            if sizing_method == "kelly":
                from src.assembled_core.portfolio.position_sizing import (
                    compute_kelly_weights,
                )
                result.target_positions = compute_kelly_weights(
                    result.signals,
                    fraction=float(sizing_cfg.get("kelly_fraction", 0.5)),
                    max_weight=float(sizing_cfg.get("max_weight", 0.25)),
                    total_capital=ctx.capital,
                    top_n=sizing_cfg.get("top_n"),
                )
            elif sizing_method == "risk_parity":
                from src.assembled_core.portfolio.position_sizing import (
                    compute_risk_parity_weights,
                )
                # Compute per-symbol volatilities from price data
                vols = _estimate_symbol_volatilities(
                    result.prices_filtered or ctx.prices,
                    lookback=int(sizing_cfg.get("vol_lookback_days", 60)),
                )
                result.target_positions = compute_risk_parity_weights(
                    result.signals,
                    vols,
                    total_capital=ctx.capital,
                    max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                    top_n=sizing_cfg.get("top_n"),
                )
            elif sizing_method == "vol_scaled":
                from src.assembled_core.portfolio.position_sizing import (
                    compute_vol_scaled_weights,
                )
                vols = _estimate_symbol_volatilities(
                    result.prices_filtered or ctx.prices,
                    lookback=int(sizing_cfg.get("vol_lookback_days", 60)),
                )
                result.target_positions = compute_vol_scaled_weights(
                    result.signals,
                    vols,
                    target_vol=float(sizing_cfg.get("target_vol", 0.15)),
                    total_capital=ctx.capital,
                    max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                    top_n=sizing_cfg.get("top_n"),
                )
            elif sizing_method == "black_litterman":
                # D1: Black-Litterman optimizer
                try:
                    from src.assembled_core.portfolio.black_litterman import (
                        BlackLittermanOptimizer,
                    )
                    prices_for_bl = result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    bl = BlackLittermanOptimizer(
                        risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)),
                        tau=float(sizing_cfg.get("tau", 0.05)),
                        max_position=float(sizing_cfg.get("max_weight", 0.15)),
                        min_position=float(sizing_cfg.get("min_position", 0.0)),
                    )
                    # Build views from signal scores
                    views = {}
                    view_confidence = {}
                    if not result.signals.empty and "symbol" in result.signals.columns:
                        for _, row in result.signals.iterrows():
                            sym = row["symbol"]
                            score = float(row.get("score", 0.0))
                            conf = float(row.get("confidence", 0.5))
                            if abs(score) > 0.01:
                                views[sym] = score * 0.10  # Map score to expected return
                                view_confidence[sym] = conf
                    if views and prices_for_bl is not None and not prices_for_bl.empty:
                        bl_weights = bl.optimize_from_scores(
                            prices=prices_for_bl,
                            signal_scores=views,
                            confidence=view_confidence,
                        )
                        # Convert to target_positions format
                        rows = []
                        for sym, w in bl_weights.items():
                            rows.append({
                                "symbol": sym,
                                "target_weight": round(w, 4),
                                "target_qty": round(w * ctx.capital, 2),
                            })
                        result.target_positions = pd.DataFrame(rows)
                    else:
                        # Fallback to default sizing if no views
                        result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_method"] = "black_litterman"
                except Exception as e:
                    log.warning("Black-Litterman sizing failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
            else:
                # Default: call position_sizing_fn (equal weight or score-based)
                result.target_positions = ctx.position_sizing_fn(
                    result.signals, ctx.capital
                )
            result.meta["sizing_method"] = sizing_method

        # D2: Barra factor risk model — post-sizing vol check
        try:
            factor_risk_cfg = policy.get("factor_risk", {})
            if factor_risk_cfg.get("enabled", False):
                from src.assembled_core.risk.factor_risk_model import FactorRiskModel
                prices_for_risk = result.prices_filtered if result.prices_filtered is not None else ctx.prices
                if prices_for_risk is not None and not prices_for_risk.empty:
                    frm = FactorRiskModel()
                    frm.fit(prices_for_risk)
                    # Get target weights as dict
                    if "target_weight" in result.target_positions.columns:
                        tw_dict = dict(zip(
                            result.target_positions["symbol"],
                            result.target_positions["target_weight"].fillna(0),
                        ))
                        portfolio_vol = frm.predict_portfolio_vol(tw_dict)
                        vol_limit = float(factor_risk_cfg.get("max_portfolio_vol", 0.25))
                        if portfolio_vol > vol_limit and portfolio_vol > 0:
                            scale = vol_limit / portfolio_vol
                            result.target_positions["target_weight"] = (
                                result.target_positions["target_weight"] * scale
                            )
                            log.info(
                                "FACTOR_RISK: portfolio_vol=%.3f > limit=%.3f → scaled by %.3f",
                                portfolio_vol, vol_limit, scale,
                            )
                        factor_contribs = frm.predict_factor_contributions(tw_dict)
                        result.meta["factor_risk"] = {
                            "portfolio_vol": round(portfolio_vol, 4),
                            "vol_limit": vol_limit,
                            "top_factor_contributors": dict(
                                sorted(factor_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                            ),
                        }
        except Exception as e:
            log.debug("factor_risk_model skipped: %s", e)

        # Validate target_positions format
        required_target_cols = ["symbol", "target_weight", "target_qty"]
        missing_target_cols = [
            c for c in required_target_cols if c not in result.target_positions.columns
        ]
        if missing_target_cols:
            # Allow missing target_weight or target_qty (at least one should be present)
            if not any(
                c in result.target_positions.columns
                for c in ["target_weight", "target_qty"]
            ):
                result.status = "error"
                result.error_message = "target_positions missing required columns: symbol and (target_weight or target_qty)"
                return result

        # Apply GeoRisk exposure overlay (scaling only, no new signals)
        try:
            policy = load_policy()
        except Exception as e:
            logger.warning("load_policy failed, using empty policy: %s", e)
            policy = {}
        geo_multiplier = compute_exposure_multiplier(ctx, policy)
        # Soft Profit Lock (INT-6.2): combine multiplicatively with GeoRisk
        pl_cfg = policy.get("profit_lock") or {}
        if (
            pl_cfg.get("enabled")
            and getattr(ctx, "equity_curve", None) is not None
            and getattr(ctx, "equity_curve_index", None) is not None
        ):
            pl_state = getattr(ctx, "profit_lock_state", None) or {}
            profit_lock_mult, pl_state_out = compute_profit_lock_multiplier(
                ctx.equity_curve,
                pl_cfg,
                ctx.equity_curve_index,
                state=pl_state,
            )
            ctx.profit_lock_state = pl_state_out
            result.meta["profit_lock_state"] = pl_state_out
            result.meta["profit_lock"] = {"multiplier": profit_lock_mult}
        else:
            profit_lock_mult = 1.0
        # Vol targeting (M6-T03): scale exposure toward target annualized vol
        vt_cfg = policy.get("vol_targeting") or {}
        if (
            vt_cfg.get("enabled", False)
            and getattr(ctx, "equity_curve", None) is not None
            and getattr(ctx, "equity_curve_index", None) is not None
        ):
            vol_scale_factor, realized_vol, target_vol = compute_vol_targeting_result(
                ctx.equity_curve,
                vt_cfg,
                now_idx=ctx.equity_curve_index,
            )
            result.meta["vol_targeting"] = {
                "scale_factor": vol_scale_factor,
                "realized_vol": realized_vol,
                "target_vol": target_vol,
            }
        else:
            vol_scale_factor = 1.0
            result.meta["vol_targeting"] = {
                "scale_factor": 1.0,
                "realized_vol": float("nan"),
                "target_vol": float("nan"),
            }
        # Market stress multiplier (MEDIUM-5.3): reduce exposure when stress is detected
        ms_multiplier = 1.0
        if ctx.market_stress:
            stress_score = int(ctx.market_stress.get("stress_score", 0))
            _ms_scaling = (policy.get("market_stress") or {}).get(
                "exposure_scaling"
            ) or {}
            if stress_score >= 2:
                ms_multiplier = float(_ms_scaling.get("stress_score_2", 0.50))
            elif stress_score >= 1:
                ms_multiplier = float(_ms_scaling.get("stress_score_1", 0.75))
            if ms_multiplier < 1.0:
                log.warning(
                    "MARKET_STRESS: stress_score=%d -> exposure multiplier=%.2f",
                    stress_score,
                    ms_multiplier,
                )
        result.meta["market_stress_multiplier"] = ms_multiplier

        # Crisis Alpha exposure reduction (Phase 1.2)
        crisis_alpha_multiplier = 1.0
        if getattr(ctx, "crisis_state_intel", None):
            crisis_mode = str(ctx.crisis_state_intel.get("mode", "NORMAL")).upper()
            ca_cfg = (
                policy.get("crisis_alpha")
                or policy.get("intel", {}).get("crisis_alpha")
                or {}
            )
            if crisis_mode == "CRISIS":
                crisis_alpha_multiplier = min(float(
                    ca_cfg.get("crisis_multiplier", 0.25)
                ), 1.0)
            elif crisis_mode == "ELEVATED":
                crisis_alpha_multiplier = min(float(
                    ca_cfg.get("elevated_multiplier", 0.60)
                ), 1.0)
            if crisis_alpha_multiplier < 1.0:
                log.warning(
                    "CRISIS_ALPHA: mode=%s -> exposure multiplier=%.2f",
                    crisis_mode,
                    crisis_alpha_multiplier,
                )
        result.meta["crisis_alpha_multiplier"] = crisis_alpha_multiplier

        final_multiplier = (
            geo_multiplier
            * profit_lock_mult
            * vol_scale_factor
            * ms_multiplier
            * crisis_alpha_multiplier
        )
        if abs(final_multiplier - 1.0) > 1e-9 and not result.target_positions.empty:
            result.target_positions = apply_exposure_multiplier_to_targets(
                result.target_positions,
                multiplier=final_multiplier,
                cash_symbol="CASH",
            )
            log.debug(
                "Exposure overlay applied: "
                f"geo={geo_multiplier:.4f}, profit_lock={profit_lock_mult:.4f}, "
                f"vol={vol_scale_factor:.4f}, stress={ms_multiplier:.4f}, "
                f"crisis_alpha={crisis_alpha_multiplier:.4f}, "
                f"final={final_multiplier:.4f}, symbols={len(result.target_positions)}"
            )
        else:
            log.debug(
                f"Target positions computed: {len(result.target_positions)} symbols"
            )
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in size_positions hook: {e}"
        return result

    # Turnover budget gate (INT-6): after GeoRisk, before order generation
    tb = policy.get("turnover_budget") or {}
    if tb.get("enabled", False) and not result.target_positions.empty:
        try:
            cap = float(tb.get("cap", 0.15) or 0.15)
            behavior = str(tb.get("behavior", "scale") or "scale")
            prices_for_turnover = (
                result.prices_latest
                if result.prices_latest is not None and not result.prices_latest.empty
                else result.prices_filtered
            )
            estimated = estimate_turnover(
                ctx.current_positions,
                result.target_positions,
                prices_for_turnover,
                portfolio_value=ctx.capital,
            )
            if estimated == float("inf"):
                result.target_positions, scale_factor = apply_turnover_gate(
                    result.target_positions,
                    ctx.current_positions,
                    cap=cap,
                    estimated_turnover=1.0,
                    behavior="block",
                    prices=prices_for_turnover,
                    portfolio_value=ctx.capital,
                )
            else:
                result.target_positions, scale_factor = apply_turnover_gate(
                    result.target_positions,
                    ctx.current_positions,
                    cap=cap,
                    estimated_turnover=estimated,
                    behavior=behavior,
                    prices=prices_for_turnover,
                    portfolio_value=ctx.capital,
                )
            result.meta["turnover_budget"] = {
                "estimated_turnover": estimated,
                "scale_factor": scale_factor,
                "cap": cap,
                "behavior": behavior,
            }
        except Exception as e:
            log.debug(f"Turnover budget gate skipped: {e}")
            result.meta["turnover_budget"] = {"error": str(e), "scale_factor": 1.0}

    # Correlation guard (M6-T07): scale down over-concentrated correlated clusters
    try:
        if not result.target_positions.empty and len(result.target_positions) >= 2:
            # Build target_weights dict from DataFrame
            tw_dict = dict(
                zip(
                    result.target_positions["symbol"],
                    result.target_positions["target_weight"],
                )
            )
            # Use prices_filtered (or prices_with_features) for correlation computation
            corr_prices = (
                result.prices_filtered
                if result.prices_filtered is not None and not result.prices_filtered.empty
                else ctx.prices
            )
            adjusted_weights, corr_reasons = apply_correlation_guard(
                tw_dict, corr_prices, policy
            )
            if corr_reasons:
                for reason in corr_reasons:
                    log.warning(reason)
                # Apply adjusted weights back to target_positions
                result.target_positions["target_weight"] = result.target_positions[
                    "symbol"
                ].map(adjusted_weights)
                if "target_qty" in result.target_positions.columns:
                    result.target_positions["target_qty"] = (
                        result.target_positions["target_weight"] * ctx.capital
                    )
                result.meta["correlation_guard"] = {
                    "clusters_scaled": len(corr_reasons),
                    "reasons": corr_reasons,
                }
                log.info(
                    "CORRELATION_GUARD: %d clusters scaled down", len(corr_reasons)
                )
    except Exception as e:
        log.debug("correlation_guard check skipped: %s", e)

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
                if (
                    result.prices_latest is not None
                    and "close" in result.prices_latest.columns
                ):
                    # Backtest mode: use pre-extracted latest prices
                    latest_prices = result.prices_latest[["symbol", "close"]].rename(
                        columns={"close": "price"}
                    )
                elif "close" in result.prices_with_features.columns:
                    # EOD mode: extract latest prices from prices_with_features
                    latest_prices = (
                        result.prices_with_features.groupby("symbol", group_keys=False)[
                            "close"
                        ]
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
                        result.orders["price"] = result.orders["price_latest"].fillna(
                            result.orders["price"]
                        )
                        result.orders = result.orders.drop(columns=["price_latest"])

        log.debug(f"Orders generated: {len(result.orders)} orders")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in generate_orders: {e}"
        return result

    # QA Gate: Block orders if qa_block_trading is True (Sprint 3 / D2)
    if ctx.qa_block_trading:
        log.warning(
            f"QA Gate: Trading blocked - {ctx.qa_block_reason or 'No reason provided'}"
        )
        # Set orders to empty DataFrame with correct schema
        result.orders = pd.DataFrame(
            columns=["timestamp", "symbol", "side", "qty", "price"]
        )
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
            log.info(
                f"Risk controls filtered orders: {len(result.orders)} -> {len(result.orders_filtered)} ({len(result.orders) - len(result.orders_filtered)} blocked)"
            )

        log.debug(f"Orders after risk controls: {len(result.orders_filtered)} orders")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in risk_controls: {e}"
        return result

    # Step 6.5: D12 — Scenario engine stress tests (post-orders, optional)
    try:
        scenario_cfg = policy.get("scenario_engine", {})
        if scenario_cfg.get("enabled", False) and not result.target_positions.empty:
            from src.assembled_core.qa.scenario_engine import run_crisis_scenarios
            import datetime as _dt
            prices_for_scenario = result.prices_filtered if result.prices_filtered is not None else ctx.prices
            if prices_for_scenario is not None and not prices_for_scenario.empty:
                crisis_type = scenario_cfg.get("crisis_type", "geopolitical_escalation")
                shock_date = ctx.as_of.replace(tzinfo=None) if ctx.as_of else _dt.datetime.utcnow()
                scenarios = run_crisis_scenarios(
                    prices=prices_for_scenario,
                    crisis_type=crisis_type,
                    shock_date=shock_date,
                )
                result.meta["scenario_engine"] = {
                    "crisis_type": crisis_type,
                    "scenarios_run": list(scenarios.keys()),
                    "n_scenarios": len(scenarios),
                }
                log.info("SCENARIO_ENGINE: %d scenarios for crisis_type=%s", len(scenarios), crisis_type)
    except Exception as e:
        log.debug("scenario_engine skipped: %s", e)

    # Step 7: Write outputs (hook point: write_outputs)
    try:
        if ctx.write_outputs:
            if "write_outputs" in hooks:
                result.output_paths = hooks["write_outputs"](
                    ctx, result.orders_filtered
                )
            else:
                # Default output writing based on output_format
                if ctx.output_format == "safe_csv":
                    try:
                        from src.assembled_core.execution.safe_bridge import (
                            write_safe_orders_csv,
                        )

                        ctx.output_dir.mkdir(parents=True, exist_ok=True)
                        out_path = write_safe_orders_csv(
                            result.orders_filtered,
                            output_path=ctx.output_dir / "orders_latest.csv",
                        )
                        result.output_paths = {"safe_csv": out_path}
                    except Exception as _oe:
                        log.warning("Default safe_csv write failed: %s", _oe)
                        result.output_paths = {}
                else:
                    # "equity_curve", "state", "none" — not yet implemented without a hook
                    log.debug(
                        "No default output writer for format '%s'; "
                        "provide a write_outputs hook to enable it.",
                        ctx.output_format,
                    )
                    result.output_paths = {}

        log.debug(f"Outputs written: {len(result.output_paths)} files")
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in write_outputs hook: {e}"
        return result

    log.info(
        f"Trading cycle completed successfully: {len(result.orders_filtered)} orders"
    )

    return result
