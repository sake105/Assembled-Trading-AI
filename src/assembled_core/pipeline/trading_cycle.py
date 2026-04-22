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
    add_moving_averages,
)
from src.assembled_core.risk.correlation_guard import (
    apply_correlation_guard,
    detect_correlation_regime_shift,
)
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
    use_factor_store: bool = True
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
    # E0.1 parity flag: when True (new default) backtest mode preserves
    # kill-switch state across bars, matching paper/live. Set False for
    # research speed-runs where a single-bar trip should not gate
    # downstream bars. Legacy behavior is False.
    kill_switch_persist: bool = True
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
            builder_fn=add_all_features if has_ohlc else add_moving_averages,
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
            from src.assembled_core.features.ta_features import add_log_returns

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

    # ---------------------------------------------------------------
    # D10: Congressional trading features (optional, Sprint 5)
    # ---------------------------------------------------------------
    if feature_cfg_obj is not None and getattr(feature_cfg_obj, "include_congress", False):
        try:
            from src.assembled_core.data.congress_trades_ingest import load_congress_sample
            from src.assembled_core.features.congress_features import add_congress_features

            congress_path = getattr(feature_cfg_obj, "congress_data_path", None)
            congress_events = load_congress_sample(path=congress_path)
            if not congress_events.empty:
                prices_with_features = add_congress_features(
                    prices_with_features,
                    congress_events,
                    as_of=ctx.as_of,
                )
                logger.debug("[Features] Congress trading features merged")
        except Exception as e:
            logger.debug("[Features] Congress features skipped: %s", e)

    return prices_with_features


def should_rebalance(
    ctx: TradingContext,
    target_positions: pd.DataFrame,
    current_weights: dict[str, float] | None = None,
    *,
    weight_drift_threshold: float = 0.05,
    vol_regime_change: bool = False,
    corr_spike: bool = False,
    scheduled: bool = True,
    drawdown_pct: float | None = None,
) -> tuple[bool, str]:
    """Determine whether rebalancing is warranted (V14).

    Checks 4 triggers:
    1. Scheduled rebalance date (existing behavior, default True)
    2. Weight drift exceeds threshold
    3. Vol regime changed (from vol_targeting)
    4. Correlation spike detected (from correlation_guard)

    Plus: drawdown-based gradual de-risking signal.

    Returns:
        (should_rebalance: bool, reason: str)
    """
    reasons: list[str] = []

    # Trigger 1: Scheduled
    if scheduled:
        reasons.append("scheduled")

    # Trigger 2: Weight drift
    if current_weights and not target_positions.empty and "symbol" in target_positions.columns:
        target_w = {}
        if "target_weight" in target_positions.columns:
            for _, row in target_positions.iterrows():
                target_w[row["symbol"]] = float(row.get("target_weight", 0.0))
        all_syms = set(current_weights.keys()) | set(target_w.keys())
        if all_syms:
            max_drift = max(
                abs(current_weights.get(s, 0.0) - target_w.get(s, 0.0))
                for s in all_syms
            )
            if max_drift > weight_drift_threshold:
                reasons.append(f"weight_drift={max_drift:.3f}")

    # Trigger 3: Vol regime change
    if vol_regime_change:
        reasons.append("vol_regime_change")

    # Trigger 4: Correlation spike
    if corr_spike:
        reasons.append("corr_spike")

    # Trigger 5: Drawdown de-risking
    if drawdown_pct is not None and drawdown_pct < -0.10:
        reasons.append(f"drawdown={drawdown_pct:.2%}")

    if not reasons:
        return False, "no_trigger"

    return True, "|".join(reasons)


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


def _evaluate_circuit_breaker_daily(
    prices: pd.DataFrame | None,
    policy: dict[str, Any] | None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """Sprint 1 / W4b — daily-bar circuit breaker (stateless).

    Checks the most recent close-to-close return of the configured market
    reference (default ``SPY``). If the drop exceeds the configured
    threshold, returns a trip dict. Returns ``None`` when no data or no
    trip.

    Policy::

        circuit_breaker:
          enabled: true
          reference_symbol: SPY
          drop_threshold_pct: 3.0   # |pct drop| that trips the breaker
    """
    cb_cfg = (policy or {}).get("circuit_breaker") or {}
    if not cb_cfg.get("enabled", False):
        return None
    if prices is None or prices.empty or "symbol" not in prices.columns:
        return None

    ref = str(cb_cfg.get("reference_symbol", "SPY")).upper()
    threshold = float(cb_cfg.get("drop_threshold_pct", 3.0))

    ref_df = prices[prices["symbol"].astype(str).str.upper() == ref]
    if ref_df.empty or "close" not in ref_df.columns:
        return None

    if "timestamp" in ref_df.columns:
        ref_df = ref_df.sort_values("timestamp")
        if as_of is not None:
            ts = pd.to_datetime(ref_df["timestamp"], utc=True)
            as_of_utc = pd.Timestamp(as_of)
            if as_of_utc.tzinfo is None:
                as_of_utc = as_of_utc.tz_localize("UTC")
            ref_df = ref_df.loc[ts <= as_of_utc]
    closes = ref_df["close"].astype(float).dropna()
    if len(closes) < 2:
        return None

    prev = float(closes.iloc[-2])
    curr = float(closes.iloc[-1])
    if prev <= 0:
        return None

    drop_pct = (prev - curr) / prev * 100.0
    if drop_pct < threshold:
        return None

    return {
        "reference_symbol": ref,
        "previous_close": prev,
        "current_close": curr,
        "drop_pct": drop_pct,
        "threshold_pct": threshold,
        "reason": (
            f"circuit_breaker: {ref} dropped {drop_pct:.2f}% "
            f"(prev={prev:.2f}, curr={curr:.2f}, threshold={threshold:.1f}%)"
        ),
    }


def _evaluate_circuit_breaker(
    ctx: "TradingContext",
    result: "TradingCycleResult",
    policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Tier-1 wiring — intraday MDD circuit breaker (default OFF).

    Reads ``policy.risk.circuit_breaker``. When enabled, feeds the recent
    equity / benchmark observations into ``risk.circuit_breaker.CircuitBreaker``
    and returns a trip decision when the window drawdown exceeds the
    configured threshold. Caller empties orders on trip. Pure function.
    """
    cfg = ((policy or {}).get("risk") or {}).get("circuit_breaker") or {}
    if not cfg.get("enabled", False):
        return None

    try:
        from src.assembled_core.risk.circuit_breaker import CircuitBreaker
    except Exception:
        return None

    observations = None
    if result is not None and isinstance(result.meta, dict):
        observations = result.meta.get("intraday_equity_observations")
    if not observations:
        observations = getattr(ctx, "intraday_equity_observations", None)
    if not observations:
        return None

    try:
        cb = CircuitBreaker(
            drop_threshold_pct=float(cfg.get("drop_threshold_pct", 3.0)),
            window_minutes=int(cfg.get("window_minutes", 15)),
            cooldown_minutes=int(cfg.get("cooldown_minutes", 30)),
        )
        tripped_on: dict[str, Any] | None = None
        for obs in observations:
            ts = obs.get("timestamp") if isinstance(obs, dict) else obs[0]
            px = float(obs["price"] if isinstance(obs, dict) else obs[1])
            if cb.observe(px, timestamp=ts):
                tripped_on = {
                    "timestamp": str(ts),
                    "price": px,
                    "trip_count": cb.trip_count,
                }
        if tripped_on is None:
            return None
        return {
            "breach": True,
            "reason": "intraday_circuit_breaker_trip",
            "tripped_on": tripped_on,
            "drop_threshold_pct": float(cfg.get("drop_threshold_pct", 3.0)),
            "window_minutes": int(cfg.get("window_minutes", 15)),
        }
    except Exception:
        return None


def _evaluate_var_gate(
    ctx: "TradingContext",
    result: "TradingCycleResult",
    policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Tier-1 wiring — parametric VaR pre-trade exposure gate.

    Reads ``policy.risk.var_gate`` (default disabled) and, when enabled,
    computes one-day parametric portfolio VaR from the current price panel
    using the intended target weights. If VaR exceeds the configured
    ``max_var_pct``, returns a decision dict — caller empties filtered
    orders. Pure function; no side effects.
    """
    cfg = ((policy or {}).get("risk") or {}).get("var_gate") or {}
    if not cfg.get("enabled", False):
        return None

    try:
        from src.assembled_core.risk.var_methods import PortfolioVaR
    except Exception:
        return None

    prices = getattr(ctx, "prices", None)
    targets = getattr(result, "target_positions", None)
    if prices is None or prices.empty or targets is None or len(targets) == 0:
        return None

    price_col = "close" if "close" in prices.columns else (
        "price" if "price" in prices.columns else None
    )
    if price_col is None:
        return None

    try:
        wide = prices.pivot_table(
            index="timestamp", columns="symbol", values=price_col, aggfunc="last"
        ).sort_index()
        returns = wide.pct_change().dropna(how="all")
        if returns.shape[0] < int(cfg.get("min_history", 20)):
            return None

        # Build weight vector from target_positions: use notional share if
        # available, otherwise equal-weight the target symbols.
        if "weight" in targets.columns:
            w = targets.set_index("symbol")["weight"].astype(float)
        elif "notional" in targets.columns and float(targets["notional"].abs().sum()) > 0:
            notl = targets.set_index("symbol")["notional"].astype(float)
            w = notl / float(notl.abs().sum())
        else:
            syms = targets["symbol"].unique()
            w = pd.Series(1.0 / max(len(syms), 1), index=syms)

        var_calc = PortfolioVaR(returns=returns.fillna(0.0), weights=w)
        alpha = float(cfg.get("confidence", 0.95))
        var_1d = float(var_calc.parametric_var(alpha=alpha, horizon=1))
    except Exception:
        return None

    max_var = float(cfg.get("max_var_pct", 0.05))
    if var_1d <= max_var:
        return None
    return {
        "breach": True,
        "var_1d": var_1d,
        "max_var_pct": max_var,
        "confidence": alpha,
        "reason": f"parametric_var_breach_{var_1d:.4f}>{max_var:.4f}",
    }


def _evaluate_auto_dd_kill_switch(
    ctx: "TradingContext",
    result: "TradingCycleResult",
    policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Sprint 1 / C7 — evaluate auto-drawdown kill-switch staircase.

    Returns a decision dict when a threshold is breached, otherwise ``None``.
    The caller is responsible for engaging the kill-switch and clearing
    orders. Pure function: no side effects, easy to unit-test.

    Staircase (``throttle_allowed_pct`` is the fraction of orders ALLOWED):
        dd ≤ kill (-18%) → 0.0  (block all)
        dd ≤ hard (-12%) → 0.2  (20% allowed)
        dd ≤ soft (-8%)  → 0.5  (50% allowed)
    """
    dd_cfg = (policy or {}).get("drawdown_policy") or {}
    if not dd_cfg.get("auto_kill_enabled", True):
        return None

    current_equity = getattr(ctx, "current_equity", None)
    peak_equity = getattr(ctx, "peak_equity", None) or getattr(ctx, "hwm_equity", None)

    dd: float | None = None
    if current_equity is not None and peak_equity and peak_equity > 0:
        dd = float((current_equity - peak_equity) / peak_equity)
    else:
        _dd_meta = result.meta.get("drawdown_pct") if result is not None else None
        if _dd_meta is not None:
            dd = float(_dd_meta)

    if dd is None or dd >= 0:
        return None

    levels = dd_cfg.get("levels") or {"soft": -0.08, "hard": -0.12, "kill": -0.18}
    soft = float(levels.get("soft", -0.08))
    hard = float(levels.get("hard", -0.12))
    kill = float(levels.get("kill", -0.18))

    if dd <= kill:
        level_name = "kill"
        throttle = 0.0
    elif dd <= hard:
        level_name = "hard"
        throttle = 0.2
    elif dd <= soft:
        level_name = "soft"
        throttle = 0.5
    else:
        return None

    return {
        "level": level_name,
        "drawdown": dd,
        "throttle_allowed_pct": throttle,
        "reason": (
            f"auto_dd_{level_name}: drawdown={dd:.2%} "
            f"(soft={soft:.0%}, hard={hard:.0%}, kill={kill:.0%})"
        ),
    }


def _apply_pre_trade_impact(
    orders: pd.DataFrame,
    prices_filtered: pd.DataFrame | None,
    impact_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Annotate each order with pre-trade implementation-shortfall estimate.

    Sprint 2 / C10. Pure helper (no ctx, no side effects on caller globals)
    so it can be unit-tested in isolation.

    Returns a modified copy of ``orders`` with a new ``expected_impact_bps``
    column, plus a metadata dict with aggregate statistics. Orders whose
    estimated cost exceeds ``max_total_cost_bps`` are scaled down
    proportionally (``qty *= max_bps / total_cost_bps``).
    """
    import numpy as np

    from src.assembled_core.execution.algo_execution import (
        ImplementationShortfallModel,
    )

    model = ImplementationShortfallModel(
        kyle_lambda=float(impact_cfg.get("kyle_lambda", 0.1)),
        timing_risk_pct=float(impact_cfg.get("timing_risk_pct", 0.5)),
        opportunity_cost_bps=float(impact_cfg.get("opportunity_cost_bps", 5.0)),
    )
    max_bps = float(impact_cfg.get("max_total_cost_bps", 50.0))
    adv_window = int(impact_cfg.get("adv_window", 60))

    adv_map: dict[str, float] = {}
    vol_map: dict[str, float] = {}
    pf = prices_filtered
    if (
        pf is not None
        and not pf.empty
        and {"symbol", "volume", "close"}.issubset(pf.columns)
    ):
        for sym, grp in pf.groupby("symbol"):
            grp_sorted = (
                grp.sort_values("timestamp") if "timestamp" in grp.columns else grp
            )
            tail = grp_sorted.tail(adv_window)
            if tail.empty:
                continue
            sym_key = str(sym).upper()
            adv_map[sym_key] = float(tail["volume"].mean())
            closes = tail["close"].astype(float)
            if len(closes) >= 5:
                rets = np.log(closes / closes.shift(1)).dropna()
                vol_map[sym_key] = float(rets.std()) if len(rets) > 0 else 0.0

    new_orders = orders.copy()
    if "expected_impact_bps" not in new_orders.columns:
        new_orders["expected_impact_bps"] = 0.0

    estimates: list[float] = []
    scaled_symbols: list[str] = []
    for idx, order in new_orders.iterrows():
        sym = str(order["symbol"]).upper()
        qty_signed = float(order.get("qty", 0.0))
        qty_abs = abs(qty_signed)
        price = float(order.get("price", 0.0) or 0.0)
        adv = adv_map.get(sym, 0.0)
        d_vol = vol_map.get(sym, 0.0)
        est = model.estimate_cost(
            quantity=qty_abs,
            adv=adv,
            daily_vol=d_vol,
            price=price,
            execution_days=1.0,
        )
        bps = float(est["total_cost_bps"])
        new_orders.at[idx, "expected_impact_bps"] = bps
        estimates.append(bps)
        if bps > max_bps and qty_abs > 0:
            scale = max_bps / bps
            new_orders.at[idx, "qty"] = qty_signed * scale
            scaled_symbols.append(sym)

    meta = {
        "n_orders": len(estimates),
        "avg_bps": float(np.mean(estimates)) if estimates else 0.0,
        "max_bps": float(np.max(estimates)) if estimates else 0.0,
        "scaled_symbols": scaled_symbols,
        "max_total_cost_bps": max_bps,
    }
    return new_orders, meta


def _apply_group_exposure_caps(
    orders: pd.DataFrame,
    security_meta_df: pd.DataFrame | None,
    group_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """Scale orders down to respect per-sector/region/currency gross caps.

    Sprint 2 / W3. Pure helper. For each group dimension configured
    (``max_sector_gross``, ``max_region_gross``, ``max_currency_gross``),
    aggregates absolute notional per group from buys-as-new-exposure
    (sign-aware), then proportionally scales symbols in over-cap groups.

    Caps are expressed as fractions of the order-book total gross notional
    (self-normalizing, does NOT require ledger equity). This keeps the
    helper independent and unit-testable; downstream risk-control still
    enforces absolute exposure against the real ledger.
    """
    import numpy as np

    if orders is None or orders.empty or security_meta_df is None or security_meta_df.empty:
        return orders, {"scaled_groups": [], "n_orders": 0 if orders is None else len(orders)}

    group_caps = {
        "sector": float(group_cfg.get("max_sector_gross", 0.0) or 0.0),
        "region": float(group_cfg.get("max_region_gross", 0.0) or 0.0),
        "currency": float(group_cfg.get("max_currency_gross", 0.0) or 0.0),
    }
    # Keep only dims with meaningful cap (>0) that exist in meta
    active_dims = [
        dim for dim, cap in group_caps.items()
        if cap > 0 and dim in security_meta_df.columns
    ]
    if not active_dims:
        return orders, {"scaled_groups": [], "n_orders": len(orders)}

    out = orders.copy()
    meta_slim = security_meta_df[["symbol", *active_dims]].drop_duplicates("symbol")
    out = out.merge(meta_slim, on="symbol", how="left")

    # notional per row (abs, for gross)
    qty = out["qty"].astype(float).values
    price = out["price"].astype(float).values if "price" in out.columns else np.ones_like(qty)
    gross = np.abs(qty * price)
    total_gross = float(gross.sum())
    if total_gross <= 0:
        out = out.drop(columns=active_dims, errors="ignore")
        return out, {"scaled_groups": [], "n_orders": len(out)}

    scale_factors = np.ones(len(out), dtype=np.float64)
    scaled_groups: list[dict] = []

    for dim in active_dims:
        cap = group_caps[dim]
        groups = out[dim].fillna("UNKNOWN").values
        # fraction per group vs total_gross
        for grp in set(groups):
            mask = groups == grp
            grp_gross = float(gross[mask].sum())
            grp_frac = grp_gross / total_gross if total_gross > 0 else 0.0
            if grp_frac > cap and grp_frac > 0:
                factor = cap / grp_frac
                scale_factors[mask] = np.minimum(scale_factors[mask], factor)
                scaled_groups.append({
                    "dim": dim,
                    "group": str(grp),
                    "fraction": round(grp_frac, 6),
                    "cap": cap,
                    "scale": round(factor, 6),
                })

    out["qty"] = (out["qty"].astype(float).values * scale_factors)
    out = out.drop(columns=active_dims, errors="ignore")

    meta = {
        "n_orders": len(out),
        "active_dims": active_dims,
        "scaled_groups": scaled_groups,
        "total_gross": round(total_gross, 2),
    }
    return out, meta


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

        # Get current_equity and peak_equity if available (for drawdown de-risking)
        current_equity = getattr(ctx, "current_equity", None)
        peak_equity = getattr(ctx, "peak_equity", None)

        # Prefer live MTM equity over initial capital so the gross-exposure cap
        # shrinks under drawdown instead of staying constant at initial-capital
        # notional. Fall back to ctx.capital on first bar / bootstrap when
        # current_equity is not yet populated. (Follow-up to P0 A6.)
        if current_equity is not None and current_equity > 0:
            equity = float(current_equity)
        else:
            equity = ctx.capital

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
                _cg = _rl.get("concentration_guard") or {}
                # P0 A6 (Deep Run v2, 2026-04-18): gross-exposure cap.
                # Prefer `risk_limits.max_gross_exposure` (general portfolio cap,
                # enforced regardless of shorts enablement). Fall back to legacy
                # `shorts.max_gross_exposure`. When both are set, take the min so
                # neither is loosened by the other.
                # Policy value is a ratio (1.0 = 100% of equity). PreTradeConfig.
                # max_gross_exposure expects a raw notional, so multiply by
                # `equity` (defined above). If equity is not positive, skip the
                # cap (the gate will no-op — acceptable for bootstrap runs).
                _rl_gross = _rl.get("max_gross_exposure")
                _sh_gross = _pol.get("shorts", {}).get("max_gross_exposure")
                if _rl_gross is not None and _sh_gross is not None:
                    _gross_ratio = min(_rl_gross, _sh_gross)
                else:
                    _gross_ratio = _rl_gross if _rl_gross is not None else _sh_gross
                if _gross_ratio is not None and equity is not None and equity > 0:
                    _gross_cap_notional = float(_gross_ratio) * float(equity)
                else:
                    _gross_cap_notional = None
                _policy_defaults = {
                    "max_weight_per_symbol": _rl.get("max_position_weight"),
                    "drawdown_threshold": _dd.get("kill"),
                    "turnover_cap": _tv.get("daily_cap"),
                    "max_sector_exposure": _cg.get("max_sector_weight"),
                    "max_gross_exposure": _gross_cap_notional,
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
                    max_gross_exposure=(
                        ctx.risk_config.get("max_gross_exposure")
                        or _policy_defaults.get("max_gross_exposure")
                    ),
                    max_sector_exposure=(
                        ctx.risk_config.get("max_sector_exposure")
                        or _policy_defaults.get("max_sector_exposure")
                    ),
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
        # CRITICAL: risk controls failed -- block ALL orders for safety.
        # Never pass through unfiltered orders when risk checks are broken.
        log = ctx.logger if ctx.logger is not None else logger
        log.critical(
            "[RISK-SAFETY] Risk controls raised %s: %s. "
            "BLOCKING all %d orders. Fix the risk module before trading.",
            type(e).__name__, e, len(orders),
        )
        return []


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

    # E0.1 parity: in backtest mode the kill-switch can either persist across
    # bars (new default — matches paper/live) or be restored per-bar (legacy
    # behavior, required only for fast research runs that should not be
    # gated by a single bar's circuit-breaker trip).
    #
    # Opt-out flag: ``ctx.kill_switch_persist=False`` restores the old
    # "reset-after-each-bar" behavior. Default True so backtest and paper
    # share the same decision logic.
    _ks_state_backup: bool | None = None
    _is_backtest = getattr(ctx, "mode", None) in ("backtest", "bt")
    _ks_persist = bool(getattr(ctx, "kill_switch_persist", True))
    _ks_restore_active = _is_backtest and not _ks_persist
    if _ks_restore_active:
        try:
            from src.assembled_core.execution.kill_switch import is_kill_switch_engaged
            _ks_state_backup = is_kill_switch_engaged()
        except Exception as _ks_err:
            log.warning("[KS-BACKUP] kill-switch state snapshot failed: %s", _ks_err)

    try:
        return _run_trading_cycle_inner(ctx, hooks=hooks, log=log)
    finally:
        if (
            _ks_restore_active
            and _ks_state_backup is not None
            and not _ks_state_backup
        ):
            try:
                from src.assembled_core.execution.kill_switch import (
                    deactivate_kill_switch,
                    is_kill_switch_engaged,
                )
                if is_kill_switch_engaged():
                    deactivate_kill_switch(
                        reason="backtest_bar_restore",
                        actor="trading_cycle_backtest_guard",
                    )
            except Exception as _ks_err:
                log.warning("[KS-RESTORE] kill-switch state restore failed: %s", _ks_err)


def _run_trading_cycle_inner(
    ctx: "TradingContext",
    *,
    hooks: "dict[str, Callable] | None" = None,
    log: "logging.Logger | None" = None,
) -> "TradingCycleResult":
    """Inner implementation of run_trading_cycle (extracted to support backtest KS guard)."""
    if log is None:
        log = logger

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

        # Phase 5.2 (Sprint 2 / W9): Market-breadth snapshot
        # Computes fraction-above-MA at the latest bar and publishes into
        # result.meta for downstream strategies / reporting. Opt-in, cheap,
        # never raises into the cycle.
        try:
            mb_cfg = policy.get("market_breadth") or {}
            if mb_cfg.get("enabled", False) and ctx.prices is not None and not ctx.prices.empty:
                from src.assembled_core.features.market_breadth import (
                    compute_market_breadth_ma,
                )

                ma_window = int(mb_cfg.get("ma_window", 50))
                breadth_df = compute_market_breadth_ma(ctx.prices, ma_window=ma_window)
                if not breadth_df.empty:
                    last_row = breadth_df.iloc[-1]
                    frac_col = f"fraction_above_ma_{ma_window}"
                    frac = float(last_row.get(frac_col, 0.0) or 0.0)
                    if frac >= 0.7:
                        regime = "strong"
                    elif frac >= 0.5:
                        regime = "neutral"
                    elif frac >= 0.3:
                        regime = "weak"
                    else:
                        regime = "narrow"
                    result.meta["market_breadth"] = {
                        "ma_window": ma_window,
                        "fraction_above_ma": round(frac, 4),
                        "count_above_ma": int(last_row.get("count_above_ma", 0) or 0),
                        "count_total": int(last_row.get("count_total", 0) or 0),
                        "regime": regime,
                    }
        except Exception as e:
            logger.debug("market_breadth snapshot skipped: %s", e)

        # Phase 5.5: Sprint 1 / W4b — Daily Circuit Breaker
        # Stateless close-to-close check against a market reference (SPY).
        # A trip engages the kill-switch with 100 % throttle (block all) and
        # marks the cycle so downstream phases know trading is halted.
        try:
            cb_trip = _evaluate_circuit_breaker_daily(ctx.prices, policy, ctx.as_of)
            if cb_trip is not None:
                from src.assembled_core.execution.kill_switch import (
                    activate_kill_switch,
                )
                activate_kill_switch(
                    throttle_pct=0.0,
                    reason=cb_trip["reason"],
                    actor="trading_cycle_circuit_breaker",
                )
                result.meta["circuit_breaker"] = cb_trip
                logger.critical(
                    "CIRCUIT_BREAKER: %s — kill-switch engaged (block all)",
                    cb_trip["reason"],
                )
        except Exception as e:
            logger.warning("[RISK-SAFETY] circuit_breaker_daily check failed: %s — breaker may not engage", e)

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

    # Step 1.8: Data versioning — hash prices_filtered for lineage + reproducibility
    try:
        from src.assembled_core.data.data_versioning import (
            compute_data_hash,
            create_lineage_record,
        )
        _price_hash = compute_data_hash(
            result.prices_filtered,
            columns=["symbol", "timestamp", "close"],
        )
        result.meta["data_lineage"] = create_lineage_record(
            data_hash=_price_hash,
            source=getattr(ctx, "data_source", "unknown"),
            n_rows=len(result.prices_filtered),
            n_symbols=int(result.prices_filtered["symbol"].nunique()) if "symbol" in result.prices_filtered.columns else 0,
            date_range=(
                f"{result.prices_filtered['timestamp'].min()} – {result.prices_filtered['timestamp'].max()}"
                if "timestamp" in result.prices_filtered.columns and not result.prices_filtered.empty
                else ""
            ),
        )
        log.debug("[DATA_LINEAGE] hash=%s rows=%d", _price_hash[:12], len(result.prices_filtered))
    except Exception as _dv_exc:
        log.debug("[DATA_LINEAGE] data_versioning skipped: %s", _dv_exc)

    # Step 1.9: Price quality check (data_quality)
    try:
        dq_cfg = policy.get("freshness_monitor") or {}
        if dq_cfg.get("enabled", False) and not result.prices_filtered.empty:
            from src.assembled_core.data.quality_checks import check_panel_quality
            _qc_results = check_panel_quality(result.prices_filtered)
            _qc_failed = [r for r in _qc_results if not r.passed]
            result.meta["price_quality_check"] = {
                "symbols_checked": len(_qc_results),
                "symbols_failed": len(_qc_failed),
                "failed_symbols": [r.symbol for r in _qc_failed[:10]],
            }
            if _qc_failed:
                log.warning(
                    "[DATA_QUALITY] %d/%d symbols failed price quality checks: %s",
                    len(_qc_failed),
                    len(_qc_results),
                    [r.symbol for r in _qc_failed[:5]],
                )
    except Exception as _dq_exc:
        log.debug("[DATA_QUALITY] price quality check skipped: %s", _dq_exc)

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

    # Step 2.2: Enhanced feature enrichment (ta_factors_core + cross-sectional normalization)
    try:
        enh_cfg = (policy.get("features") or {}).get("enhanced_factors") or {}
        if enh_cfg.get("enabled", False) and not result.prices_with_features.empty:
            # ta_factors_core: multi-horizon returns, trend strength, short-term reversal
            if enh_cfg.get("ta_factors_core", True):
                from src.assembled_core.features.ta_factors_core import build_core_ta_factors
                n_before = len(result.prices_with_features.columns)
                result.prices_with_features = build_core_ta_factors(
                    result.prices_with_features,
                    price_col="close",
                    group_col="symbol",
                    timestamp_col="timestamp",
                )
                n_added = len(result.prices_with_features.columns) - n_before
                log.info("[FEATURE-ENH] ta_factors_core: +%d columns", n_added)
            # cross_sectional: rank-normalize key signal features across symbols per day
            if enh_cfg.get("cross_sectional_rank", True):
                from src.assembled_core.features.cross_sectional import rank_cross_sectional
                rank_cols = [
                    c for c in enh_cfg.get("rank_cols", [
                        "trend_ema_spread", "mom_rsi_centered", "mom_12_1",
                        "low_vol_rank", "quality_score",
                        "trend_strength_20", "trend_strength_50", "momentum_12m_excl_1m",
                    ])
                    if c in result.prices_with_features.columns
                ]
                if rank_cols:
                    result.prices_with_features = rank_cross_sectional(
                        result.prices_with_features,
                        feature_cols=rank_cols,
                        timestamp_col="timestamp",
                        normalize_to=enh_cfg.get("rank_normalize_to", "symmetric"),
                    )
                    log.info(
                        "[FEATURE-ENH] cross_sectional_rank: %d features → _xrank columns",
                        len(rank_cols),
                    )
    except Exception as _enh_exc:
        log.debug("[FEATURE-ENH] enhanced feature enrichment skipped: %s", _enh_exc)

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

    # Step 2.3: Stale-feature detection — flag features constant for N days (data-feed outage proxy)
    try:
        fm_cfg = policy.get("freshness_monitor") or {}
        if fm_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.data.freshness_monitor import detect_stale_features
            _feat_cols = [
                c for c in result.prices_with_features.columns
                if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
                and result.prices_with_features[c].dtype in ("float64", "float32")
            ][:30]  # cap columns to keep it fast
            if _feat_cols:
                _stale = detect_stale_features(
                    result.prices_with_features,
                    _feat_cols,
                    stale_days=int(fm_cfg.get("stale_feature_days", 5)),
                )
                result.meta["freshness_check"] = {
                    "stale_count": len(_stale),
                    "stale_items": _stale[:10],
                }
                if _stale:
                    log.warning(
                        "[FRESHNESS] %d stale feature/symbol pairs detected",
                        len(_stale),
                    )
    except Exception as _fm_exc:
        log.debug("[FRESHNESS] freshness_monitor skipped: %s", _fm_exc)

    # Step 2.4: Feature drift detection (KS-test, shadow observability)
    try:
        drift_cfg = (policy.get("ml") or {}).get("drift_detection") or {}
        if drift_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.ml.model_monitoring import detect_feature_drift
            _feat_cols_drift = [
                c for c in result.prices_with_features.columns
                if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
                and result.prices_with_features[c].dtype in ("float64", "float32")
            ][:20]
            if _feat_cols_drift and len(result.prices_with_features) >= 40:
                _n_recent = min(30, len(result.prices_with_features) // 5)
                _train_df = result.prices_with_features.iloc[:-_n_recent]
                _recent_df = result.prices_with_features.iloc[-_n_recent:]
                _drift_result = detect_feature_drift(
                    _train_df,
                    _recent_df,
                    _feat_cols_drift,
                    p_value_threshold=float(drift_cfg.get("p_value_threshold", 0.01)),
                )
                result.meta["feature_drift"] = _drift_result
                if _drift_result.get("alert_level") in ("CRITICAL", "WARNING"):
                    log.warning(
                        "[DRIFT] %s — %.0f%% features drifted (%d/%d)",
                        _drift_result["alert_level"],
                        _drift_result.get("drift_score", 0) * 100,
                        len(_drift_result.get("drifted_features", [])),
                        _drift_result.get("n_tested", 0),
                    )
    except Exception as _dr_exc:
        log.debug("[DRIFT] feature_drift detection skipped: %s", _dr_exc)

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
                # [D2] Shadow-mode observation window (Ultra-Plan Part D).
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )

                zk_shadow = is_shadow_only(policy, "zombie_killer")
                zombie_symbols = {pos["symbol"] for pos, _reason in zombies}
                record_shadow(
                    "zombie_killer",
                    {
                        "zombie_symbols": sorted(zombie_symbols),
                        "would_force_flat": sorted(zombie_symbols),
                    },
                    as_of=str(ctx.as_of) if ctx.as_of is not None else None,
                    meta={
                        "zombies_found": len(zombies),
                        "reasons": [r for _, r in zombies],
                        "applied": not zk_shadow,
                    },
                )
                for _pos, reason in zombies:
                    log.warning(reason)
                if not zk_shadow:
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
                    "shadow_only": zk_shadow,
                }
                log.info(
                    "ZOMBIE_KILLER: %d positions flagged for exit (shadow=%s): %s",
                    len(zombies), zk_shadow, sorted(zombie_symbols),
                )
    except Exception as e:
        log.debug("zombie_killer check skipped: %s", e)

    # Step 3.1: Intel signal layer (Phase 9 / Plan 1.5)
    # Gated by policy intel.signal_layer.enabled — skip cleanly if missing
    try:
        intel_sig_cfg = (policy.get("intel") or {}).get("signal_layer") or {}
        if intel_sig_cfg.get("enabled", False) and not result.signals.empty:
            from src.assembled_core.signals.intel_signal_adapter import (
                IntelSignalAdapter,
                compute_symbol_intel_scores,
            )
            # Gather intel dimensions from ctx if wired
            sector_impacts = getattr(ctx, "intel_sector_impacts", None)
            supply_vuln = getattr(ctx, "intel_supply_vulnerability", None)
            sanctions_ben = getattr(ctx, "intel_sanctions_beneficiary", None)
            chokepoint_exp = getattr(ctx, "intel_chokepoint_exposure", None)
            intel_conf = getattr(ctx, "intel_confidence", None)

            if any(x is not None for x in [sector_impacts, supply_vuln, sanctions_ben, chokepoint_exp]):
                raw_scores = compute_symbol_intel_scores(
                    sector_impacts=sector_impacts,
                    supply_chain_vulnerability=supply_vuln,
                    sanctions_beneficiary=sanctions_ben,
                    chokepoint_exposure=chokepoint_exp,
                    confidence=intel_conf,
                )
                if raw_scores and "score" in result.signals.columns:
                    intel_weight = float(intel_sig_cfg.get("weight", 0.15))
                    for idx, row in result.signals.iterrows():
                        sym = row.get("symbol", "")
                        if sym in raw_scores:
                            result.signals.at[idx, "score"] = (
                                float(row["score"]) + intel_weight * raw_scores[sym]
                            )
                    result.meta["intel_signal_layer"] = {
                        "n_symbols": len(raw_scores),
                        "weight": intel_weight,
                    }
                    log.info("[INTEL] signal layer applied: %d symbols scored", len(raw_scores))

            # Also wire shock beneficiaries if active shocks in context
            active_shocks = getattr(ctx, "intel_active_shocks", None)
            if active_shocks:
                adapter = IntelSignalAdapter(
                    allow_short_signals=intel_sig_cfg.get("allow_short", False),
                    min_confidence=float(intel_sig_cfg.get("min_confidence", 0.50)),
                )
                shock_df = adapter.enrich_signals_with_shock_beneficiaries(
                    active_shocks,
                    base_confidence=float(intel_sig_cfg.get("shock_confidence", 0.60)),
                )
                if not shock_df.empty:
                    # Add shock-derived symbols not already in signals
                    existing_syms = set(result.signals["symbol"].values)
                    new_shock = shock_df[~shock_df["symbol"].isin(existing_syms)].copy()
                    if not new_shock.empty:
                        new_shock["timestamp"] = ctx.as_of or pd.Timestamp.now("UTC")
                        new_shock["direction"] = "LONG"
                        result.signals = pd.concat([result.signals, new_shock[["timestamp", "symbol", "direction", "score"]]], ignore_index=True)
                        log.info("[INTEL] %d shock beneficiary signals added", len(new_shock))
    except Exception as _e:
        log.debug("[INTEL] intel_signal_layer skipped: %s", _e)

    # Step 3.2: Sector rotation signals (Phase 9)
    # Gated by policy signal_generation.sector_rotation.enabled
    try:
        sr_cfg = (policy.get("signal_generation") or {}).get("sector_rotation") or {}
        if sr_cfg.get("enabled", False):
            from src.assembled_core.signals.sector_rotation import (
                generate_sector_rotation_signals,
                get_sector_weights,
            )
            # Build a scores_row from available prices or context
            scores_row = getattr(ctx, "sector_rotation_scores", None)
            if scores_row is not None:
                sr_signals = generate_sector_rotation_signals(scores_row)
                sr_weights = get_sector_weights(
                    sr_signals,
                    long_weight=float(sr_cfg.get("long_weight", 0.12)),
                    short_weight=float(sr_cfg.get("short_weight", 0.08)),
                )
                if sr_weights:
                    ts_now = ctx.as_of or pd.Timestamp.now("UTC")
                    existing_syms = set(result.signals["symbol"].values) if not result.signals.empty else set()
                    sr_rows = []
                    for sym, w in sr_weights.items():
                        if sym not in existing_syms:
                            sr_rows.append({
                                "timestamp": ts_now,
                                "symbol": sym,
                                "direction": "LONG" if w > 0 else "SHORT",
                                "score": round(w, 4),
                            })
                    if sr_rows:
                        result.signals = pd.concat(
                            [result.signals, pd.DataFrame(sr_rows)], ignore_index=True
                        )
                    result.meta["sector_rotation"] = {
                        "longs": sr_signals.longs,
                        "shorts": sr_signals.shorts,
                        "is_risk_off": sr_signals.is_risk_off,
                        "negative_count": sr_signals.negative_count,
                    }
                    log.info(
                        "[SIGNAL-DIAG] sector_rotation: longs=%s shorts=%s risk_off=%s",
                        sr_signals.longs, sr_signals.shorts, sr_signals.is_risk_off,
                    )
    except Exception as _e:
        log.debug("[SIGNAL-DIAG] sector_rotation skipped: %s", _e)

    # Step 3.3: Earnings guard — suppress signals pre-earnings (Phase 9)
    # Gated by policy signal_generation.earnings_guard.enabled
    try:
        eg_cfg = (policy.get("signal_generation") or {}).get("earnings_guard") or {}
        if eg_cfg.get("enabled", False) and not result.signals.empty:
            from src.assembled_core.signals.earnings_integration import apply_earnings_integration

            earnings_calendar = getattr(ctx, "earnings_calendar", None)
            earnings_events = getattr(ctx, "earnings_events", None)
            as_of_for_earnings = ctx.as_of or pd.Timestamp.now("UTC")

            if earnings_calendar is not None or earnings_events is not None:
                adjusted_signals, earnings_result = apply_earnings_integration(
                    result.signals,
                    earnings_calendar=earnings_calendar,
                    earnings_events=earnings_events,
                    as_of=as_of_for_earnings,
                    suppress_window=int(eg_cfg.get("suppress_window", 3)),
                    pead_window_days=int(eg_cfg.get("pead_window_days", 60)),
                    pead_weight=float(eg_cfg.get("pead_weight", 0.15)),
                )
                result.signals = adjusted_signals
                result.meta["earnings_guard"] = {
                    "suppressed": earnings_result.suppressed_symbols,
                    "pead_count": len(earnings_result.pead_signals),
                    "concentration_warning": earnings_result.concentration_warning,
                    "pct_near_earnings": earnings_result.pct_near_earnings,
                }
                if earnings_result.suppressed_symbols:
                    log.info(
                        "[SIGNAL-DIAG] earnings_guard: suppressed %d symbols %s",
                        len(earnings_result.suppressed_symbols),
                        earnings_result.suppressed_symbols,
                    )
    except Exception as _e:
        log.debug("[SIGNAL-DIAG] earnings_guard skipped: %s", _e)

    # Step 3.35: News→Signal bridge (Part B deeper wiring)
    # Gated by policy intel.news_signal_bridge.enabled
    try:
        from src.assembled_core.signals.news_signal_bridge import (
            load_and_apply_news_signals,
        )
        root_for_news = Path(ctx.data_root) if getattr(ctx, "data_root", None) else Path.cwd()
        result.signals, news_bridge_meta = load_and_apply_news_signals(
            result.signals,
            root=root_for_news,
            policy=policy,
            as_of=ctx.as_of,
        )
        if news_bridge_meta.get("enabled"):
            result.meta["news_signal_bridge"] = news_bridge_meta
            if news_bridge_meta.get("applied", 0) or news_bridge_meta.get("added", 0):
                log.info(
                    "[SIGNAL-DIAG] news_signal_bridge: applied=%d added=%d |Δ|=%.2f",
                    news_bridge_meta.get("applied", 0),
                    news_bridge_meta.get("added", 0),
                    news_bridge_meta.get("total_delta_abs", 0.0),
                )
    except Exception as _e:
        log.debug("[SIGNAL-DIAG] news_signal_bridge skipped: %s", _e)

    # Step 3.4: Bayesian signal confidence scoring (Phase 9 / Plan 1.9)
    # Applied when policy signal_generation.bayesian_confidence.enabled is true
    try:
        bc_cfg = (policy.get("signal_generation") or {}).get("bayesian_confidence") or {}
        if bc_cfg.get("enabled", False) and not result.signals.empty and "score" in result.signals.columns:
            from src.assembled_core.signals.signal_confidence import (
                compute_signal_confidence,
                confidence_position_scaler,
            )
            current_scores = result.signals.set_index("symbol")["score"].dropna()
            if len(current_scores) >= 2:
                historical_scores = getattr(ctx, "signal_historical_scores", None)
                confidences = compute_signal_confidence(
                    current_scores,
                    historical_scores=historical_scores,
                    ci_level=float(bc_cfg.get("ci_level", 0.90)),
                )
                # Scale scores by confidence width (narrow CI = more confident = larger position)
                for idx, row in result.signals.iterrows():
                    sym = row.get("symbol", "")
                    if sym in confidences:
                        scaler = confidence_position_scaler(
                            confidences[sym],
                            max_scale=float(bc_cfg.get("max_scale", 1.5)),
                            min_scale=float(bc_cfg.get("min_scale", 0.5)),
                        )
                        result.signals.at[idx, "score"] = float(row["score"]) * scaler
                avg_width = sum(c.confidence_width for c in confidences.values()) / max(len(confidences), 1)
                result.meta["bayesian_confidence"] = {
                    "n_symbols": len(confidences),
                    "avg_ci_width": round(avg_width, 4),
                }
                log.info("[SIGNAL-DIAG] bayesian_confidence: %d symbols, avg_ci_width=%.4f", len(confidences), avg_width)
    except Exception as _e:
        log.debug("[SIGNAL-DIAG] bayesian_confidence skipped: %s", _e)

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

    # Step 3.6: Ranking hysteresis (anti-churn) — reduces symbol rotation noise
    try:
        anti_churn_cfg = policy.get("anti_churn") or {}
        if anti_churn_cfg.get("ranking_hysteresis_enabled", False) and not result.signals.empty:
            from src.assembled_core.paper.ranking_hysteresis import apply_ranking_hysteresis
            held_symbols: set[str] = set()
            if (
                ctx.current_positions is not None
                and not ctx.current_positions.empty
                and "symbol" in ctx.current_positions.columns
            ):
                held_symbols = set(ctx.current_positions["symbol"].tolist())
            result.signals, _rh_meta = apply_ranking_hysteresis(
                result.signals,
                held_symbols,
                entry_n=int(anti_churn_cfg.get("entry_n", 5)),
                hold_n=int(anti_churn_cfg.get("hold_n", 7)),
            )
            result.meta["ranking_hysteresis"] = _rh_meta
            log.info(
                "[ANTI-CHURN] ranking_hysteresis: kept=%d blocked_entry=%d held=%d",
                _rh_meta.get("kept_by_hysteresis", 0),
                _rh_meta.get("blocked_entry", 0),
                len(held_symbols),
            )
    except Exception as _rh_exc:
        log.debug("[ANTI-CHURN] ranking_hysteresis skipped: %s", _rh_exc)

    # Step 3.7: Meta-model confidence scoring (shadow — needs pre-trained model)
    try:
        mm_cfg = (policy.get("signals") or {}).get("meta_model_scoring") or {}
        if mm_cfg.get("enabled", False) and not result.signals.empty:
            from src.assembled_core.signals.meta_model import load_meta_model
            import pathlib as _pathlib
            _mm_path = _pathlib.Path(mm_cfg.get("model_path", "output/models/meta/meta_model.joblib"))
            if _mm_path.exists():
                _meta_model = load_meta_model(_mm_path)
                _score_cols = [c for c in _meta_model.feature_names if c in result.signals.columns]
                if _score_cols:
                    _X_mm = result.signals[_score_cols].fillna(0.0)
                    _conf_scores = _meta_model.predict_proba(_X_mm)
                    result.signals = result.signals.copy()
                    result.signals["meta_confidence"] = _conf_scores.values
                    result.meta["meta_model_scores"] = {
                        "model_path": str(_mm_path),
                        "n_scored": int(len(_conf_scores)),
                        "mean_confidence": round(float(_conf_scores.mean()), 4),
                        "shadow_only": True,
                    }
                    log.info(
                        "[META-MODEL] scored %d signals, mean_confidence=%.4f",
                        len(_conf_scores),
                        float(_conf_scores.mean()),
                    )
            else:
                log.debug("[META-MODEL] model not found at %s — skipping", _mm_path)
    except Exception as _mm_exc:
        log.debug("[META-MODEL] meta_model scoring skipped: %s", _mm_exc)

    # Step 4: Size positions (hook point: size_positions)
    try:
        if "size_positions" in hooks:
            result.target_positions = hooks["size_positions"](ctx, result.signals)
        else:
            # Policy-driven sizing method dispatch
            sizing_cfg = policy.get("position_sizing") or {}
            sizing_method = sizing_cfg.get("method", "default")
            # Policy-driven sizing takes precedence over caller fn.
            # Only fall back to caller fn when policy says "default".

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
                    from src.assembled_core.portfolio.covariance import estimate_covariance

                    prices_for_bl = result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    bl = BlackLittermanOptimizer(
                        risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)),
                        tau=float(sizing_cfg.get("tau", 0.05)),
                        max_position=float(sizing_cfg.get("max_weight", 0.15)),
                        min_position=float(sizing_cfg.get("min_position", 0.0)),
                    )
                    # Build scores from signals
                    scores_dict: dict[str, float] = {}
                    if not result.signals.empty and "symbol" in result.signals.columns:
                        for _, row in result.signals.iterrows():
                            sym = row["symbol"]
                            score = float(row.get("score", 0.0))
                            if abs(score) > 0.01:
                                scores_dict[sym] = score
                    if scores_dict and prices_for_bl is not None and not prices_for_bl.empty:
                        # Compute covariance matrix using Ledoit-Wolf shrinkage
                        _ts_col = "timestamp" if "timestamp" in prices_for_bl.columns else prices_for_bl.columns[0]
                        _sym_col = "symbol" if "symbol" in prices_for_bl.columns else None
                        if _sym_col and "close" in prices_for_bl.columns:
                            _pivot = prices_for_bl.pivot_table(
                                index=_ts_col, columns=_sym_col, values="close"
                            )
                            _returns = _pivot.pct_change().dropna(how="all")
                            _cov_method = sizing_cfg.get("cov_method", "ledoit_wolf")
                            sigma = estimate_covariance(_returns, method=_cov_method)
                        else:
                            sigma = pd.DataFrame()

                        if not sigma.empty:
                            scores_series = pd.Series(scores_dict)
                            bl_conf = float(sizing_cfg.get("bl_confidence", 0.5))
                            bl_weights = bl.optimize_from_scores(
                                scores=scores_series,
                                sigma=sigma,
                                confidence=bl_conf,
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
                            log.warning("BL: could not compute covariance — using default sizing")
                            result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    else:
                        # Fallback to default sizing if no views
                        result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_method"] = "black_litterman"
                except Exception as e:
                    log.warning("Black-Litterman sizing failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_fallback"] = True
                    result.meta["sizing_fallback_reason"] = str(e)
            elif sizing_method == "cost_aware":
                # [RISK-WIRE] Cost-aware optimizer (turnover-penalized MVO)
                try:
                    from src.assembled_core.portfolio.cost_aware_optimizer import (
                        OptimizerConfig,
                        optimize_portfolio,
                    )
                    from src.assembled_core.portfolio.covariance import estimate_covariance

                    prices_for_cao = (
                        result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    )
                    if (
                        prices_for_cao is not None
                        and not prices_for_cao.empty
                        and not result.signals.empty
                        and "symbol" in result.signals.columns
                    ):
                        _ts_col = "timestamp" if "timestamp" in prices_for_cao.columns else prices_for_cao.columns[0]
                        if "close" in prices_for_cao.columns and "symbol" in prices_for_cao.columns:
                            _pivot_cao = prices_for_cao.pivot_table(
                                index=_ts_col, columns="symbol", values="close"
                            )
                            _rets_cao = _pivot_cao.pct_change().dropna(how="all")
                            sigma_cao = estimate_covariance(_rets_cao, method="ledoit_wolf")
                            mu_cao = result.signals.set_index("symbol")["score"] if "score" in result.signals.columns else pd.Series(dtype=float)
                            mu_cao = mu_cao.reindex(sigma_cao.index).fillna(0.0)
                            cao_cfg = OptimizerConfig(
                                risk_aversion=float(sizing_cfg.get("risk_aversion", 1.0)),
                                turnover_penalty=float(sizing_cfg.get("turnover_penalty", 0.001)),
                                max_weight=float(sizing_cfg.get("max_weight", 0.10)),
                            )
                            # Build current weights inline (current_w computed later in cycle)
                            _cao_cur_w: dict[str, float] = {}
                            if hasattr(ctx, "current_positions") and ctx.current_positions is not None:
                                if isinstance(ctx.current_positions, dict):
                                    _cao_cur_w = ctx.current_positions
                                elif isinstance(ctx.current_positions, pd.DataFrame) and "symbol" in ctx.current_positions.columns:
                                    for _, _cao_row in ctx.current_positions.iterrows():
                                        _cao_cur_w[_cao_row["symbol"]] = float(
                                            _cao_row.get("weight", _cao_row.get("target_weight", 0.0))
                                        )
                            cao_result = optimize_portfolio(
                                expected_returns=mu_cao,
                                covariance=sigma_cao,
                                current_weights=_cao_cur_w,
                                config=cao_cfg,
                            )
                            rows = [
                                {
                                    "symbol": s,
                                    "target_weight": round(w, 4),
                                    "target_qty": round(w * ctx.capital, 2),
                                }
                                for s, w in cao_result.weights.items()
                                if abs(w) > 1e-6
                            ]
                            result.target_positions = pd.DataFrame(rows)
                            result.meta["cost_aware_optimizer"] = {
                                "method": cao_result.method,
                                "solver_status": cao_result.solver_status,
                                "turnover_cost": cao_result.turnover_cost,
                            }
                            log.info(
                                "[RISK-WIRE] cost_aware sizing: method=%s status=%s turnover_cost=%.6f",
                                cao_result.method,
                                cao_result.solver_status,
                                cao_result.turnover_cost,
                            )
                        else:
                            raise ValueError("cost_aware: missing close/symbol columns in prices")
                    else:
                        raise ValueError("cost_aware: insufficient data — falling back")
                except Exception as e:
                    log.warning("[RISK-WIRE] cost_aware_optimizer failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
            elif sizing_method == "erc":
                # [RISK-WIRE] Equal-Risk-Contribution (Maillard-Roncalli-Teïletche)
                try:
                    from src.assembled_core.portfolio.risk_budgeting import (
                        compute_erc_weights,
                    )
                    from src.assembled_core.portfolio.covariance import estimate_covariance

                    prices_for_erc = (
                        result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    )
                    if (
                        prices_for_erc is not None
                        and not prices_for_erc.empty
                        and not result.signals.empty
                        and "symbol" in result.signals.columns
                        and "close" in prices_for_erc.columns
                        and "symbol" in prices_for_erc.columns
                    ):
                        _ts_col_erc = "timestamp" if "timestamp" in prices_for_erc.columns else prices_for_erc.columns[0]
                        # Only ERC over symbols that both have a signal and appear in prices
                        _sig_syms = [
                            s for s in result.signals["symbol"].tolist()
                            if s in prices_for_erc["symbol"].unique()
                        ]
                        if len(_sig_syms) >= 2:
                            _pivot_erc = prices_for_erc[
                                prices_for_erc["symbol"].isin(_sig_syms)
                            ].pivot_table(
                                index=_ts_col_erc, columns="symbol", values="close"
                            )
                            _rets_erc = _pivot_erc.pct_change().dropna(how="all")
                            if len(_rets_erc) >= 3:
                                sigma_erc = estimate_covariance(_rets_erc, method="ledoit_wolf")
                                erc_res = compute_erc_weights(
                                    sigma_erc,
                                    symbols=list(sigma_erc.columns),
                                    long_only=True,
                                    max_weight=float(sizing_cfg.get("max_weight", 0.25)),
                                )
                                rows = [
                                    {
                                        "symbol": s,
                                        "target_weight": round(w, 6),
                                        "target_qty": round(w * ctx.capital, 2),
                                    }
                                    for s, w in erc_res.weights.items()
                                    if abs(w) > 1e-6
                                ]
                                result.target_positions = pd.DataFrame(rows)
                                result.meta["erc_sizing"] = {
                                    "method": erc_res.method,
                                    "converged": erc_res.converged,
                                    "max_rc_deviation": erc_res.max_rc_deviation,
                                    "portfolio_volatility": erc_res.portfolio_volatility,
                                }
                                log.info(
                                    "[RISK-WIRE] erc sizing: method=%s converged=%s max_rc_dev=%.6f",
                                    erc_res.method,
                                    erc_res.converged,
                                    erc_res.max_rc_deviation,
                                )
                            else:
                                raise ValueError("erc: insufficient return history (<3 bars)")
                        else:
                            raise ValueError("erc: fewer than 2 symbols overlap signals and prices")
                    else:
                        raise ValueError("erc: insufficient data — falling back")
                except Exception as e:
                    log.warning("[RISK-WIRE] erc sizing failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_fallback"] = True
                    result.meta["sizing_fallback_reason"] = str(e)
            elif sizing_method == "bl_blend":
                # [RISK-WIRE] Black-Litterman score-blend wrapper (apply_bl_sizing).
                # Distinct from existing "black_litterman" branch — this uses the
                # bl_sizing wrapper that blends BL posterior with score weights.
                try:
                    from src.assembled_core.portfolio.bl_sizing import apply_bl_sizing

                    base_tp = ctx.position_sizing_fn(result.signals, ctx.capital)
                    prices_for_bl = (
                        result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    )
                    if (
                        base_tp is not None
                        and not base_tp.empty
                        and "target_weight" in base_tp.columns
                        and prices_for_bl is not None
                        and not prices_for_bl.empty
                    ):
                        score_w = {
                            str(r["symbol"]): float(r["target_weight"])
                            for _, r in base_tp.iterrows()
                            if pd.notna(r.get("target_weight"))
                        }
                        bl_w, reasons = apply_bl_sizing(
                            score_w,
                            prices_for_bl,
                            lookback_days=int(sizing_cfg.get("lookback_days", 60)),
                            risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)),
                            tau=float(sizing_cfg.get("tau", 0.05)),
                            max_position=float(sizing_cfg.get("max_weight", 0.15)),
                            confidence=float(sizing_cfg.get("bl_confidence", 0.5)),
                            return_scale=float(sizing_cfg.get("return_scale", 0.10)),
                            target_invested_pct=float(sizing_cfg.get("target_invested_pct", 1.0)),
                        )
                        rows = [
                            {
                                "symbol": s,
                                "target_weight": round(w, 6),
                                "target_qty": round(w * ctx.capital, 2),
                            }
                            for s, w in bl_w.items()
                            if abs(w) > 1e-6
                        ]
                        result.target_positions = pd.DataFrame(rows)
                        result.meta["bl_blend_sizing"] = {
                            "reasons": reasons,
                            "n_symbols": len(bl_w),
                        }
                        log.info(
                            "[RISK-WIRE] bl_blend sizing: %d symbols (reasons=%d)",
                            len(bl_w), len(reasons),
                        )
                    else:
                        raise ValueError("bl_blend: insufficient data or no target_weight")
                except Exception as e:
                    log.warning("[RISK-WIRE] bl_blend sizing failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_fallback"] = True
                    result.meta["sizing_fallback_reason"] = str(e)
            elif sizing_method == "hrp":
                # [RISK-WIRE] Hierarchical Risk Parity blend on score-based sizing.
                try:
                    from src.assembled_core.portfolio.hrp_sizing import apply_hrp_sizing
                    base_tp = ctx.position_sizing_fn(result.signals, ctx.capital)
                    prices_for_hrp = (
                        result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    )
                    if (
                        base_tp is not None
                        and not base_tp.empty
                        and "target_weight" in base_tp.columns
                        and prices_for_hrp is not None
                        and not prices_for_hrp.empty
                    ):
                        score_w = {
                            str(r["symbol"]): float(r["target_weight"])
                            for _, r in base_tp.iterrows()
                            if pd.notna(r.get("target_weight"))
                        }
                        blended, reasons = apply_hrp_sizing(
                            score_w,
                            prices_for_hrp,
                            lookback_days=int(sizing_cfg.get("lookback_days", 60)),
                            blend=float(sizing_cfg.get("blend", 0.7)),
                            target_invested_pct=float(sizing_cfg.get("target_invested_pct", 1.0)),
                            min_weight=float(sizing_cfg.get("min_weight", 0.0)),
                            max_weight=float(sizing_cfg.get("max_weight", 1.0)),
                        )
                        rows = [
                            {
                                "symbol": s,
                                "target_weight": round(w, 6),
                                "target_qty": round(w * ctx.capital, 2),
                            }
                            for s, w in blended.items()
                            if abs(w) > 1e-6
                        ]
                        result.target_positions = pd.DataFrame(rows)
                        result.meta["hrp_sizing"] = {
                            "reasons": reasons,
                            "n_symbols": len(blended),
                        }
                        log.info(
                            "[RISK-WIRE] hrp sizing: %d symbols blended (reasons=%d)",
                            len(blended), len(reasons),
                        )
                    else:
                        raise ValueError("hrp: insufficient data or no target_weight column")
                except Exception as e:
                    log.warning("[RISK-WIRE] hrp sizing failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_fallback"] = True
                    result.meta["sizing_fallback_reason"] = str(e)
            elif sizing_method == "mvo":
                # [RISK-WIRE] Markowitz MVO with cardinality constraint.
                try:
                    import numpy as _np
                    from src.assembled_core.portfolio.mvo_optimizer import (
                        mvo_with_cardinality,
                    )
                    from src.assembled_core.portfolio.covariance import estimate_covariance

                    prices_for_mvo = (
                        result.prices_filtered if result.prices_filtered is not None else ctx.prices
                    )
                    if (
                        prices_for_mvo is not None
                        and not prices_for_mvo.empty
                        and not result.signals.empty
                        and "symbol" in result.signals.columns
                        and "close" in prices_for_mvo.columns
                        and "symbol" in prices_for_mvo.columns
                    ):
                        _ts_col_mvo = (
                            "timestamp" if "timestamp" in prices_for_mvo.columns else prices_for_mvo.columns[0]
                        )
                        _mvo_syms = [
                            s for s in result.signals["symbol"].tolist()
                            if s in prices_for_mvo["symbol"].unique()
                        ]
                        if len(_mvo_syms) >= 2:
                            _pivot_mvo = prices_for_mvo[
                                prices_for_mvo["symbol"].isin(_mvo_syms)
                            ].pivot_table(
                                index=_ts_col_mvo, columns="symbol", values="close"
                            )
                            _rets_mvo = _pivot_mvo.pct_change().dropna(how="all")
                            if len(_rets_mvo) >= 3:
                                sigma_mvo_df = estimate_covariance(
                                    _rets_mvo, method="ledoit_wolf"
                                )
                                mvo_syms = list(sigma_mvo_df.columns)
                                sigma_mvo = sigma_mvo_df.values.astype(float)
                                mu_series = (
                                    result.signals.set_index("symbol")["score"]
                                    if "score" in result.signals.columns
                                    else pd.Series(dtype=float)
                                )
                                mu_mvo = _np.asarray(
                                    mu_series.reindex(mvo_syms).fillna(0.0).values,
                                    dtype=float,
                                )
                                w_arr = mvo_with_cardinality(
                                    mu_mvo,
                                    sigma_mvo,
                                    max_positions=int(sizing_cfg.get("max_positions", 20)),
                                    risk_aversion=float(sizing_cfg.get("risk_aversion", 1.0)),
                                    min_weight=float(sizing_cfg.get("min_weight", 0.01)),
                                )
                                rows = [
                                    {
                                        "symbol": s,
                                        "target_weight": round(float(w_arr[i]), 6),
                                        "target_qty": round(float(w_arr[i]) * ctx.capital, 2),
                                    }
                                    for i, s in enumerate(mvo_syms)
                                    if abs(w_arr[i]) > 1e-6
                                ]
                                result.target_positions = pd.DataFrame(rows)
                                result.meta["mvo_sizing"] = {
                                    "n_symbols": len(rows),
                                    "max_positions": int(sizing_cfg.get("max_positions", 20)),
                                    "risk_aversion": float(sizing_cfg.get("risk_aversion", 1.0)),
                                }
                                log.info(
                                    "[RISK-WIRE] mvo sizing: %d active positions (max=%d)",
                                    len(rows),
                                    int(sizing_cfg.get("max_positions", 20)),
                                )
                            else:
                                raise ValueError("mvo: insufficient return history (<3 bars)")
                        else:
                            raise ValueError("mvo: fewer than 2 symbols overlap signals and prices")
                    else:
                        raise ValueError("mvo: insufficient data — falling back")
                except Exception as e:
                    log.warning("[RISK-WIRE] mvo sizing failed, using default: %s", e)
                    result.target_positions = ctx.position_sizing_fn(result.signals, ctx.capital)
                    result.meta["sizing_fallback"] = True
                    result.meta["sizing_fallback_reason"] = str(e)
            else:
                # Default: call position_sizing_fn (equal weight or score-based)
                result.target_positions = ctx.position_sizing_fn(
                    result.signals, ctx.capital
                )
            # Attribution truthfulness: if any sizing branch fell through to the default
            # position_sizing_fn, label the method truthfully so downstream reports
            # (tca/attribution) don't claim black_litterman/hrp/mvo when equal-weight ran.
            if result.meta.get("sizing_fallback"):
                result.meta["sizing_method_requested"] = sizing_method
                result.meta["sizing_method"] = "default_fallback"
            else:
                result.meta["sizing_method"] = sizing_method

        # Part B deeper wiring (2026-04-22): kelly_uncertainty shadow
        # Computes half-Kelly weights with conformal-uncertainty discount IN
        # PARALLEL to the active sizing method. Purely observational — written
        # to result.meta.kelly_uncertainty_shadow for ML training data only.
        # No change to result.target_positions.
        try:
            ku_cfg = (policy.get("position_sizing") or {}).get("kelly_uncertainty_shadow") or {}
            if ku_cfg.get("enabled", False) and not result.signals.empty and "score" in result.signals.columns:
                from src.assembled_core.portfolio.position_sizing import (
                    compute_kelly_weights_with_uncertainty,
                )
                edges = result.signals.set_index("symbol")["score"].dropna().astype(float)
                if not edges.empty:
                    vol_lookback = int(ku_cfg.get("vol_lookback_days", 60))
                    vols_map = _estimate_symbol_volatilities(
                        result.prices_filtered if result.prices_filtered is not None else ctx.prices,
                        lookback=vol_lookback,
                    )
                    variances = pd.Series(
                        {s: max((vols_map.get(s, 0.20)) ** 2, 1e-6) for s in edges.index},
                        name="variance",
                    )
                    conf_widths = getattr(ctx, "conformal_half_widths", None)
                    ref_width = ku_cfg.get("reference_half_width")
                    shadow_weights = compute_kelly_weights_with_uncertainty(
                        edges=edges,
                        variances=variances,
                        conformal_half_widths=conf_widths,
                        reference_half_width=float(ref_width) if ref_width is not None else None,
                        fractional_kelly=float(ku_cfg.get("fractional_kelly", 0.5)),
                        max_fraction=float(ku_cfg.get("max_fraction", 0.25)),
                        normalize=bool(ku_cfg.get("normalize", True)),
                    )
                    result.meta["kelly_uncertainty_shadow"] = {
                        "n_symbols": int(len(shadow_weights)),
                        "mean_abs_weight": float(shadow_weights.abs().mean()),
                        "max_abs_weight": float(shadow_weights.abs().max()) if len(shadow_weights) else 0.0,
                        "weights": {k: round(float(v), 6) for k, v in shadow_weights.items()},
                    }
                    log.info(
                        "[SIZING-SHADOW] kelly_uncertainty: %d symbols, mean|w|=%.4f",
                        len(shadow_weights), float(shadow_weights.abs().mean()),
                    )
        except Exception as _ku_exc:
            log.debug("[SIZING-SHADOW] kelly_uncertainty_shadow skipped: %s", _ku_exc)

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

        # Phase 9.5: Sprint 1 / W2 — Liquidity-aware position scaling
        # -----------------------------------------------------------
        # Applied immediately after sizing so every downstream overlay
        # (GeoRisk, profit-lock, vol-targeting, turnover) operates on
        # liquidity-adjusted weights. Gated by policy.liquidity_scoring
        # (default disabled) to keep behaviour unchanged until opt-in.
        try:
            liq_cfg = policy.get("liquidity_scoring") or {}
            if (
                liq_cfg.get("enabled", False)
                and not result.target_positions.empty
                and "target_weight" in result.target_positions.columns
                and result.prices_filtered is not None
                and not result.prices_filtered.empty
            ):
                from src.assembled_core.risk.liquidity_scoring import (
                    apply_liquidity_adjusted_sizing,
                    compute_liquidity_scores,
                )
                liq_scores = compute_liquidity_scores(
                    result.prices_filtered,
                    lookback_days=int(liq_cfg.get("lookback_days", 60)),
                )
                if liq_scores:
                    tw_map = {
                        str(r["symbol"]).upper(): float(r["target_weight"])
                        for _, r in result.target_positions.iterrows()
                    }
                    # Normalise symbol case for the score list too
                    for s in liq_scores:
                        s.symbol = s.symbol.upper()
                    adjusted_tw = apply_liquidity_adjusted_sizing(
                        target_weights=tw_map,
                        liquidity_scores=liq_scores,
                        alpha=float(liq_cfg.get("alpha", 0.5)),
                        min_score_threshold=float(
                            liq_cfg.get("min_score_threshold", 0.1)
                        ),
                    )
                    result.target_positions["target_weight"] = (
                        result.target_positions["symbol"]
                        .astype(str)
                        .str.upper()
                        .map(adjusted_tw)
                        .fillna(result.target_positions["target_weight"])
                    )
                    result.meta["liquidity_scoring"] = {
                        "applied": True,
                        "n_symbols": len(liq_scores),
                        "tiers": {
                            t: sum(1 for s in liq_scores if s.tier == t)
                            for t in ["mega", "large", "mid", "small", "micro"]
                        },
                    }
                    log.info(
                        "LIQUIDITY_SCORING: adjusted %d symbols, tiers=%s",
                        len(liq_scores),
                        result.meta["liquidity_scoring"]["tiers"],
                    )
        except Exception as e:
            log.debug("liquidity_scoring skipped: %s", e)

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
        # Floor: never compress exposure below 5% -- prevents near-zero orders
        # when multiple overlays fire simultaneously (EVT+Copula+Barbell+geo+vol)
        _MIN_EXPOSURE_MULT = 0.05
        if final_multiplier < _MIN_EXPOSURE_MULT:
            log.warning(
                "[RISK-WIRE] Cumulative exposure multiplier %.4f below floor %.2f "
                "-- clamping to floor. Components: geo=%.3f profit=%.3f vol=%.3f "
                "stress=%.3f crisis=%.3f",
                final_multiplier, _MIN_EXPOSURE_MULT,
                geo_multiplier, profit_lock_mult, vol_scale_factor,
                ms_multiplier, crisis_alpha_multiplier,
            )
            final_multiplier = _MIN_EXPOSURE_MULT
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

    # Phase 11.5: Sprint 1 / W1 — Trailing Stops (regime-adaptive ATR)
    # -------------------------------------------------------------------
    # After profit-lock / vol / stress overlays, the trailing-stops module
    # decides which existing positions should be exited or partially
    # reduced based on regime-scaled ATR rules and VIX.
    try:
        ts_cfg = policy.get("trailing_stops") or {}
        if ts_cfg.get("enabled", False) and not result.target_positions.empty:
            current_positions_df = ctx.current_positions
            if current_positions_df is not None and not current_positions_df.empty:
                from src.assembled_core.risk.trailing_stops import (
                    apply_stop_reductions_to_weights,
                    compute_trailing_stops,
                )
                pos_map: dict[str, dict] = {}
                for _, row in current_positions_df.iterrows():
                    sym = str(row.get("symbol", "")).upper()
                    if not sym:
                        continue
                    entry = (
                        row.get("avg_entry_price")
                        or row.get("entry_price")
                        or row.get("price")
                    )
                    if entry is None:
                        continue
                    pos_map[sym] = {
                        "entry_price": float(entry),
                        "qty": float(row.get("qty", 0.0) or 0.0),
                        "weight": float(row.get("weight", 0.0) or 0.0),
                    }

                rs_meta = result.meta.get("risk_state") or {}
                regime_label = str(rs_meta.get("regime", "unknown")).lower()
                vix_level = None
                if ctx.market_stress:
                    vix_level = ctx.market_stress.get("vix_level")

                if pos_map and result.prices_filtered is not None:
                    ts_result = compute_trailing_stops(
                        positions=pos_map,
                        prices_df=result.prices_filtered,
                        regime=regime_label,
                        atr_window=int(ts_cfg.get("atr_window", 14)),
                        vix_level=vix_level,
                    )
                    if ts_result.triggered_symbols or ts_result.reduction_symbols:
                        tw_col = (
                            "target_weight"
                            if "target_weight" in result.target_positions.columns
                            else "weight"
                        )
                        if tw_col in result.target_positions.columns:
                            weights_map = {
                                str(r["symbol"]).upper(): float(r[tw_col])
                                for _, r in result.target_positions.iterrows()
                            }
                            adjusted = apply_stop_reductions_to_weights(
                                weights_map, ts_result
                            )
                            result.target_positions[tw_col] = (
                                result.target_positions["symbol"]
                                .astype(str)
                                .str.upper()
                                .map(adjusted)
                                .fillna(result.target_positions[tw_col])
                            )
                            if "target_qty" in result.target_positions.columns:
                                for sym in ts_result.triggered_symbols:
                                    mask = (
                                        result.target_positions["symbol"]
                                        .astype(str)
                                        .str.upper()
                                        == sym
                                    )
                                    result.target_positions.loc[
                                        mask, "target_qty"
                                    ] = 0.0
                        result.meta["trailing_stops"] = {
                            "regime": regime_label,
                            "vix_level": vix_level,
                            "triggered": ts_result.triggered_symbols,
                            "reductions": ts_result.reduction_symbols,
                        }
                        log.warning(
                            "TRAILING_STOPS: regime=%s triggered=%s reductions=%s",
                            regime_label,
                            ts_result.triggered_symbols,
                            ts_result.reduction_symbols,
                        )
    except Exception as e:
        log.debug("trailing_stops check skipped: %s", e)

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
            # Compute invested_pct for ramp-up acceleration
            _invested_pct = None
            if ctx.capital > 0:
                _invested_notional = 0.0
                if (
                    ctx.current_positions is not None
                    and not ctx.current_positions.empty
                    and "qty" in ctx.current_positions.columns
                ):
                    _price_s = prices_for_turnover.groupby("symbol")["close"].last() if (prices_for_turnover is not None and not prices_for_turnover.empty and "close" in prices_for_turnover.columns) else pd.Series(dtype=float)
                    for _, _row in ctx.current_positions.iterrows():
                        _sym = _row.get("symbol", "")
                        _qty = float(_row.get("qty", 0) or 0)
                        _px = float(_price_s.get(_sym, 0) or 0) if not _price_s.empty else 0.0
                        _invested_notional += _qty * _px
                    _invested_pct = _invested_notional / ctx.capital
            _target_inv = float(tb.get("target_invested_pct", 0.80) or 0.80)
            if estimated == float("inf"):
                result.target_positions, scale_factor = apply_turnover_gate(
                    result.target_positions,
                    ctx.current_positions,
                    cap=cap,
                    estimated_turnover=1.0,
                    behavior="block",
                    prices=prices_for_turnover,
                    portfolio_value=ctx.capital,
                    invested_pct=_invested_pct,
                    target_invested_pct=_target_inv,
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
                    invested_pct=_invested_pct,
                    target_invested_pct=_target_inv,
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
                # [D1] Shadow-mode observation window (Ultra-Plan Part D).
                # policy.correlation_guard.shadow_only: True logs what would
                # have been scaled but does not touch target_positions.
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )

                cg_shadow = is_shadow_only(policy, "correlation_guard")
                deltas = {
                    sym: {"old": tw_dict[sym], "new": adjusted_weights[sym]}
                    for sym in tw_dict
                    if abs(tw_dict[sym] - adjusted_weights[sym]) > 1e-9
                }
                record_shadow(
                    "correlation_guard",
                    {"adjusted_weights": adjusted_weights, "deltas": deltas},
                    as_of=str(ctx.order_timestamp) if ctx.order_timestamp else None,
                    meta={
                        "clusters_scaled": len(corr_reasons),
                        "reasons": corr_reasons,
                        "applied": not cg_shadow,
                    },
                )
                for reason in corr_reasons:
                    log.warning(reason)
                if not cg_shadow:
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
                    "shadow_only": cg_shadow,
                }
                log.info(
                    "CORRELATION_GUARD: %d clusters scaled down (shadow=%s)",
                    len(corr_reasons),
                    cg_shadow,
                )

            # V4: Apply correlation regime shift exposure scaling
            symbols_in_portfolio = list(
                result.target_positions["symbol"].unique()
            )
            if len(symbols_in_portfolio) >= 2:
                shift_result = detect_correlation_regime_shift(
                    corr_prices, symbols_in_portfolio
                )
                if shift_result.get("regime_shift_detected", False):
                    exp_scale = shift_result["exposure_scale"]
                    result.target_positions["target_weight"] *= exp_scale
                    if "target_qty" in result.target_positions.columns:
                        result.target_positions["target_qty"] *= exp_scale
                    result.meta["correlation_regime_shift"] = {
                        "avg_corr_short": shift_result["avg_corr_short"],
                        "avg_corr_long": shift_result["avg_corr_long"],
                        "shift": shift_result["shift"],
                        "exposure_scale": exp_scale,
                    }
                    log.warning(
                        "CORR_REGIME_SHIFT: shift=%.3f, scaling exposure by %.2f",
                        shift_result["shift"],
                        exp_scale,
                    )
    except Exception as e:
        log.debug("correlation_guard check skipped: %s", e)

    # [D3] Crash-prediction long-equity cap (Ultra-Plan Part D).
    # When crash_prob exceeds a threshold, scale long gross exposure down.
    # Shadow-only by default — the scaling would-be is recorded but not applied
    # until policy.crash_prediction.shadow_only=False.
    try:
        cp_meta = result.meta.get("crash_prediction", {}) or {}
        crash_prob = float(cp_meta.get("crash_probability", 0.0) or 0.0)
        cp_cfg = (policy.get("crash_prediction", {}) or {})
        eq_cap_enabled = bool(cp_cfg.get("equity_cap_enabled", False))
        if (
            eq_cap_enabled
            and not result.target_positions.empty
            and "target_weight" in result.target_positions.columns
        ):
            threshold = float(cp_cfg.get("equity_cap_threshold", 0.4))
            if crash_prob > threshold:
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )

                base_long_gross = float(cp_cfg.get("base_long_gross", 1.0))
                cap = max(0.5 - crash_prob, 0.0) * base_long_gross
                long_mask = result.target_positions["target_weight"] > 0
                current_long_gross = float(
                    result.target_positions.loc[long_mask, "target_weight"].sum()
                )
                # When ``cap == 0`` (max crash_prob), scale must go to 0 — the
                # previous guard ``current_long_gross > cap > 0`` silently
                # bypassed the cap at the moment it was most aggressive.
                if current_long_gross > 0.0:
                    scale = min(cap / current_long_gross, 1.0)
                else:
                    scale = 1.0
                cp_shadow = is_shadow_only(policy, "crash_prediction")
                record_shadow(
                    "crash_prediction_cap",
                    {
                        "cap": cap,
                        "current_long_gross": current_long_gross,
                        "scale": scale,
                    },
                    as_of=str(ctx.order_timestamp) if ctx.order_timestamp else None,
                    meta={
                        "crash_prob": crash_prob,
                        "threshold": threshold,
                        "applied": (not cp_shadow) and scale < 1.0,
                    },
                )
                if not cp_shadow and scale < 1.0:
                    result.target_positions.loc[long_mask, "target_weight"] *= scale
                    if "target_qty" in result.target_positions.columns:
                        result.target_positions.loc[long_mask, "target_qty"] *= scale
                result.meta["crash_prediction_cap"] = {
                    "crash_prob": crash_prob,
                    "cap": cap,
                    "scale": scale,
                    "shadow_only": cp_shadow,
                }
                log.info(
                    "CRASH_EQ_CAP: prob=%.3f cap=%.3f scale=%.3f (shadow=%s)",
                    crash_prob, cap, scale, cp_shadow,
                )
    except Exception as e:
        log.debug("crash_prediction equity cap skipped: %s", e)

    # [D4] Inverse-ETF tail hedge (Ultra-Plan Part D).
    # VIX + crash_prob gate. 1x inverse ETFs only in Phase 1 (no 2x/3x).
    # Shadow-only by default — records hedge recommendation but does not add
    # orders until policy.inverse_etf.shadow_only=False.
    try:
        ie_cfg = (policy.get("inverse_etf", {}) or {})
        ie_enabled = bool(ie_cfg.get("enabled", False))
        cp_meta = result.meta.get("crash_prediction", {}) or {}
        crash_prob = float(cp_meta.get("crash_probability", 0.0) or 0.0)

        vix_val = None
        if ctx.prices is not None and not ctx.prices.empty:
            if "VIX" in ctx.prices.columns:
                try:
                    vix_val = float(ctx.prices["VIX"].iloc[-1])
                except Exception:
                    vix_val = None

        vix_threshold = float(ie_cfg.get("vix_threshold", 25.0))
        cp_threshold = float(ie_cfg.get("crash_prob_threshold", 0.4))

        if (
            ie_enabled
            and vix_val is not None
            and vix_val > vix_threshold
            and crash_prob > cp_threshold
        ):
            from src.assembled_core.ops.shadow_recorder import (
                is_shadow_only,
                record_shadow,
            )
            from src.assembled_core.portfolio.inverse_etf_selector import (
                InverseETFSelector,
            )

            selector = InverseETFSelector(allow_2x=False, allow_3x=False)
            hedge_sym = selector.select_best_short_instrument(
                sector="BROAD",
                severity=float(cp_meta.get("severity", 0.5) or 0.5),
                holding_period_days=int(ie_cfg.get("max_holding_days", 5)),
            )
            hedge_ratio = float(ie_cfg.get("hedge_ratio", 0.1))
            ie_shadow = is_shadow_only(policy, "inverse_etf")
            record_shadow(
                "inverse_etf",
                {
                    "hedge_symbol": hedge_sym,
                    "hedge_weight": hedge_ratio,
                },
                as_of=str(ctx.order_timestamp) if ctx.order_timestamp else None,
                meta={
                    "vix": vix_val,
                    "crash_prob": crash_prob,
                    "vix_threshold": vix_threshold,
                    "cp_threshold": cp_threshold,
                    "applied": (not ie_shadow) and hedge_sym is not None,
                },
            )
            if not ie_shadow and hedge_sym and "target_weight" in result.target_positions.columns:
                if hedge_sym not in result.target_positions["symbol"].values:
                    new_row = pd.DataFrame([{
                        "symbol": hedge_sym,
                        "target_weight": hedge_ratio,
                        "target_qty": hedge_ratio * ctx.capital,
                    }])
                    result.target_positions = pd.concat(
                        [result.target_positions, new_row], ignore_index=True
                    )
            result.meta["inverse_etf"] = {
                "hedge_symbol": hedge_sym,
                "hedge_ratio": hedge_ratio,
                "vix": vix_val,
                "crash_prob": crash_prob,
                "shadow_only": ie_shadow,
            }
            log.info(
                "INVERSE_ETF: hedge=%s ratio=%.3f vix=%.1f crash_prob=%.3f (shadow=%s)",
                hedge_sym, hedge_ratio, vix_val, crash_prob, ie_shadow,
            )
    except Exception as e:
        log.debug("inverse_etf hedge skipped: %s", e)

    # [RISK-WIRE] Quantile asymmetry sizing — reduce positions with high downside skew
    try:
        qm_cfg = (policy.get("risk", {}) or {}).get("quantile_sizing", {}) or {}
        if (
            qm_cfg.get("enabled", False)
            and not result.target_positions.empty
            and "target_weight" in result.target_positions.columns
            and not result.prices_with_features.empty
        ):
            from src.assembled_core.ml.quantile_models import predict_quantiles

            _feature_cols = qm_cfg.get("feature_cols", [])
            _target_col = qm_cfg.get("target_col", "return_1d")
            _asym_threshold = float(qm_cfg.get("asymmetry_threshold", 1.5))
            _asym_reduction = float(qm_cfg.get("asymmetry_reduction", 0.5))

            if _feature_cols and _target_col in result.prices_with_features.columns:
                _valid_fcols = [c for c in _feature_cols if c in result.prices_with_features.columns]
                if _valid_fcols:
                    _qpreds = predict_quantiles(
                        result.prices_with_features,
                        target_col=_target_col,
                        feature_cols=_valid_fcols,
                    )
                    _asym_map: dict[str, float] = {qp.symbol: qp.asymmetry for qp in _qpreds}
                    _reduced: list[str] = []
                    for idx, row in result.target_positions.iterrows():
                        sym = row.get("symbol", "")
                        asym = _asym_map.get(sym, 0.0)
                        if asym > _asym_threshold:
                            result.target_positions.at[idx, "target_weight"] *= _asym_reduction
                            if "target_qty" in result.target_positions.columns:
                                result.target_positions.at[idx, "target_qty"] *= _asym_reduction
                            _reduced.append(sym)
                    if _reduced:
                        log.info(
                            "[RISK-WIRE] quantile_asymmetry: reduced %d positions with asymmetry>%.1f: %s",
                            len(_reduced), _asym_threshold, _reduced,
                        )
                    result.meta["quantile_asymmetry"] = {
                        "reduced_symbols": _reduced,
                        "asymmetry_threshold": _asym_threshold,
                    }
    except Exception as e:
        log.debug("[RISK-WIRE] quantile_asymmetry skipped: %s", e)

    # [RISK-WIRE] Crowding detector — HHI cap and weight ceiling
    try:
        if not result.target_positions.empty and "target_weight" in result.target_positions.columns:
            from src.assembled_core.risk.crowding_detector import compute_hhi

            _tw_dict_crowd = dict(
                zip(
                    result.target_positions["symbol"],
                    result.target_positions["target_weight"].fillna(0.0),
                )
            )
            _hhi = compute_hhi(_tw_dict_crowd)
            result.meta["crowding_hhi"] = round(_hhi, 4)

            if _hhi > 0.15:
                log.warning(
                    "[RISK-WIRE] crowding_detector: HHI=%.4f > 0.15 (concentration warning)",
                    _hhi,
                )
            if _hhi > 0.25 and len(_tw_dict_crowd) >= 5:
                # Cap each position at 10% (only meaningful for portfolios of 5+ names)
                _capped = 0
                _max_w = 0.10
                for idx, row in result.target_positions.iterrows():
                    if abs(float(row.get("target_weight", 0.0))) > _max_w:
                        result.target_positions.at[idx, "target_weight"] = _max_w
                        if "target_qty" in result.target_positions.columns:
                            result.target_positions.at[idx, "target_qty"] = _max_w * ctx.capital
                        _capped += 1
                if _capped:
                    log.warning(
                        "[RISK-WIRE] crowding_detector: HHI=%.4f > 0.25 — capped %d positions to 10%%",
                        _hhi, _capped,
                    )
                result.meta["crowding_hhi_capped"] = _capped
    except Exception as e:
        log.debug("[RISK-WIRE] crowding_detector skipped: %s", e)

    # Step 4.5 (V14): Check rebalancing triggers — skip order generation if no trigger
    rebal_scheduled = True  # Default: always rebalance (backward compatible)
    vol_regime_changed = bool(result.meta.get("vol_targeting", {}).get("regime_changed", False))
    corr_spiked = bool(result.meta.get("correlation_regime_shift", {}).get("exposure_scale", 1.0) < 1.0)
    dd_pct = result.meta.get("drawdown_pct")
    current_w: dict[str, float] = {}
    if hasattr(ctx, "current_positions") and ctx.current_positions is not None:
        if isinstance(ctx.current_positions, dict):
            current_w = ctx.current_positions
        elif isinstance(ctx.current_positions, pd.DataFrame) and "symbol" in ctx.current_positions.columns:
            for _, row in ctx.current_positions.iterrows():
                current_w[row["symbol"]] = float(row.get("weight", row.get("target_weight", 0.0)))

    do_rebal, rebal_reason = should_rebalance(
        ctx, result.target_positions,
        current_weights=current_w,
        weight_drift_threshold=float(policy.get("rebalancing", {}).get("weight_drift_threshold", 0.05)),
        vol_regime_change=vol_regime_changed,
        corr_spike=corr_spiked,
        scheduled=rebal_scheduled,
        drawdown_pct=float(dd_pct) if dd_pct is not None else None,
    )
    result.meta["rebalance_decision"] = {"triggered": do_rebal, "reason": rebal_reason}
    if not do_rebal:
        log.info("REBALANCE SKIPPED: %s — no orders generated", rebal_reason)
        result.orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        result.orders_filtered = result.orders.copy()
        # Skip to outputs
    else:
        log.info("REBALANCE TRIGGERED: %s", rebal_reason)

    # T4.1: Crisis-Alpha v1 wiring (policy-gated, default shadow-only).
    # Inserted after signal generation + rebalance decision, before order generation.
    # shadow_only=True (default): pipeline called dry_run=True, result logged only.
    # shadow_only=False: pipeline called dry_run=False, target_weights capped
    #   conservatively — crisis_alpha can only reduce weights, never increase them.
    if (policy or {}).get("intel", {}).get("crisis_alpha", {}).get("enabled", False):
        try:
            from src.assembled_core.events.crisis_alpha.pipeline import (
                run_crisis_alpha_pipeline,
            )
            from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext

            # Build a neutral CrisisAlphaContext from available trading-cycle data.
            # TradingContext and CrisisAlphaContext are separate types; we use
            # CrisisAlphaContext.empty() so the pipeline has a valid input contract.
            # Callers that want richer geo/stress inputs should provide a pre-built
            # CrisisAlphaContext via ctx.meta["crisis_alpha_ctx"].
            _ca_ctx = ctx.meta.get("crisis_alpha_ctx") if hasattr(ctx, "meta") else None
            if _ca_ctx is None:
                _as_of_dt = pd.to_datetime(ctx.as_of, utc=True).to_pydatetime() if getattr(ctx, "as_of", None) is not None else None
                _ca_ctx = CrisisAlphaContext.empty(timestamp_utc=_as_of_dt)

            shadow_only = (policy or {}).get("intel", {}).get("crisis_alpha", {}).get("shadow_only", True)

            # dry_run=True when shadow — state is never persisted in shadow mode.
            ca_result = run_crisis_alpha_pipeline(_ca_ctx, policy=policy, dry_run=shadow_only)

            if shadow_only:
                logger.info("[SHADOW-T4.1] crisis_alpha result (log-only): state=%s targets=%d",
                            ca_result.get("state"), len(ca_result.get("target_weights") or {}))
            else:
                # Step 3: apply conservative weight cap from crisis_alpha target_weights.
                # Invariant: never increase a weight via crisis_alpha — only reduce.
                ca_weights: dict[str, float] = {
                    k.upper(): v for k, v in (ca_result.get("target_weights") or {}).items()
                }
                if ca_weights and not result.target_positions.empty:
                    n_adjusted = 0
                    for idx, row in result.target_positions.iterrows():
                        sym = str(row["symbol"]).upper()
                        ca_cap = ca_weights.get(sym)
                        if ca_cap is not None:
                            old_w = float(row["target_weight"])
                            new_w = min(old_w, float(ca_cap))
                            if new_w < old_w:
                                result.target_positions.at[idx, "target_weight"] = new_w
                                n_adjusted += 1
                    logger.info("[OK] T4.1 crisis_alpha weights applied: %d symbols adjusted", n_adjusted)
                    result.meta["crisis_alpha_weight_cap"] = {
                        "applied": True,
                        "n_adjusted": n_adjusted,
                        "ca_state": ca_result.get("state"),
                    }
                else:
                    logger.info("[OK] T4.1 crisis_alpha: no target_weights to apply (state=%s)",
                                ca_result.get("state"))
                    result.meta["crisis_alpha_weight_cap"] = {
                        "applied": False,
                        "n_adjusted": 0,
                        "ca_state": ca_result.get("state"),
                    }
        except Exception as exc:
            logger.warning("[WARN] T4.1 crisis_alpha_pipeline failed, continuing without cap: %s", exc)

    # Step 4.9: ML training snapshot — per-symbol signals + target weights
    try:
        ml_snap: dict = {}
        if not result.signals.empty and "symbol" in result.signals.columns:
            sig_cols = [c for c in ("symbol", "direction", "score") if c in result.signals.columns]
            sig_snap = (
                result.signals[sig_cols]
                .dropna(subset=["symbol"])
                .sort_values("score", ascending=False)
                if "score" in result.signals.columns
                else result.signals[sig_cols].dropna(subset=["symbol"])
            )
            ml_snap["signal_snapshot"] = sig_snap.to_dict(orient="records")
        if not result.target_positions.empty and "symbol" in result.target_positions.columns:
            w_col = next(
                (c for c in ("target_weight", "weight", "target_pct") if c in result.target_positions.columns),
                None,
            )
            if w_col:
                ml_snap["target_weights"] = (
                    result.target_positions[["symbol", w_col]]
                    .rename(columns={w_col: "weight"})
                    .dropna()
                    .to_dict(orient="records")
                )
        if ml_snap:
            result.meta["ml_training_snapshot"] = ml_snap
            log.debug(
                "[ML-SNAP] captured %d signal rows, %d target rows",
                len(ml_snap.get("signal_snapshot", [])),
                len(ml_snap.get("target_weights", [])),
            )
    except Exception as _ms_exc:
        log.debug("[ML-SNAP] ml_training_snapshot skipped: %s", _ms_exc)

    # Step 4.85: Cost-aware weight shrinkage (reduces oversizing on expensive trades)
    try:
        caw_cfg = policy.get("cost_aware_wrapper") or {}
        if caw_cfg.get("enabled", False) and not result.target_positions.empty:
            from src.assembled_core.portfolio.cost_aware_wrapper import (
                apply_cost_aware_from_policy,
            )
            w_col = next(
                (c for c in ("target_weight", "weight", "target_pct") if c in result.target_positions.columns),
                None,
            )
            if w_col and "symbol" in result.target_positions.columns:
                _target_w = {
                    str(r["symbol"]): float(r[w_col])
                    for _, r in result.target_positions.iterrows()
                    if pd.notna(r.get(w_col))
                }
                _curr_w: dict[str, float] = {}
                if ctx.current_positions is not None and not ctx.current_positions.empty:
                    if "symbol" in ctx.current_positions.columns and w_col in ctx.current_positions.columns:
                        _curr_w = {
                            str(r["symbol"]): float(r[w_col])
                            for _, r in ctx.current_positions.iterrows()
                            if pd.notna(r.get(w_col))
                        }
                    elif "symbol" in ctx.current_positions.columns and "weight" in ctx.current_positions.columns:
                        _curr_w = {
                            str(r["symbol"]): float(r["weight"])
                            for _, r in ctx.current_positions.iterrows()
                            if pd.notna(r.get("weight"))
                        }
                _adj_w, _caw_reasons = apply_cost_aware_from_policy(
                    _target_w, _curr_w, policy,
                    current_invested_pct=float(sum(abs(v) for v in _target_w.values())),
                )
                if _caw_reasons:
                    # Apply shrunken weights back to target_positions
                    result.target_positions = result.target_positions.copy()
                    result.target_positions[w_col] = result.target_positions["symbol"].map(
                        lambda s: _adj_w.get(str(s), _target_w.get(str(s), 0.0))
                    )
                    result.meta["cost_aware_wrapper"] = {
                        "n_shrunken": len(_caw_reasons),
                        "reasons": _caw_reasons[:5],
                    }
                    log.info(
                        "[COST-AWARE] shrinkage applied to %d positions",
                        len(_caw_reasons),
                    )
    except Exception as _caw_exc:
        log.debug("[COST-AWARE] cost_aware_wrapper skipped: %s", _caw_exc)

    # Step 5: Generate orders (hook point: generate_orders)
    try:
        if not do_rebal:
            pass  # Already set empty orders above
        elif "generate_orders" in hooks:
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

    # Phase 17.8 (Sprint 2 / C10): Pre-Trade Impact estimate
    try:
        impact_cfg = (policy.get("execution", {}) or {}).get("pre_trade_impact", {}) or {}
        if impact_cfg.get("enabled", False) and not result.orders.empty:
            new_orders, impact_meta = _apply_pre_trade_impact(
                result.orders, result.prices_filtered, impact_cfg
            )
            result.orders = new_orders
            result.meta["pre_trade_impact"] = impact_meta
            if impact_meta.get("scaled_symbols"):
                log.info(
                    "[PRE_TRADE_IMPACT] scaled %d orders exceeding %.1fbps: %s",
                    len(impact_meta["scaled_symbols"]),
                    impact_meta.get("max_total_cost_bps", 0.0),
                    impact_meta["scaled_symbols"],
                )
    except Exception as e:
        log.debug("pre_trade_impact skipped: %s", e)

    # Phase 17.85 (Sprint 5 / C12): Optional TWAP order slicing
    try:
        exec_cfg = (policy.get("execution", {}) or {}).get("algo", {}) or {}
        algo_mode = str(exec_cfg.get("mode", "market")).lower()
        if algo_mode.startswith("twap") and not result.orders.empty:
            from datetime import datetime, timedelta, timezone

            from src.assembled_core.execution.algo_execution import TWAPScheduler

            n_slices = int(exec_cfg.get("n_slices", 10))
            window_minutes = int(exec_cfg.get("window_minutes", 60))
            scheduler = TWAPScheduler(n_slices=n_slices, randomize=True)
            sliced_orders = []
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            for _, order in result.orders.iterrows():
                slices = scheduler.schedule(
                    symbol=str(order.get("symbol", "")),
                    total_qty=abs(float(order.get("qty", 0))),
                    side=str(order.get("side", "BUY")),
                    start_time=now,
                    end_time=now + timedelta(minutes=window_minutes),
                )
                sliced_orders.extend(s.to_dict() for s in slices)
            result.meta["twap_slices"] = len(sliced_orders)
            result.meta["twap_parent_orders"] = len(result.orders)
            log.info(
                "[TWAP] Sliced %d orders into %d TWAP slices (%d min window)",
                len(result.orders), len(sliced_orders), window_minutes,
            )
    except Exception as e:
        log.debug("twap slicing skipped: %s", e)

    # Phase 17.9 (Sprint 2 / W3): Group-Exposure caps (sector/region/currency)
    try:
        group_cfg = (policy.get("risk", {}) or {}).get("group_limits", {}) or {}
        if group_cfg.get("enabled", False) and not result.orders.empty:
            sec_meta = None
            try:
                from src.assembled_core.data.security_master import (
                    load_security_master,
                )
                sec_meta = load_security_master(
                    group_cfg.get("security_master_path") or None
                )
            except Exception as _e:
                log.debug("security_master load failed: %s", _e)
                sec_meta = None
            if sec_meta is not None:
                new_orders, grp_meta = _apply_group_exposure_caps(
                    result.orders, sec_meta, group_cfg
                )
                result.orders = new_orders
                result.meta["group_exposures"] = grp_meta
                if grp_meta.get("scaled_groups"):
                    log.info(
                        "[GROUP_EXPOSURES] scaled %d groups: %s",
                        len(grp_meta["scaled_groups"]),
                        [g["group"] for g in grp_meta["scaled_groups"]],
                    )
    except Exception as e:
        log.debug("group_exposures skipped: %s", e)

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

    # [RISK-WIRE] EVT tail VaR — post-portfolio exposure check
    try:
        if not result.orders.empty:
            _prices_for_evt = (
                result.prices_filtered if result.prices_filtered is not None else ctx.prices
            )
            if _prices_for_evt is not None and not _prices_for_evt.empty and "close" in _prices_for_evt.columns:
                from src.assembled_core.risk.evt_tail_var import evt_var

                # Build portfolio return series (equal-weighted proxy)
                _pivot_evt = _prices_for_evt.pivot_table(
                    index="timestamp" if "timestamp" in _prices_for_evt.columns else _prices_for_evt.columns[0],
                    columns="symbol" if "symbol" in _prices_for_evt.columns else None,
                    values="close",
                )
                _rets_evt = _pivot_evt.pct_change().dropna(how="all")
                if len(_rets_evt) >= 60 and not _rets_evt.empty:
                    _port_rets = _rets_evt.mean(axis=1).dropna()
                    _losses = (-_port_rets).values  # positive = loss
                    # Historical VaR (99%) as baseline
                    import numpy as _np_evt
                    _hist_var_99 = float(_np_evt.quantile(_losses, 0.99))
                    try:
                        _evt_var_99 = evt_var(_losses, alpha=0.99, threshold_pct=0.90)
                    except Exception as _evt_err:
                        log.debug("[RISK-WIRE] evt_var fit failed: %s", _evt_err)
                        _evt_var_99 = None

                    if _evt_var_99 is not None:
                        result.meta["evt_var_99"] = round(float(_evt_var_99), 6)
                        result.meta["hist_var_99"] = round(_hist_var_99, 6)
                        log.info(
                            "[RISK-WIRE] EVT VaR 99%%=%.4f  Hist VaR 99%%=%.4f",
                            _evt_var_99, _hist_var_99,
                        )
                        # If EVT VaR > 2x historical VaR: reduce order exposure by 20%
                        if _hist_var_99 > 1e-8 and _evt_var_99 > 2.0 * _hist_var_99:
                            _scale_evt = 0.80
                            result.orders["qty"] = result.orders["qty"] * _scale_evt
                            log.warning(
                                "[RISK-WIRE] EVT VaR %.4f > 2x Hist VaR %.4f — reducing exposure by 20%%",
                                _evt_var_99, _hist_var_99,
                            )
                            result.meta["evt_exposure_reduction"] = _scale_evt
    except Exception as e:
        log.debug("[RISK-WIRE] evt_tail_var skipped: %s", e)

    # [RISK-WIRE] Copula tail dependence — additional exposure cut if avg_lower_tail_dep > 0.5
    try:
        if not result.orders.empty:
            _prices_for_cop = (
                result.prices_filtered if result.prices_filtered is not None else ctx.prices
            )
            if _prices_for_cop is not None and not _prices_for_cop.empty and "close" in _prices_for_cop.columns:
                from src.assembled_core.ml.copula_models import compute_portfolio_tail_risk

                _pivot_cop = _prices_for_cop.pivot_table(
                    index="timestamp" if "timestamp" in _prices_for_cop.columns else _prices_for_cop.columns[0],
                    columns="symbol" if "symbol" in _prices_for_cop.columns else None,
                    values="close",
                )
                _rets_cop = _pivot_cop.pct_change().dropna(how="all")
                # Only run if enough data and not too many symbols (avoid O(n^2) explosion)
                if len(_rets_cop) >= 60 and 1 < _rets_cop.shape[1] <= 30:
                    _cop_metrics = compute_portfolio_tail_risk(_rets_cop)
                    _avg_ltd = float(_cop_metrics.get("avg_lower_tail_dep", 0.0))
                    result.meta["copula_tail_risk"] = _cop_metrics
                    log.info(
                        "[RISK-WIRE] Copula avg_lower_tail_dep=%.4f max=%.4f n_pairs=%d",
                        _avg_ltd,
                        float(_cop_metrics.get("max_lower_tail_dep", 0.0)),
                        int(_cop_metrics.get("n_pairs", 0)),
                    )
                    if _avg_ltd > 0.5:
                        _scale_cop = 0.80
                        result.orders["qty"] = result.orders["qty"] * _scale_cop
                        log.warning(
                            "[RISK-WIRE] Copula avg_lower_tail_dep=%.4f > 0.5 — reducing exposure by additional 20%%",
                            _avg_ltd,
                        )
                        result.meta["copula_exposure_reduction"] = _scale_cop
    except Exception as e:
        log.debug("[RISK-WIRE] copula_tail_risk skipped: %s", e)

    # [RISK-WIRE] Barbell strategy — crisis overlay when composite tail risk score > 0.30
    try:
        _tail_score_for_barbell = 0.0
        _barbell_reasons: list[str] = []
        # Gather available signals into barbell score
        _evt_var_meta = result.meta.get("evt_var_99", 0.0) or 0.0
        _hist_var_meta = result.meta.get("hist_var_99", 0.0) or 0.0
        _cop_ltd_meta = float((result.meta.get("copula_tail_risk") or {}).get("avg_lower_tail_dep", 0.0))

        from src.assembled_core.portfolio.barbell_strategy import (
            build_barbell_allocation,
            compute_tail_risk_score,
        )

        _bb_score, _bb_reasons = compute_tail_risk_score(
            evt_var_99=float(_evt_var_meta),
            evt_var_99_historical_avg=float(_hist_var_meta),
            hmm_crisis_prob=0.0,  # not wired yet
            vix_current=0.0,      # not wired yet
            vix_5d_change=0.0,
            avg_copula_tail_dep=_cop_ltd_meta,
        )
        result.meta["barbell_tail_risk_score"] = round(_bb_score, 4)

        if _bb_score > 0.30 and not result.orders.empty:
            # Build alpha scores from signals for speculative sleeve
            _alpha_scores: dict[str, float] = {}
            if not result.signals.empty and "symbol" in result.signals.columns and "score" in result.signals.columns:
                _alpha_scores = dict(zip(result.signals["symbol"], result.signals["score"].fillna(0.0)))

            _bb_alloc = build_barbell_allocation(
                tail_risk_score=_bb_score,
                trigger_reasons=_bb_reasons,
                alpha_scores=_alpha_scores,
            )
            if _bb_alloc.active:
                # Scale down all orders to reflect barbell exposure compression
                _bb_scale = _bb_alloc.speculative_weight
                result.orders["qty"] = result.orders["qty"] * _bb_scale
                result.meta["barbell"] = {
                    "active": True,
                    "tail_risk_score": _bb_score,
                    "safe_weight": _bb_alloc.safe_weight,
                    "speculative_weight": _bb_alloc.speculative_weight,
                    "trigger_reasons": _bb_reasons,
                    "speculative_symbols": _bb_alloc.speculative_symbols,
                }
                log.warning(
                    "[RISK-WIRE] Barbell ACTIVATED: score=%.3f safe=%.0f%% spec=%.0f%% triggers=%s",
                    _bb_score,
                    _bb_alloc.safe_weight * 100,
                    _bb_alloc.speculative_weight * 100,
                    ", ".join(_bb_reasons),
                )
    except Exception as e:
        log.debug("[RISK-WIRE] barbell_strategy skipped: %s", e)

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

    # Step 6.35: Tier-1 — parametric VaR exposure gate (default OFF).
    try:
        var_decision = _evaluate_var_gate(ctx, result, policy)
        if var_decision is not None:
            result.meta["var_gate"] = var_decision
            log.warning(
                "[RISK-WIRE] VaR gate breach: var_1d=%.4f > %.4f (%.0f%% conf) — %s",
                var_decision["var_1d"],
                var_decision["max_var_pct"],
                var_decision["confidence"] * 100.0,
                var_decision["reason"],
            )
            # Block all orders for this cycle when breach is detected.
            result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        # A risk-path gate silently skipping at DEBUG level is invisible at
        # default log verbosity. Emit WARNING and stamp ``result.meta`` so the
        # manifest / QA consumer can tell "gate ran clean" apart from "gate
        # crashed and orders passed unfiltered".
        log.warning("[RISK-WIRE] var_gate evaluation raised — gate no-op: %s", e)
        result.meta["var_gate"] = {"status": "error", "error": str(e)}

    # Step 6.4: Sprint 1 / C7 — Auto-Drawdown Kill-Switch trigger
    try:
        dd_decision = _evaluate_auto_dd_kill_switch(ctx, result, policy)
        if dd_decision is not None:
            from src.assembled_core.execution.kill_switch import (
                activate_kill_switch,
                is_kill_switch_engaged,
            )
            activate_kill_switch(
                throttle_pct=dd_decision["throttle_allowed_pct"],
                reason=dd_decision["reason"],
                actor="trading_cycle_auto_dd",
            )
            result.meta["auto_dd_kill_switch"] = dd_decision
            log.warning(
                "AUTO_DD_KILL_SWITCH: level=%s drawdown=%.2f%% throttle_allowed=%.0f%% — %s",
                dd_decision["level"],
                dd_decision["drawdown"] * 100,
                dd_decision["throttle_allowed_pct"] * 100,
                "kill switch engaged" if is_kill_switch_engaged() else "pending",
            )
            # Kill-mode blocks ALL orders for this cycle too.
            if dd_decision["level"] == "kill":
                result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        log.warning(
            "[RISK-WIRE] auto_dd_kill_switch evaluation raised — gate no-op: %s",
            e,
        )
        result.meta["auto_dd_kill_switch"] = {"status": "error", "error": str(e)}

    # Step 6.45: Tier-1 — intraday circuit breaker (default OFF).
    try:
        cb_decision = _evaluate_circuit_breaker(ctx, result, policy)
        if cb_decision is not None:
            result.meta["circuit_breaker"] = cb_decision
            log.warning(
                "[RISK-WIRE] Circuit breaker TRIPPED: threshold=%.1f%% window=%dmin — %s",
                cb_decision["drop_threshold_pct"],
                cb_decision["window_minutes"],
                cb_decision["reason"],
            )
            result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        log.warning(
            "[RISK-WIRE] circuit_breaker evaluation raised — gate no-op: %s", e
        )
        result.meta["circuit_breaker"] = {"status": "error", "error": str(e)}

    # Step 6.5: D12 — Scenario engine stress tests (post-orders, optional)
    try:
        scenario_cfg = policy.get("scenario_engine", {})
        if scenario_cfg.get("enabled", False) and not result.target_positions.empty:
            from src.assembled_core.qa.scenario_engine import run_crisis_scenarios
            import datetime as _dt
            prices_for_scenario = result.prices_filtered if result.prices_filtered is not None else ctx.prices
            if prices_for_scenario is not None and not prices_for_scenario.empty:
                crisis_type = scenario_cfg.get("crisis_type", "geopolitical_escalation")
                shock_date = ctx.as_of.replace(tzinfo=None) if ctx.as_of else _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
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

    # Step 6.6: Anti-churn order filters (deadzone + min-notional)
    try:
        anti_churn_cfg = policy.get("anti_churn") or {}
        if not result.orders_filtered.empty:
            if anti_churn_cfg.get("deadzone_enabled", False):
                from src.assembled_core.paper.deadzone_rebalance import filter_deadzone_orders
                _dz_positions = (
                    ctx.current_positions[["symbol", "qty"]].copy()
                    if ctx.current_positions is not None
                    and not ctx.current_positions.empty
                    and "qty" in ctx.current_positions.columns
                    else None
                )
                result.orders_filtered, _dz_meta = filter_deadzone_orders(
                    result.orders_filtered,
                    _dz_positions,
                    deadzone_pct=float(anti_churn_cfg.get("deadzone_pct", 0.05)),
                )
                result.meta["deadzone_rebalance"] = _dz_meta
                if _dz_meta.get("orders_dropped", 0):
                    log.info(
                        "[ANTI-CHURN] deadzone_rebalance: dropped=%d kept=%d",
                        _dz_meta["orders_dropped"],
                        _dz_meta["orders_after"],
                    )
            if anti_churn_cfg.get("rebalance_filter_enabled", False) and not result.orders_filtered.empty:
                from src.assembled_core.paper.rebalance_filter import filter_small_rebalances
                _prices_for_filter = result.prices_filtered if result.prices_filtered is not None else ctx.prices
                result.orders_filtered, _rf_meta = filter_small_rebalances(
                    result.orders_filtered,
                    min_notional=float(anti_churn_cfg.get("min_notional", 500.0)),
                    prices=_prices_for_filter,
                )
                result.meta["rebalance_filter"] = _rf_meta
                if _rf_meta.get("orders_dropped", 0):
                    log.info(
                        "[ANTI-CHURN] rebalance_filter: dropped=%d kept=%d min_notional=%.0f",
                        _rf_meta["orders_dropped"],
                        _rf_meta["orders_after"],
                        _rf_meta["min_notional"],
                    )
    except Exception as _ac_exc:
        log.debug("[ANTI-CHURN] order filters skipped: %s", _ac_exc)

    # Step 6.7: Fat-finger guard — hard notional + qty-multiple cap (pre-submission)
    try:
        ffg_cfg = policy.get("fat_finger_guard") or {}
        if ffg_cfg.get("enabled", False) and not result.orders_filtered.empty:
            from src.assembled_core.execution.fat_finger_guard import (
                apply_fat_finger_guard_from_policy,
            )
            _ffg_orders, _ffg_reasons = apply_fat_finger_guard_from_policy(
                result.orders_filtered, policy
            )
            n_rejected = len(result.orders_filtered) - len(_ffg_orders)
            result.orders_filtered = _ffg_orders
            result.meta["fat_finger_guard"] = {
                "n_rejected": n_rejected,
                "reasons": _ffg_reasons,
            }
            if n_rejected:
                log.warning(
                    "[FAT-FINGER] Rejected %d orders: %s",
                    n_rejected,
                    _ffg_reasons[:3],
                )
    except Exception as _ffg_exc:
        log.debug("[FAT-FINGER] fat_finger_guard skipped: %s", _ffg_exc)

    # Step 6.8: Borrow cost estimate for short positions (observability + accounting)
    try:
        bc_cfg = policy.get("borrow_costs") or {}
        if bc_cfg.get("enabled", True) and not result.orders_filtered.empty:
            from src.assembled_core.execution.borrow_costs import (
                BorrowRateTable,
                compute_borrow_cost,
            )
            _brt = BorrowRateTable(
                default_rate_bps=float(bc_cfg.get("default_rate_bps", 50.0)),
                htb_rate_bps=float(bc_cfg.get("htb_rate_bps", 500.0)),
            )
            _total_borrow_usd = 0.0
            _short_count = 0
            for _, _ord_row in result.orders_filtered.iterrows():
                _qty = float(_ord_row.get("qty", 0))
                _px = float(_ord_row.get("price", 0))
                if _qty < 0 and _px > 0:
                    _sym = str(_ord_row.get("symbol", ""))
                    _cost = compute_borrow_cost(_qty, _px, _brt.rate_bps(_sym))
                    _total_borrow_usd += _cost
                    _short_count += 1
            if _short_count > 0:
                result.meta["borrow_costs"] = {
                    "n_short_orders": _short_count,
                    "estimated_daily_borrow_usd": round(_total_borrow_usd, 4),
                }
                log.info(
                    "[BORROW] %d short orders, estimated daily borrow cost: $%.4f",
                    _short_count, _total_borrow_usd,
                )
    except Exception as _bc_exc:
        log.debug("[BORROW] borrow_costs skipped: %s", _bc_exc)

    # Step 6.9: Order lifecycle tracking (audit trail for submitted orders)
    try:
        if not result.orders_filtered.empty:
            from src.assembled_core.execution.order_lifecycle import (
                OrderLifecycleTracker,
                OrderState,
            )
            _olt = OrderLifecycleTracker()
            _olt_ids = []
            for _, _ord_row in result.orders_filtered.iterrows():
                _oid = _olt.create(
                    symbol=str(_ord_row.get("symbol", "")),
                    side=str(_ord_row.get("side", "buy")),
                    quantity=float(_ord_row.get("qty", 0)),
                    price=float(_ord_row.get("price", 0)) or None,
                    source="trading_cycle",
                )
                _olt.transition(_oid, OrderState.VALIDATED)
                _olt.transition(_oid, OrderState.SUBMITTED)
                _olt_ids.append(_oid)
            result.meta["order_lifecycle"] = {
                "n_orders_tracked": len(_olt_ids),
                "state": "SUBMITTED",
            }
            log.debug("[ORDER_LIFECYCLE] tracked %d orders", len(_olt_ids))
    except Exception as _ol_exc:
        log.debug("[ORDER_LIFECYCLE] order_lifecycle tracking skipped: %s", _ol_exc)

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

    # Phase 9: Signal diagnostics (end of cycle) — write signal_health.json
    try:
        sd_cfg = (policy.get("signal_generation") or {}).get("signal_diagnostics") or {}
        if sd_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.signals.signal_diagnostics import (
                compute_signal_health,
                generate_signal_health_alerts,
                save_signal_health_artifact,
            )
            # Only run if forward_returns col is present (backtest mode typically)
            fwd_col = sd_cfg.get("forward_returns_col", "return_1d")
            if fwd_col in result.prices_with_features.columns:
                factor_cols = [
                    c for c in result.prices_with_features.columns
                    if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume", fwd_col}
                    and result.prices_with_features[c].dtype in ("float64", "float32")
                ][:20]  # cap at 20 factors
                if factor_cols and "timestamp" in result.prices_with_features.columns:
                    health_df = compute_signal_health(
                        result.prices_with_features,
                        forward_returns_col=fwd_col,
                        factor_cols=factor_cols,
                    )
                    alerts = generate_signal_health_alerts(
                        health_df,
                        ic_alert_threshold=float(sd_cfg.get("ic_alert_threshold", 0.0)),
                    )
                    diag_dir = sd_cfg.get("output_dir", "output/diagnostics")
                    save_signal_health_artifact(
                        health_df,
                        alerts,
                        output_dir=str(ctx.output_dir / "diagnostics") if ctx.write_outputs else diag_dir,
                        run_date=ctx.as_of.strftime("%Y-%m-%d") if ctx.as_of else None,
                    )
                    result.meta["signal_diagnostics"] = {
                        "n_factors": len(factor_cols),
                        "n_alerts": len(alerts),
                    }
                    if alerts:
                        log.warning("[SIGNAL-DIAG] %d signal health alerts: %s", len(alerts), alerts)
                    else:
                        log.info("[SIGNAL-DIAG] signal health: %d factors, no alerts", len(factor_cols))
    except Exception as _e:
        log.debug("[SIGNAL-DIAG] signal_diagnostics skipped: %s", _e)

    # Phase 10: Monte Carlo bootstrap (end of cycle, gated by policy monte_carlo.auto_run)
    try:
        mc_cfg = policy.get("monte_carlo") or {}
        if mc_cfg.get("auto_run", False):
            from src.assembled_core.qa.monte_carlo import bootstrap_returns
            # Build return series from prices_filtered if available
            _prices_mc = result.prices_filtered if result.prices_filtered is not None else ctx.prices
            if _prices_mc is not None and not _prices_mc.empty and "close" in _prices_mc.columns and "symbol" in _prices_mc.columns:
                _pivot_mc = _prices_mc.pivot_table(
                    index="timestamp" if "timestamp" in _prices_mc.columns else _prices_mc.columns[0],
                    columns="symbol",
                    values="close",
                )
                _rets_mc = _pivot_mc.pct_change().dropna(how="all")
                _port_rets_mc = _rets_mc.mean(axis=1).dropna()
                if len(_port_rets_mc) >= 20:
                    mc_result = bootstrap_returns(
                        _port_rets_mc,
                        n_paths=int(mc_cfg.get("n_paths", 500)),
                        confidence_level=float(mc_cfg.get("confidence_level", 0.95)),
                        seed=mc_cfg.get("seed", None),
                    )
                    ci_sharpe = mc_result.confidence_intervals["sharpe"]
                    ci_cagr = mc_result.confidence_intervals["cagr"]
                    ci_mdd = mc_result.confidence_intervals["max_drawdown"]
                    result.meta["monte_carlo"] = {
                        "n_paths": mc_result.n_paths,
                        "sharpe_point": round(ci_sharpe.point_estimate, 4),
                        "sharpe_ci_lower": round(ci_sharpe.ci_lower, 4),
                        "sharpe_ci_upper": round(ci_sharpe.ci_upper, 4),
                        "cagr_point": round(ci_cagr.point_estimate, 4),
                        "cagr_ci_lower": round(ci_cagr.ci_lower, 4),
                        "cagr_ci_upper": round(ci_cagr.ci_upper, 4),
                        "max_dd_point": round(ci_mdd.point_estimate, 4),
                        "p_value_vs_zero": round(mc_result.p_value_vs_zero, 4),
                        "confidence_level": mc_result.confidence_intervals["sharpe"].confidence_level,
                    }
                    log.info(
                        "[MONTE-CARLO] bootstrap: paths=%d sharpe=%.3f [%.3f, %.3f] p_vs_zero=%.3f",
                        mc_result.n_paths,
                        ci_sharpe.point_estimate,
                        ci_sharpe.ci_lower,
                        ci_sharpe.ci_upper,
                        mc_result.p_value_vs_zero,
                    )
    except Exception as _e:
        log.debug("[MONTE-CARLO] monte_carlo bootstrap skipped: %s", _e)

    # Phase 11: KPI export — Prometheus metrics after each cycle
    try:
        kpi_cfg = policy.get("kpi_export") or {}
        if kpi_cfg.get("enabled", False):
            from src.assembled_core.ops.metrics_exporter import export_metrics
            kpi_metrics: dict[str, float] = {}
            # Collect available numeric meta values
            kpi_metrics["assembled_orders_generated_total"] = float(len(result.orders_filtered))
            kpi_metrics["assembled_targets_count"] = float(len(result.target_positions))
            kpi_metrics["assembled_signals_count"] = float(len(result.signals))
            # Turnover
            tb_meta = result.meta.get("turnover_budget") or {}
            if "estimated_turnover" in tb_meta and tb_meta["estimated_turnover"] != float("inf"):
                kpi_metrics["assembled_turnover_estimated"] = float(tb_meta["estimated_turnover"])
            # Vol targeting
            vt_meta = result.meta.get("vol_targeting") or {}
            if "realized_vol" in vt_meta:
                kpi_metrics["assembled_realized_vol"] = float(vt_meta["realized_vol"])
            # Monte Carlo if available
            mc_meta = result.meta.get("monte_carlo") or {}
            if "sharpe_point" in mc_meta:
                kpi_metrics["assembled_mc_sharpe"] = float(mc_meta["sharpe_point"])
            # GeoRisk multiplier
            if "georisk_overlay" in result.meta:
                kpi_metrics["assembled_georisk_multiplier"] = float(result.meta["georisk_overlay"].get("multiplier", 1.0))
            metrics_dir = ctx.output_dir / "metrics" if ctx.write_outputs else None
            export_result = export_metrics(
                kpi_metrics,
                labels={"strategy": ctx.strategy_name or "unknown", "mode": ctx.mode},
                path=metrics_dir / "assembled.prom" if metrics_dir else None,
            )
            result.meta["kpi_export"] = {"file": export_result.get("file"), "n_metrics": export_result.get("metrics_count", 0)}
            log.info("[KPI] metrics exported: %d metrics to %s", export_result.get("metrics_count", 0), export_result.get("file"))
    except Exception as _e:
        log.debug("[KPI] kpi_export skipped: %s", _e)

    # [RISK-WIRE] Tail-hedge recommendation — SHADOW-ONLY by default.
    # When enabled the cycle computes a tail-hedge recommendation (VIX-based
    # + vol-elevated) and writes it to result.meta["tail_hedge_recommendation"]
    # without altering orders. Flag `shadow_only=False` would let a future
    # sprint translate the recommendation into an actual hedge position.
    try:
        th_cfg = policy.get("tail_hedging") or {}
        if th_cfg.get("enabled", False):
            from src.assembled_core.risk.tail_hedging import (
                TailHedgeConfig,
                recommend_hedge,
            )
            current_vix = float(th_cfg.get("current_vix", 0.0))
            # Prefer policy-provided vix; otherwise derive from result.meta if available
            if current_vix <= 0.0:
                vix_meta = result.meta.get("vol_targeting") or {}
                current_vix = float(vix_meta.get("current_vix", 0.0))
            portfolio_value = float(getattr(ctx, "capital", 0.0) or 0.0)
            portfolio_vol = float(
                (result.meta.get("vol_targeting") or {}).get("realized_vol", 0.0) or 0.0
            )
            recent_dd = float(
                (result.meta.get("profit_lock_state") or {}).get("max_drawdown", 0.0) or 0.0
            )
            if current_vix > 0.0 and portfolio_value > 0.0:
                cfg_th = TailHedgeConfig(
                    tail_risk_budget_pct=float(th_cfg.get("tail_risk_budget_pct", 1.0)),
                    vix_hedge_trigger=float(th_cfg.get("vix_hedge_trigger", 25.0)),
                    vix_full_hedge_level=float(th_cfg.get("vix_full_hedge_level", 35.0)),
                    max_hedge_ratio=float(th_cfg.get("max_hedge_ratio", 0.30)),
                    min_hedge_ratio=float(th_cfg.get("min_hedge_ratio", 0.05)),
                    put_otm_pct=float(th_cfg.get("put_otm_pct", 0.05)),
                )
                rec = recommend_hedge(
                    portfolio_value=portfolio_value,
                    current_vix=current_vix,
                    portfolio_vol=portfolio_vol,
                    recent_max_drawdown=recent_dd,
                    config=cfg_th,
                )
                result.meta["tail_hedge_recommendation"] = {
                    "hedge_ratio": rec.hedge_ratio,
                    "trigger_reason": rec.trigger_reason,
                    "estimated_annual_cost_pct": rec.estimated_annual_cost_pct,
                    "notional_to_hedge": rec.notional_to_hedge,
                    "put_strike_pct": rec.put_strike_pct,
                    "urgency": rec.urgency,
                    "shadow_only": bool(th_cfg.get("shadow_only", True)),
                }
                log.info(
                    "[RISK-WIRE] tail_hedge_shadow: ratio=%.4f urgency=%.2f reason=%s",
                    rec.hedge_ratio, rec.urgency, rec.trigger_reason,
                )
            else:
                log.debug(
                    "[RISK-WIRE] tail_hedge: skipped (vix=%.1f, pv=%.1f)",
                    current_vix, portfolio_value,
                )
    except Exception as _e:
        log.debug("[RISK-WIRE] tail_hedge skipped: %s", _e)

    # [RISK-WIRE] Attribution report — decompose returns/vol per symbol.
    # Policy-flag-guarded; default OFF so existing callers see no behavior
    # change. Populates result.meta["attribution"] when enabled.
    try:
        attr_cfg = policy.get("attribution") or {}
        if (
            attr_cfg.get("enabled", False)
            and not result.target_positions.empty
            and "target_weight" in result.target_positions.columns
        ):
            from src.assembled_core.risk.attribution import (
                compute_attribution_report,
            )

            weights_map: dict[str, float] = {
                str(r["symbol"]): float(r["target_weight"])
                for _, r in result.target_positions.iterrows()
                if pd.notna(r.get("target_weight"))
            }
            # Per-symbol latest return: 1-bar pct_change from prices_filtered
            returns_map: dict[str, float] = {}
            prices_for_attr = ctx.prices if ctx.prices is not None else result.prices_filtered
            if (
                prices_for_attr is not None
                and not prices_for_attr.empty
                and {"timestamp", "symbol", "close"}.issubset(prices_for_attr.columns)
            ):
                for sym in weights_map:
                    sym_rows = prices_for_attr[prices_for_attr["symbol"] == sym].sort_values("timestamp")
                    if len(sym_rows) >= 2:
                        last = float(sym_rows["close"].iloc[-1])
                        prev = float(sym_rows["close"].iloc[-2])
                        returns_map[sym] = (last - prev) / prev if prev > 0 else 0.0

            attr_report = compute_attribution_report(
                weights=weights_map,
                returns=returns_map,
                prices=prices_for_attr if prices_for_attr is not None else pd.DataFrame(),
                policy=policy,
            )
            result.meta["attribution"] = {
                "status": attr_report.get("status"),
                "portfolio_return": attr_report.get("portfolio_return"),
                "portfolio_vol": attr_report.get("portfolio_vol"),
                "return_contributions": attr_report.get("return_contributions"),
                "vol_contributions": attr_report.get("vol_contributions"),
            }
            log.info(
                "[RISK-WIRE] attribution: status=%s port_ret=%.6f port_vol=%s",
                attr_report.get("status"),
                float(attr_report.get("portfolio_return") or 0.0),
                attr_report.get("portfolio_vol"),
            )
    except Exception as _e:
        log.debug("[RISK-WIRE] attribution skipped: %s", _e)

    # [RISK-WIRE] Portfolio-execution batching — SHADOW-ONLY meta.
    # Policy-flag-guarded; default OFF. Computes execution batches
    # (correlated opposing trades grouped) and writes to
    # result.meta["execution_batches"] without reordering the live
    # order stream. A future sprint can promote this to an actual
    # batch-aware broker dispatcher.
    try:
        pe_cfg = policy.get("portfolio_execution") or {}
        if (
            pe_cfg.get("enabled", False)
            and not result.orders_filtered.empty
            and {"symbol", "qty"}.issubset(result.orders_filtered.columns)
        ):
            from src.assembled_core.execution.portfolio_execution import (
                optimize_execution_sequence,
            )

            max_parallel = int(pe_cfg.get("max_parallel", 5))
            corr_lookback = int(pe_cfg.get("corr_lookback_days", 60))

            # Build correlation matrix from filtered prices (pivot wide).
            corr_mtx: pd.DataFrame | None = None
            prices_for_pe = result.prices_filtered if result.prices_filtered is not None else ctx.prices
            if (
                prices_for_pe is not None
                and not prices_for_pe.empty
                and {"timestamp", "symbol", "close"}.issubset(prices_for_pe.columns)
            ):
                wide = (
                    prices_for_pe.pivot_table(
                        index="timestamp", columns="symbol", values="close", aggfunc="last"
                    )
                    .sort_index()
                    .tail(corr_lookback)
                    .pct_change()
                    .dropna(how="all")
                )
                if len(wide) >= 5 and wide.shape[1] >= 2:
                    corr_mtx = wide.corr()

            batched = optimize_execution_sequence(
                result.orders_filtered.reset_index(drop=True),
                correlation_matrix=corr_mtx,
                max_parallel=max_parallel,
            )
            n_batches = int(batched["execution_batch"].nunique()) if not batched.empty else 0
            result.meta["execution_batches"] = {
                "n_orders": int(len(batched)),
                "n_batches": n_batches,
                "max_parallel": max_parallel,
                "shadow_only": True,
                "batches": batched[["symbol", "qty", "execution_batch"]].to_dict(
                    orient="records"
                ) if not batched.empty else [],
            }
            log.info(
                "[RISK-WIRE] portfolio_execution (shadow): %d orders -> %d batches",
                len(batched), n_batches,
            )
    except Exception as _e:
        log.debug("[RISK-WIRE] portfolio_execution skipped: %s", _e)

    # [RISK-WIRE] Almgren-Chriss impact-cost estimate — SHADOW-ONLY meta.
    # Policy-flag-guarded; default OFF. Per-order estimate plus aggregate.
    # This is pre-trade insight, not a replacement of the paper engine's
    # fill model. Populates result.meta["almgren_chriss_impact"].
    try:
        ac_cfg = policy.get("almgren_chriss") or {}
        if (
            ac_cfg.get("enabled", False)
            and not result.orders_filtered.empty
            and {"symbol", "qty", "price"}.issubset(result.orders_filtered.columns)
        ):
            from src.assembled_core.execution.almgren_chriss import (
                estimate_impact_cost,
            )

            default_adv = float(ac_cfg.get("default_adv", 1_000_000.0))
            gamma = float(ac_cfg.get("gamma", 0.1))
            eta = float(ac_cfg.get("eta", 0.05))
            horizon_days = float(ac_cfg.get("horizon_days", 1.0))
            sigma_default = float(ac_cfg.get("sigma_default", 0.02))

            # Per-symbol realised sigma from prices_filtered if available.
            sigma_map: dict[str, float] = {}
            prices_for_ac = result.prices_filtered if result.prices_filtered is not None else ctx.prices
            if (
                prices_for_ac is not None
                and not prices_for_ac.empty
                and {"timestamp", "symbol", "close"}.issubset(prices_for_ac.columns)
            ):
                wide_ac = (
                    prices_for_ac.pivot_table(
                        index="timestamp", columns="symbol", values="close", aggfunc="last"
                    )
                    .sort_index()
                    .tail(60)
                    .pct_change()
                    .dropna(how="all")
                )
                if not wide_ac.empty:
                    sigma_series = wide_ac.std(ddof=0)
                    sigma_map = {
                        str(s): float(v) for s, v in sigma_series.items() if pd.notna(v)
                    }

            per_order: list[dict] = []
            tot_notional = 0.0
            tot_cost_usd = 0.0
            for _idx, _row in result.orders_filtered.iterrows():
                sym = str(_row["symbol"])
                qty = float(_row["qty"])
                px = float(_row["price"])
                if qty == 0 or px <= 0:
                    continue
                sigma = sigma_map.get(sym, sigma_default)
                est = estimate_impact_cost(
                    total_shares=qty,
                    price=px,
                    adv=default_adv,
                    sigma=sigma,
                    horizon_days=horizon_days,
                    gamma=gamma,
                    eta=eta,
                )
                per_order.append({"symbol": sym, **est})
                tot_notional += abs(qty) * px
                tot_cost_usd += est["total_cost_usd"]
            total_bps = (tot_cost_usd / tot_notional * 10_000) if tot_notional > 0 else 0.0
            result.meta["almgren_chriss_impact"] = {
                "total_notional_usd": round(tot_notional, 2),
                "total_cost_usd": round(tot_cost_usd, 2),
                "aggregate_bps": round(total_bps, 2),
                "per_order": per_order,
                "shadow_only": True,
            }
            log.info(
                "[RISK-WIRE] almgren_chriss (shadow): notional=%.0f cost=%.2f bps=%.2f",
                tot_notional, tot_cost_usd, total_bps,
            )
    except Exception as _e:
        log.debug("[RISK-WIRE] almgren_chriss skipped: %s", _e)

    log.info(
        f"Trading cycle completed successfully: {len(result.orders_filtered)} orders"
    )

    return result
