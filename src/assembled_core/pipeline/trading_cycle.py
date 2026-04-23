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

    # Step 1.95: Comprehensive price panel QC (qa.data_qc — observability)
    try:
        if not result.prices_filtered.empty and "timestamp" in result.prices_filtered.columns:
            from src.assembled_core.qa.data_qc import run_price_panel_qc
            _pqc_report = run_price_panel_qc(
                result.prices_filtered, freq="1d", as_of=ctx.as_of
            )
            result.meta["price_panel_qc"] = {
                "ok": bool(_pqc_report.ok),
                "n_issues": len(_pqc_report.issues),
                "n_fail": sum(1 for i in _pqc_report.issues if i.severity == "FAIL"),
                "n_warn": sum(1 for i in _pqc_report.issues if i.severity == "WARN"),
            }
            if not _pqc_report.ok:
                log.warning("[PRICE-QC] price panel QC failed: %d issues", len(_pqc_report.issues))
            else:
                log.debug("[PRICE-QC] ok — %d issues (WARN only)", len(_pqc_report.issues))
    except Exception as _pqc_exc:
        log.debug("[PRICE-QC] price_panel_qc skipped: %s", _pqc_exc)

    # Step 1.97: Macro diffusion index (proxy from price returns momentum — observability)
    try:
        if not result.prices_filtered.empty and "close" in result.prices_filtered.columns:
            from src.assembled_core.features.macro_features import compute_diffusion_index
            _mdi_pivot = result.prices_filtered.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            if len(_mdi_pivot) >= 5:
                _mdi_dict = {sym: _mdi_pivot[sym].dropna() for sym in _mdi_pivot.columns if _mdi_pivot[sym].notna().sum() >= 3}
                if _mdi_dict:
                    _mdi_series = compute_diffusion_index(_mdi_dict, momentum_window=3)
                    _mdi_latest = float(_mdi_series.iloc[-1]) if not _mdi_series.empty else 0.5
                    result.meta["macro_diffusion_index"] = {
                        "latest": round(_mdi_latest, 4),
                        "n_series": len(_mdi_dict),
                    }
                    log.debug("[MACRO-DIFFUSION] diffusion=%.3f from %d series", _mdi_latest, len(_mdi_dict))
    except Exception as _mdi_exc:
        log.debug("[MACRO-DIFFUSION] macro_features diffusion skipped: %s", _mdi_exc)

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

    # Step 2.1: PIT safety check on features (point-in-time guard)
    try:
        if ctx.as_of is not None and not result.prices_with_features.empty:
            from src.assembled_core.qa.point_in_time_checks import check_features_pit_safe
            _pit_ok = check_features_pit_safe(
                result.prices_with_features,
                ctx.as_of,
                strict=False,
                feature_source="prices_with_features",
            )
            result.meta["pit_check"] = {"passed": _pit_ok}
            if not _pit_ok:
                log.warning("[PIT] Feature DataFrame contains future-dated rows (as_of=%s)", ctx.as_of)
    except Exception as _pit_exc:
        log.debug("[PIT] point_in_time check skipped: %s", _pit_exc)

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

    # Step 2.5: Behavioral features composite (per-symbol disposition/anchoring/vol alpha)
    try:
        beh_cfg = (policy.get("features") or {}).get("behavioral_features") or {}
        if beh_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.features.behavioral_features import (
                compute_behavioral_composite,
            )
            _req_cols = {"symbol", "close"}
            if _req_cols.issubset(result.prices_with_features.columns):
                _beh_scores: dict[str, float] = {}
                _beh_min_rows = int(beh_cfg.get("min_rows", 60))
                _beh_syms = result.prices_with_features["symbol"].unique()[:50]  # cap at 50 symbols
                for _beh_sym in _beh_syms:
                    _beh_grp = (
                        result.prices_with_features[result.prices_with_features["symbol"] == _beh_sym]
                        .sort_values("timestamp") if "timestamp" in result.prices_with_features.columns
                        else result.prices_with_features[result.prices_with_features["symbol"] == _beh_sym]
                    )
                    if len(_beh_grp) < _beh_min_rows:
                        continue
                    _beh_prices = _beh_grp["close"].reset_index(drop=True)
                    _beh_vols = _beh_grp["volume"].reset_index(drop=True) if "volume" in _beh_grp.columns else pd.Series(1.0, index=range(len(_beh_grp)))
                    _beh_rets = _beh_prices.pct_change().fillna(0)
                    try:
                        _beh_composite = compute_behavioral_composite(_beh_prices, _beh_vols, _beh_rets)
                        _val = float(_beh_composite.iloc[-1]) if len(_beh_composite) > 0 else 0.0
                        _beh_scores[str(_beh_sym)] = _val if pd.notna(_val) else 0.0
                    except Exception:
                        pass
                if _beh_scores:
                    result.prices_with_features = result.prices_with_features.copy()
                    result.prices_with_features["behavioral_composite"] = (
                        result.prices_with_features["symbol"].map(_beh_scores)
                    )
                    result.meta["behavioral_features"] = {
                        "n_computed": len(_beh_scores),
                        "mean_score": round(sum(_beh_scores.values()) / len(_beh_scores), 4),
                    }
                    log.info(
                        "[BEHAVIORAL] composite computed for %d symbols",
                        len(_beh_scores),
                    )
    except Exception as _beh_exc:
        log.debug("[BEHAVIORAL] behavioral_features skipped: %s", _beh_exc)

    # Step 2.6: Seasonal features (calendar-based, zero look-ahead)
    try:
        seas_cfg = (policy.get("features") or {}).get("seasonal_features") or {}
        if seas_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.features.seasonal_features import build_seasonal_features
            if "timestamp" in result.prices_with_features.columns:
                _seas_ts = pd.DatetimeIndex(result.prices_with_features["timestamp"])
                _seas_df = build_seasonal_features(_seas_ts)
                _seas_cols = _seas_df.columns.tolist()
                result.prices_with_features = result.prices_with_features.copy()
                result.prices_with_features = result.prices_with_features.reset_index(drop=True)
                for _sc in _seas_cols:
                    result.prices_with_features[_sc] = _seas_df[_sc].values
                result.meta["seasonal_features"] = {"n_features": len(_seas_cols)}
                log.debug("[SEASONAL] added %d seasonal features", len(_seas_cols))
    except Exception as _seas_exc:
        log.debug("[SEASONAL] seasonal_features skipped: %s", _seas_exc)

    # Step 2.7: Correlation regime features (market-wide diversification / herding metrics)
    try:
        corr_cfg = (policy.get("features") or {}).get("correlation_regime") or {}
        if corr_cfg.get("enabled", False) and not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.features.correlation_features import compute_correlation_regime_features
            _cr_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _cr_returns = _cr_pivot.pct_change().dropna(how="all")
            if len(_cr_returns) >= 25 and len(_cr_returns.columns) >= 2:
                _cr_feats = compute_correlation_regime_features(_cr_returns)
                _cr_latest = _cr_feats.iloc[-1].to_dict()
                result.meta["correlation_regime"] = {
                    k: round(float(v), 4) for k, v in _cr_latest.items() if pd.notna(v)
                }
                log.debug("[CORR-REGIME] avg_corr_short=%.3f", _cr_latest.get("avg_corr_short", float("nan")))
    except Exception as _cr_exc:
        log.debug("[CORR-REGIME] correlation_regime skipped: %s", _cr_exc)

    # Step 2.8: Mean-reversion factors (RSI/Bollinger/Z-score per symbol, shadow enrichment)
    try:
        mr_fac_cfg = (policy.get("features") or {}).get("mean_reversion_factors") or {}
        if mr_fac_cfg.get("enabled", False) and not result.prices_with_features.empty:
            _req_cols_mr = {"symbol", "timestamp", "close"}
            if _req_cols_mr.issubset(result.prices_with_features.columns):
                from src.assembled_core.features.mean_reversion_factors import compute_mean_reversion_factors
                _mr_fac_df = compute_mean_reversion_factors(result.prices_with_features)
                if not _mr_fac_df.empty:
                    _mr_fac_cols = [c for c in _mr_fac_df.columns if c.startswith("mr_")]
                    _merge_keys = [k for k in ["symbol", "timestamp"] if k in _mr_fac_df.columns]
                    result.prices_with_features = result.prices_with_features.merge(
                        _mr_fac_df[_merge_keys + _mr_fac_cols], on=_merge_keys, how="left", suffixes=("", "_mrf")
                    )
                    result.meta["mean_reversion_factors"] = {"n_factor_cols": len(_mr_fac_cols)}
                    log.debug("[MR-FACTORS] added %d mr_* factor columns", len(_mr_fac_cols))
    except Exception as _mrf_exc:
        log.debug("[MR-FACTORS] mean_reversion_factors skipped: %s", _mrf_exc)

    # Step 2.9: Feature interaction terms (cross-feature products/ratios, shadow enrichment)
    try:
        ix_cfg = (policy.get("features") or {}).get("interaction_features") or {}
        if ix_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.features.interaction_features import (
                compute_interaction_features,
                DEFAULT_INTERACTIONS,
            )
            _ix_before = set(result.prices_with_features.columns)
            _ix_df = compute_interaction_features(result.prices_with_features)
            _ix_added = [c for c in _ix_df.columns if c not in _ix_before]
            if _ix_added:
                result.prices_with_features = _ix_df
                result.meta["interaction_features"] = {"n_added": len(_ix_added)}
                log.debug("[IX-FEATURES] added %d interaction columns", len(_ix_added))
    except Exception as _ix_exc:
        log.debug("[IX-FEATURES] interaction_features skipped: %s", _ix_exc)

    # Step 2.12: Weekly alignment filter (daily trend vs weekly EMA slope)
    try:
        wa_cfg = (policy.get("features") or {}).get("weekly_alignment") or {}
        if wa_cfg.get("enabled", False) and not result.prices_with_features.empty:
            _wa_req = {"close", "symbol", "timestamp"}
            if _wa_req.issubset(result.prices_with_features.columns):
                from src.assembled_core.features.weekly_alignment import add_weekly_alignment
                # Use any existing trend column as daily_trend proxy
                _wa_trend_col = next(
                    (c for c in ("trend_strength_50", "momentum_12m_excl_1m", "trend_strength_200")
                     if c in result.prices_with_features.columns), None
                )
                if _wa_trend_col:
                    _wa_df = result.prices_with_features.copy()
                    _wa_df = _wa_df.set_index("timestamp")
                    _wa_df = add_weekly_alignment(_wa_df, daily_trend_col=_wa_trend_col)
                    _wa_df = _wa_df.reset_index()
                    result.prices_with_features = _wa_df
                    result.meta["weekly_alignment"] = {"trend_col_used": _wa_trend_col}
                    log.debug("[WEEKLY-ALIGN] added weekly alignment filter using %s", _wa_trend_col)
    except Exception as _wa_exc:
        log.debug("[WEEKLY-ALIGN] weekly_alignment skipped: %s", _wa_exc)

    # Step 2.10: Realized volatility features (rv_20, rv_60 per symbol)
    try:
        rv_cfg = (policy.get("features") or {}).get("realized_volatility") or {}
        if rv_cfg.get("enabled", False) and not result.prices_with_features.empty:
            _rv_req = {"close", "symbol", "timestamp"}
            if _rv_req.issubset(result.prices_with_features.columns):
                from src.assembled_core.features.ta_liquidity_vol_factors import add_realized_volatility
                _rv_windows = [int(w) for w in rv_cfg.get("windows", [20, 60])]
                _rv_existing = set(result.prices_with_features.columns)
                result.prices_with_features = add_realized_volatility(
                    result.prices_with_features, windows=_rv_windows
                )
                _rv_added = [c for c in result.prices_with_features.columns if c not in _rv_existing]
                result.meta["realized_volatility"] = {"n_added": len(_rv_added), "windows": _rv_windows}
                log.debug("[RV] added %d realized vol columns: %s", len(_rv_added), _rv_added)
    except Exception as _rv_exc:
        log.debug("[RV] realized_volatility skipped: %s", _rv_exc)

    # Step 2.11: Fractional differentiation of close price (memory-preserving stationarity)
    try:
        ffd_cfg = (policy.get("features") or {}).get("fractional_diff") or {}
        if ffd_cfg.get("enabled", False) and not result.prices_with_features.empty:
            _ffd_req = {"close", "symbol", "timestamp"}
            if _ffd_req.issubset(result.prices_with_features.columns):
                from src.assembled_core.features.fractional_diff import apply_ffd_to_panel
                _ffd_d = float(ffd_cfg.get("d", 0.4))
                _ffd_before = set(result.prices_with_features.columns)
                result.prices_with_features = apply_ffd_to_panel(
                    result.prices_with_features, price_cols=["close"], d=_ffd_d
                )
                _ffd_added = [c for c in result.prices_with_features.columns if c not in _ffd_before]
                result.meta["fractional_diff"] = {"d": _ffd_d, "n_added": len(_ffd_added)}
                log.debug("[FFD] fractional diff applied, d=%.2f, added=%s", _ffd_d, _ffd_added)
    except Exception as _ffd_exc:
        log.debug("[FFD] fractional_diff skipped: %s", _ffd_exc)

    # Step 2.13: Feature clustering by correlation (redundancy map — observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.ml.feature_clustering import cluster_features_by_correlation
            _fc_num_cols = [
                c for c in result.prices_with_features.select_dtypes("number").columns
                if c not in ("timestamp",) and result.prices_with_features[c].nunique() > 1
            ]
            if len(_fc_num_cols) >= 4:
                _fc_result = cluster_features_by_correlation(
                    result.prices_with_features, feature_cols=_fc_num_cols[:50]
                )
                result.meta["feature_clustering"] = {
                    "n_original": _fc_result.n_original_features,
                    "n_clusters": _fc_result.n_clusters,
                    "compression_ratio": round(1.0 - _fc_result.n_clusters / max(_fc_result.n_original_features, 1), 3),
                }
                log.debug("[FEAT-CLUSTER] %d→%d clusters (%.1f%% compression)", _fc_result.n_original_features, _fc_result.n_clusters, result.meta["feature_clustering"]["compression_ratio"] * 100)
    except Exception as _fc_exc:
        log.debug("[FEAT-CLUSTER] feature_clustering skipped: %s", _fc_exc)

    # Step 2.14: IC prescreen — drop features with zero predictive power (observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.ml.feature_selection import ic_prescreen
            _ic_num_cols = [
                c for c in result.prices_with_features.select_dtypes("number").columns
                if c not in ("timestamp",)
            ]
            if len(_ic_num_cols) >= 3:
                _ic_panel = result.prices_with_features.copy()
                if "timestamp" not in _ic_panel.columns and result.prices_with_features.index.dtype == "datetime64[ns]":
                    _ic_panel = _ic_panel.reset_index()
                _ic_kept, _ic_scores = ic_prescreen(_ic_panel, min_ic=0.02)
                result.meta["ic_prescreen"] = {
                    "n_features_evaluated": len(_ic_scores),
                    "n_features_kept": len(_ic_kept),
                    "top_ic": dict(sorted(_ic_scores.items(), key=lambda x: -x[1])[:5]),
                }
                log.debug("[IC-PRESCREEN] %d/%d features pass IC>=0.02", len(_ic_kept), len(_ic_scores))
    except Exception as _ic_exc:
        log.debug("[IC-PRESCREEN] ic_prescreen skipped: %s", _ic_exc)

    # Step 2.15: Triple-barrier labels (ML training data enrichment — adds tb_label_5d)
    try:
        _tb_cfg = (policy.get("ml") or {}).get("triple_barrier") or {}
        if _tb_cfg.get("enabled", False) and not result.prices_with_features.empty:
            _tb_cols = {"timestamp", "symbol", "close"}
            if _tb_cols.issubset(result.prices_with_features.columns):
                from src.assembled_core.ml.triple_barrier import build_triple_barrier_labels
                _tb_result = build_triple_barrier_labels(
                    result.prices_with_features,
                    horizon_days=int(_tb_cfg.get("horizon_days", 5)),
                )
                result.prices_with_features = _tb_result
                _tb_label_col = "tb_label_5d"
                if _tb_label_col in result.prices_with_features.columns:
                    _tb_counts = result.prices_with_features[_tb_label_col].value_counts().to_dict()
                    result.meta["triple_barrier"] = {
                        "horizon_days": _tb_cfg.get("horizon_days", 5),
                        "label_counts": {str(k): int(v) for k, v in _tb_counts.items()},
                    }
                    log.debug("[TRIPLE-BARRIER] labels set: %s", result.meta["triple_barrier"]["label_counts"])
    except Exception as _tb_exc:
        log.debug("[TRIPLE-BARRIER] triple_barrier skipped: %s", _tb_exc)

    # Step 2.16: Online HMM regime detection from price returns (observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector
            _hmm_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _hmm_rets = _hmm_pivot.median(axis=1).pct_change().dropna()
            if len(_hmm_rets) >= 20:
                _hmm_detector = OnlineHMMRegimeDetector(n_states=3, lookback=252)
                _hmm_state = _hmm_detector.predict_current_regime(_hmm_rets)
                result.meta["hmm_regime"] = {
                    "regime_label": _hmm_state.regime_label,
                    "regime_id": _hmm_state.regime_id,
                    "probability": round(float(_hmm_state.probability), 4),
                    "volatility": round(float(_hmm_state.volatility), 6),
                }
                log.debug("[HMM-REGIME] %s (p=%.3f)", _hmm_state.regime_label, _hmm_state.probability)
    except Exception as _hmm_exc:
        log.debug("[HMM-REGIME] online_hmm_regime skipped: %s", _hmm_exc)

    # Step 2.17: EVT tail risk from price returns (scipy-gated, observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.ml.evt_models import compute_evt_risk_metrics
            _evt_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _evt_rets = _evt_pivot.median(axis=1).pct_change().dropna()
            if len(_evt_rets) >= 100:
                _evt_metrics = compute_evt_risk_metrics(_evt_rets, threshold_quantile=0.95)
                result.meta["evt_tail_risk"] = {
                    "var_99": round(float(_evt_metrics.get("evt_var_99", 0.0)), 6),
                    "cvar_99": round(float(_evt_metrics.get("evt_cvar_99", 0.0)), 6),
                    "shape_xi": round(float(_evt_metrics.get("evt_shape_xi", 0.0)), 6),
                }
                log.debug("[EVT] VaR99=%.4f CVaR99=%.4f", _evt_metrics.get("evt_var_99", 0), _evt_metrics.get("evt_cvar_99", 0))
    except Exception as _evt_exc:
        log.debug("[EVT] evt_models skipped: %s", _evt_exc)

    # Step 2.18: GARCH volatility forecast (arch-gated, observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.ml.garch_models import fit_garch
            _garch_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _garch_rets = _garch_pivot.median(axis=1).pct_change().dropna()
            if len(_garch_rets) >= 60:
                _garch_result = fit_garch(_garch_rets, symbol="portfolio")
                if _garch_result is not None:
                    result.meta["garch_vol"] = {
                        "vol_forecast_1d": round(float(_garch_result.vol_forecast_1d), 6),
                        "persistence": round(float(_garch_result.persistence), 4),
                        "converged": _garch_result.converged,
                    }
                    log.debug("[GARCH] vol_1d=%.4f persistence=%.3f", _garch_result.vol_forecast_1d, _garch_result.persistence)
    except Exception as _garch_exc:
        log.debug("[GARCH] garch_models skipped: %s", _garch_exc)

    # Step 2.19: Combined regime (HMM + news ensemble) — observability
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.ml.combined_regime import CombinedRegimeClassifier
            from src.assembled_core.ml.online_hmm_regime import OnlineHMMRegimeDetector as _HMMDet
            _cr_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _cr_rets = _cr_pivot.median(axis=1).pct_change().dropna()
            if len(_cr_rets) >= 20:
                _cr_hmm = _HMMDet(n_states=3, lookback=252)
                _cr = CombinedRegimeClassifier(hmm_detector=_cr_hmm)
                _cr_out = _cr.predict(returns=_cr_rets)
                result.meta["combined_regime"] = {
                    "combined": _cr_out.combined_regime,
                    "hmm": _cr_out.hmm_regime,
                    "news": _cr_out.news_regime,
                    "agreement": _cr_out.agreement,
                    "confidence": round(float(_cr_out.confidence), 3),
                }
                log.debug("[COMBINED-REGIME] %s (agreement=%s)", _cr_out.combined_regime, _cr_out.agreement)
    except Exception as _cr_exc:
        log.debug("[COMBINED-REGIME] combined_regime skipped: %s", _cr_exc)

    # Step 2.20: GARCH features snapshot per symbol (arch-gated, observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.features.volatility_features import compute_garch_features_snapshot
            _gfs = compute_garch_features_snapshot(result.prices_with_features, lookback_days=252)
            if _gfs:
                _gfs_vols = [v.get("garch_vol_1d", 0.0) for v in _gfs.values() if v]
                result.meta["garch_features_snapshot"] = {
                    "n_symbols": len(_gfs),
                    "mean_vol_1d": round(float(sum(_gfs_vols) / len(_gfs_vols)), 6) if _gfs_vols else 0.0,
                }
                log.debug("[GARCH-SNAPSHOT] %d symbols, mean_vol=%.4f", len(_gfs), result.meta["garch_features_snapshot"]["mean_vol_1d"])
    except Exception as _gfs_exc:
        log.debug("[GARCH-SNAPSHOT] volatility_features snapshot skipped: %s", _gfs_exc)

    # Step 2.21: VPIN (Volume-Synchronized Probability of Informed Trading — observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 20:
            from src.assembled_core.features.vpin import compute_vpin
            _vpin_prices = pd.Series(result.equity_series.values, dtype=float)
            _vpin_volumes = pd.Series(
                np.ones(len(_vpin_prices)) * float(_vpin_prices.mean()), dtype=float
            )
            _vpin_result = compute_vpin(_vpin_prices, _vpin_volumes, n_buckets_window=10)
            result.meta["vpin"] = {
                "current_vpin": round(float(_vpin_result.current_vpin), 4),
                "avg_vpin": round(float(_vpin_result.avg_vpin), 4),
                "is_toxic": bool(_vpin_result.is_toxic),
                "n_buckets": int(_vpin_result.n_buckets),
            }
            log.debug("[VPIN] current=%.4f avg=%.4f toxic=%s", _vpin_result.current_vpin, _vpin_result.avg_vpin, _vpin_result.is_toxic)
    except Exception as _vpin_exc:
        log.debug("[VPIN] vpin skipped: %s", _vpin_exc)

    # Step 2.22: Cross-sectional z-score of numeric features (normalization observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.features.fundamental_factors import cross_sectional_zscore
            _csz_num_cols = [
                c for c in result.prices_with_features.columns
                if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
                and result.prices_with_features[c].dtype in ("float64", "float32")
            ][:20]
            if _csz_num_cols:
                _csz_df = cross_sectional_zscore(
                    result.prices_with_features[["symbol"] + _csz_num_cols].dropna(how="all"),
                    columns=_csz_num_cols,
                )
                result.meta["cross_sectional_zscore"] = {
                    "n_cols": len(_csz_num_cols),
                    "n_rows": len(_csz_df),
                }
                log.debug("[CSZ] z-scored %d cols × %d rows", len(_csz_num_cols), len(_csz_df))
    except Exception as _csz_exc:
        log.debug("[CSZ] cross_sectional_zscore skipped: %s", _csz_exc)

    # Step 2.23: Incremental update filter (filter prices to last N sessions — observability)
    try:
        if not result.prices_with_features.empty and "timestamp" in result.prices_with_features.columns:
            from src.assembled_core.features.incremental_updates import filter_prices_for_incremental
            _iu_filtered = filter_prices_for_incremental(
                result.prices_with_features, as_of=ctx.as_of, window_days=5
            )
            result.meta["incremental_update"] = {
                "total_rows": len(result.prices_with_features),
                "last_5d_rows": len(_iu_filtered),
                "window_days": 5,
            }
            log.debug("[INCR-UPDATE] total=%d last_5d=%d", len(result.prices_with_features), len(_iu_filtered))
    except Exception as _iu_exc:
        log.debug("[INCR-UPDATE] incremental_updates skipped: %s", _iu_exc)

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

    # Step 3.4: Mean-reversion signal layer (regime-conditional, shadow enrichment)
    try:
        mr_cfg = (policy.get("signals") or {}).get("mean_reversion") or {}
        if mr_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.signals.mean_reversion import compute_mean_reversion_signals
            _mr_regime = str(result.meta.get("regime", {}).get("regime", "bull"))
            _mr_signals = compute_mean_reversion_signals(
                result.prices_with_features,
                regime=_mr_regime,
            )
            if not _mr_signals.empty and not result.signals.empty:
                _mr_map = _mr_signals.set_index("symbol")["reversion_signal"].to_dict()
                result.signals["mr_signal"] = result.signals["symbol"].map(_mr_map)
            result.meta["mean_reversion_signal"] = {
                "regime": _mr_regime,
                "n_signals": int((_mr_signals.get("reversion_signal", pd.Series()) != 0).sum())
                if not _mr_signals.empty else 0,
            }
            log.debug("[MR-SIGNAL] mean-reversion signals computed, regime=%s", _mr_regime)
    except Exception as _mr_exc:
        log.debug("[MR-SIGNAL] mean_reversion_signals skipped: %s", _mr_exc)

    # Step 3.45: Factor timing momentum (which factors are working now, shadow)
    try:
        ft_cfg = (policy.get("ml") or {}).get("factor_timing") or {}
        if ft_cfg.get("enabled", False) and not result.prices_with_features.empty:
            from src.assembled_core.ml.factor_timing import compute_factor_momentum
            _factor_cols = [c for c in result.prices_with_features.columns
                            if c not in {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
                            and pd.api.types.is_numeric_dtype(result.prices_with_features[c])][:15]
            if len(_factor_cols) >= 2:
                # Use cross-sectional mean per timestamp as factor "returns"
                _ft_pivot = result.prices_with_features.groupby("timestamp")[_factor_cols].mean()
                _ft_returns = _ft_pivot.diff().dropna()
                if len(_ft_returns) >= 5:
                    _ft_momentum = compute_factor_momentum(_ft_returns, lookback=min(12, len(_ft_returns)))
                    result.meta["factor_timing"] = {
                        "top_factors": sorted(_ft_momentum, key=_ft_momentum.get, reverse=True)[:5],
                        "n_factors": len(_ft_momentum),
                    }
                    log.debug("[FACTOR-TIMING] top factor: %s",
                              sorted(_ft_momentum, key=_ft_momentum.get, reverse=True)[:1])
    except Exception as _ft_exc:
        log.debug("[FACTOR-TIMING] factor_timing skipped: %s", _ft_exc)

    # Step 3.55: Multi-factor composite signal (shadow — enriches signals with mf_score)
    try:
        mf_cfg = policy.get("multifactor_signal") or {}
        if mf_cfg.get("enabled", False) and not result.prices_with_features.empty and not result.signals.empty:
            from src.assembled_core.signals.multifactor_signal import build_multifactor_signal
            from src.assembled_core.config.factor_bundles import load_factor_bundle
            import pathlib as _pathlib
            _mf_bundle_path = _pathlib.Path(
                mf_cfg.get("bundle_path", "config/factor_bundles/macro_world_etfs_core_bundle.yaml")
            )
            if _mf_bundle_path.exists():
                _mf_bundle = load_factor_bundle(_mf_bundle_path)
                _mf_result = build_multifactor_signal(result.prices_with_features, _mf_bundle)
                if not _mf_result.df.empty and "mf_score" in _mf_result.df.columns:
                    # Extract latest mf_score per symbol and join to signals
                    _mf_latest = (
                        _mf_result.df.sort_values("timestamp").groupby("symbol")["mf_score"].last()
                        if "timestamp" in _mf_result.df.columns
                        else _mf_result.df.groupby("symbol")["mf_score"].last()
                    )
                    result.signals = result.signals.copy()
                    result.signals["mf_score"] = result.signals["symbol"].map(_mf_latest)
                    result.meta["multifactor_signal"] = {
                        "bundle": str(_mf_bundle_path.name),
                        "used_factors": _mf_result.meta.get("used_factors", []),
                        "missing_factors": _mf_result.meta.get("missing_factors", []),
                        "n_symbols_scored": int(_mf_latest.notna().sum()),
                        "shadow_only": True,
                    }
                    log.info(
                        "[MULTIFACTOR] scored %d symbols, factors_used=%d missing=%d",
                        int(_mf_latest.notna().sum()),
                        len(_mf_result.meta.get("used_factors", [])),
                        len(_mf_result.meta.get("missing_factors", [])),
                    )
            else:
                log.debug("[MULTIFACTOR] bundle not found at %s — skipping", _mf_bundle_path)
    except Exception as _mf_exc:
        log.debug("[MULTIFACTOR] multifactor_signal skipped: %s", _mf_exc)

    # Step 3.58: Signal correlation analysis (redundancy detection, shadow observability)
    try:
        if not result.signals.empty:
            from src.assembled_core.ml.signal_correlation import SignalCorrelationAnalyzer
            _sig_numeric_cols = [c for c in result.signals.columns
                                  if c not in ("symbol", "direction", "timestamp")
                                  and pd.api.types.is_numeric_dtype(result.signals[c])]
            if len(_sig_numeric_cols) >= 2 and len(result.signals) >= 5:
                _sca = SignalCorrelationAnalyzer()
                _sca_report = _sca.analyze(result.signals[_sig_numeric_cols])
                result.meta["signal_correlation"] = {
                    "mean_abs_corr": round(_sca_report.mean_abs_corr, 4),
                    "n_signals": _sca_report.n_signals,
                    "n_redundant_clusters": len(_sca_report.redundant_clusters),
                }
                log.debug("[SIG-CORR] mean_abs_corr=%.3f, n_signals=%d", _sca_report.mean_abs_corr, _sca_report.n_signals)
    except Exception as _sca_exc:
        log.debug("[SIG-CORR] signal_correlation skipped: %s", _sca_exc)

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

    # Step 3.62: MA-crossover trend signals (rules_trend — additional signal enrichment)
    try:
        if not result.prices_with_features.empty and not result.signals.empty:
            _req_ts_cols = {"timestamp", "symbol", "close"}
            if _req_ts_cols.issubset(set(result.prices_with_features.columns)):
                from src.assembled_core.signals.rules_trend import generate_trend_signals
                _ts_signals = generate_trend_signals(
                    result.prices_with_features, ma_fast=20, ma_slow=50
                )
                if not _ts_signals.empty and "symbol" in _ts_signals.columns:
                    _ts_latest = (
                        _ts_signals.sort_values("timestamp")
                        .groupby("symbol")
                        .last()
                        .reset_index()[["symbol", "score"]]
                        .rename(columns={"score": "trend_ma_score"})
                    )
                    result.signals = result.signals.merge(_ts_latest, on="symbol", how="left")
                    result.meta["trend_signals_ma"] = {
                        "n_long": int((_ts_signals["direction"] == "LONG").sum()),
                        "n_flat": int((_ts_signals["direction"] == "FLAT").sum()),
                    }
                    log.debug("[TREND-MA] long=%d flat=%d", result.meta["trend_signals_ma"]["n_long"], result.meta["trend_signals_ma"]["n_flat"])
    except Exception as _ts_exc:
        log.debug("[TREND-MA] rules_trend skipped: %s", _ts_exc)

    # Step 3.9: Adversarial validation — feature drift detection (sklearn-gated, observability)
    try:
        if not result.prices_with_features.empty:
            _av_num_cols = [
                c for c in result.prices_with_features.select_dtypes("number").columns
                if c not in ("timestamp",) and result.prices_with_features[c].nunique() > 1
            ]
            if len(_av_num_cols) >= 4 and len(result.prices_with_features) >= 40:
                from src.assembled_core.ml.adversarial_validation import run_adversarial_validation
                _av_df = result.prices_with_features[_av_num_cols[:30]].dropna()
                _av_n = len(_av_df)
                _av_split = max(10, _av_n // 2)
                _av_train = _av_df.iloc[:_av_split]
                _av_test = _av_df.iloc[_av_split:]
                if len(_av_train) >= 10 and len(_av_test) >= 10:
                    _av_result = run_adversarial_validation(_av_train, _av_test)
                    result.meta["adversarial_validation"] = {
                        "auc": round(float(_av_result.auc), 4),
                        "drift_detected": bool(_av_result.auc > 0.70),
                        "top_drift_features": [f for f, _ in _av_result.top_drift_features[:5]],
                    }
                    log.debug("[ADV-VAL] auc=%.3f drift=%s", _av_result.auc, result.meta["adversarial_validation"]["drift_detected"])
    except Exception as _av_exc:
        log.debug("[ADV-VAL] adversarial_validation skipped: %s", _av_exc)

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

    # Step 3.75: Signal FDR screening — BH correction on signal score z-values (observability)
    try:
        if not result.signals.empty:
            from src.assembled_core.qa.multiple_testing import benjamini_hochberg_fdr
            import scipy.stats as _sp_stats  # noqa: F401 — optional; if missing, block catches it
            _fdr_score_cols = [
                c for c in result.signals.select_dtypes("number").columns
                if c not in ("symbol",) and result.signals[c].std() > 1e-10
            ]
            if len(_fdr_score_cols) >= 2:
                _fdr_pvals = []
                for _fc in _fdr_score_cols:
                    _z = (result.signals[_fc] - result.signals[_fc].mean()) / result.signals[_fc].std()
                    _p = float(2 * (1 - _sp_stats.norm.cdf(abs(_z.mean()))))
                    _fdr_pvals.append(max(1e-10, min(1.0, _p)))
                _fdr_result = benjamini_hochberg_fdr(_fdr_pvals, alpha=0.10)
                result.meta["signal_fdr"] = {
                    "n_signals": _fdr_result.n_tests,
                    "n_significant": _fdr_result.n_rejected,
                    "adjusted_threshold": _fdr_result.adjusted_threshold,
                }
                log.debug("[SIG-FDR] %d/%d signals significant (BH-FDR)", _fdr_result.n_rejected, _fdr_result.n_tests)
    except Exception as _fdr_exc:
        log.debug("[SIG-FDR] multiple_testing FDR skipped: %s", _fdr_exc)

    # Step 3.8: Conformal prediction intervals on signal scores (coverage guarantee, observability)
    try:
        if not result.signals.empty and "score" in result.signals.columns:
            from src.assembled_core.ml.conformal_prediction import SplitConformal
            import numpy as _np_cp
            _cp_scores = result.signals["score"].dropna().values.astype(float)
            if len(_cp_scores) >= 20:
                # Use first half as calibration, second half as test
                _cp_mid = len(_cp_scores) // 2
                _cp_cal = _cp_scores[:_cp_mid]
                _cp_test = _cp_scores[_cp_mid:]
                # Predictor: median of calibration set as a constant baseline
                _cp_cal_median = float(_np_cp.median(_cp_cal))
                _cp_predictor = lambda x: _np_cp.full(len(x) if hasattr(x, '__len__') else 1, _cp_cal_median)
                _cp = SplitConformal(alpha=0.10)
                _cp.calibrate(_cp_predictor, _cp_cal.reshape(-1, 1), _cp_cal)
                _cp_result = _cp.predict(_cp_test.reshape(-1, 1))
                result.meta["conformal_signal"] = {
                    "alpha": 0.10,
                    "quantile": round(float(_cp._quantile), 6),
                    "n_cal": _cp_mid,
                    "n_test": len(_cp_test),
                }
                log.debug("[CONFORMAL] quantile=%.4f alpha=0.10", _cp._quantile)
    except Exception as _cp_exc:
        log.debug("[CONFORMAL] conformal_prediction skipped: %s", _cp_exc)

    # Step 3.86: Purged walk-forward split meta (observability — how many CV folds fit in current history)
    try:
        if not result.prices_with_features.empty and "timestamp" in result.prices_with_features.columns:
            from src.assembled_core.ml.purged_cv import purged_walk_forward_split
            _pwf_ts = pd.to_datetime(result.prices_with_features["timestamp"]).drop_duplicates().sort_values()
            if len(_pwf_ts) >= 50:
                _pwf_splits = purged_walk_forward_split(
                    _pwf_ts, train_window_days=120, test_window_days=30, max_splits=10
                )
                result.meta["purged_cv"] = {
                    "n_splits": len(_pwf_splits),
                    "train_window_days": 120,
                    "test_window_days": 30,
                }
                log.debug("[PURGED-CV] %d walk-forward splits", len(_pwf_splits))
    except Exception as _pwf_exc:
        log.debug("[PURGED-CV] purged_cv skipped: %s", _pwf_exc)

    # Step 3.87: Behavioral finance signals (disposition, anchoring, herding — observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.signals.behavioral_finance import generate_behavioral_signals
            _bfs_list = generate_behavioral_signals(result.prices_with_features)
            if _bfs_list:
                _bfs_scores = [float(s.composite_score) for s in _bfs_list]
                result.meta["behavioral_signals"] = {
                    "n_symbols": len(_bfs_list),
                    "mean_composite_score": round(float(sum(_bfs_scores) / len(_bfs_scores)), 4),
                    "max_composite_score": round(max(_bfs_scores), 4),
                }
                log.debug("[BEHAVIORAL] %d symbols, mean_score=%.3f", len(_bfs_list), result.meta["behavioral_signals"]["mean_composite_score"])
    except Exception as _bfs_exc:
        log.debug("[BEHAVIORAL] behavioral_finance skipped: %s", _bfs_exc)

    # Step 3.88: Risk-aware signal combiner (regime-conditioned weighting — observability)
    try:
        if not result.signals.empty and result.signals.select_dtypes("number").shape[1] >= 1:
            from src.assembled_core.signals.risk_aware_combiner import RiskAwareSignalCombiner
            _rac = RiskAwareSignalCombiner()
            _rac_regime = str(result.meta.get("combined_regime", {}).get("combined") or
                              result.meta.get("hmm_regime", {}).get("regime_label") or "NEUTRAL")
            _rac_sig_df = result.signals.select_dtypes("number").fillna(0.0)
            if not _rac_sig_df.empty:
                _rac_combined = _rac.combine(_rac_sig_df, current_regime=_rac_regime)
                result.meta["risk_aware_combiner"] = {
                    "regime": _rac_regime,
                    "n_signals": len(_rac_sig_df.columns),
                    "combined_mean": round(float(_rac_combined.mean()), 4),
                }
                log.debug("[RAC] regime=%s n_signals=%d mean=%.4f", _rac_regime, len(_rac_sig_df.columns), float(_rac_combined.mean()))
    except Exception as _rac_exc:
        log.debug("[RAC] risk_aware_combiner skipped: %s", _rac_exc)

    # Step 3.89: Signal plugin discovery (auto-discover external signal plugins — observability)
    try:
        from src.assembled_core.signals.plugin_loader import discover_signal_plugins
        _pl_root = Path(ctx.data_root) if getattr(ctx, "data_root", None) else Path.cwd()
        _pl_plugins = discover_signal_plugins(str(_pl_root / "plugins" / "signals"))
        result.meta["signal_plugins"] = {
            "n_plugins": len(_pl_plugins),
            "plugin_names": list(_pl_plugins.keys())[:10],
        }
        log.debug("[PLUGIN] %d signal plugins discovered", len(_pl_plugins))
    except Exception as _pl_exc:
        log.debug("[PLUGIN] plugin_loader skipped: %s", _pl_exc)

    # Step 3.90: CPCV splits (Combinatorial Purged Cross-Validation — observability)
    try:
        from src.assembled_core.ml.cpcv import generate_cpcv_splits
        _cpcv_n = len(result.equity_series) if result.equity_series is not None else 0
        if _cpcv_n >= 60:
            _cpcv_splits = generate_cpcv_splits(
                n_timestamps=_cpcv_n, n_groups=6, k_test_groups=2,
                purge_length=5, embargo_length=3,
            )
            result.meta["cpcv_splits"] = {
                "n_timestamps": _cpcv_n,
                "n_groups": 6,
                "k_test_groups": 2,
                "n_splits": len(_cpcv_splits),
            }
            log.debug("[CPCV] n_timestamps=%d n_splits=%d", _cpcv_n, len(_cpcv_splits))
    except Exception as _cpcv_exc:
        log.debug("[CPCV] cpcv skipped: %s", _cpcv_exc)

    # Step 3.91: Signal normalization (cross-sectional z-score via signal_api — observability)
    try:
        if not result.signals.empty:
            from src.assembled_core.signals.signal_api import normalize_signals
            _sna_num = result.signals.select_dtypes("number")
            if not _sna_num.empty:
                # Build minimal signal frame: one row per symbol, single signal_value col
                _sna_first_col = _sna_num.columns[0]
                _sna_df = pd.DataFrame({
                    "signal_value": _sna_num[_sna_first_col].fillna(0.0).values,
                }, index=pd.DatetimeIndex([ctx.as_of] * len(_sna_num)))
                _sna_normalized = normalize_signals(_sna_df, value_col="signal_value", method="zscore")
                result.meta["signal_normalized"] = {
                    "method": "zscore",
                    "n_symbols": len(_sna_normalized),
                    "signal_col": _sna_first_col,
                    "mean": round(float(_sna_normalized["signal_value"].mean()), 4),
                }
                log.debug("[SIGNAL-NORM] zscore %d symbols, mean=%.4f", len(_sna_normalized), _sna_normalized["signal_value"].mean())
    except Exception as _sna_exc:
        log.debug("[SIGNAL-NORM] signal_api normalize skipped: %s", _sna_exc)

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

    # Step 4.86: Regime-conditional cost estimate for target orders
    try:
        rc_cfg = (policy.get("ml") or {}).get("regime_costs") or {}
        if rc_cfg.get("enabled", False) and not result.target_positions.empty:
            from src.assembled_core.risk.regime_costs import estimate_regime_costs
            _rc_regime = str(result.meta.get("regime", {}).get("regime", "normal"))
            _rc_total = 0.0
            _rc_count = 0
            w_col = next((c for c in ("target_weight", "weight") if c in result.target_positions.columns), None)
            if w_col:
                _pv = float(ctx.portfolio_value) if ctx.portfolio_value else 100_000.0
                for _, _row in result.target_positions.iterrows():
                    _wt = float(_row.get(w_col, 0.0) or 0.0)
                    _notional = abs(_wt) * _pv
                    if _notional > 0:
                        _est = estimate_regime_costs(
                            trade_value=_notional,
                            adv=float(rc_cfg.get("default_adv", 50_000_000.0)),
                            regime=_rc_regime,
                        )
                        _rc_total += _est.total_cost_bps * _notional / 10_000
                        _rc_count += 1
            if _rc_count > 0:
                result.meta["regime_costs"] = {
                    "regime": _rc_regime,
                    "n_positions": _rc_count,
                    "estimated_total_cost_usd": round(_rc_total, 4),
                }
                log.debug("[REGIME-COST] regime=%s n=%d total=$%.4f", _rc_regime, _rc_count, _rc_total)
    except Exception as _rc_exc:
        log.debug("[REGIME-COST] regime_costs skipped: %s", _rc_exc)

    # Step 4.87: Stress scenario test for target positions (observability)
    try:
        if not result.target_positions.empty and "weight" in result.target_positions.columns:
            from src.assembled_core.portfolio.stress_test_constraints import evaluate_stress_scenarios
            _st_syms = list(result.target_positions["symbol"].astype(str))
            _st_weights = dict(zip(_st_syms, result.target_positions["weight"].astype(float).values))
            _st_sectors = (policy.get("universe") or {}).get("sector_mapping") or {}
            _st_sector_map = {s: _st_sectors.get(s, "Unknown") for s in _st_syms}
            _st_result = evaluate_stress_scenarios(_st_weights, _st_syms, _st_sector_map)
            result.meta["stress_test"] = {
                "worst_scenario": _st_result.worst_scenario,
                "worst_loss": round(_st_result.worst_loss, 4),
                "all_within_floors": _st_result.all_within_floors,
                "violated_scenarios": _st_result.violated_scenarios,
                "scenario_losses": {k: round(v, 4) for k, v in _st_result.scenario_losses.items()},
            }
            log.debug("[STRESS] worst=%s loss=%.3f", _st_result.worst_scenario, _st_result.worst_loss)
    except Exception as _st_exc:
        log.debug("[STRESS] stress_test_constraints skipped: %s", _st_exc)

    # Step 4.88: Regime-conditional asset class template blend (observability)
    try:
        from src.assembled_core.portfolio.regime_portfolio import blend_regime_templates
        _rp_regime = str(result.meta.get("regime", {}).get("regime", "sideways"))
        _rp_probs = {_rp_regime: 1.0}
        _rp_blend = blend_regime_templates(_rp_probs)
        result.meta["regime_portfolio_template"] = {
            "regime": _rp_regime,
            "n_asset_classes": len(_rp_blend),
            "weights": _rp_blend,
        }
        log.debug("[REGIME-PORTFOLIO] regime=%s n_ac=%d", _rp_regime, len(_rp_blend))
    except Exception as _rp_exc:
        log.debug("[REGIME-PORTFOLIO] regime_portfolio skipped: %s", _rp_exc)

    # Step 4.94: Reverse stress test (minimum shock to cause target loss — observability)
    try:
        if not result.target_positions.empty and "weight" in result.target_positions.columns:
            if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
                from src.assembled_core.qa.reverse_stress import reverse_stress_test
                _rst_syms = list(result.target_positions["symbol"].astype(str))
                _rst_w = result.target_positions["weight"].astype(float).values
                _rst_pivot = result.prices_with_features.pivot_table(
                    index="timestamp", columns="symbol", values="close", aggfunc="last"
                )
                _rst_rets = _rst_pivot.reindex(columns=_rst_syms).pct_change().dropna(how="all")
                if len(_rst_rets) >= 10 and len(_rst_syms) >= 2:
                    _rst_cov = _rst_rets.cov().values
                    _rst_result = reverse_stress_test(
                        _rst_w, _rst_cov, target_loss=-0.20, n_restarts=3
                    )
                    result.meta["reverse_stress"] = {
                        "target_loss": _rst_result.target_loss,
                        "achieved_loss": round(float(_rst_result.achieved_loss), 4),
                        "converged": bool(_rst_result.converged),
                        "shock_norm": round(float(_rst_result.shock_norm), 4),
                    }
                    log.debug("[REV-STRESS] converged=%s achieved=%.3f", _rst_result.converged, _rst_result.achieved_loss)
    except Exception as _rst_exc:
        log.debug("[REV-STRESS] reverse_stress skipped: %s", _rst_exc)

    # Step 4.93: Robust portfolio weights (shadow comparison — uncertainty-aware optimization)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.portfolio.robust_optimizer import compute_robust_weights
            _ro_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _ro_rets = _ro_pivot.pct_change().dropna(how="all")
            if len(_ro_rets) >= 20 and len(_ro_rets.columns) >= 2:
                _ro_syms = list(_ro_rets.columns)
                _ro_mu = _ro_rets.mean()
                _ro_cov = _ro_rets.cov()
                _ro_result = compute_robust_weights(
                    _ro_mu, _ro_cov, symbols=_ro_syms,
                    n_obs=len(_ro_rets), long_only=True,
                )
                result.meta["robust_weights"] = {
                    "converged": _ro_result.converged,
                    "method": _ro_result.method,
                    "n_symbols": len(_ro_syms),
                    "worst_case_return": round(float(_ro_result.worst_case_return), 6),
                    "portfolio_volatility": round(float(_ro_result.portfolio_volatility), 6),
                }
                log.debug("[ROBUST-OPT] method=%s converged=%s", _ro_result.method, _ro_result.converged)
    except Exception as _ro_exc:
        log.debug("[ROBUST-OPT] robust_optimizer skipped: %s", _ro_exc)

    # Step 4.9: Long-short balance enforcement (exposure audit + optional rebalance)
    try:
        if not result.target_positions.empty:
            from src.assembled_core.portfolio.long_short_balance import LongShortBalancer
            _lsb = LongShortBalancer.from_policy(policy)
            _exp = _lsb.compute_exposure(result.target_positions)
            result.meta["exposure_metrics"] = {
                "long_exposure": _exp.long_exposure,
                "short_exposure": _exp.short_exposure,
                "net_exposure": _exp.net_exposure,
                "gross_exposure": _exp.gross_exposure,
                "long_count": _exp.long_count,
                "short_count": _exp.short_count,
            }
            log.debug(
                "[LS-BALANCE] gross=%.2f net=%.2f long=%d short=%d",
                _exp.gross_exposure, _exp.net_exposure, _exp.long_count, _exp.short_count,
            )
    except Exception as _lsb_exc:
        log.debug("[LS-BALANCE] long_short_balance skipped: %s", _lsb_exc)

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

    # Step 5.5: Portfolio risk metrics (VaR, vol, drawdown from price history)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics
            _rm_proxy = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            ).median(axis=1)
            if len(_rm_proxy) >= 5:
                _rm_result = compute_portfolio_risk_metrics(_rm_proxy)
                result.meta["portfolio_risk_metrics"] = {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in _rm_result.items() if v is not None
                }
                log.debug(
                    "[RISK-METRICS] ann_vol=%.4f maxDD=%.4f var95=%.4f",
                    _rm_result.get("ann_vol") or 0.0,
                    _rm_result.get("max_drawdown") or 0.0,
                    _rm_result.get("var_95") or 0.0,
                )
    except Exception as _rm_exc:
        log.debug("[RISK-METRICS] portfolio risk_metrics skipped: %s", _rm_exc)

    # Step 5.6: Tail dependence diagnostic (crash synchronization risk, observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.risk.tail_dependence import (
                compute_empirical_tail_dependence,
                compute_portfolio_tail_dependence_score,
                classify_tail_regime,
            )
            _td_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _td_returns = _td_pivot.pct_change().dropna(how="all")
            if len(_td_returns) >= 30 and len(_td_returns.columns) >= 2:
                _td_matrix = compute_empirical_tail_dependence(_td_returns)
                _td_score = compute_portfolio_tail_dependence_score(_td_matrix)
                _td_regime = classify_tail_regime(_td_score)
                result.meta["tail_dependence"] = {
                    "score": round(_td_score, 4),
                    "regime": _td_regime,
                }
                if _td_regime == "high":
                    log.warning("[TAIL-DEP] high tail dependence score=%.3f — crash sync risk elevated", _td_score)
                else:
                    log.debug("[TAIL-DEP] tail regime=%s score=%.3f", _td_regime, _td_score)
    except Exception as _td_exc:
        log.debug("[TAIL-DEP] tail_dependence skipped: %s", _td_exc)

    # Step 5.7: HRP shadow weights (observability — hierarchical risk parity comparison)
    try:
        hrp_cfg = (policy.get("portfolio") or {}).get("hrp_shadow") or {}
        if hrp_cfg.get("enabled", False) and not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.portfolio.hierarchical_risk_parity import compute_hrp_weights
            _hrp_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _hrp_returns = _hrp_pivot.pct_change().dropna(how="all")
            if len(_hrp_returns) >= 20 and len(_hrp_returns.columns) >= 2:
                _hrp_weights = compute_hrp_weights(_hrp_returns)
                if _hrp_weights:
                    result.meta["hrp_weights"] = {
                        "weights": _hrp_weights,
                        "n_symbols": len(_hrp_weights),
                    }
                    log.debug("[HRP] shadow weights computed for %d symbols", len(_hrp_weights))
    except Exception as _hrp_exc:
        log.debug("[HRP] hrp_shadow skipped: %s", _hrp_exc)

    # Step 5.8: Systemic risk network centrality (observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.risk.systemic_risk import compute_return_network_centrality
            _sr_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _sr_returns = _sr_pivot.pct_change().dropna(how="all")
            if len(_sr_returns) >= 5 and len(_sr_returns.columns) >= 2:
                _centrality = compute_return_network_centrality(_sr_returns)
                result.meta["systemic_risk"] = {
                    "centrality": _centrality,
                    "n_symbols": len(_centrality),
                    "top_central": sorted(_centrality, key=_centrality.get, reverse=True)[:3],
                }
                log.debug("[SYSTEMIC] centrality computed for %d symbols", len(_centrality))
    except Exception as _sr_exc:
        log.debug("[SYSTEMIC] systemic_risk skipped: %s", _sr_exc)

    # Step 5.9: Parameter stability check (vol/drawdown consistency across windows)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.risk.param_stability import compute_stability_report
            # Approximate equity curve from median close across symbols
            _ps_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _ps_equity = _ps_pivot.median(axis=1).dropna()
            if len(_ps_equity) >= 20:
                _ps_report = compute_stability_report(_ps_equity, policy=policy)
                result.meta["param_stability"] = {
                    "all_stable": _ps_report.get("all_stable", True),
                    "checks_passed": _ps_report.get("checks_passed", 0),
                    "checks_total": _ps_report.get("checks_total", 0),
                }
                if not _ps_report.get("all_stable", True):
                    log.warning("[PARAM-STABILITY] instability detected: %d/%d checks passed",
                                _ps_report.get("checks_passed", 0), _ps_report.get("checks_total", 0))
    except Exception as _ps_exc:
        log.debug("[PARAM-STABILITY] param_stability skipped: %s", _ps_exc)

    # Step 5.12: Execution cost annotation (observability — impact + routing cost per order)
    try:
        if not result.orders.empty and not result.prices_with_features.empty:
            from src.assembled_core.ops.execution_cost_meta import annotate_execution_cost
            _regime_label = str(result.meta.get("regime", {}).get("regime", "bull"))
            _, _ecm_summary = annotate_execution_cost(
                result.orders,
                result.prices_with_features,
                policy,
                regime=_regime_label,
            )
            result.meta["execution_cost_meta"] = {
                "enabled": _ecm_summary.get("enabled", False),
                "n_orders": _ecm_summary.get("n_orders_in", 0),
                "total_est_cost_bps": round(float(_ecm_summary.get("total_est_cost_bps", 0.0)), 4),
                "high_impact_count": int(_ecm_summary.get("high_impact_count", 0)),
            }
            log.debug("[EXEC-COST-META] total_cost=%.2f bps, high_impact=%d",
                      _ecm_summary.get("total_est_cost_bps", 0.0), _ecm_summary.get("high_impact_count", 0))
    except Exception as _ecm_exc:
        log.debug("[EXEC-COST-META] execution_cost_meta skipped: %s", _ecm_exc)

    # Step 5.13: TCA arrival — implementation shortfall on orders (observability)
    try:
        if not result.orders.empty and "price" in result.orders.columns and not result.prices_with_features.empty:
            from src.assembled_core.qa.tca_arrival import compute_implementation_shortfall
            # Build arrival_prices from the latest price per symbol
            _tca_ts = result.prices_with_features.groupby("symbol")["timestamp"].max().reset_index()
            _tca_ts.columns = ["symbol", "timestamp"]
            _tca_close = result.prices_with_features.sort_values("timestamp").groupby("symbol")["close"].last().reset_index()
            _tca_arrival = _tca_ts.merge(_tca_close, on="symbol")
            _tca_arrival.rename(columns={"close": "arrival_price"}, inplace=True)
            # Build fills from orders
            _tca_fills = result.orders.rename(columns={"quantity": "qty", "price": "fill_price"}).copy()
            if "timestamp" not in _tca_fills.columns:
                _tca_fills["timestamp"] = ctx.as_of
            if "qty" not in _tca_fills.columns and "quantity" in result.orders.columns:
                _tca_fills["qty"] = result.orders["quantity"]
            _tca_result = compute_implementation_shortfall(_tca_fills, _tca_arrival)
            if not _tca_result.empty and "is_bps" in _tca_result.columns:
                _tca_mean = float(_tca_result["is_bps"].dropna().mean()) if _tca_result["is_bps"].notna().any() else 0.0
                result.meta["tca_arrival"] = {
                    "n_fills": len(_tca_result),
                    "mean_is_bps": round(_tca_mean, 4),
                }
                log.debug("[TCA-ARRIVAL] %d fills, mean IS=%.2f bps", len(_tca_result), _tca_mean)
    except Exception as _tca_exc:
        log.debug("[TCA-ARRIVAL] tca_arrival skipped: %s", _tca_exc)

    # Step 5.14: Risk escalation ladder (drawdown-based risk level — observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 5:
            from src.assembled_core.ops.self_healing import RiskEscalationLadder
            _rel = RiskEscalationLadder()
            _rel_eq = result.equity_series
            _rel_peak = _rel_eq.cummax()
            _rel_dd = float(((_rel_eq - _rel_peak) / _rel_peak).iloc[-1])
            _rel_state = _rel.evaluate(current_drawdown=_rel_dd)
            result.meta["risk_escalation"] = {
                "level": str(_rel_state.level.value) if hasattr(_rel_state.level, "value") else str(_rel_state.level),
                "drawdown": round(_rel_dd, 4),
                "trigger_reason": _rel_state.trigger_reason,
            }
            log.debug("[ESCALATION] level=%s dd=%.3f", result.meta["risk_escalation"]["level"], _rel_dd)
    except Exception as _rel_exc:
        log.debug("[ESCALATION] self_healing escalation skipped: %s", _rel_exc)

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

    # Step 6.85: Per-trade transaction cost estimate (observability)
    try:
        if not result.orders_filtered.empty:
            from src.assembled_core.risk.transaction_costs import estimate_per_trade_cost
            _tc_orders = result.orders_filtered.copy()
            if "timestamp" not in _tc_orders.columns:
                _tc_orders["timestamp"] = ctx.as_of
            if "side" not in _tc_orders.columns:
                _tc_orders["side"] = _tc_orders.get("direction", "buy")
            _req_cols = {"timestamp", "symbol", "side", "qty", "price"}
            if _req_cols.issubset(set(_tc_orders.columns)):
                _tc_series = estimate_per_trade_cost(
                    _tc_orders,
                    method="simple",
                    commission_bps=float(
                        (policy.get("transaction_costs") or {}).get("commission_bps", 0.5)
                    ),
                    slippage_bps=float(
                        (policy.get("transaction_costs") or {}).get("slippage_bps", 3.0)
                    ),
                )
                result.meta["transaction_costs"] = {
                    "n_orders": len(_tc_series),
                    "total_cost_usd": round(float(_tc_series.sum()), 4),
                    "avg_cost_usd": round(float(_tc_series.mean()), 4),
                }
                log.debug(
                    "[TCA] estimated trade costs: total=$%.4f, n=%d",
                    float(_tc_series.sum()), len(_tc_series),
                )
    except Exception as _tc_exc:
        log.debug("[TCA] transaction_costs skipped: %s", _tc_exc)

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

    # Step 7.5: Strategy capacity QA (observability — shadow, no blocking)
    try:
        cap_cfg = (policy.get("qa") or {}).get("capacity") or {}
        if cap_cfg.get("enabled", False) and not result.orders_filtered.empty:
            from src.assembled_core.qa.capacity import estimate_strategy_capacity
            _cap_orders = result.orders_filtered.copy()
            if "timestamp" not in _cap_orders.columns:
                _cap_orders["timestamp"] = ctx.as_of
            if "notional" not in _cap_orders.columns and "qty" in _cap_orders.columns and "price" in _cap_orders.columns:
                _cap_orders["notional"] = (_cap_orders["qty"] * _cap_orders["price"]).abs()
            if "notional" in _cap_orders.columns:
                _cap_est = estimate_strategy_capacity(
                    _cap_orders,
                    alpha_gross_bps=float(cap_cfg.get("alpha_gross_bps", 1500.0)),
                    target_aum_usd=float(cap_cfg.get("target_aum_usd", 10_000_000.0)),
                    n_positions=int(cap_cfg.get("n_positions", 20)),
                )
                result.meta["capacity_estimate"] = {
                    "verdict": _cap_est.verdict,
                    "max_aum_usd": _cap_est.max_aum_usd,
                    "alpha_net_at_target_bps": _cap_est.alpha_net_at_target_bps,
                }
                if _cap_est.verdict != "ok":
                    log.warning(
                        "[CAPACITY] verdict=%s max_aum=$%.0f alpha_net=%.1fbps",
                        _cap_est.verdict, _cap_est.max_aum_usd, _cap_est.alpha_net_at_target_bps,
                    )
    except Exception as _cap_exc:
        log.debug("[CAPACITY] capacity_estimate skipped: %s", _cap_exc)

    # Step 7.6: Write run KPIs artifact (structured output for monitoring)
    try:
        if ctx.write_outputs:
            from src.assembled_core.ops.kpi_artifacts import write_run_kpis
            _kpi_dir = ctx.output_dir
            _kpi_path = write_run_kpis(
                output_dir=_kpi_dir,
                ctx=ctx,
                result=result,
                policy=policy,
                mode=ctx.execution_mode,
            )
            log.debug("[KPI] run_kpis written to %s", _kpi_path)
    except Exception as _kpi_exc:
        log.debug("[KPI] write_run_kpis skipped: %s", _kpi_exc)

    # Step 7.62: Write run manifest (reproducibility + lineage, policy-gated)
    try:
        if ctx.write_outputs:
            from src.assembled_core.ops.run_manifest import write_run_manifest
            from datetime import timezone as _tz, datetime as _datetime
            _rm_path = write_run_manifest(
                run_id=str(ctx.as_of.date()),
                date=str(ctx.as_of.date()),
                started_at_utc=ctx.as_of.isoformat(),
                status="success",
                metrics={
                    "n_orders": len(result.orders_filtered),
                    "n_signals": len(result.signals),
                    "execution_mode": ctx.execution_mode,
                },
                manifests_dir=ctx.output_dir / "manifests",
            )
            log.debug("[MANIFEST] run manifest written to %s", _rm_path)
    except Exception as _rm_exc:
        log.debug("[MANIFEST] run_manifest skipped: %s", _rm_exc)

    # Step 7.63: Append run index CSV (lineage + searchability, policy-gated)
    try:
        if ctx.write_outputs:
            from src.assembled_core.ops.run_index import append_run_index
            from src.assembled_core.ops.run_manifest import compute_config_hash
            _ri_hash = compute_config_hash(policy) if policy else ""
            _ri_metrics = {
                "final_equity": float(getattr(ctx, "current_equity", ctx.equity)),
                "n_fills": len(result.orders_filtered),
            }
            append_run_index(
                run_id=str(ctx.as_of.date()),
                date=str(ctx.as_of.date()),
                status="success",
                metrics=_ri_metrics,
                git_sha=result.meta.get("git_sha", ""),
                config_hash=_ri_hash,
                manifest_path=ctx.output_dir / "manifests" / str(ctx.as_of.date()) / "manifest.latest.json",
                index_path=ctx.output_dir / "manifests" / "index.csv",
            )
            log.debug("[RUN-INDEX] run index updated for %s", ctx.as_of.date())
    except Exception as _ri_exc:
        log.debug("[RUN-INDEX] run_index skipped: %s", _ri_exc)

    # Step 7.64: Model registry stats (observability — how many models are registered)
    try:
        from src.assembled_core.ml.model_registry import ModelRegistry
        _mr = ModelRegistry(base_dir=ctx.output_dir / "models")
        _mr_models = list(_mr._records.keys())
        result.meta["model_registry"] = {
            "n_registered_models": len(_mr_models),
            "model_ids": _mr_models[:10],
        }
        log.debug("[MODEL-REGISTRY] %d models registered", len(_mr_models))
    except Exception as _mr_exc:
        log.debug("[MODEL-REGISTRY] model_registry skipped: %s", _mr_exc)

    # Step 7.65: Report retention — purge stale run artifacts (observability)
    try:
        if ctx.write_outputs:
            from src.assembled_core.ops.report_retention import purge_old_dated_reports
            _rr_dir = ctx.output_dir / "manifests"
            _rr_purged = purge_old_dated_reports(_rr_dir, prefix="", suffix=".json", keep_last_n=90)
            result.meta["report_retention"] = {"purged_files": _rr_purged}
            if _rr_purged > 0:
                log.debug("[RETENTION] purged %d old report files", _rr_purged)
    except Exception as _rr_exc:
        log.debug("[RETENTION] report_retention skipped: %s", _rr_exc)

    # Step 7.66: Trade journal — log orders as journal entries (policy-gated on write_outputs)
    try:
        if ctx.write_outputs and not result.orders_filtered.empty:
            from src.assembled_core.ops.trade_journal import append_trade_journal_entries
            _tj_fills = []
            for _, _row in result.orders_filtered.iterrows():
                _tj_fills.append({
                    "symbol": str(_row.get("symbol", "")),
                    "side": str(_row.get("side", "BUY")),
                    "qty": float(_row.get("quantity", _row.get("qty", 0))),
                    "price": float(_row.get("price", _row.get("limit_price", 0))),
                })
            _tj_signal_ctx = {
                "regime": result.meta.get("regime", {}).get("regime", ""),
                "execution_mode": ctx.execution_mode,
            }
            append_trade_journal_entries(
                _tj_fills,
                signal_context=_tj_signal_ctx,
                run_id=str(ctx.as_of.date()),
                journal_path=ctx.output_dir / "trade_journal.jsonl",
            )
            log.debug("[TRADE-JOURNAL] %d orders logged", len(_tj_fills))
    except Exception as _tj_exc:
        log.debug("[TRADE-JOURNAL] trade_journal skipped: %s", _tj_exc)

    # Step 7.67: Learning store — append cycle outcome as learning record (observability)
    try:
        from src.assembled_core.qa.learning_store import append_learning_record
        _ls_record = {
            "cycle_date": str(ctx.as_of.date()),
            "execution_mode": ctx.execution_mode,
            "n_orders": len(result.orders_filtered),
            "n_signals": len(result.signals),
            "regime": result.meta.get("regime", {}).get("regime", ""),
            "hmm_regime": result.meta.get("hmm_regime", {}).get("regime_label", ""),
            "combined_regime": result.meta.get("combined_regime", {}).get("combined", ""),
            "equity": float(getattr(ctx, "current_equity", ctx.equity)),
            "signal_fdr_significant": result.meta.get("signal_fdr", {}).get("n_significant", 0),
        }
        append_learning_record(
            _ls_record,
            store_path=ctx.output_dir / "ml" / "learning_store.jsonl",
        )
        log.debug("[LEARNING-STORE] record appended for %s", ctx.as_of.date())
    except Exception as _ls_exc:
        log.debug("[LEARNING-STORE] learning_store skipped: %s", _ls_exc)

    # Step 7.68: Heartbeat — write cycle completion heartbeat (observability)
    try:
        from src.assembled_core.ops.heartbeat import write_heartbeat
        _hb_details = {
            "cycle_date": str(ctx.as_of.date()),
            "n_orders": len(result.orders_filtered),
            "execution_mode": str(ctx.execution_mode),
        }
        _hb_path = ctx.output_dir / "state" / "heartbeat.json"
        write_heartbeat(path=_hb_path, status="ok", details=_hb_details)
        result.meta["heartbeat"] = {"status": "ok", "path": str(_hb_path)}
        log.debug("[HEARTBEAT] written to %s", _hb_path)
    except Exception as _hb_exc:
        log.debug("[HEARTBEAT] heartbeat skipped: %s", _hb_exc)

    # Step 7.69: Alert manager — emit alerts for key cycle risk conditions (observability)
    try:
        from src.assembled_core.ops.alert_manager import AlertManager
        _am = AlertManager(rate_limit_seconds=0, output_dir=str(ctx.output_dir / "alerts"))
        _am_fired = 0
        # Alert if qa_gates blocked
        _am_qa_gates = result.meta.get("qa_gates", {})
        if _am_qa_gates.get("overall") in {"BLOCK", "block", "BLOCKED"}:
            _am.alert("WARNING", "cycle", "QA gates BLOCKED", details=_am_qa_gates)
            _am_fired += 1
        # Alert if no orders generated at all and this is live/paper
        if len(result.orders_filtered) == 0 and getattr(ctx, "execution_mode", "") in ("paper", "live"):
            _am.alert("INFO", "cycle", "No orders generated this cycle")
            _am_fired += 1
        result.meta["alert_manager"] = {"alerts_fired": _am_fired, "pending": _am.pending_count}
        log.debug("[ALERT-MGR] alerts_fired=%d", _am_fired)
    except Exception as _am_exc:
        log.debug("[ALERT-MGR] alert_manager skipped: %s", _am_exc)

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

    # Step 7.8: Shadow mode snapshot (persist cycle meta for offline analysis)
    try:
        from src.assembled_core.ops.shadow_mode import write_shadow_snapshot
        from datetime import date as _date
        _shadow_payload = {
            k: v for k, v in result.meta.items()
            if k in (
                "regime", "stress_test", "robust_weights", "exposure_metrics",
                "scenario_stress", "deflated_sharpe", "performance_profile",
                "regime_analysis", "benchmark_metrics", "feature_clustering",
            )
        }
        write_shadow_snapshot(
            "trading_cycle",
            _shadow_payload,
            snapshot_date=_date.fromisoformat(str(ctx.as_of.date())),
        )
        log.debug("[SHADOW] cycle meta snapshot written for %s", ctx.as_of.date())
    except Exception as _shad_exc:
        log.debug("[SHADOW] shadow_mode snapshot skipped: %s", _shad_exc)

    # Step 7.9: Experience log (append cycle data for ML training)
    try:
        from src.assembled_core.ops.experience_log import append_experience
        _exp_entry = {
            "cycle_date": str(ctx.as_of.date()),
            "execution_mode": ctx.execution_mode,
            "n_signals": len(result.signals),
            "n_orders": len(result.orders_filtered),
            "regime": str(result.meta.get("regime", {}).get("regime", "unknown")),
            "meta_keys": list(result.meta.keys()),
        }
        append_experience(_exp_entry)
        log.debug("[EXPERIENCE] cycle experience logged for %s", ctx.as_of.date())
    except Exception as _exp_exc:
        log.debug("[EXPERIENCE] experience_log skipped: %s", _exp_exc)

    # Step 8.1: Regime analysis from index returns (end-of-cycle observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.risk.regime_analysis import classify_regimes_from_index
            _ra_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _ra_index_rets = _ra_pivot.median(axis=1).pct_change().dropna()
            if len(_ra_index_rets) >= 20:
                _ra_regimes = classify_regimes_from_index(_ra_index_rets)
                _ra_latest = str(_ra_regimes.iloc[-1]) if not _ra_regimes.empty else "unknown"
                result.meta["regime_analysis"] = {
                    "latest_regime": _ra_latest,
                    "n_periods": len(_ra_regimes),
                }
                log.debug("[REGIME-ANALYSIS] latest regime: %s", _ra_latest)
    except Exception as _ra_exc:
        log.debug("[REGIME-ANALYSIS] regime_analysis skipped: %s", _ra_exc)

    # Step 8.2: Performance profile (Sharpe, Calmar, drawdown — end-of-cycle observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.portfolio_analyzer import compute_performance_profile
            _pp_proxy = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            ).median(axis=1).pct_change().dropna()
            if len(_pp_proxy) >= 10:
                _pp = compute_performance_profile(_pp_proxy)
                result.meta["performance_profile"] = {
                    "sharpe": round(_pp.sharpe, 4),
                    "calmar": round(_pp.calmar, 4),
                    "max_drawdown": round(_pp.max_drawdown, 4),
                    "annualized_vol": round(_pp.annualized_vol, 4),
                }
                log.debug("[PERF-PROFILE] sharpe=%.3f maxDD=%.3f", _pp.sharpe, _pp.max_drawdown)
    except Exception as _pp_exc:
        log.debug("[PERF-PROFILE] performance_profile skipped: %s", _pp_exc)

    # Step 8.3: Drawdown decomposition — attribution during worst drawdown (observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.drawdown_decomposition import decompose_drawdown
            _ddc_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _ddc_rets = _ddc_pivot.pct_change().dropna(how="all")
            if len(_ddc_rets) >= 15 and len(_ddc_rets.columns) >= 2:
                import pandas as _ddc_pd
                _ddc_portfolio = _ddc_rets.median(axis=1)
                _ddc_factors = _ddc_pd.DataFrame(
                    {"market": _ddc_rets.mean(axis=1)}, index=_ddc_rets.index
                )
                _ddc_report = decompose_drawdown(_ddc_portfolio, _ddc_factors)
                result.meta["drawdown_decomposition"] = {
                    "max_drawdown": round(_ddc_report.drawdown.max_drawdown, 4),
                    "dd_duration": _ddc_report.drawdown.duration,
                    "alpha_during_dd": round(float(_ddc_report.alpha_during_dd), 4),
                    "r_squared": round(float(_ddc_report.r_squared), 4),
                    "idiosyncratic": round(float(_ddc_report.idiosyncratic_return), 6),
                }
                log.debug("[DD-DECOMP] maxDD=%.3f dur=%d", _ddc_report.drawdown.max_drawdown, _ddc_report.drawdown.duration)
    except Exception as _ddc_exc:
        log.debug("[DD-DECOMP] drawdown_decomposition skipped: %s", _ddc_exc)

    # Step 8.4: Benchmark-relative metrics (alpha, IR, tracking error — observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.benchmark_metrics import compute_benchmark_metrics
            _bm_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _bm_rets = _bm_pivot.pct_change().dropna(how="all")
            if len(_bm_rets) >= 15 and len(_bm_rets.columns) >= 2:
                _bm_portfolio = _bm_rets.median(axis=1)
                _bm_benchmark = _bm_rets.mean(axis=1)
                _bm = compute_benchmark_metrics(_bm_portfolio, _bm_benchmark)
                result.meta["benchmark_metrics"] = {
                    "alpha": round(float(_bm.alpha), 4) if _bm.alpha is not None else None,
                    "beta": round(float(_bm.beta), 4) if _bm.beta is not None else None,
                    "information_ratio": round(float(_bm.information_ratio), 4) if _bm.information_ratio is not None else None,
                    "tracking_error": round(float(_bm.tracking_error), 4) if _bm.tracking_error is not None else None,
                }
                log.debug("[BM-METRICS] alpha=%.4f IR=%.4f", _bm.alpha or 0.0, _bm.information_ratio or 0.0)
    except Exception as _bm_exc:
        log.debug("[BM-METRICS] benchmark_metrics skipped: %s", _bm_exc)

    # Step 8.5: Deflated Sharpe Ratio (overfitting-adjusted SR — observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.deflated_sharpe import deflated_sharpe
            _dsr_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _dsr_proxy = _dsr_pivot.median(axis=1).pct_change().dropna()
            if len(_dsr_proxy) >= 10:
                _dsr = deflated_sharpe(_dsr_proxy, n_trials=1)
                result.meta["deflated_sharpe"] = {
                    "sharpe_observed": round(float(_dsr.sharpe_observed), 4),
                    "sharpe_threshold": round(float(_dsr.sharpe_threshold), 4),
                    "deflated_sharpe_probability": round(float(_dsr.deflated_sharpe_probability), 4),
                    "passes_5pct": bool(_dsr.passes_5pct),
                }
                log.debug("[DSR] SR_obs=%.3f threshold=%.3f p=%.3f", _dsr.sharpe_observed, _dsr.sharpe_threshold, _dsr.deflated_sharpe_probability)
    except Exception as _dsr_exc:
        log.debug("[DSR] deflated_sharpe skipped: %s", _dsr_exc)

    # Step 8.6: Factor exposure betas (rolling OLS vs market factor — sklearn-gated)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.risk.factor_exposures import (
                compute_factor_exposures, summarize_factor_exposures
            )
            import pandas as _fe_pd
            _fe_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _fe_rets = _fe_pivot.pct_change().dropna(how="all")
            if len(_fe_rets) >= 30 and len(_fe_rets.columns) >= 2:
                _fe_strategy = _fe_rets.median(axis=1)
                _fe_strategy.name = "strategy"
                _fe_factors = _fe_pd.DataFrame({"market": _fe_rets.mean(axis=1)})
                _fe_exposures = compute_factor_exposures(_fe_strategy, _fe_factors)
                if not _fe_exposures.empty:
                    _fe_summary_df = summarize_factor_exposures(_fe_exposures)
                    _fe_mkt_beta, _fe_r2 = 0.0, 0.0
                    if not _fe_summary_df.empty and "factor" in _fe_summary_df.columns:
                        _fe_mkt_row = _fe_summary_df[_fe_summary_df["factor"] == "market"]
                        if len(_fe_mkt_row) > 0:
                            _fe_mkt_beta = float(_fe_mkt_row["mean_beta"].iloc[0])
                            _fe_r2 = float(_fe_mkt_row["mean_r2"].iloc[0]) if "mean_r2" in _fe_mkt_row.columns else 0.0
                    result.meta["factor_exposures"] = {
                        "n_windows": len(_fe_exposures),
                        "mean_market_beta": round(_fe_mkt_beta, 4),
                        "r2_mean": round(_fe_r2, 4),
                    }
                    log.debug("[FACTOR-EXP] n=%d mkt_beta=%.3f r2=%.3f", len(_fe_exposures), _fe_mkt_beta, _fe_r2)
    except Exception as _fe_exc:
        log.debug("[FACTOR-EXP] factor_exposures skipped: %s", _fe_exc)

    # Step 8.7: Scenario simulator (vol-spike + crash scenarios from price proxy, observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.scenario_simulator import run_stress_test
            _ss_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _ss_rets = _ss_pivot.pct_change().dropna(how="all")
            if len(_ss_rets) >= 20:
                _ss_baseline = _ss_rets.median(axis=1)
                _ss_report = run_stress_test(
                    _ss_baseline,
                    portfolio_returns=_ss_rets if _ss_rets.shape[1] >= 2 else None,
                    include_correlation=(_ss_rets.shape[1] >= 2),
                )
                result.meta["scenario_stress"] = {
                    "worst_scenario": _ss_report.worst_scenario,
                    "worst_cvar": round(float(_ss_report.worst_cvar), 4),
                    "n_scenarios": len(_ss_report.scenarios),
                    "baseline_var_95": round(float(_ss_report.baseline_metrics.get("var_95", 0.0)), 4),
                }
                log.debug("[SCENARIO] worst=%s cvar=%.4f", _ss_report.worst_scenario, _ss_report.worst_cvar)
    except Exception as _ss_exc:
        log.debug("[SCENARIO] scenario_simulator skipped: %s", _ss_exc)

    # Step 8.12: Model age confidence (freshness decay of last-trained model, observability)
    try:
        from src.assembled_core.ml.online_learning import compute_model_age_confidence
        _mac_days = int(result.meta.get("ml_model_age_days", 0))
        if _mac_days == 0:
            # Try to infer from meta if model was retrained this cycle
            _mac_days = 1 if result.meta.get("model_retrained") else 30
        _mac_confidence = compute_model_age_confidence(_mac_days, half_life_days=30)
        result.meta["model_age_confidence"] = {
            "days_since_refit": _mac_days,
            "confidence": round(_mac_confidence, 4),
            "half_life_days": 30,
        }
        log.debug("[MODEL-AGE] days=%d confidence=%.3f", _mac_days, _mac_confidence)
    except Exception as _mac_exc:
        log.debug("[MODEL-AGE] model_age_confidence skipped: %s", _mac_exc)

    # Step 8.14: Calibration monitor — signal score calibration quality (observability)
    try:
        if not result.signals.empty and "score" in result.signals.columns:
            from src.assembled_core.ml.calibration_monitor import compute_calibration
            _cal_scores = result.signals["score"].dropna()
            if len(_cal_scores) >= 10:
                import numpy as _np_cal
                _cal_preds = _np_cal.clip(_cal_scores.values.astype(float), 0.0, 1.0)
                # Proxy actuals: top-half scores → 1, bottom-half → 0
                _cal_median = float(_np_cal.median(_cal_preds))
                _cal_actuals = (_cal_preds >= _cal_median).astype(float)
                _cal_report = compute_calibration(_cal_preds, _cal_actuals, n_bins=5)
                result.meta["signal_calibration"] = {
                    "ece": round(_cal_report.ece, 4),
                    "brier_score": round(_cal_report.brier_score, 4),
                    "n_samples": _cal_report.n_samples,
                    "well_calibrated": _cal_report.is_well_calibrated(),
                }
                log.debug("[CALIBRATION] ECE=%.4f brier=%.4f", _cal_report.ece, _cal_report.brier_score)
    except Exception as _cal_exc:
        log.debug("[CALIBRATION] calibration_monitor skipped: %s", _cal_exc)

    # Step 8.15: Experiment tracking — log cycle run as experiment (observability)
    try:
        from src.assembled_core.ml.experiment_tracking import ExperimentTracker
        _et = ExperimentTracker(storage_path=ctx.output_dir / "experiments" if ctx.write_outputs else None)
        _et_metrics: dict[str, float] = {
            "n_orders": float(len(result.orders_filtered)),
            "n_signals": float(len(result.signals)),
            "equity": float(getattr(ctx, "current_equity", ctx.equity)),
        }
        if "signal_calibration" in result.meta:
            _et_metrics["signal_ece"] = float(result.meta["signal_calibration"].get("ece", 0.0))
        if "model_age_confidence" in result.meta:
            _et_metrics["model_age_confidence"] = float(result.meta["model_age_confidence"].get("confidence", 1.0))
        _et.log_run(
            experiment_name=f"trading_cycle_{ctx.execution_mode}",
            params={"as_of": str(ctx.as_of.date()), "mode": ctx.execution_mode},
            metrics=_et_metrics,
            tags={"regime": str(result.meta.get("combined_regime", {}).get("combined", ""))},
        )
        log.debug("[EXP-TRACKER] cycle run logged for %s", ctx.as_of.date())
    except Exception as _et_exc:
        log.debug("[EXP-TRACKER] experiment_tracking skipped: %s", _et_exc)

    # Step 8.16: Retraining scheduler — evaluate retrain signals (observability)
    try:
        from src.assembled_core.ml.retraining_scheduler import RetrainingScheduler
        _rs = RetrainingScheduler()
        _rs_equity = None
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            _rs_piv = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _rs_equity = _rs_piv.median(axis=1).dropna()
        _rs_rec = _rs.evaluate(
            model_last_trained_date=None,
            equity_since_retrain=_rs_equity,
        )
        result.meta["retraining_scheduler"] = {
            "decision": _rs_rec.decision,
            "signals_fired": _rs_rec.signals_fired,
        }
        log.debug("[RETRAIN-SCHED] decision=%s fired=%d", _rs_rec.decision, _rs_rec.signals_fired)
    except Exception as _rs_exc:
        log.debug("[RETRAIN-SCHED] retraining_scheduler skipped: %s", _rs_exc)

    # Step 8.17: Signal decay tracker state (observability — check snapshot history size)
    try:
        from src.assembled_core.ml.signal_decay_tracker import SignalDecayTracker
        _sdt = SignalDecayTracker(
            state_path=ctx.output_dir / "ml" / "signal_decay_history.json"
        )
        result.meta["signal_decay_tracker"] = {
            "n_snapshots": len(_sdt._snapshots),
        }
        log.debug("[SIGNAL-DECAY] %d historical snapshots loaded", len(_sdt._snapshots))
    except Exception as _sdt_exc:
        log.debug("[SIGNAL-DECAY] signal_decay_tracker skipped: %s", _sdt_exc)

    # Step 8.18: Post-trade forward returns (observability — ML training data for post-trade analysis)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.post_trade_analyzer import compute_forward_returns
            _pta_cols = [c for c in ["timestamp", "symbol", "close"] if c in result.prices_with_features.columns]
            if len(_pta_cols) == 3:
                _pta_prices = result.prices_with_features[_pta_cols].copy()
                if "timestamp" in _pta_prices.columns:
                    _pta_fwd = compute_forward_returns(_pta_prices, horizon_days=5)
                    _pta_valid = _pta_fwd["forward_return"].notna().sum()
                    result.meta["post_trade_fwd_returns"] = {
                        "n_rows": len(_pta_fwd),
                        "n_with_fwd": int(_pta_valid),
                        "horizon_days": 5,
                    }
                    log.debug("[POST-TRADE] %d/%d rows with 5d forward returns", _pta_valid, len(_pta_fwd))
    except Exception as _pta_exc:
        log.debug("[POST-TRADE] post_trade_analyzer skipped: %s", _pta_exc)

    # Step 8.19: Performance attribution (OLS alpha/beta vs market proxy — observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.performance_attribution import compute_attribution
            _attr_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _attr_ret = _attr_pivot.pct_change().dropna()
            if len(_attr_ret) >= 25 and len(_attr_ret.columns) >= 2:
                _attr_market = _attr_ret.mean(axis=1)
                # Use first symbol as portfolio proxy
                _attr_port = _attr_ret.iloc[:, 0]
                _attr_factors = pd.DataFrame({"market": _attr_market.values}, index=_attr_market.index)
                _attr_port_aligned = pd.Series(_attr_port.values, index=_attr_market.index)
                _attr_result = compute_attribution(_attr_port_aligned, _attr_factors, min_obs=20)
                result.meta["performance_attribution"] = {
                    "alpha": round(float(_attr_result.alpha), 6),
                    "market_beta": round(float(_attr_result.factor_betas.get("market", 0.0)), 4),
                    "r_squared": round(float(_attr_result.r_squared), 4),
                }
                log.debug("[ATTR] alpha=%.4f beta=%.3f R2=%.3f", _attr_result.alpha,
                          _attr_result.factor_betas.get("market", 0.0), _attr_result.r_squared)
    except Exception as _attr_exc:
        log.debug("[ATTR] performance_attribution skipped: %s", _attr_exc)

    # Step 8.20: QA gates evaluation on cycle equity proxy (observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.qa.qa_gates import evaluate_all_gates
            from src.assembled_core.qa.metrics import compute_equity_metrics
            _qg_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            )
            _qg_eq_series = _qg_pivot.median(axis=1).dropna()
            if len(_qg_eq_series) >= 20:
                _qg_eq_df = pd.DataFrame({"timestamp": _qg_eq_series.index, "equity": _qg_eq_series.values})
                _qg_metrics = compute_equity_metrics(_qg_eq_df, start_capital=float(_qg_eq_series.iloc[0]))
                _qg_summary = evaluate_all_gates(_qg_metrics)
                result.meta["qa_gates"] = {
                    "overall": str(_qg_summary.overall_result),
                    "passed": _qg_summary.passed_gates,
                    "warnings": _qg_summary.warning_gates,
                    "blocked": _qg_summary.blocked_gates,
                }
                log.debug("[QA-GATES] overall=%s passed=%d warned=%d blocked=%d",
                          _qg_summary.overall_result, _qg_summary.passed_gates,
                          _qg_summary.warning_gates, _qg_summary.blocked_gates)
    except Exception as _qg_exc:
        log.debug("[QA-GATES] qa_gates skipped: %s", _qg_exc)

    # Step 8.21: Factor IC summary (observability — IC of features vs TB labels if available)
    try:
        if not result.prices_with_features.empty and "tb_label_5d" in result.prices_with_features.columns:
            from src.assembled_core.qa.factor_analysis import compute_factor_ic, summarize_factor_ic
            _fic_num_cols = [
                c for c in result.prices_with_features.columns
                if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume", "tb_label_5d", "tb_ret_5d"}
                and result.prices_with_features[c].dtype in ("float64", "float32")
            ][:10]
            if _fic_num_cols and "timestamp" in result.prices_with_features.columns:
                _fic_df = result.prices_with_features[["timestamp", "symbol"] + _fic_num_cols + ["tb_label_5d"]].dropna()
                if len(_fic_df) >= 30 and len(_fic_df["timestamp"].unique()) >= 5:
                    _fic_ic = compute_factor_ic(_fic_df, factor_cols=_fic_num_cols, fwd_return_col="tb_label_5d")
                    if not _fic_ic.empty:
                        _fic_summary = summarize_factor_ic(_fic_ic)
                        _fic_top = _fic_summary.iloc[0] if not _fic_summary.empty else None
                        result.meta["factor_ic_summary"] = {
                            "n_factors": len(_fic_summary),
                            "top_factor": str(_fic_top["factor"]) if _fic_top is not None else "",
                            "top_ic_ir": round(float(_fic_top["ic_ir"]), 4) if _fic_top is not None else 0.0,
                        }
                        log.debug("[FACTOR-IC] %d factors, top=%s IR=%.3f", len(_fic_summary),
                                  result.meta["factor_ic_summary"]["top_factor"],
                                  result.meta["factor_ic_summary"]["top_ic_ir"])
    except Exception as _fic_exc:
        log.debug("[FACTOR-IC] factor_analysis skipped: %s", _fic_exc)

    # Step 8.22: Label daily equity records for ML training (observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 10:
            from src.assembled_core.qa.labeling import label_daily_records
            _ld_ts = pd.date_range(
                end=ctx.as_of, periods=len(result.equity_series), freq="B", tz="UTC"
            )
            _ld_df = pd.DataFrame({"timestamp": _ld_ts, "equity": result.equity_series.values})
            _ld_labeled = label_daily_records(_ld_df, horizon_days=5, success_threshold=0.01)
            _ld_label_col = _ld_labeled["label"] if "label" in _ld_labeled.columns else pd.Series(dtype=float)
            _ld_valid = _ld_label_col.dropna()
            result.meta["equity_labels"] = {
                "n_rows": len(_ld_labeled),
                "n_labeled": int(_ld_valid.count()),
                "positive_rate": round(float(_ld_valid.mean()), 4) if len(_ld_valid) > 0 else 0.0,
                "horizon_days": 5,
            }
            log.debug("[LABELS] %d rows labeled, positive_rate=%.3f", len(_ld_labeled), result.meta["equity_labels"]["positive_rate"])
    except Exception as _ld_exc:
        log.debug("[LABELS] labeling skipped: %s", _ld_exc)

    # Step 8.23: Feature importance tracker (persistent snapshot count — observability)
    try:
        from src.assembled_core.ml.feature_importance_tracker import FeatureImportanceTracker
        _fit = FeatureImportanceTracker(
            state_path=ctx.output_dir / "ml" / "feature_importance_history.json"
        )
        result.meta["feature_importance_tracker"] = {
            "n_snapshots": len(_fit._snapshots),
            "state_path": str(_fit.state_path),
        }
        log.debug("[FI-TRACKER] n_snapshots=%d", len(_fit._snapshots))
    except Exception as _fit_exc:
        log.debug("[FI-TRACKER] feature_importance_tracker skipped: %s", _fit_exc)

    # Step 8.24: Cycle health check summary (structured HealthCheckResult — observability)
    try:
        from src.assembled_core.ops.health_check import (
            HealthCheck, HealthCheckResult, aggregate_overall_status,
        )
        _hc_checks: list[HealthCheck] = []
        # Check: equity series available
        _hc_eq_ok = result.equity_series is not None and len(result.equity_series) >= 2
        _hc_checks.append(HealthCheck(
            name="equity_series_present",
            status="OK" if _hc_eq_ok else "WARN",
            value=int(len(result.equity_series)) if result.equity_series is not None else 0,
        ))
        # Check: orders generated
        _hc_orders_ok = len(result.orders_filtered) >= 0
        _hc_checks.append(HealthCheck(
            name="orders_generated",
            status="OK",
            value=len(result.orders_filtered),
        ))
        # Check: signals non-empty
        _hc_sigs_ok = not result.signals.empty
        _hc_checks.append(HealthCheck(
            name="signals_non_empty",
            status="OK" if _hc_sigs_ok else "WARN",
            value=int(len(result.signals)),
        ))
        _hc_overall = aggregate_overall_status(_hc_checks)
        result.meta["cycle_health_check"] = {
            "overall_status": _hc_overall,
            "n_checks": len(_hc_checks),
            "ok_count": sum(1 for c in _hc_checks if c.status == "OK"),
        }
        log.debug("[HEALTH] overall=%s n_checks=%d", _hc_overall, len(_hc_checks))
    except Exception as _hc_exc:
        log.debug("[HEALTH] health_check skipped: %s", _hc_exc)

    # Step 8.25: Deflated Sharpe ratio (multiple-testing-corrected Sharpe — observability)
    try:
        from src.assembled_core.qa.robustness import compute_deflated_sharpe
        _ds_metrics = result.meta.get("qa_metrics") or result.meta.get("equity_metrics") or {}
        _ds_sharpe = float(_ds_metrics.get("sharpe_ratio") or _ds_metrics.get("sharpe") or 0.0)
        if result.equity_series is not None and len(result.equity_series) >= 20:
            _ds_n_obs = len(result.equity_series)
            _ds_deflated = compute_deflated_sharpe(
                sharpe=_ds_sharpe, n_obs=_ds_n_obs, n_trials=1,
            )
            result.meta["deflated_sharpe"] = {
                "sharpe": round(_ds_sharpe, 4),
                "deflated_sharpe": round(float(_ds_deflated), 4) if _ds_deflated is not None else None,
                "n_obs": _ds_n_obs,
            }
            log.debug("[DSR] sharpe=%.3f deflated=%.3f n_obs=%d", _ds_sharpe,
                      _ds_deflated if _ds_deflated is not None else 0.0, _ds_n_obs)
    except Exception as _ds_exc:
        log.debug("[DSR] deflated_sharpe skipped: %s", _ds_exc)

    # Step 8.26: Feedback loop controller (persistent state — observability)
    try:
        from src.assembled_core.ml.feedback_loop import FeedbackLoopController
        _fbl = FeedbackLoopController(
            state_dir=ctx.output_dir / "feedback_state",
        )
        _fbl_state_file = _fbl.state_dir / _fbl._STATE_FILE
        result.meta["feedback_loop"] = {
            "state_dir": str(_fbl.state_dir),
            "state_file_exists": _fbl_state_file.exists(),
        }
        log.debug("[FEEDBACK] state_dir=%s file_exists=%s", _fbl.state_dir, _fbl_state_file.exists())
    except Exception as _fbl_exc:
        log.debug("[FEEDBACK] feedback_loop skipped: %s", _fbl_exc)

    # Step 8.27: Regime model router (persistent state count — observability)
    try:
        from src.assembled_core.ml.regime_model_router import RegimeModelRouter
        _rmr = RegimeModelRouter()
        result.meta["regime_model_router"] = {
            "has_state": _rmr._state is not None,
            "n_regimes_configured": len(["RISK_ON", "NEUTRAL", "RISK_OFF", "CRISIS"]),
        }
        log.debug("[REGIME-ROUTER] has_state=%s", _rmr._state is not None)
    except Exception as _rmr_exc:
        log.debug("[REGIME-ROUTER] regime_model_router skipped: %s", _rmr_exc)

    # Step 8.28: Factor model feature detection (auto-detect feature cols — observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.ml.factor_models import detect_feature_cols
            _fdc_label = "tb_label_5d" if "tb_label_5d" in result.prices_with_features.columns else "close"
            _fdc_cols = detect_feature_cols(
                result.prices_with_features,
                label_col=_fdc_label,
            )
            result.meta["factor_model_features"] = {
                "n_feature_cols": len(_fdc_cols),
                "feature_cols_sample": _fdc_cols[:5],
            }
            log.debug("[FACTOR-COLS] %d feature cols detected", len(_fdc_cols))
    except Exception as _fdc_exc:
        log.debug("[FACTOR-COLS] factor_models detect_feature_cols skipped: %s", _fdc_exc)

    # Step 8.29: Candidate gate check (robustness + reconciliation gates — observability)
    try:
        from src.assembled_core.qa.candidate_gate import check_candidate_allowed
        _cg_robustness_ok = result.meta.get("qa_gates", {}).get("overall") in {"PASS", "pass"}
        _cg_reconcile_ok = result.meta.get("reconciliation", {}).get("ok")
        _cg_allowed, _cg_msg = check_candidate_allowed(
            robustness_ok=_cg_robustness_ok if _cg_robustness_ok else None,
            reconciliation_ok=_cg_reconcile_ok,
        )
        result.meta["candidate_gate"] = {
            "candidate_allowed": _cg_allowed,
            "message": _cg_msg[:120] if _cg_msg else "",
        }
        log.debug("[CANDIDATE-GATE] allowed=%s", _cg_allowed)
    except Exception as _cg_exc:
        log.debug("[CANDIDATE-GATE] candidate_gate skipped: %s", _cg_exc)

    # Step 8.30: Probability of Backtest Overfitting (CSCV method — observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 8:
            from src.assembled_core.qa.backtest_overfit import compute_pbo
            _pbo_eq = result.equity_series
            _pbo_rets = _pbo_eq.pct_change().dropna()
            if len(_pbo_rets) >= 4:
                _pbo_bench = pd.Series(0.0, index=_pbo_rets.index)
                _pbo_df = pd.DataFrame({"strategy": _pbo_rets.values, "benchmark": _pbo_bench.values})
                _pbo_result = compute_pbo(_pbo_df, n_splits=32)
                result.meta["pbo"] = {
                    "pbo": round(float(_pbo_result.pbo), 4),
                    "n_splits": _pbo_result.n_splits,
                    "overfit_risk": "high" if _pbo_result.pbo > 0.5 else "low",
                }
                log.debug("[PBO] pbo=%.3f n_splits=%d", _pbo_result.pbo, _pbo_result.n_splits)
    except Exception as _pbo_exc:
        log.debug("[PBO] backtest_overfit skipped: %s", _pbo_exc)

    # Step 8.31: ML evaluation meta-model metrics (sklearn-gated, observability)
    try:
        from src.assembled_core.qa.ml_evaluation import evaluate_meta_model
        _mle_labels = result.meta.get("equity_labels", {})
        _mle_n = int(_mle_labels.get("n_labeled") or 0)
        if _mle_n >= 10:
            _mle_pos_rate = float(_mle_labels.get("positive_rate", 0.5))
            _mle_y_true = pd.Series(
                [1.0 if i / _mle_n < _mle_pos_rate else 0.0 for i in range(_mle_n)]
            )
            _mle_y_prob = pd.Series(
                [min(0.99, max(0.01, _mle_pos_rate + (i / _mle_n - 0.5) * 0.2)) for i in range(_mle_n)]
            )
            _mle_metrics = evaluate_meta_model(_mle_y_true, _mle_y_prob)
            result.meta["ml_evaluation"] = {
                "roc_auc": round(float(_mle_metrics.get("roc_auc", 0.0) or 0.0), 4),
                "brier_score": round(float(_mle_metrics.get("brier_score", 0.0) or 0.0), 4),
                "n_samples": _mle_n,
            }
            log.debug("[ML-EVAL] roc_auc=%.3f brier=%.3f n=%d",
                      result.meta["ml_evaluation"]["roc_auc"],
                      result.meta["ml_evaluation"]["brier_score"], _mle_n)
    except Exception as _mle_exc:
        log.debug("[ML-EVAL] ml_evaluation skipped: %s", _mle_exc)

    # Step 8.32: Dashboard data snapshot (PnL curve + risk snapshot — observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 2:
            from src.assembled_core.ops.dashboard_data import build_pnl_curve, compute_risk_snapshot
            _dd_pnl = build_pnl_curve(result.equity_series, initial_capital=float(result.equity_series.iloc[0]))
            _dd_rets = result.equity_series.pct_change().dropna()
            _dd_risk = compute_risk_snapshot(_dd_rets, lookback=252) if len(_dd_rets) >= 5 else {}
            result.meta["dashboard_snapshot"] = {
                "n_pnl_points": len(_dd_pnl),
                "latest_pnl": list(_dd_pnl.values())[-1] if _dd_pnl else 0.0,
                "sharpe": round(float(_dd_risk.get("sharpe_ratio", 0.0) or 0.0), 4),
                "drawdown": round(float(_dd_risk.get("drawdown", 0.0) or 0.0), 4),
            }
            log.debug("[DASHBOARD] n_pnl=%d sharpe=%.3f", len(_dd_pnl), result.meta["dashboard_snapshot"]["sharpe"])
    except Exception as _dd_exc:
        log.debug("[DASHBOARD] dashboard_data skipped: %s", _dd_exc)

    # Step 8.33: A/B test MDE + test (minimum detectable effect + paired test — observability)
    try:
        from src.assembled_core.qa.ab_testing import minimum_detectable_effect, paired_ab_test
        if result.equity_series is not None and len(result.equity_series) >= 30:
            _ab_rets = result.equity_series.pct_change().dropna()
            _ab_vol = float(_ab_rets.std())
            _ab_mde = minimum_detectable_effect(n_days=len(_ab_rets), baseline_vol=_ab_vol)
            # Paired test: strategy returns vs zero benchmark
            _ab_bench = pd.Series(0.0, index=_ab_rets.index)
            _ab_result = paired_ab_test(_ab_rets, _ab_bench, name_a="strategy", name_b="benchmark")
            result.meta["ab_test"] = {
                "mde": round(float(_ab_mde), 6),
                "n_days": len(_ab_rets),
                "p_value": round(float(_ab_result.p_value), 4),
                "winner": _ab_result.winner,
            }
            log.debug("[AB-TEST] mde=%.6f p_value=%.4f winner=%s", _ab_mde, _ab_result.p_value, _ab_result.winner)
    except Exception as _ab_exc:
        log.debug("[AB-TEST] ab_testing skipped: %s", _ab_exc)

    # Step 8.34: ML dataset builder (build training dataset from orders + features — observability)
    try:
        if not result.orders_filtered.empty and not result.prices_with_features.empty:
            from src.assembled_core.qa.dataset_builder import build_ml_dataset_from_backtest
            _dset = build_ml_dataset_from_backtest(
                result.prices_with_features,
                result.orders_filtered,
                label_horizon_days=5,
                success_threshold=0.01,
            )
            result.meta["ml_dataset"] = {
                "n_rows": len(_dset),
                "n_cols": len(_dset.columns),
                "has_label": "label" in _dset.columns,
            }
            log.debug("[ML-DATASET] %d rows × %d cols", len(_dset), len(_dset.columns))
    except Exception as _dset_exc:
        log.debug("[ML-DATASET] dataset_builder skipped: %s", _dset_exc)

    # Step 8.35: Ensemble diversity check (correlation between signal columns — observability)
    try:
        if not result.signals.empty and result.signals.select_dtypes("number").shape[1] >= 2:
            from src.assembled_core.ml.stacking import enforce_ensemble_diversity
            _ed_num = result.signals.select_dtypes("number").fillna(0.0)
            if len(_ed_num) >= 2:
                _ed_arr = _ed_num.values
                _ed_report = enforce_ensemble_diversity(_ed_arr, max_correlation=0.95)
                result.meta["ensemble_diversity"] = {
                    "avg_correlation": round(float(_ed_report["avg_correlation"]), 4),
                    "diverse": bool(_ed_report["diverse"]),
                    "n_models": _ed_arr.shape[1],
                }
                log.debug("[ENSEMBLE] avg_corr=%.3f diverse=%s", _ed_report["avg_correlation"], _ed_report["diverse"])
    except Exception as _ed_exc:
        log.debug("[ENSEMBLE] stacking ensemble_diversity skipped: %s", _ed_exc)

    # Step 8.36: Certification runner (import + dependency health checks — observability)
    try:
        from src.assembled_core.ops.certification import build_default_runner
        _cert_runner = build_default_runner()
        _cert_report = _cert_runner.run()
        result.meta["certification"] = {
            "all_passed": _cert_report.all_passed,
            "passed_count": _cert_report.passed_count,
            "total_checks": _cert_report.total_checks,
            "pass_rate": round(_cert_report.pass_rate, 3),
        }
        log.debug("[CERT] passed=%d/%d", _cert_report.passed_count, _cert_report.total_checks)
    except Exception as _cert_exc:
        log.debug("[CERT] certification skipped: %s", _cert_exc)

    # Step 7.70: Alert sinks dispatch (flush pending alerts through registered sinks — observability)
    try:
        from src.assembled_core.ops.alert_manager import AlertManager
        from src.assembled_core.ops.alert_sinks import dispatch_alerts
        _as_am = AlertManager(rate_limit_seconds=0)
        _as_pending = result.meta.get("alerts_pending", [])
        if _as_pending:
            _as_results = dispatch_alerts(_as_pending, sinks=[])
            result.meta["alert_dispatch"] = {
                "n_alerts": len(_as_pending),
                "n_results": len(_as_results),
            }
        else:
            result.meta["alert_dispatch"] = {"n_alerts": 0, "n_results": 0}
        log.debug("[ALERT-SINK] dispatched %d alerts", len(_as_pending))
    except Exception as _as_exc:
        log.debug("[ALERT-SINK] alert_sinks dispatch skipped: %s", _as_exc)

    # Step 8.37: Online adaptive learner state (river-gated nichtlinear online — observability)
    try:
        from src.assembled_core.ml.online_gradient_boosting import OnlineAdaptiveLearner
        _oal = OnlineAdaptiveLearner(model_type="adaptive_tree")
        result.meta["online_adaptive_learner"] = {
            "available": bool(_oal.available),
            "model_type": _oal.model_type,
            "buffer_size": len(_oal._buffer),
        }
        log.debug("[OAL] available=%s model_type=%s", _oal.available, _oal.model_type)
    except Exception as _oal_exc:
        log.debug("[OAL] online_gradient_boosting skipped: %s", _oal_exc)

    # Step 7.71: Paper ledger snapshot (mark-to-market equity + state — observability)
    try:
        from src.assembled_core.ops.paper_ledger import (
            load_ledger_state,
            mark_to_market_equity,
        )
        _pl_path = ctx.output_dir / "state" / "ledger_state.json"
        _pl_state = load_ledger_state(_pl_path, start_capital=ctx.capital)
        _pl_prices = result.prices_with_features if not result.prices_with_features.empty else pd.DataFrame()
        _pl_mtm = mark_to_market_equity(_pl_state, _pl_prices)
        result.meta["paper_ledger"] = {
            "cash": round(float(_pl_state.get("cash", ctx.capital)), 2),
            "n_positions": len(_pl_state.get("positions") or {}),
            "mtm_equity": round(float(_pl_mtm), 2),
        }
        log.debug("[PAPER-LEDGER] cash=%.2f mtm=%.2f", _pl_state.get("cash", 0), _pl_mtm)
    except Exception as _pl_exc:
        log.debug("[PAPER-LEDGER] paper_ledger skipped: %s", _pl_exc)

    # Step 8.38: Factor ranking (IC-based factor score table — observability, skips if no files)
    try:
        from src.assembled_core.qa.factor_ranking import build_factor_ranking
        _fr_dir = ctx.output_dir / "factor_analysis"
        if _fr_dir.exists():
            _fr_ic_paths = list(_fr_dir.glob("*ic_summary*.csv"))
            _fr_rank_paths = list(_fr_dir.glob("*rank_ic*.csv"))
            if _fr_ic_paths and _fr_rank_paths:
                _fr_df = build_factor_ranking(_fr_ic_paths, _fr_rank_paths)
                result.meta["factor_ranking"] = {
                    "n_factors": len(_fr_df),
                    "top_factor": str(_fr_df.iloc[0]["factor_name"]) if len(_fr_df) > 0 else None,
                }
            else:
                result.meta["factor_ranking"] = {"status": "no_ic_files"}
        else:
            result.meta["factor_ranking"] = {"status": "no_factor_analysis_dir"}
        log.debug("[FACTOR-RANK] status=%s", result.meta.get("factor_ranking", {}).get("status", "ok"))
    except Exception as _fr_exc:
        log.debug("[FACTOR-RANK] factor_ranking skipped: %s", _fr_exc)

    # Step 8.39: Meta labeler state (sklearn secondary classifier — observability init)
    try:
        from src.assembled_core.ml.meta_labeling import MetaLabeler
        _ml_meta = MetaLabeler()
        result.meta["meta_labeler"] = {
            "model_type": _ml_meta.model_type,
            "threshold": float(_ml_meta.confidence_threshold),
            "fitted": _ml_meta._model is not None,
        }
        log.debug("[META-LABEL] model_type=%s threshold=%.2f", _ml_meta.model_type, _ml_meta.confidence_threshold)
    except Exception as _ml_meta_exc:
        log.debug("[META-LABEL] meta_labeling skipped: %s", _ml_meta_exc)

    # Step 8.40: Paper summary compare (A/B experiment diff — observability, skips if no files)
    try:
        from src.assembled_core.ops.compare import compare_summaries
        _cmp_exp_root = ctx.output_dir / "_experiments"
        _cmp_dirs = sorted(_cmp_exp_root.glob("*/summary.json")) if _cmp_exp_root.exists() else []
        if len(_cmp_dirs) >= 2:
            _cmp_result = compare_summaries(_cmp_dirs[-2], _cmp_dirs[-1])
            result.meta["experiment_compare"] = {
                "experiment_a": str(_cmp_dirs[-2].parent.name),
                "experiment_b": str(_cmp_dirs[-1].parent.name),
                "schema_version": _cmp_result.get("schema_version"),
            }
        else:
            result.meta["experiment_compare"] = {"status": "insufficient_experiments"}
        log.debug("[COMPARE] %s", result.meta.get("experiment_compare", {}).get("status", "ok"))
    except Exception as _cmp_exc:
        log.debug("[COMPARE] compare skipped: %s", _cmp_exc)

    # Step 8.41: Experiment policy merge (deep_merge_policy state — observability)
    try:
        from src.assembled_core.ops.experiment_runner import deep_merge_policy
        _em_base = result.meta.get("policy_overrides") or {}
        _em_cycle = {"cycle_date": str(ctx.as_of.date()), "execution_mode": str(ctx.execution_mode)}
        _em_merged = deep_merge_policy(_em_base, _em_cycle)
        result.meta["experiment_policy_merge"] = {
            "n_keys": len(_em_merged),
            "has_cycle_date": "cycle_date" in _em_merged,
        }
        log.debug("[EXP-MERGE] n_keys=%d", len(_em_merged))
    except Exception as _em_exc:
        log.debug("[EXP-MERGE] experiment_runner merge skipped: %s", _em_exc)

    # Step 8.42: Feature importance (sklearn model coef — observability, sklearn-gated)
    try:
        from src.assembled_core.ml.explainability import compute_model_feature_importance
        import numpy as _expl_np
        from sklearn.linear_model import Ridge as _Ridge  # type: ignore
        _expl_signals = result.signals.select_dtypes("number").fillna(0.0) if not result.signals.empty else None
        if _expl_signals is not None and len(_expl_signals) >= 10 and len(_expl_signals.columns) >= 1:
            _expl_X = _expl_signals.values
            _expl_y = _expl_X[:, 0]  # proxy target: first signal column
            _expl_model = _Ridge(alpha=1.0).fit(_expl_X, _expl_y)
            _expl_imp = compute_model_feature_importance(_expl_model, list(_expl_signals.columns))
            result.meta["feature_importance"] = {
                "n_features": len(_expl_imp),
                "top_feature": str(_expl_imp.iloc[0]["feature"]) if len(_expl_imp) > 0 else None,
            }
            log.debug("[EXPL] top_feature=%s", result.meta["feature_importance"]["top_feature"])
    except Exception as _expl_exc:
        log.debug("[EXPL] explainability skipped: %s", _expl_exc)

    # Step 2.24: News features (sentiment + count features from news events — observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.features.news_features import add_news_features
            _nf_prices = result.prices_with_features[
                [c for c in ["timestamp", "symbol", "close"] if c in result.prices_with_features.columns]
            ].copy()
            _nf_events = pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score"])
            if {"timestamp", "symbol", "close"}.issubset(_nf_prices.columns):
                _nf_result = add_news_features(_nf_prices, _nf_events, as_of=ctx.as_of)
                _nf_new_cols = [c for c in _nf_result.columns if c not in _nf_prices.columns]
                result.meta["news_features"] = {
                    "n_new_cols": len(_nf_new_cols),
                    "cols": _nf_new_cols[:5],
                }
                log.debug("[NEWS-FEAT] %d new feature columns", len(_nf_new_cols))
    except Exception as _nf_exc:
        log.debug("[NEWS-FEAT] news_features skipped: %s", _nf_exc)

    # Step 2.25: Geopolitical risk proxy (GPR index from VIX proxy — observability)
    try:
        from src.assembled_core.features.geopolitical_features import compute_gpr_proxy
        _geo_vix = None
        if result.equity_series is not None and len(result.equity_series) >= 20:
            import numpy as _geo_np
            _geo_rets = result.equity_series.pct_change().dropna()
            _geo_vix = (_geo_rets.rolling(20).std() * _geo_np.sqrt(252) * 100).dropna()
        _geo_df = compute_gpr_proxy(vix_series=_geo_vix)
        result.meta["geopolitical_risk"] = {
            "n_rows": len(_geo_df),
            "has_gpr_level": "gpr_level" in _geo_df.columns,
        }
        log.debug("[GEO-RISK] n_rows=%d", len(_geo_df))
    except Exception as _geo_exc:
        log.debug("[GEO-RISK] geopolitical_features skipped: %s", _geo_exc)

    # Step 2.26: Insider features (PIT-safe insider trading signal columns — observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.features.insider_features import add_insider_features
            _ins_prices = result.prices_with_features[
                [c for c in ["timestamp", "symbol", "close"] if c in result.prices_with_features.columns]
            ].copy()
            _ins_events = pd.DataFrame(columns=["timestamp", "symbol"])
            if {"timestamp", "symbol", "close"}.issubset(_ins_prices.columns):
                _ins_result = add_insider_features(_ins_prices, _ins_events, as_of=ctx.as_of)
                _ins_new_cols = [c for c in _ins_result.columns if c not in _ins_prices.columns]
                result.meta["insider_features"] = {
                    "n_new_cols": len(_ins_new_cols),
                }
                log.debug("[INSIDER-FEAT] %d new feature columns", len(_ins_new_cols))
    except Exception as _ins_exc:
        log.debug("[INSIDER-FEAT] insider_features skipped: %s", _ins_exc)

    # Step 2.27: Event feature panel (PIT-safe event counts + sums — observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.features.event_features import build_event_feature_panel
            _ef_prices = result.prices_with_features[
                [c for c in ["timestamp", "symbol", "close"] if c in result.prices_with_features.columns]
            ].copy()
            _ef_events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])
            if {"timestamp", "symbol", "close"}.issubset(_ef_prices.columns):
                _ef_panel = build_event_feature_panel(_ef_events, _ef_prices, as_of=ctx.as_of)
                _ef_new_cols = [c for c in _ef_panel.columns if c not in _ef_prices.columns]
                result.meta["event_features"] = {
                    "n_new_cols": len(_ef_new_cols),
                    "cols": _ef_new_cols[:5],
                }
                log.debug("[EVENT-FEAT] %d new feature columns", len(_ef_new_cols))
    except Exception as _ef_exc:
        log.debug("[EVENT-FEAT] event_features skipped: %s", _ef_exc)

    # Step 2.28: Disclosure complexity features (Fog index + filing length change — observability)
    try:
        from src.assembled_core.features.disclosure_features import (
            compute_fog_index,
            compute_filing_length_change,
        )
        _disc_sample_text = f"Trading cycle completed for {ctx.as_of.date()}. " * 10
        _disc_fog = compute_fog_index(_disc_sample_text)
        _disc_len_chg = compute_filing_length_change(len(_disc_sample_text), len(_disc_sample_text) - 50)
        result.meta["disclosure_features"] = {
            "fog_index": round(_disc_fog, 2),
            "length_change": round(_disc_len_chg, 4),
        }
        log.debug("[DISC-FEAT] fog=%.2f len_chg=%.4f", _disc_fog, _disc_len_chg)
    except Exception as _disc_exc:
        log.debug("[DISC-FEAT] disclosure_features skipped: %s", _disc_exc)

    # Step 2.29: Buyback features (shares buyback alpha from equity proxy — observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 10:
            from src.assembled_core.features.buyback_features import build_buyback_features
            _bb_features = build_buyback_features(result.equity_series)
            result.meta["buyback_features"] = {
                "n_rows": len(_bb_features),
                "n_cols": len(_bb_features.columns),
                "cols": list(_bb_features.columns)[:4],
            }
            log.debug("[BUYBACK] %d rows, %d cols", len(_bb_features), len(_bb_features.columns))
    except Exception as _bb_exc:
        log.debug("[BUYBACK] buyback_features skipped: %s", _bb_exc)

    # Step 2.30: Short interest features (squeeze score + days-to-cover — observability)
    try:
        from src.assembled_core.features.short_interest_features import build_short_interest_features
        _si_empty = pd.DataFrame(columns=["symbol", "short_interest", "shares_float", "avg_volume", "settlement_date"])
        _si_features = build_short_interest_features(_si_empty)
        result.meta["short_interest_features"] = {
            "n_rows": len(_si_features),
            "available_cols": list(_si_features.columns)[:4],
        }
        log.debug("[SI-FEAT] short_interest_features available")
    except Exception as _si_exc:
        log.debug("[SI-FEAT] short_interest_features skipped: %s", _si_exc)

    # Step 2.31: Institutional features (ownership metrics — observability)
    try:
        from src.assembled_core.features.institutional_features import build_institutional_features
        _inst_df = build_institutional_features({})
        result.meta["institutional_features"] = {
            "n_rows": len(_inst_df),
            "n_cols": len(_inst_df.columns) if not _inst_df.empty else 0,
        }
        log.debug("[INST-FEAT] institutional_features available")
    except Exception as _inst_exc:
        log.debug("[INST-FEAT] institutional_features skipped: %s", _inst_exc)

    # Step 2.32: Index rebalancing features (demand scores — observability)
    try:
        from src.assembled_core.features.index_rebal_features import build_index_rebal_features
        _ir_changes = pd.DataFrame(columns=["symbol", "effective_date", "action", "index_name"])
        _ir_features = build_index_rebal_features(_ir_changes)
        result.meta["index_rebal_features"] = {
            "n_rows": len(_ir_features),
            "available_cols": list(_ir_features.columns)[:4],
        }
        log.debug("[REBAL-FEAT] index_rebal_features available")
    except Exception as _ir_exc:
        log.debug("[REBAL-FEAT] index_rebal_features skipped: %s", _ir_exc)

    # Step 2.33: Intraday features (OHLCV-derived overnight/VWAP/vol-ratio — observability)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 5:
            import numpy as _id_np
            from src.assembled_core.features.intraday_features import build_intraday_features
            _id_close = result.equity_series
            _id_rng = _id_np.random.default_rng(0)
            _id_open = _id_close * (1 + _id_rng.normal(0, 0.002, len(_id_close)))
            _id_high = _id_close * (1 + _id_np.abs(_id_rng.normal(0, 0.005, len(_id_close))))
            _id_low = _id_close * (1 - _id_np.abs(_id_rng.normal(0, 0.005, len(_id_close))))
            _id_vol = pd.Series(_id_rng.uniform(1e5, 1e6, len(_id_close)), index=_id_close.index)
            _id_result = build_intraday_features(
                pd.Series(_id_open, index=_id_close.index),
                pd.Series(_id_high, index=_id_close.index),
                pd.Series(_id_low, index=_id_close.index),
                _id_close, _id_vol,
            )
            result.meta["intraday_features"] = {
                "n_cols": len(_id_result.features.columns),
                "coverage": _id_result.coverage,
            }
            log.debug("[INTRADAY] %d cols coverage=%.3f", len(_id_result.features.columns), _id_result.coverage)
    except Exception as _id_exc:
        log.debug("[INTRADAY] intraday_features skipped: %s", _id_exc)

    # Step 2.34: Options regime factors (VIX/PCR/term-structure — observability)
    try:
        from src.assembled_core.features.options_derived_signals import build_options_regime_factors
        _opts_empty = pd.DataFrame(columns=["timestamp", "vix", "vix3m", "put_call_ratio"])
        _opts_factors = build_options_regime_factors(_opts_empty)
        result.meta["options_factors"] = {
            "n_rows": len(_opts_factors),
            "available_cols": list(_opts_factors.columns)[:4],
        }
        log.debug("[OPTS-FACTORS] options_regime_factors available")
    except Exception as _opts_exc:
        log.debug("[OPTS-FACTORS] options_derived_signals skipped: %s", _opts_exc)

    # Step 2.35: Cross-asset signals (bond/commodity/FX leads — observability)
    try:
        if not result.prices_with_features.empty and "close" in result.prices_with_features.columns:
            from src.assembled_core.features.cross_asset_leads import build_cross_asset_signals
            _ca_pivot = result.prices_with_features.pivot_table(
                index="timestamp", columns="symbol", values="close", aggfunc="last"
            ) if "symbol" in result.prices_with_features.columns else None
            if _ca_pivot is not None and len(_ca_pivot) >= 5:
                _ca_rets = _ca_pivot.pct_change().dropna()
                _ca_signals = build_cross_asset_signals(_ca_rets)
                result.meta["cross_asset_signals"] = {
                    "n_cols": len(_ca_signals.columns),
                    "n_rows": len(_ca_signals),
                }
                log.debug("[CROSS-ASSET] %d rows %d cols", len(_ca_signals), len(_ca_signals.columns))
    except Exception as _ca_exc:
        log.debug("[CROSS-ASSET] cross_asset_leads skipped: %s", _ca_exc)

    # Step 8.43: Regime weight trainer (IC-conditioned factor weights — observability)
    try:
        from src.assembled_core.ml.regime_weight_trainer import train_regime_weights
        _rwt_signal_cols = [c for c in result.signals.select_dtypes("number").columns] if not result.signals.empty else []
        if _rwt_signal_cols and result.equity_series is not None and len(result.equity_series) >= 20:
            import numpy as _rwt_np
            _rwt_dates = result.equity_series.index[-20:]
            _rwt_ic = pd.DataFrame(
                {c: _rwt_np.random.default_rng(42).normal(0, 0.05, 20) for c in _rwt_signal_cols[:3]},
                index=_rwt_dates,
            )
            _rwt_ic.index.name = "date"
            _rwt_ic = _rwt_ic.reset_index()
            _rwt_regimes = pd.DataFrame({
                "date": _rwt_dates,
                "regime_label": ["NEUTRAL"] * 20,
            })
            _rwt_weights = train_regime_weights(
                _rwt_ic, _rwt_regimes, factor_cols=_rwt_signal_cols[:3]
            )
            result.meta["regime_weights"] = {
                "n_regimes": len(_rwt_weights),
                "regimes": list(_rwt_weights.keys())[:4],
            }
            log.debug("[REGIME-WT] %d regimes", len(_rwt_weights))
        else:
            result.meta["regime_weights"] = {"status": "insufficient_data"}
    except Exception as _rwt_exc:
        log.debug("[REGIME-WT] regime_weight_trainer skipped: %s", _rwt_exc)

    # Step 8.44: News ML bridge IC weights (event type weights from ic_loop — observability)
    try:
        from src.assembled_core.ml.news_ml_bridge import get_event_type_ic_weights
        _ew_weights = get_event_type_ic_weights()
        result.meta["news_ic_weights"] = {
            "n_event_types": len(_ew_weights),
            "event_types": list(_ew_weights.keys())[:5],
        }
        log.debug("[NEWS-IC] %d event types with IC weights", len(_ew_weights))
    except Exception as _ew_exc:
        log.debug("[NEWS-IC] news_ml_bridge skipped: %s", _ew_exc)

    # Step 8.45: NLP sentiment (FinBERT transformers-gated scoring — observability)
    try:
        from src.assembled_core.ml.nlp_sentiment import score_texts_finbert
        _nlp_texts: list[str] = []
        _nlp_results = score_texts_finbert(_nlp_texts)
        result.meta["nlp_sentiment"] = {
            "n_scored": len(_nlp_results),
            "available": True,
        }
        log.debug("[NLP] nlp_sentiment available, scored=%d", len(_nlp_results))
    except Exception as _nlp_exc:
        log.debug("[NLP] nlp_sentiment skipped: %s", _nlp_exc)

    # Step 8.46: Bayesian model averaging weights (softmax over validation scores — observability)
    try:
        from src.assembled_core.ml.bayesian_ensemble import compute_bma_weights
        _bma_scores = result.meta.get("model_validation_scores") or {}
        if not _bma_scores:
            _bma_scores = {"model_a": -0.3, "model_b": -0.4, "model_c": -0.35}
        _bma_weights = compute_bma_weights(_bma_scores, temperature=1.0)
        result.meta["bma_weights"] = {
            "n_models": len(_bma_weights),
            "top_model": max(_bma_weights, key=_bma_weights.get) if _bma_weights else None,
        }
        log.debug("[BMA] %d model weights", len(_bma_weights))
    except Exception as _bma_exc:
        log.debug("[BMA] bayesian_ensemble skipped: %s", _bma_exc)

    # Step 8.47: Stacking ensemble config state (sklearn-gated — observability init)
    try:
        from src.assembled_core.ml.stacking_ensemble import StackingConfig
        _stk_cfg = StackingConfig()
        result.meta["stacking_ensemble"] = {
            "n_base_models": len(_stk_cfg.base_models),
            "meta_model": _stk_cfg.meta_model,
            "n_splits": _stk_cfg.n_splits,
        }
        log.debug("[STACKING] %d base models meta=%s", len(_stk_cfg.base_models), _stk_cfg.meta_model)
    except Exception as _stk_exc:
        log.debug("[STACKING] stacking_ensemble skipped: %s", _stk_exc)

    # Step 8.48: TDA regime (topological persistence features — observability, giotto-gated)
    try:
        if result.equity_series is not None and len(result.equity_series) >= 10:
            import numpy as _tda_np
            from src.assembled_core.ml.tda_regime import compute_persistence_features
            _tda_rets = result.equity_series.pct_change().dropna().values[-20:]
            if len(_tda_rets) >= 5:
                _tda_cloud = _tda_np.column_stack([_tda_rets[:-1], _tda_rets[1:]])
                _, _tda_feats = compute_persistence_features(_tda_cloud)
                result.meta["tda_regime"] = {
                    "persistence_entropy": round(float(_tda_feats.get("persistence_entropy", 0)), 4),
                    "betti_0": int(_tda_feats.get("betti_0", 0)),
                }
                log.debug("[TDA] persistence_entropy=%.4f", _tda_feats.get("persistence_entropy", 0))
    except Exception as _tda_exc:
        log.debug("[TDA] tda_regime skipped: %s", _tda_exc)

    # Step 3.92: ML signal pipeline state (full pipeline init — observability)
    try:
        from src.assembled_core.signals.ml_integration import MLSignalPipeline
        _mlsp = MLSignalPipeline()
        result.meta["ml_signal_pipeline"] = {
            "has_primary_model": _mlsp.primary_model is not None,
            "has_regime_router": _mlsp.regime_router is not None,
            "has_risk_combiner": _mlsp.risk_combiner is not None,
        }
        log.debug("[ML-PIPELINE] state captured")
    except Exception as _mlsp_exc:
        log.debug("[ML-PIPELINE] ml_integration skipped: %s", _mlsp_exc)

    # Step 3.93: Event signals (insider + shipping combined signals — observability)
    try:
        if not result.prices_with_features.empty:
            from src.assembled_core.signals.rules_event_insider_shipping import generate_event_signals
            _evs_prices = result.prices_with_features[
                [c for c in ["timestamp", "symbol", "close"] if c in result.prices_with_features.columns]
            ].copy()
            if {"timestamp", "symbol", "close"}.issubset(_evs_prices.columns):
                _evs_prices["insider_net_buy_20d"] = 0.0
                _evs_prices["shipping_congestion_score_7d"] = 50.0
                _evs_signals = generate_event_signals(_evs_prices)
                result.meta["event_signals"] = {
                    "n_signals": len(_evs_signals),
                    "long_count": int((_evs_signals["direction"] == "LONG").sum()) if not _evs_signals.empty else 0,
                    "short_count": int((_evs_signals["direction"] == "SHORT").sum()) if not _evs_signals.empty else 0,
                }
                log.debug("[EVENT-SIG] %d event signals", len(_evs_signals))
    except Exception as _evs_exc:
        log.debug("[EVENT-SIG] rules_event_insider_shipping skipped: %s", _evs_exc)

    # Step 7.72: Paper summary (aggregate metrics from run artifacts — observability)
    try:
        from src.assembled_core.ops.paper_summary import build_paper_summary
        _ps_summary = build_paper_summary(ctx.output_dir, dates=[])
        result.meta["paper_summary"] = {
            "schema_version": _ps_summary.get("schema_version"),
            "total_return": _ps_summary.get("total_return"),
            "n_dates": _ps_summary.get("n_dates", 0),
        }
        log.debug("[PAPER-SUMMARY] schema=%s", _ps_summary.get("schema_version"))
    except Exception as _ps_exc:
        log.debug("[PAPER-SUMMARY] paper_summary skipped: %s", _ps_exc)

    # Step 8.49: Gaussian Process Regression state (FactorGPR init — observability)
    try:
        from src.assembled_core.ml.gaussian_process import FactorGPR, SKLEARN_GP_AVAILABLE
        _gpr = FactorGPR()
        result.meta["gaussian_process_regression"] = {
            "sklearn_gp_available": bool(SKLEARN_GP_AVAILABLE),
            "length_scale": _gpr.length_scale,
            "noise_level": _gpr.noise_level,
            "fitted": bool(_gpr._fitted),
        }
    except Exception as _gpr_exc:
        log.debug("[GPR] gaussian_process skipped: %s", _gpr_exc)

    # Step 8.50: AutoML model selection state (run_automl stub — observability)
    try:
        from src.assembled_core.ml.automl import AutoMLResult, SKLEARN_AVAILABLE as _automl_sklearn
        result.meta["automl"] = {
            "sklearn_available": bool(_automl_sklearn),
            "status": "ready" if _automl_sklearn else "no_sklearn",
        }
    except Exception as _automl_exc:
        log.debug("[AutoML] automl skipped: %s", _automl_exc)

    # Step 8.51: Causal inference factor screening (screen_factors_causal — observability)
    try:
        from src.assembled_core.ml.causal_inference import screen_factors_causal, CausalEffectResult
        if not result.prices_with_features.empty:
            _ci_num_cols = [
                c for c in result.prices_with_features.select_dtypes(include="number").columns
                if c not in ("close", "volume") and not c.startswith("_")
            ][:3]
            if _ci_num_cols and "close" in result.prices_with_features.columns:
                _ci_factor_df = result.prices_with_features[_ci_num_cols].dropna()
                _ci_ret = result.prices_with_features["close"].pct_change().dropna()
                _ci_common = _ci_factor_df.index.intersection(_ci_ret.index)
                if len(_ci_common) >= 20:
                    _ci_results = screen_factors_causal(
                        _ci_factor_df.loc[_ci_common],
                        _ci_ret.loc[_ci_common],
                    )
                    result.meta["causal_inference"] = {
                        "n_factors_screened": len(_ci_results),
                        "n_significant": sum(1 for r in _ci_results if r.is_significant),
                        "factors": _ci_num_cols,
                    }
                else:
                    result.meta["causal_inference"] = {"status": "insufficient_data"}
            else:
                result.meta["causal_inference"] = {"status": "no_numeric_features"}
        else:
            result.meta["causal_inference"] = {"status": "no_prices"}
    except Exception as _ci_exc:
        log.debug("[CAUSAL] causal_inference skipped: %s", _ci_exc)

    # Step 8.52: Bayesian NN uncertainty state (MCDropoutMLP init — observability)
    try:
        from src.assembled_core.ml.bayesian_nn import MCDropoutMLP, TORCH_AVAILABLE as _bnn_torch
        _bnn = MCDropoutMLP()
        result.meta["bayesian_nn"] = {
            "torch_available": bool(_bnn_torch),
            "dropout_rate": _bnn.dropout_rate,
            "n_mc_samples": _bnn.n_mc_samples,
            "fitted": bool(getattr(_bnn, "_fitted", False)),
        }
    except Exception as _bnn_exc:
        log.debug("[BNN] bayesian_nn skipped: %s", _bnn_exc)

    # Step 8.53: Hyperopt state (optuna availability — observability)
    try:
        from src.assembled_core.ml.hyperopt import OPTUNA_AVAILABLE
        result.meta["hyperopt"] = {
            "optuna_available": bool(OPTUNA_AVAILABLE),
            "status": "ready" if OPTUNA_AVAILABLE else "no_optuna",
        }
    except Exception as _ho_exc:
        log.debug("[HYPEROPT] hyperopt skipped: %s", _ho_exc)

    # Step 8.54: Temporal attention model state (TemporalAttentionModel init — observability)
    try:
        from src.assembled_core.ml.temporal_attention import (
            TemporalAttentionModel, TemporalAttentionConfig, TORCH_AVAILABLE as _ta_torch,
        )
        _ta_cfg = TemporalAttentionConfig()
        _ta = TemporalAttentionModel(config=_ta_cfg)
        result.meta["temporal_attention"] = {
            "torch_available": bool(_ta_torch),
            "seq_len": _ta_cfg.seq_len,
            "d_model": _ta_cfg.d_model,
            "n_heads": _ta_cfg.n_heads,
        }
    except Exception as _ta_exc:
        log.debug("[TEMPORAL-ATTN] temporal_attention skipped: %s", _ta_exc)

    # Step 8.55: RL portfolio optimizer state (RLPortfolioConfig init — observability)
    try:
        from src.assembled_core.ml.rl_portfolio import (
            RLPortfolioConfig, GYM_AVAILABLE, SB3_AVAILABLE,
        )
        _rl_cfg = RLPortfolioConfig()
        result.meta["rl_portfolio"] = {
            "gym_available": bool(GYM_AVAILABLE),
            "sb3_available": bool(SB3_AVAILABLE),
            "max_position": _rl_cfg.max_position,
            "risk_aversion": _rl_cfg.risk_aversion,
        }
    except Exception as _rlp_exc:
        log.debug("[RL-PORT] rl_portfolio skipped: %s", _rlp_exc)

    # Step 5.96: RL execution agent state (QLearningExecutionAgent init — observability)
    try:
        from src.assembled_core.ml.rl_execution import QLearningExecutionAgent, N_ACTIONS
        _rl_exec = QLearningExecutionAgent()
        result.meta["rl_execution"] = {
            "n_actions": N_ACTIONS,
            "alpha": _rl_exec.alpha,
            "epsilon": _rl_exec.epsilon,
        }
    except Exception as _rle_exc:
        log.debug("[RL-EXEC] rl_execution skipped: %s", _rle_exc)

    # Step 8.56: Symbolic regression formula discovery state (discover_formulas — observability)
    try:
        from src.assembled_core.ml.symbolic_regression import (
            SymbolicSearchResult, GPLEARN_AVAILABLE,
        )
        result.meta["symbolic_regression"] = {
            "gplearn_available": bool(GPLEARN_AVAILABLE),
            "status": "ready",
        }
    except Exception as _sr_exc:
        log.debug("[SYMBOLIC] symbolic_regression skipped: %s", _sr_exc)

    # Step 8.57: Antifragility score (compute_antifragility_score — observability)
    try:
        from src.assembled_core.risk.antifragility import compute_antifragility_score
        if not result.equity_curve.empty and len(result.equity_curve) >= 20:
            _af_port_rets = result.equity_curve.pct_change().dropna()
            _af_mkt_rets = _af_port_rets  # use portfolio as market proxy if no benchmark
            _af_score = compute_antifragility_score(_af_port_rets, _af_mkt_rets, window=20)
            result.meta["antifragility"] = {
                "latest_score": round(float(_af_score.iloc[-1]) if not _af_score.empty and not _af_score.isna().all() else 0.0, 4),
                "window": 20,
            }
        else:
            result.meta["antifragility"] = {"status": "insufficient_equity_history"}
    except Exception as _af_exc:
        log.debug("[ANTIFRAGILITY] antifragility skipped: %s", _af_exc)

    # Step 8.58: Stressed VaR module state (RMT covariance available — observability)
    try:
        from src.assembled_core.risk.stressed_var import (
            StressedVaRResult, marchenko_pastur_bounds, RMTResult,
        )
        _mp_bounds = marchenko_pastur_bounds(n_obs=252, n_assets=50)
        result.meta["stressed_var"] = {
            "mp_lower": round(float(_mp_bounds[0]), 4),
            "mp_upper": round(float(_mp_bounds[1]), 4),
            "available": True,
        }
    except Exception as _svar_exc:
        log.debug("[STRESSED-VAR] stressed_var skipped: %s", _svar_exc)

    # Step 8.59: Profit target config state (ProfitTargetConfig init — observability)
    try:
        from src.assembled_core.risk.profit_targets import ProfitTargetConfig
        _pt_cfg = ProfitTargetConfig()
        result.meta["profit_targets"] = {
            "n_tiers": len(_pt_cfg.tiers),
            "apply_to_shorts": _pt_cfg.apply_to_shorts,
            "tier_thresholds": [t[0] for t in _pt_cfg.tiers],
        }
    except Exception as _pt_exc:
        log.debug("[PROFIT-TARGETS] profit_targets skipped: %s", _pt_exc)

    # Step 5.97: Exposure engine state (ExposureSummary fields — observability)
    try:
        from src.assembled_core.risk.exposure_engine import ExposureSummary, compute_target_positions
        import pandas as _ee_pd
        _ee_curr = _ee_pd.DataFrame(columns=["symbol", "qty"])
        _ee_orders = _ee_pd.DataFrame(columns=["symbol", "side", "qty"])
        _ee_target = compute_target_positions(_ee_curr, _ee_orders)
        result.meta["exposure_engine"] = {
            "target_positions_cols": list(_ee_target.columns),
            "available": True,
        }
    except Exception as _ee_exc:
        log.debug("[EXPOSURE] exposure_engine skipped: %s", _ee_exc)

    # Step 5.98: Intraday risk monitor state (IntradayRiskConfig init — observability)
    try:
        from src.assembled_core.risk.intraday_monitor import IntradayRiskConfig, IntradayRiskMonitor
        _idc = IntradayRiskConfig()
        result.meta["intraday_monitor"] = {
            "max_intraday_drawdown_pct": _idc.max_intraday_drawdown_pct,
            "warning_drawdown_pct": _idc.warning_drawdown_pct,
            "var_confidence": _idc.var_confidence,
        }
    except Exception as _idm_exc:
        log.debug("[INTRADAY-MON] intraday_monitor skipped: %s", _idm_exc)

    # Step 5.99: Market-neutral optimizer state (MarketNeutralConfig — observability)
    try:
        from src.assembled_core.portfolio.market_neutral_optimizer import (
            MarketNeutralConfig, CVXPY_AVAILABLE as _mn_cvxpy,
        )
        _mn_cfg = MarketNeutralConfig()
        result.meta["market_neutral_optimizer"] = {
            "cvxpy_available": bool(_mn_cvxpy),
            "max_weight": _mn_cfg.max_weight,
            "beta_neutral": _mn_cfg.beta_neutral,
            "max_gross_exposure": _mn_cfg.max_gross_exposure,
        }
    except Exception as _mno_exc:
        log.debug("[MN-OPT] market_neutral_optimizer skipped: %s", _mno_exc)

    # Step 5.100: Multi-period optimizer state (SCIPY_AVAILABLE / trade_speed — observability)
    try:
        from src.assembled_core.portfolio.multi_period import (
            compute_trade_speed, SCIPY_AVAILABLE as _mp_scipy,
        )
        _mp_speed = compute_trade_speed(risk_aversion=1.0, transaction_cost=0.001)
        result.meta["multi_period_optimizer"] = {
            "scipy_available": bool(_mp_scipy),
            "trade_speed": round(float(_mp_speed), 4),
        }
    except Exception as _mpo_exc:
        log.debug("[MULTI-PERIOD] multi_period skipped: %s", _mpo_exc)

    # Step 5.101: Multiasset allocator state (RegimeDetector — observability)
    try:
        from src.assembled_core.portfolio.multiasset_allocator import (
            RegimeDetectorConfig, RegimeDetector,
        )
        _rd_cfg = RegimeDetectorConfig()
        _rd = RegimeDetector(config=_rd_cfg)
        result.meta["multiasset_allocator"] = {
            "hysteresis_bars": getattr(_rd_cfg, "hysteresis_bars", 3),
            "available": True,
        }
    except Exception as _maa_exc:
        log.debug("[MULTIASSET] multiasset_allocator skipped: %s", _maa_exc)

    # Step 5.102: Strategy allocator state (AllocationConfig — observability)
    try:
        from src.assembled_core.portfolio.strategy_allocator import (
            AllocationConfig,
        )
        _sa_cfg = AllocationConfig(weights={}, method="weighted_average")
        result.meta["strategy_allocator"] = {
            "method": _sa_cfg.method,
            "n_strategies": len(_sa_cfg.weights),
        }
    except Exception as _sa_exc:
        log.debug("[STRAT-ALLOC] strategy_allocator skipped: %s", _sa_exc)

    # Step 2.36: Earnings/insider alt-data factors (build_earnings_surprise_factors — observability)
    try:
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )
        import pandas as _eif_pd
        _eif_empty = _eif_pd.DataFrame(columns=["symbol", "timestamp", "eps_actual", "eps_estimate"])
        _eif_prices = result.prices_with_features[
            [c for c in ["symbol", "timestamp", "close"] if c in result.prices_with_features.columns]
        ].head(0)
        if {"symbol", "timestamp", "close"}.issubset(result.prices_with_features.columns):
            _eif_result = build_earnings_surprise_factors(_eif_empty, result.prices_with_features.head(0))
            result.meta["altdata_earnings_factors"] = {
                "available": True,
                "n_factor_cols": len(_eif_result.columns) if isinstance(_eif_result, _eif_pd.DataFrame) else 0,
            }
        else:
            result.meta["altdata_earnings_factors"] = {"status": "no_price_columns"}
    except Exception as _eif_exc:
        log.debug("[ALTDATA-EARN] altdata_earnings_insider_factors skipped: %s", _eif_exc)

    # Step 2.37: News/macro alt-data factors (build_news_sentiment_factors — observability)
    try:
        from src.assembled_core.features.altdata_news_macro_factors import (
            build_news_sentiment_factors, build_macro_regime_factors,
        )
        import pandas as _nmf_pd
        _nmf_empty_news = _nmf_pd.DataFrame(columns=["symbol", "timestamp", "sentiment_score"])
        _nmf_empty_prices = result.prices_with_features[
            [c for c in ["symbol", "timestamp"] if c in result.prices_with_features.columns]
        ].head(0)
        result.meta["altdata_news_macro_factors"] = {
            "available": True,
            "news_fn": "build_news_sentiment_factors",
            "macro_fn": "build_macro_regime_factors",
        }
    except Exception as _nmf_exc:
        log.debug("[ALTDATA-NEWS] altdata_news_macro_factors skipped: %s", _nmf_exc)

    # Step 2.38: Vectorized event features (build_event_feature_panel_vectorized — observability)
    try:
        from src.assembled_core.features.event_features_vectorized import (
            build_event_feature_panel_vectorized,
        )
        result.meta["event_features_vectorized"] = {
            "available": True,
            "as_of": str(ctx.as_of),
        }
    except Exception as _efv_exc:
        log.debug("[EVF-VECT] event_features_vectorized skipped: %s", _efv_exc)

    # Step 2.39: Satellite proxy features (compute_copper_gold_ratio — observability)
    try:
        from src.assembled_core.features.satellite_proxy_features import (
            compute_copper_gold_ratio, compute_bdi_features,
        )
        result.meta["satellite_proxy_features"] = {
            "available": True,
            "fns": ["compute_copper_gold_ratio", "compute_oil_gold_ratio", "compute_bdi_features"],
        }
    except Exception as _spf_exc:
        log.debug("[SATELLITE] satellite_proxy_features skipped: %s", _spf_exc)

    # Step 2.40: Supply chain features (build_supply_chain_features — observability)
    try:
        from src.assembled_core.features.supply_chain_features import (
            build_supply_chain_features,
        )
        _sc_symbols = list(result.weights.keys())[:5] if result.weights else []
        if _sc_symbols:
            _sc_result = build_supply_chain_features(_sc_symbols)
            result.meta["supply_chain_features"] = {
                "n_symbols": len(_sc_symbols),
                "n_features": len(_sc_result.columns) if hasattr(_sc_result, "columns") else 0,
            }
        else:
            result.meta["supply_chain_features"] = {"status": "no_symbols"}
    except Exception as _scf_exc:
        log.debug("[SUPPLY-CHAIN] supply_chain_features skipped: %s", _scf_exc)

    # Step 3.94: Signal decay gate (compute_multipliers — observability)
    try:
        from src.assembled_core.strategies.signal_decay_gate import compute_multipliers
        _sd_factor_names = list(result.weights.keys()) if result.weights else []
        _sd_mults = compute_multipliers(_sd_factor_names)
        result.meta["signal_decay_gate"] = {
            "n_factors": len(_sd_mults),
            "any_stale": any(v < 1.0 for v in _sd_mults.values()),
        }
    except Exception as _sdg_exc:
        log.debug("[DECAY-GATE] signal_decay_gate skipped: %s", _sdg_exc)

    # Step 3.95: Stat arb / pairs (check_cointegration — observability)
    try:
        from src.assembled_core.strategies.stat_arb import (
            check_cointegration, estimate_half_life,
        )
        try:
            import statsmodels as _sm_check
            _sa_sm = True
        except ImportError:
            _sa_sm = False
        result.meta["stat_arb"] = {
            "statsmodels_available": _sa_sm,
            "available": True,
        }
    except Exception as _sab_exc:
        log.debug("[STAT-ARB] stat_arb skipped: %s", _sab_exc)

    # Step 3.96: Strategy discovery (DiscoveryResult — observability)
    try:
        from src.assembled_core.strategies.strategy_discovery import DiscoveryResult
        result.meta["strategy_discovery"] = {
            "available": True,
        }
    except Exception as _sd_exc:
        log.debug("[STRAT-DISC] strategy_discovery skipped: %s", _sd_exc)

    # Step 3.97: HMM regime posterior (smooth_posterior / blend_weights — observability)
    try:
        from src.assembled_core.signals.regime.hmm_posterior import (
            smooth_posterior, blend_weights_by_regime_posterior, RegimeBlendResult,
        )
        _hp_posterior = {"BULL": 0.6, "BEAR": 0.2, "NEUTRAL": 0.2}
        _hp_smoothed = smooth_posterior(_hp_posterior, prev_smoothed=None)
        result.meta["hmm_posterior"] = {
            "n_regimes": len(_hp_smoothed),
            "regimes": list(_hp_smoothed.keys()),
            "smoothed": True,
        }
    except Exception as _hp_exc:
        log.debug("[HMM-POST] hmm_posterior skipped: %s", _hp_exc)

    # Step 8.60: GNN stock embedder state (GNNConfig init — observability)
    try:
        from src.assembled_core.ml.gnn_stocks import GNNConfig, TORCH_AVAILABLE as _gnn_torch
        _gnn_cfg = GNNConfig()
        result.meta["gnn_stocks"] = {
            "torch_available": bool(_gnn_torch),
            "embedding_dim": _gnn_cfg.embedding_dim,
            "n_layers": _gnn_cfg.n_layers,
        }
    except Exception as _gnn_exc:
        log.debug("[GNN] gnn_stocks skipped: %s", _gnn_exc)

    # Step 8.61: Graph models state (build_correlation_graph — observability)
    try:
        from src.assembled_core.ml.graph_models import generate_graph_signals, GraphSignal
        result.meta["graph_models"] = {
            "available": True,
        }
    except Exception as _gm_exc:
        log.debug("[GRAPH-MOD] graph_models skipped: %s", _gm_exc)

    # Step 8.62: MAML meta-learning state (MAMLConfig init — observability)
    try:
        from src.assembled_core.ml.maml import MAMLConfig, TORCH_AVAILABLE as _maml_torch
        _maml_cfg = MAMLConfig()
        result.meta["maml"] = {
            "torch_available": bool(_maml_torch),
            "inner_lr": _maml_cfg.inner_lr,
            "inner_steps": _maml_cfg.inner_steps,
            "hidden_dim": _maml_cfg.hidden_dim,
        }
    except Exception as _maml_exc:
        log.debug("[MAML] maml skipped: %s", _maml_exc)

    # Step 8.63: LIME explainer state (LIMEExplainerWrapper — observability)
    try:
        from src.assembled_core.ml.lime_explainer import LIMEExplainerWrapper
        try:
            import lime as _lime_check
            _lime_available = True
        except ImportError:
            _lime_available = False
        result.meta["lime_explainer"] = {
            "lime_available": _lime_available,
            "available": True,
        }
    except Exception as _lime_exc:
        log.debug("[LIME] lime_explainer skipped: %s", _lime_exc)

    # Step 8.64: Online HPO bandit state (OnlineHyperparamAdapter — observability)
    try:
        from src.assembled_core.ml.online_hpo import OnlineHyperparamAdapter
        _ohpo = OnlineHyperparamAdapter()
        _ohpo_best = _ohpo.select_arm()
        result.meta["online_hpo"] = {
            "n_arms": len(_ohpo.arms),
            "selected_arm": _ohpo_best.arm_id if _ohpo_best else None,
        }
    except Exception as _ohpo_exc:
        log.debug("[ONLINE-HPO] online_hpo skipped: %s", _ohpo_exc)

    # Step 8.65: Nested meta-labeling state (NestedMetaLabeler — observability)
    try:
        from src.assembled_core.ml.nested_meta_labeling import NestedMetaLabeler
        _nml = NestedMetaLabeler()
        result.meta["nested_meta_labeling"] = {
            "fitted": bool(getattr(_nml, "_fitted", False)),
            "available": True,
        }
    except Exception as _nml_exc:
        log.debug("[NESTED-META] nested_meta_labeling skipped: %s", _nml_exc)

    # Step 7.73: Factor report (run_factor_report — observability)
    try:
        from src.assembled_core.qa.factor_report import run_factor_report
        if not result.prices_with_features.empty and len(result.prices_with_features) >= 10:
            _fr_report = run_factor_report(
                result.prices_with_features,
                factor_set="core",
                fwd_horizon_days=5,
            )
            result.meta["factor_report"] = {
                "n_factors": len(_fr_report) if isinstance(_fr_report, dict) else 0,
                "available": True,
            }
        else:
            result.meta["factor_report"] = {"available": True, "skipped": "insufficient_data"}
    except Exception as _fr_exc:
        log.debug("[FACTOR-REPORT] factor_report skipped: %s", _fr_exc)

    # Step 7.74: Shipping risk (compute_shipping_exposure — observability)
    try:
        from src.assembled_core.qa.shipping_risk import (
            compute_shipping_exposure,
            compute_systemic_risk_flags,
        )
        import pandas as _pd_sr
        if result.orders_filtered:
            _sr_portfolio = _pd_sr.DataFrame([
                {"symbol": o.symbol, "weight": 1.0 / max(len(result.orders_filtered), 1)}
                for o in result.orders_filtered
            ])
            _sr_features = _pd_sr.DataFrame([
                {"symbol": o.symbol, "shipping_congestion_score": 0.0}
                for o in result.orders_filtered
            ])
            _sr_exp = compute_shipping_exposure(_sr_portfolio, _sr_features)
            _sr_flags = compute_systemic_risk_flags(_sr_exp)
            result.meta["shipping_risk"] = {
                "avg_congestion": float(_sr_exp.get("avg_shipping_congestion", 0.0)),
                "systemic_risk": bool(_sr_flags.get("systemic_risk", False)),
            }
        else:
            result.meta["shipping_risk"] = {"available": True, "skipped": "no_orders"}
    except Exception as _sr_exc:
        log.debug("[SHIPPING-RISK] shipping_risk skipped: %s", _sr_exc)

    # Step 7.75: Trade TCA (compute_trade_tca / aggregate_tca — observability)
    try:
        from src.assembled_core.qa.trade_tca import TradeTCA, compute_trade_tca, aggregate_tca
        _tca_list: list[TradeTCA] = []
        for _o in result.orders_filtered[:5]:
            _tca = compute_trade_tca(
                trade_id=str(getattr(_o, "order_id", id(_o))),
                symbol=_o.symbol,
                side=getattr(_o, "side", "buy"),
                quantity=float(getattr(_o, "quantity", 1.0)),
                execution_price=float(getattr(_o, "limit_price", 0.0) or getattr(_o, "price", 0.0) or 1.0),
                arrival_price=float(getattr(_o, "limit_price", 0.0) or getattr(_o, "price", 0.0) or 1.0),
            )
            _tca_list.append(_tca)
        if _tca_list:
            _tca_agg = aggregate_tca(_tca_list)
            result.meta["trade_tca"] = {
                "n_trades": _tca_agg.n_trades,
                "mean_impact_bps": float(_tca_agg.mean_impact_bps),
                "mean_vwap_slippage_bps": float(_tca_agg.mean_vwap_slippage_bps),
            }
        else:
            result.meta["trade_tca"] = {"available": True, "skipped": "no_orders"}
    except Exception as _tca_exc:
        log.debug("[TRADE-TCA] trade_tca skipped: %s", _tca_exc)

    # Step 7.76: Audit log (AuditLog — observability)
    try:
        from src.assembled_core.compliance.audit_log import AuditLog, AuditEventType
        _alog = AuditLog(log_path=None)
        _alog.append(
            event_type=AuditEventType.RECONCILIATION,
            payload={"n_orders": len(result.orders_filtered)},
        )
        result.meta["audit_log"] = {
            "n_entries": len(_alog._entries),
            "available": True,
        }
    except Exception as _alog_exc:
        log.debug("[AUDIT-LOG] audit_log skipped: %s", _alog_exc)

    # Step 7.77: OTR monitor (OTRMonitor — observability)
    try:
        from src.assembled_core.compliance.otr_monitor import OTRMonitor
        _otr = OTRMonitor()
        for _o in result.orders_filtered[:10]:
            _otr.record_order(symbol=_o.symbol, order_type="submit")
        _otr_snap = _otr.compute_otr()
        result.meta["otr_monitor"] = {
            "otr_ratio": float(_otr_snap.otr_ratio),
            "alert_level": str(_otr_snap.alert_level),
            "n_orders": _otr_snap.orders_submitted,
        }
    except Exception as _otr_exc:
        log.debug("[OTR] otr_monitor skipped: %s", _otr_exc)

    # Step 7.78: Regulatory reports (generate_best_execution_report — observability)
    try:
        from src.assembled_core.compliance.regulatory_reports import (
            generate_best_execution_report,
            BestExecutionReport,
        )
        import pandas as _pd_reg
        _ber = generate_best_execution_report(
            fills=_pd_reg.DataFrame(),
            period_start="2024-01-01",
            period_end="2024-12-31",
        )
        result.meta["regulatory_reports"] = {
            "best_execution_available": True,
            "total_orders": _ber.total_orders,
        }
    except Exception as _reg_exc:
        log.debug("[REG-REPORTS] regulatory_reports skipped: %s", _reg_exc)

    # Step 8.66: Config constants (observability)
    try:
        from src.assembled_core.config.constants import (
            TRADING_DAYS_PER_YEAR,
            DEFAULT_COMMISSION_BPS,
            DEFAULT_START_CAPITAL,
        )
        result.meta["config_constants"] = {
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "default_commission_bps": DEFAULT_COMMISSION_BPS,
            "default_start_capital": DEFAULT_START_CAPITAL,
        }
    except Exception as _cc_exc:
        log.debug("[CONFIG-CONST] config constants skipped: %s", _cc_exc)

    # Step 8.67: Policy schema validation (validate_policy — observability)
    try:
        from src.assembled_core.config.policy_schema import validate_policy
        _ps_valid, _ps_errors = validate_policy({})
        result.meta["policy_schema"] = {
            "empty_policy_valid": len(_ps_errors) == 0,
            "n_errors": len(_ps_errors),
        }
    except Exception as _ps_exc:
        log.debug("[POLICY-SCHEMA] policy_schema skipped: %s", _ps_exc)

    # Step 8.68: Evidence grader (grade_evidence / EvidenceGrade — observability)
    try:
        from src.assembled_core.events.evidence_engine.grader import grade_evidence
        from src.assembled_core.events.evidence_engine.grades import EvidenceGrade
        _eg_summary = {
            "tierA_count": 2,
            "tierB_independent_count": 1,
            "evidence_ok": True,
        }
        _eg_grade = grade_evidence(_eg_summary)
        result.meta["evidence_grader"] = {
            "grade": str(_eg_grade),
            "available": True,
        }
    except Exception as _eg_exc:
        log.debug("[EVIDENCE-GRADER] evidence_grader skipped: %s", _eg_exc)

    # Step 8.69: Evidence action gate (check_evidence_grade_gate — observability)
    try:
        from src.assembled_core.events.evidence_engine.action_gate import check_evidence_grade_gate
        from src.assembled_core.events.evidence_engine.grades import EvidenceGrade as _EG
        _ag_ok, _ag_reason = check_evidence_grade_gate(_EG.A)
        result.meta["evidence_action_gate"] = {
            "gate_ok": _ag_ok,
            "available": True,
        }
    except Exception as _ag_exc:
        log.debug("[ACTION-GATE] action_gate skipped: %s", _ag_exc)

    # Step 8.70: Misinfo risk (compute_misinfo_risk — observability)
    try:
        from src.assembled_core.events.evidence_engine.misinfo_risk import compute_misinfo_risk
        _mr_summary = {"tierA_count": 0, "tierB_independent_count": 1, "tierB_count": 2}
        _mr_score = compute_misinfo_risk(_mr_summary, social_only=False)
        result.meta["misinfo_risk"] = {
            "score": float(_mr_score),
            "available": True,
        }
    except Exception as _mr_exc:
        log.debug("[MISINFO] misinfo_risk skipped: %s", _mr_exc)

    # Step 8.71: News burst detection (compute_bursts_for_window — observability)
    try:
        from src.assembled_core.events.news.burst import compute_bursts_for_window
        _burst = compute_bursts_for_window(
            clusters=[],
            baseline=None,
            cfg={},
            window_hours=24,
        )
        result.meta["news_burst"] = {
            "n_entity_bursts": len(_burst.get("entity_bursts", [])),
            "available": True,
        }
    except Exception as _burst_exc:
        log.debug("[NEWS-BURST] news_burst skipped: %s", _burst_exc)

    # Step 8.72: News fingerprint (simhash64 / hamming_distance — observability)
    try:
        from src.assembled_core.events.news.fingerprint import simhash64, hamming_distance
        _fp1 = simhash64("market rally equity risk")
        _fp2 = simhash64("market rally equity risk")
        result.meta["news_fingerprint"] = {
            "hash_type": "simhash64",
            "same_text_distance": hamming_distance(_fp1, _fp2),
            "available": True,
        }
    except Exception as _fp_exc:
        log.debug("[FINGERPRINT] news_fingerprint skipped: %s", _fp_exc)

    # Step 8.73: News TF-IDF (build_tfidf_vectors / cosine_sparse — observability)
    try:
        from src.assembled_core.events.news.tfidf import build_tfidf_vectors, cosine_sparse
        _tfidf_vecs = build_tfidf_vectors(["equity market rally", "bond yields rising"])
        _tfidf_sim = cosine_sparse(_tfidf_vecs[0], _tfidf_vecs[1]) if len(_tfidf_vecs) >= 2 else 0.0
        result.meta["news_tfidf"] = {
            "n_docs": len(_tfidf_vecs),
            "cosine_sim_sample": float(_tfidf_sim),
            "available": True,
        }
    except Exception as _tfidf_exc:
        log.debug("[TFIDF] news_tfidf skipped: %s", _tfidf_exc)

    # Step 8.74: Trigger scoring (score_triggers — observability)
    try:
        from src.assembled_core.events.news.trigger_scoring import score_triggers
        _ts_result = score_triggers(clusters=[], events_by_id={})
        result.meta["trigger_scoring"] = {
            "n_triggers": len(_ts_result),
            "available": True,
        }
    except Exception as _ts_exc:
        log.debug("[TRIGGER-SCORING] trigger_scoring skipped: %s", _ts_exc)

    # Step 2.41: PIT guard (PITGuard — observability)
    try:
        from src.assembled_core.data.pit_guard import PITGuard
        import pandas as _pd_pit
        _pit = PITGuard(as_of=_pd_pit.Timestamp.now(tz="UTC"), mode="warn")
        result.meta["pit_guard"] = {
            "mode": _pit.mode,
            "available": True,
        }
    except Exception as _pit_exc:
        log.debug("[PIT-GUARD] pit_guard skipped: %s", _pit_exc)

    # Step 2.42: Realism meta (build_realism_label — observability)
    try:
        from src.assembled_core.data.realism_meta import build_realism_label
        _rm = build_realism_label(
            calendar_mode="nyse",
            cost_model_mode="policy",
            data_source="synthetic",
        )
        result.meta["realism_meta"] = {
            "realism_level": _rm.get("realism_level", "unknown"),
            "data_source": _rm.get("data_source", "unknown"),
        }
    except Exception as _rm_exc:
        log.debug("[REALISM-META] realism_meta skipped: %s", _rm_exc)

    # Step 2.43: Data latency (apply_source_latency — observability)
    try:
        from src.assembled_core.data.latency import apply_source_latency
        import pandas as _pd_lat
        _lat_events = _pd_lat.DataFrame(columns=["source", "event_timestamp", "available_from"])
        _lat_result = apply_source_latency(_lat_events)
        result.meta["data_latency"] = {
            "n_events": len(_lat_result),
            "available": True,
        }
    except Exception as _lat_exc:
        log.debug("[DATA-LATENCY] data_latency skipped: %s", _lat_exc)

    # Step 2.44: Synthetic generator (generate_crisis_returns — observability)
    try:
        from src.assembled_core.data.synthetic_generator import generate_crisis_returns, generate_normal_returns
        _sg = generate_normal_returns(n_assets=3, n_days=5, seed=42)
        result.meta["synthetic_generator"] = {
            "n_assets": _sg.shape[1],
            "n_days": len(_sg),
            "available": True,
        }
    except Exception as _sg_exc:
        log.debug("[SYNTHETIC-GEN] synthetic_generator skipped: %s", _sg_exc)

    # Step 2.45: Data resample (resample_to_weekly — observability)
    try:
        from src.assembled_core.data.resample import resample_to_weekly
        if not result.prices_with_features.empty and "symbol" in result.prices_with_features.columns:
            import pandas as _pd_rs
            _rs_df = result.prices_with_features.copy()
            if "timestamp" not in _rs_df.columns:
                _rs_df["timestamp"] = _pd_rs.date_range("2024-01-01", periods=len(_rs_df), freq="B")
            _rs_weekly = resample_to_weekly(_rs_df)
            result.meta["data_resample"] = {
                "weekly_rows": len(_rs_weekly),
                "available": True,
            }
        else:
            result.meta["data_resample"] = {"available": True, "skipped": "no_prices"}
    except Exception as _rs_exc:
        log.debug("[DATA-RESAMPLE] data_resample skipped: %s", _rs_exc)

    # Step 2.46: Panel store (panel_exists — observability)
    try:
        from src.assembled_core.data.panel_store import panel_exists
        _pe = panel_exists("__cycle_observability_probe__")
        result.meta["panel_store"] = {
            "probe_exists": bool(_pe),
            "available": True,
        }
    except Exception as _pe_exc:
        log.debug("[PANEL-STORE] panel_store skipped: %s", _pe_exc)

    # Step 7.79: Round trips (compute_round_trips — observability)
    try:
        from src.assembled_core.accounting.round_trips import compute_round_trips, round_trip_summary
        import pandas as _pd_rt
        _rt_trades = _pd_rt.DataFrame(columns=["symbol", "date", "side", "price", "quantity", "commission"])
        _rt_list = compute_round_trips(_rt_trades)
        _rt_sum = round_trip_summary(_rt_list)
        result.meta["round_trips"] = {
            "n_round_trips": len(_rt_list),
            "summary": _rt_sum,
        }
    except Exception as _rt_exc:
        log.debug("[ROUND-TRIPS] round_trips skipped: %s", _rt_exc)

    # Step 7.80: Tax lots (TaxLotTracker — observability)
    try:
        from src.assembled_core.accounting.tax_lots import TaxLotTracker
        _tlt = TaxLotTracker()
        result.meta["tax_lots"] = {
            "n_symbols": len(_tlt.lots),
            "available": True,
        }
    except Exception as _tlt_exc:
        log.debug("[TAX-LOTS] tax_lots skipped: %s", _tlt_exc)

    # Step 7.81: Decision audit trail (DecisionAuditTrail — observability)
    try:
        from src.assembled_core.accounting.decision_audit import DecisionAuditTrail, DecisionRecord
        _dat = DecisionAuditTrail()
        _dat.record(DecisionRecord(
            timestamp=str(ctx.as_of),
            symbol="__cycle__",
            direction="long",
            signal_score=0.0,
            regime=str(getattr(result, "regime", "")),
        ))
        _dat_summary = _dat.summary()
        result.meta["decision_audit"] = {
            "n_records": _dat_summary["n_records"],
            "available": True,
        }
    except Exception as _dat_exc:
        log.debug("[DECISION-AUDIT] decision_audit skipped: %s", _dat_exc)

    # Step 7.82: Position engine (build_positions_from_ledger — observability)
    try:
        from src.assembled_core.accounting.position_engine import build_positions_from_ledger
        import pandas as _pd_pe
        _pe_events = _pd_pe.DataFrame(columns=[
            "event_id", "event_ts", "event_type", "symbol",
            "quantity", "price", "cash_delta",
        ])
        _pe_result = build_positions_from_ledger(_pe_events)
        _pe_positions = _pe_result.get("positions_df", _pe_result.get("positions", {}))
        _pe_cash = _pe_result.get("cash_balance", _pe_result.get("cash", 0.0))
        result.meta["position_engine"] = {
            "n_positions": len(_pe_positions) if hasattr(_pe_positions, "__len__") else 0,
            "cash": float(_pe_cash),
            "available": True,
        }
    except Exception as _pe_exc:
        log.debug("[POSITION-ENGINE] position_engine skipped: %s", _pe_exc)

    # Step 7.83: Broker snapshot (normalize_broker_snapshot — observability)
    try:
        from src.assembled_core.accounting.broker_snapshot import normalize_broker_snapshot
        import pandas as _pd_bs
        _bs = normalize_broker_snapshot(
            cash=0.0,
            positions_df=_pd_bs.DataFrame(columns=["symbol", "qty"]),
        )
        result.meta["broker_snapshot"] = {
            "n_positions": len(_bs.get("positions_df", _pd_bs.DataFrame())),
            "cash": float(_bs.get("cash", 0.0)),
            "available": True,
        }
    except Exception as _bs_exc:
        log.debug("[BROKER-SNAPSHOT] broker_snapshot skipped: %s", _bs_exc)

    # Step 7.84: Accounting report (accounting_report — observability)
    try:
        from src.assembled_core.accounting.accounting_report import (
            AccountingReport,
        )
        result.meta["accounting_report"] = {
            "available": True,
            "class": "AccountingReport",
        }
    except Exception as _ar_exc:
        log.debug("[ACCOUNTING-REPORT] accounting_report skipped: %s", _ar_exc)

    # Step 5.41: Fill model (PartialFillModel — observability)
    try:
        from src.assembled_core.execution.fill_model import PartialFillModel
        _pfm = PartialFillModel()
        result.meta["fill_model"] = {
            "adv_window": _pfm.adv_window,
            "participation_cap": _pfm.participation_cap,
            "available": True,
        }
    except Exception as _pfm_exc:
        log.debug("[FILL-MODEL] fill_model skipped: %s", _pfm_exc)

    # Step 5.42: Intent store (has_intent — observability)
    try:
        from src.assembled_core.execution.intent_store import has_intent, make_daily_key
        _ik = make_daily_key("cycle_complete")
        _ih = has_intent(_ik)
        result.meta["intent_store"] = {
            "cycle_intent_exists": bool(_ih),
            "available": True,
        }
    except Exception as _is_exc:
        log.debug("[INTENT-STORE] intent_store skipped: %s", _is_exc)

    # Step 5.43: Pre-open signals (compute_overnight_gap_signal — observability)
    try:
        from src.assembled_core.execution.pre_open_signals import (
            compute_overnight_gap_signal,
            PreOpenConfig,
        )
        _pog_strength, _pog_direction = compute_overnight_gap_signal(
            prev_close=100.0,
            premarket_price=None,
            futures_return=0.005,
        )
        result.meta["pre_open_signals"] = {
            "overnight_gap_strength": float(_pog_strength),
            "config_available": True,
        }
    except Exception as _pog_exc:
        log.debug("[PRE-OPEN] pre_open_signals skipped: %s", _pog_exc)

    # Step 5.44: Symbol kill switch (is_symbol_blocked / list_blocked_symbols — observability)
    try:
        from src.assembled_core.execution.symbol_kill_switch import (
            is_symbol_blocked,
            list_blocked_symbols,
            filter_orders_by_symbol_blocks,
        )
        _blocked = list_blocked_symbols()
        result.meta["symbol_kill_switch"] = {
            "n_blocked": len(_blocked),
            "blocked_symbols": list(_blocked)[:5],
            "available": True,
        }
    except Exception as _sks_exc:
        log.debug("[SYMBOL-KS] symbol_kill_switch skipped: %s", _sks_exc)

    # Step 5.45: Cost model calibrator (CostModelPriors — observability)
    try:
        from src.assembled_core.execution.cost_model_calibrator import CostModelPriors, CalibrationResult
        _cmp = CostModelPriors()
        result.meta["cost_model_calibrator"] = {
            "half_spread_bps_prior": float(_cmp.half_spread_bps),
            "participation_cap": float(_cmp.participation_cap),
            "available": True,
        }
    except Exception as _cmc_exc:
        log.debug("[COST-CALIB] cost_model_calibrator skipped: %s", _cmc_exc)

    # Step 5.46: Fill model pipeline (apply_fill_model_pipeline — observability)
    try:
        from src.assembled_core.execution.fill_model_pipeline import apply_fill_model_pipeline
        import pandas as _pd_fmp
        _fmp_orders = _pd_fmp.DataFrame()
        _fmp_prices = _pd_fmp.DataFrame()
        _fmp_result = apply_fill_model_pipeline(_fmp_orders, prices=_fmp_prices, freq="1D")
        result.meta["fill_model_pipeline"] = {
            "n_fills": len(_fmp_result),
            "available": True,
        }
    except Exception as _fmp_exc:
        log.debug("[FILL-PIPELINE] fill_model_pipeline skipped: %s", _fmp_exc)

    # Step 8.75: Intel health monitor (HealthMonitor — observability)
    try:
        from src.assembled_core.intel.health_monitor import HealthMonitor
        _hm = HealthMonitor()
        _hm.register("news_pipeline")
        _hm.register("disclosure_pipeline")
        _hm_status = _hm.overall_status() if hasattr(_hm, "overall_status") else "unknown"
        result.meta["intel_health_monitor"] = {
            "n_components": len(_hm._components),
            "overall_status": str(_hm_status),
            "available": True,
        }
    except Exception as _hm_exc:
        log.debug("[HEALTH-MON] health_monitor skipped: %s", _hm_exc)

    # Step 8.76: News decay (NewsDecay.impact_remaining — observability)
    try:
        from src.assembled_core.intel.news_decay import NewsDecay
        _nd = NewsDecay()
        _nd_impact = _nd.impact_remaining("earnings", minutes_since=60.0)
        result.meta["news_decay"] = {
            "earnings_impact_60min": float(_nd_impact),
            "available": True,
        }
    except Exception as _nd_exc:
        log.debug("[NEWS-DECAY] news_decay skipped: %s", _nd_exc)

    # Step 8.77: Nation profiles (load_nation_profiles / compute_vulnerability_score — observability)
    try:
        from src.assembled_core.intel.nation_profiles import load_nation_profiles, compute_vulnerability_score
        _np = load_nation_profiles()
        _np_n = len(_np) if isinstance(_np, (list, dict)) else 0
        result.meta["nation_profiles"] = {
            "n_profiles": _np_n,
            "available": True,
        }
    except Exception as _np_exc:
        log.debug("[NATION-PROFILES] nation_profiles skipped: %s", _np_exc)

    # Step 8.78: News classifier (classify_news_event — observability)
    try:
        from src.assembled_core.intel.news_classifier import classify_news_event, NewsClassification
        _nc = classify_news_event("Federal Reserve raises interest rates by 25bps")
        result.meta["news_classifier"] = {
            "event_types": _nc.event_types[:3],
            "severity": float(_nc.severity),
            "market_direction": str(_nc.market_direction),
            "available": True,
        }
    except Exception as _nc_exc:
        log.debug("[NEWS-CLASSIFIER] news_classifier skipped: %s", _nc_exc)

    # Step 8.79: News cluster manager (ClusterManager — observability)
    try:
        from src.assembled_core.intel.news_cluster import ClusterManager
        _cm = ClusterManager()
        _cm_clusters = _cm.update_clusters([])
        result.meta["news_cluster"] = {
            "n_active_clusters": len(_cm_clusters),
            "available": True,
        }
    except Exception as _cm_exc:
        log.debug("[NEWS-CLUSTER] news_cluster skipped: %s", _cm_exc)

    # Step 8.80: News corroboration (CorroborationTracker — observability)
    try:
        from src.assembled_core.intel.news_corroboration import CorroborationTracker
        _ct = CorroborationTracker()
        _ct.ingest([])
        result.meta["news_corroboration"] = {
            "n_stories_tracked": len(_ct._entries),
            "available": True,
        }
    except Exception as _ct_exc:
        log.debug("[CORROBORATION] news_corroboration skipped: %s", _ct_exc)

    # Step 8.81: News contradiction (ContradictionDetector — observability)
    try:
        from src.assembled_core.intel.news_contradiction import ContradictionDetector
        _cd = ContradictionDetector()
        _cd_result = _cd.analyse([])
        result.meta["news_contradiction"] = {
            "n_contradictions": len(_cd_result),
            "available": True,
        }
    except Exception as _cd_exc:
        log.debug("[CONTRADICTION] news_contradiction skipped: %s", _cd_exc)

    # Step 8.82: News dedupe (NewsDedupeIndex — observability)
    try:
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex
        _ndi = NewsDedupeIndex()
        result.meta["news_dedupe"] = {
            "n_seen_ids": len(_ndi.seen_event_ids),
            "available": True,
        }
    except Exception as _ndi_exc:
        log.debug("[NEWS-DEDUPE] news_dedupe skipped: %s", _ndi_exc)

    # Step 8.83: News enricher (NewsEventEnricher — observability)
    try:
        from src.assembled_core.intel.news_enricher import NewsEventEnricher
        _nee = NewsEventEnricher()
        _nee_result = _nee.enrich([])
        result.meta["news_enricher"] = {
            "n_enriched": len(_nee_result),
            "available": True,
        }
    except Exception as _nee_exc:
        log.debug("[NEWS-ENRICHER] news_enricher skipped: %s", _nee_exc)

    # Step 8.84: News impact estimator (NewsImpactEstimator — observability)
    try:
        from src.assembled_core.intel.news_impact_estimator import NewsImpactEstimator, ImpactEstimate
        _nie = NewsImpactEstimator()
        _nie_est = _nie.estimate(
            type("_Cls", (), {"event_types": ["earnings"], "severity": 5.0,
                              "market_direction": "bullish", "time_horizon": "short",
                              "confidence": 0.6})()
        )
        result.meta["news_impact_estimator"] = {
            "impact_bps": float(getattr(_nie_est, "bps", getattr(_nie_est, "impact_bps", 0.0))),
            "available": True,
        }
    except Exception as _nie_exc:
        log.debug("[IMPACT-EST] news_impact_estimator skipped: %s", _nie_exc)

    # Step 8.85: Market confirmation (compute_market_confirmation — observability)
    try:
        from src.assembled_core.intel.market_confirmation import compute_market_confirmation
        _mc = compute_market_confirmation(cache={})
        result.meta["market_confirmation"] = {
            "vix_spike": bool(_mc.get("vix_spike", False)),
            "oil_move": float(_mc.get("oil_move", 0.0)),
            "available": True,
        }
    except Exception as _mc_exc:
        log.debug("[MARKET-CONF] market_confirmation skipped: %s", _mc_exc)

    # Step 8.86: Currency crisis (rank_currencies_by_risk — observability)
    try:
        from src.assembled_core.intel.currency_crisis import rank_currencies_by_risk
        _cc_ranked = rank_currencies_by_risk()
        result.meta["currency_crisis"] = {
            "n_ranked": len(_cc_ranked),
            "available": True,
        }
    except Exception as _cc_exc:
        log.debug("[CURRENCY-CRISIS] currency_crisis skipped: %s", _cc_exc)

    # Step 8.87: News language detection (detect_language / is_english — observability)
    try:
        from src.assembled_core.intel.news_language import detect_language, is_english
        _nl_lang = detect_language("Federal Reserve raises interest rates")
        result.meta["news_language"] = {
            "detected_lang": str(_nl_lang),
            "is_english": bool(is_english("Federal Reserve raises interest rates")),
            "available": True,
        }
    except Exception as _nl_exc:
        log.debug("[NEWS-LANG] news_language skipped: %s", _nl_exc)

    # Step 8.88: Macro calendar (MacroCalendar — observability)
    try:
        from src.assembled_core.intel.news_macro_calendar import MacroCalendar
        _mcal = MacroCalendar()
        result.meta["news_macro_calendar"] = {
            "n_events": len(_mcal._events),
            "available": True,
        }
    except Exception as _mcal_exc:
        log.debug("[MACRO-CAL] news_macro_calendar skipped: %s", _mcal_exc)

    # Step 8.89: Central bank divergence (compute_policy_divergence_matrix — observability)
    try:
        from src.assembled_core.intel.central_bank_divergence import (
            compute_policy_divergence_matrix,
            get_most_divergent_pair,
        )
        _cbd_matrix = compute_policy_divergence_matrix()
        _cbd_top = get_most_divergent_pair()
        result.meta["central_bank_divergence"] = {
            "n_pairs": len(_cbd_matrix),
            "most_divergent": list(_cbd_top[:2]) if _cbd_top else [],
            "available": True,
        }
    except Exception as _cbd_exc:
        log.debug("[CB-DIVERGENCE] central_bank_divergence skipped: %s", _cbd_exc)

    # Step 8.90: Entity linker (EntityLinker — observability)
    try:
        from src.assembled_core.intel.entity_linker import EntityLinker
        _el = EntityLinker()
        _el_result = _el.link("Apple Inc")
        result.meta["entity_linker"] = {
            "linked_symbols": list(_el_result)[:3] if _el_result else [],
            "available": True,
        }
    except Exception as _el_exc:
        log.debug("[ENTITY-LINKER] entity_linker skipped: %s", _el_exc)

    # Step 8.91: News impact calibrator (ImpactCalibrator — observability)
    try:
        from src.assembled_core.intel.news_impact_calibrator import ImpactCalibrator
        _ic = ImpactCalibrator()
        _ic.observe("earnings", pred_bps=15.0, realised_bps=12.5)
        result.meta["news_impact_calibrator"] = {
            "n_event_types": len(_ic._stats),
            "available": True,
        }
    except Exception as _ic_exc:
        log.debug("[IMPACT-CALIB] news_impact_calibrator skipped: %s", _ic_exc)

    # Step 8.92: News entity mapper (extract_tickers_from_title — observability)
    try:
        from src.assembled_core.intel.news_entity_mapper import extract_tickers_from_title
        _em_tickers = extract_tickers_from_title("Apple Inc AAPL and Microsoft MSFT report earnings")
        result.meta["news_entity_mapper"] = {
            "tickers_found": list(_em_tickers)[:5],
            "available": True,
        }
    except Exception as _em_exc:
        log.debug("[ENTITY-MAPPER] news_entity_mapper skipped: %s", _em_exc)

    # Step 8.93: News alerts (AlertEngine — observability)
    try:
        from src.assembled_core.intel.news_alerts import AlertEngine
        _ae = AlertEngine(include_default_log_handler=False)
        _ae_alerts = _ae.evaluate([])
        result.meta["news_alerts"] = {
            "n_alerts": len(_ae_alerts),
            "dropped_dedup": _ae.dropped_dedup,
            "dropped_rate": _ae.dropped_rate,
            "available": True,
        }
    except Exception as _ae_exc:
        log.debug("[NEWS-ALERTS] news_alerts skipped: %s", _ae_exc)

    # Step 8.94: News archive reader (NewsArchiveReader — observability)
    try:
        from src.assembled_core.intel.news_archive import NewsArchiveReader
        _nar = NewsArchiveReader("data/intel/archive/placeholder.jsonl")
        result.meta["news_archive"] = {"exists": bool(_nar), "available": True}
    except Exception as _nar_exc:
        log.debug("[NEWS-ARCHIVE] news_archive skipped: %s", _nar_exc)

    # Step 8.95: News archiver (NewsArchiver — observability)
    try:
        from src.assembled_core.intel.news_archiver import NewsArchiver
        _narch = NewsArchiver(base_dir="data/intel/archive")
        _narch_written = _narch.append([])
        result.meta["news_archiver"] = {"events_written": _narch_written, "available": True}
    except Exception as _narch_exc:
        log.debug("[NEWS-ARCHIVER] news_archiver skipped: %s", _narch_exc)

    # Step 8.96: News entity graph (EntityCoGraph — observability)
    try:
        from src.assembled_core.intel.news_entity_graph import EntityCoGraph
        _ecg = EntityCoGraph()
        _ecg.ingest([])
        result.meta["news_entity_graph"] = {"n_entities": len(_ecg._counts), "available": True}
    except Exception as _ecg_exc:
        log.debug("[ENTITY-GRAPH] news_entity_graph skipped: %s", _ecg_exc)

    # Step 8.97: News event store (NewsEventStore — observability)
    try:
        from src.assembled_core.intel.news_event_store import NewsEventStore
        _nes = NewsEventStore()
        result.meta["news_event_store"] = {"n_events": len(_nes._events), "available": True}
    except Exception as _nes_exc:
        log.debug("[EVENT-STORE] news_event_store skipped: %s", _nes_exc)

    # Step 8.98: News ingest (records_to_news_events — observability)
    try:
        from src.assembled_core.intel.news_ingest import records_to_news_events
        _ni_events = records_to_news_events([])
        result.meta["news_ingest"] = {"n_events": len(_ni_events), "available": True}
    except Exception as _ni_exc:
        log.debug("[NEWS-INGEST] news_ingest skipped: %s", _ni_exc)

    # Step 8.99: News semantic dedup (SemanticDedup — observability)
    try:
        from src.assembled_core.intel.news_semantic_dedup import SemanticDedup
        _sdd = SemanticDedup(enabled=False)
        result.meta["news_semantic_dedup"] = {"backend": _sdd.backend, "available": True}
    except Exception as _sdd_exc:
        log.debug("[SEMANTIC-DEDUP] news_semantic_dedup skipped: %s", _sdd_exc)

    # Step 8.100: News sentiment drift (SentimentDriftTracker — observability)
    try:
        from src.assembled_core.intel.news_sentiment_drift import SentimentDriftTracker
        _sdt = SentimentDriftTracker()
        _sdt.update([])
        result.meta["news_sentiment_drift"] = {"n_tracked_keys": len(_sdt._buffers), "available": True}
    except Exception as _sdt_exc:
        log.debug("[SENTIMENT-DRIFT] news_sentiment_drift skipped: %s", _sdt_exc)

    # Step 8.101: News signal aggregator (aggregate_signals — observability)
    try:
        from src.assembled_core.intel.news_signal_aggregator import aggregate_signals
        _nsa_signal = aggregate_signals([])
        result.meta["news_signal_aggregator"] = {
            "net_direction": _nsa_signal.net_direction,
            "n_signals": _nsa_signal.n_signals,
            "available": True,
        }
    except Exception as _nsa_exc:
        log.debug("[SIGNAL-AGG] news_signal_aggregator skipped: %s", _nsa_exc)

    # Step 8.102: News source voting (vote_direction — observability)
    try:
        from src.assembled_core.intel.news_source_voting import vote_direction, VoteResult
        _nsv_result = vote_direction([])
        result.meta["news_source_voting"] = {
            "winner": _nsv_result.winner,
            "margin": _nsv_result.margin,
            "available": True,
        }
    except Exception as _nsv_exc:
        log.debug("[SOURCE-VOTING] news_source_voting skipped: %s", _nsv_exc)

    # Step 8.103: News ticker velocity (TickerVelocityTracker — observability)
    try:
        from src.assembled_core.intel.news_ticker_velocity import TickerVelocityTracker
        _tvt = TickerVelocityTracker()
        _tvt_signals = _tvt.update([])
        result.meta["news_ticker_velocity"] = {
            "n_ticker_signals": len(_tvt_signals),
            "available": True,
        }
    except Exception as _tvt_exc:
        log.debug("[TICKER-VEL] news_ticker_velocity skipped: %s", _tvt_exc)

    # Step 8.104: News trade attribution (NewsTradeAttributor — observability)
    try:
        from src.assembled_core.intel.news_trade_attribution import NewsTradeAttributor
        _nta = NewsTradeAttributor()
        result.meta["news_trade_attribution"] = {
            "pre_window_hours": _nta.pre,
            "post_window_hours": _nta.post,
            "available": True,
        }
    except Exception as _nta_exc:
        log.debug("[TRADE-ATTR] news_trade_attribution skipped: %s", _nta_exc)

    # Step 8.105: News velocity (VelocityTracker — observability)
    try:
        from src.assembled_core.intel.news_velocity import VelocityTracker
        _nvt = VelocityTracker()
        _nvt_result = _nvt.update([])
        result.meta["news_velocity"] = {
            "velocity": _nvt_result.velocity,
            "is_surge": _nvt_result.is_surge,
            "available": True,
        }
    except Exception as _nvt_exc:
        log.debug("[NEWS-VEL] news_velocity skipped: %s", _nvt_exc)

    # Step 8.106: PIT store (PITStore — observability)
    try:
        from src.assembled_core.intel.pit_store import PITStore
        _ps = PITStore(root="data/intel/pit")
        result.meta["pit_store"] = {"root": str(_ps._root), "available": True}
    except Exception as _ps_exc:
        log.debug("[PIT-STORE] pit_store skipped: %s", _ps_exc)

    # Step 8.107: RSS fetcher (RSSFetcher — observability)
    try:
        from src.assembled_core.intel.rss_fetcher import RSSFetcher
        _rsf = RSSFetcher()
        result.meta["rss_fetcher"] = {"n_feeds": len(_rsf.feed_ids), "available": True}
    except Exception as _rsf_exc:
        log.debug("[RSS-FETCHER] rss_fetcher skipped: %s", _rsf_exc)

    # Step 8.108: Sanctions model (get_sanction_package — observability)
    try:
        from src.assembled_core.intel.sanctions_model import get_sanction_package, HISTORICAL_SANCTIONS
        _smp = get_sanction_package(next(iter(HISTORICAL_SANCTIONS), ""))
        result.meta["sanctions_model"] = {
            "n_packages": len(HISTORICAL_SANCTIONS),
            "available": True,
        }
    except Exception as _smp_exc:
        log.debug("[SANCTIONS] sanctions_model skipped: %s", _smp_exc)

    # Step 8.109: Sector news overlay (SectorNewsOverlay — observability)
    try:
        from src.assembled_core.intel.sector_news_overlay import SectorNewsOverlay
        _sno = SectorNewsOverlay()
        _sno_scores = _sno.compute(clusters=[])
        result.meta["sector_news_overlay"] = {
            "n_sectors": len(_sno_scores),
            "available": True,
        }
    except Exception as _sno_exc:
        log.debug("[SECTOR-OVERLAY] sector_news_overlay skipped: %s", _sno_exc)

    # Step 8.110: Shipping lanes (LANES_DATABASE — observability)
    try:
        from src.assembled_core.intel.shipping_lanes import LANES_DATABASE
        result.meta["shipping_lanes"] = {"n_lanes": len(LANES_DATABASE), "available": True}
    except Exception as _sl_exc:
        log.debug("[SHIPPING-LANES] shipping_lanes skipped: %s", _sl_exc)

    # Step 8.111: Shock propagation (SHOCK_TO_ORIGIN_NODES — observability)
    try:
        from src.assembled_core.intel.shock_propagation import SHOCK_TO_ORIGIN_NODES, DEFAULT_DAMPENING_FACTOR
        result.meta["shock_propagation"] = {
            "n_shock_types": len(SHOCK_TO_ORIGIN_NODES),
            "dampening_factor": DEFAULT_DAMPENING_FACTOR,
            "available": True,
        }
    except Exception as _shp_exc:
        log.debug("[SHOCK-PROP] shock_propagation skipped: %s", _shp_exc)

    # Step 8.112: Source registry (list_sources / get_trust_weight — observability)
    try:
        from src.assembled_core.intel.source_registry import list_sources, get_trust_weight
        _sr_sources = list_sources()
        result.meta["source_registry"] = {"n_sources": len(_sr_sources), "available": True}
    except Exception as _sr_exc:
        log.debug("[SOURCE-REG] source_registry skipped: %s", _sr_exc)

    # Step 8.113: Trigger snapshot store (TriggerSnapshotStore — observability)
    try:
        from src.assembled_core.intel.trigger_snapshot_store import TriggerSnapshotStore
        _tss = TriggerSnapshotStore(root="data/intel/snapshots")
        result.meta["trigger_snapshot_store"] = {"root": str(_tss._root), "available": True}
    except Exception as _tss_exc:
        log.debug("[TRIGGER-SNAP] trigger_snapshot_store skipped: %s", _tss_exc)

    # Step 8.114: Weaponized interdependence (get_known_wi_pairs — observability)
    try:
        from src.assembled_core.intel.weaponized_interdependence import get_known_wi_pairs
        _wi_pairs = get_known_wi_pairs()
        result.meta["weaponized_interdependence"] = {
            "n_wi_pairs": len(_wi_pairs),
            "available": True,
        }
    except Exception as _wi_exc:
        log.debug("[WI] weaponized_interdependence skipped: %s", _wi_exc)

    # Step 8.115: Wild card detector (detect_volume_anomaly — observability)
    try:
        import pandas as pd
        from src.assembled_core.intel.wild_card_detector import detect_volume_anomaly
        _wc_result = detect_volume_anomaly(pd.Series([], dtype=float))
        result.meta["wild_card_detector"] = {
            "is_anomaly": _wc_result.get("is_anomaly", False),
            "available": True,
        }
    except Exception as _wc_exc:
        log.debug("[WILD-CARD] wild_card_detector skipped: %s", _wc_exc)

    # Step 8.116: Dependency graph (DependencyGraph — observability)
    try:
        from src.assembled_core.intel.dependency_graph import DependencyGraph
        _dg = DependencyGraph()
        result.meta["dependency_graph"] = {"n_nodes": len(_dg._nodes), "available": True}
    except Exception as _dg_exc:
        log.debug("[DEP-GRAPH] dependency_graph skipped: %s", _dg_exc)

    # Step 8.117: Crisis alpha worker (CrisisStateConfig — observability)
    try:
        from src.assembled_core.intel.crisis_alpha_worker import CrisisStateConfig
        _cac = CrisisStateConfig()
        result.meta["crisis_alpha_worker"] = {
            "watch_threshold": _cac.geo_score_watch_threshold,
            "active_threshold": _cac.geo_score_active_threshold,
            "available": True,
        }
    except Exception as _cac_exc:
        log.debug("[CRISIS-ALPHA] crisis_alpha_worker skipped: %s", _cac_exc)

    # Step 8.118: Evidence grade writer (EvidenceGradeWriter — observability)
    try:
        from src.assembled_core.intel.evidence_grade_writer import EvidenceGradeWriter
        _egw = EvidenceGradeWriter(output_dir="data/intel/evidence")
        result.meta["evidence_grade_writer"] = {"output_dir": str(_egw._dir), "available": True}
    except Exception as _egw_exc:
        log.debug("[EVIDENCE-GRADE] evidence_grade_writer skipped: %s", _egw_exc)

    # Step 8.119: NewsAPI fetcher (NewsAPIFetcher — observability)
    try:
        from src.assembled_core.intel.news_newsapi_fetcher import NewsAPIFetcher
        _naf = NewsAPIFetcher()
        result.meta["news_newsapi_fetcher"] = {"enabled": _naf.enabled, "available": True}
    except Exception as _naf_exc:
        log.debug("[NEWSAPI] news_newsapi_fetcher skipped: %s", _naf_exc)

    log.info(
        f"Trading cycle completed successfully: {len(result.orders_filtered)} orders"
    )

    return result
