"""trading_cycle_shared — shared types and helper functions.

TradingContext, TradingCycleResult, and the 12 helper functions used by
both trading_cycle.py (legacy) and trading_cycle_v2.py (active).

This file was extracted from trading_cycle.py so that trading_cycle_v2.py
can be self-contained without importing from the legacy monolith.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

# Import existing modules (no duplication)
from src.assembled_core.config import get_base_dir  # re-export for test monkeypatching  # noqa: F401
from src.assembled_core.config.models import (
    FeatureConfig,
    ensure_feature_config,
)
from src.assembled_core.config.policy_loader import load_policy

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

    # EDCL — Event-Driven Conviction Layer (Phase A-H)
    # edcl_state: populated by _load_intel() from active_triggers + geo_confidence.
    # Consumed by _sp_compute_final_multiplier via compute_edcl_conviction_multiplier().
    edcl_state: dict[str, Any] | None = None
    # raw_news_events: optional list[NewsEvent] supplied by intel pipeline for full
    # keyword-based basket scoring. Falls back to active_triggers when None.
    raw_news_events: list | None = None
    # options_iv_skew_z: Options IV skew Z-score for Phase H triple-confirmation.
    # Populated by options_iv pipeline when available; 0.0 means no IV data.
    options_iv_skew_z: float = 0.0

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

    Bar-Convention (D5):
        timestamp refers to the *open* of the bar (bar-open convention).
        A bar with timestamp=T contains price information for the period [T, T+freq).
        The `<= as_of` filter is INCLUSIVE on timestamp, meaning a bar whose open falls
        exactly ON as_of IS included. This matches EOD data convention where the bar at
        date D represents that full trading day and is available at close of day D.

        Example — as_of = 2024-03-15 (EOD):
          bar 2024-03-15 is INCLUDED (daily bar for that trading day is available)
          bar 2024-03-18 (next Monday) is EXCLUDED

        Implication for signals: signals computed from `prices_filtered` in backtest
        mode are based on closes available at or before as_of. No look-ahead bias
        as long as as_of represents the decision point (e.g., market close).

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
                from src.assembled_core.features.ta_candlestick import (
                    build_candlestick_features,
                )
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
            from src.assembled_core.data.congress_trades_ingest import (
                load_congress_sample,
            )
            from src.assembled_core.features.congress_features import (
                add_congress_features,
            )

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
        returns = wide.pct_change(fill_method=None).dropna(how="all")
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
            logger.warning("[VAR-GATE] no weight/notional column in target_positions — using equal weights for VaR")

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


