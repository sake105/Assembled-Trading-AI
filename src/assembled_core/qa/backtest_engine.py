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
- Nutzt pipeline.portfolio.simulate_with_costs (mit cost-params=0) für kostenfreie
  Simulation — single-source-of-truth, cash-aware (audit §9.6(d) 2026-05-19).
- Nutzt pipeline.portfolio.simulate_with_costs für kostenbewusste Simulation
- Nutzt execution.order_generation für Order-Generierung
- Erweitert um Walk-Forward-Analyse, Monte-Carlo-Simulation, etc.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Type-only imports to avoid circular dependency
from typing import TYPE_CHECKING

from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.data.factor_store import compute_universe_key
from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.execution.transaction_costs import (
    SlippageModel,
    SpreadModel,
    add_cost_columns_to_trades,
    commission_model_from_cost_params,
)
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.features.ta_features import (
    add_all_features,
    add_log_returns,
    add_moving_averages,
)

# 2026-05-19 audit §9.6(d): no-costs runs no longer call simulate_equity here
# (legacy unconstrained path); both with-costs and no-costs route through
# simulate_with_costs with appropriate cost params for one source of truth.
from src.assembled_core.pipeline.portfolio import simulate_with_costs

if TYPE_CHECKING:
    from src.assembled_core.pipeline.trading_cycle_shared import (
        TradingContext,
        TradingCycleResult,
    )
from src.assembled_core.config.settings import get_settings
from src.assembled_core.utils.timing import timed_step

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
                aggregate_position_deltas_numba,
                compute_position_deltas_numba,
            )

            if NUMBA_AVAILABLE:
                # Convert to numpy arrays for Numba
                symbols_list = orders["symbol"].unique().tolist()
                symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols_list)}

                # Map sides to integers (0=BUY, 1=SELL)
                side_map = {"BUY": 0, "SELL": 1}
                sides = orders["side"].map(side_map).fillna(0).values.astype(np.int32)
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
            logger.warning(
                "[BacktestEngine] numba fill simulation failed, falling back to pandas: %s",
                exc,
            )

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
    enable_risk_controls: bool = True,
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
        >>> from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
        >>> from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle
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
        from src.assembled_core.pipeline.trading_cycle_v2 import (
            run_trading_cycle as run_trading_cycle_fn,
        )

    _enable_risk_controls: bool = enable_risk_controls

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


# ---------------------------------------------------------------------------
# _pb_* private helpers for run_portfolio_backtest
# ---------------------------------------------------------------------------


def _pb_compute_features(
    prices: pd.DataFrame,
    compute_features: bool,
    cycle_fn: Any,
    feature_config: dict[str, Any] | None,
    use_factor_store: bool,
    factor_store_root: Path | None,
    factor_group: str,
    rebalance_freq: str,
) -> pd.DataFrame:
    """Return prices_with_features; falls back to prices copy when features are skipped."""
    if not (compute_features and len(prices) > 0 and cycle_fn is None):
        return prices.copy()

    config = feature_config or {}
    has_ohlc = all(col in prices.columns for col in ["high", "low", "open"])

    if use_factor_store:
        logger.info(
            "Using factor store: group=%s, root=%s", factor_group, factor_store_root
        )
        universe_symbols = sorted(prices["symbol"].unique().tolist())
        universe_key = compute_universe_key(symbols=universe_symbols)
        start_date = prices["timestamp"].min()
        end_date = prices["timestamp"].max()
        return build_or_load_factors(
            prices=prices,
            factor_group=factor_group,
            freq=rebalance_freq,
            universe_key=universe_key,
            start_date=start_date,
            end_date=end_date,
            as_of=None,
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
    if has_ohlc:
        return add_all_features(
            prices,
            ma_windows=config.get("ma_windows", (20, 50, 200)),
            atr_window=config.get("atr_window", 14),
            rsi_window=config.get("rsi_window", 14),
            include_rsi=config.get("include_rsi", True),
        )
    prices_wf = add_log_returns(prices.copy())
    return add_moving_averages(
        prices_wf, windows=config.get("ma_windows", (20, 50, 200))
    )


def _pb_generate_signals(
    prices_with_features: pd.DataFrame,
    signal_fn: Callable | None,
    cycle_fn: Any,
    *,
    use_meta_model: bool,
    meta_model: Any | None,
    meta_model_path: str | None,
    meta_min_confidence: float,
    meta_ensemble_mode: str,
) -> pd.DataFrame:
    """Generate signals and optionally apply meta-model ensemble. Returns signals DataFrame."""
    if cycle_fn is None:
        if signal_fn is None:
            raise ValueError("signal_fn is required when cycle_fn is not provided")
        signals = signal_fn(prices_with_features)
    else:
        signals = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    if cycle_fn is None:
        required_signal_cols = ["timestamp", "symbol", "direction"]
        missing_signal = [c for c in required_signal_cols if c not in signals.columns]
        if missing_signal:
            raise KeyError(
                f"signal_fn must return DataFrame with columns: {required_signal_cols}. Missing: {missing_signal}"
            )
        signals = signals.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if not (use_meta_model and cycle_fn is None):
        return signals

    logger.info("Applying meta-model ensemble...")
    _mm = meta_model
    if _mm is None and meta_model_path is not None:
        try:
            from src.assembled_core.signals.meta_model import load_meta_model

            _mm = load_meta_model(meta_model_path)
            logger.info("Loaded meta-model from %s", meta_model_path)
        except Exception as e:
            raise ValueError(f"Failed to load meta-model: {e}") from e
    if _mm is None:
        raise ValueError(
            "use_meta_model=True but no meta_model or meta_model_path provided"
        )

    feature_cols = _mm.feature_names
    available_features = [f for f in feature_cols if f in prices_with_features.columns]
    missing_features = [
        f for f in feature_cols if f not in prices_with_features.columns
    ]

    if missing_features:
        logger.warning(
            "Missing %d features for meta-model: %s...",
            len(missing_features),
            missing_features[:5],
        )
        logger.warning(
            "Meta-model ensemble may not work correctly. Continuing anyway..."
        )

    if not available_features:
        logger.error("No features available for meta-model. Disabling ensemble.")
        return signals

    signals_with_features = signals.merge(
        prices_with_features[["timestamp", "symbol"] + available_features],
        on=["timestamp", "symbol"],
        how="inner",
    )
    if signals_with_features.empty:
        logger.warning(
            "No signals matched with features. Disabling meta-model ensemble."
        )
        return signals

    features_subset = signals_with_features[available_features].copy()
    if missing_features:
        for feat in missing_features:
            features_subset[feat] = 0.0
        features_subset = features_subset[_mm.feature_names]

    from src.assembled_core.signals.ensemble import (
        apply_meta_filter,
        apply_meta_scaling,
    )

    original_signal_count = len(signals_with_features)
    original_long_count = (signals_with_features["direction"] == "LONG").sum()

    if meta_ensemble_mode == "filter":
        signals_with_features = apply_meta_filter(
            signals=signals_with_features,
            meta_model=_mm,
            features=features_subset,
            min_confidence=meta_min_confidence,
            join_keys=["timestamp", "symbol"],
        )
    elif meta_ensemble_mode == "scaling":
        signals_with_features = apply_meta_scaling(
            signals=signals_with_features,
            meta_model=_mm,
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

    meta_cols = ["timestamp", "symbol", "direction", "meta_confidence"]
    if "final_score" in signals_with_features.columns:
        meta_cols.append("final_score")
    signals = signals.merge(
        signals_with_features[meta_cols],
        on=["timestamp", "symbol"],
        how="left",
        suffixes=("", "_meta"),
    )
    if "direction_meta" in signals.columns:
        signals["direction"] = signals["direction_meta"].fillna(signals["direction"])
        signals = signals.drop(columns=["direction_meta"])
    if "final_score" in signals.columns:
        if "score" not in signals.columns:
            signals["score"] = 0.0
        signals["score"] = signals["final_score"].fillna(signals["score"])
        signals = signals.drop(columns=["final_score"])

    filtered_signal_count = len(signals_with_features)
    filtered_long_count = (signals_with_features["direction"] == "LONG").sum()
    dropped_count = original_long_count - filtered_long_count
    logger.info("Meta-model ensemble applied:")
    logger.info(
        "  Original signals: %d (LONG: %d)", original_signal_count, original_long_count
    )
    logger.info(
        "  After filtering: %d (LONG: %d)", filtered_signal_count, filtered_long_count
    )
    logger.info("  Dropped signals: %d", dropped_count)
    logger.info(
        "  Mode: %s, Min confidence: %s", meta_ensemble_mode, meta_min_confidence
    )
    return signals


def _pb_run_cycle_fn_loop(
    *,
    cycle_fn: Callable,
    timeline: list,
    rebalance_timestamps_set: set,
    start_capital: float,
    use_numba: bool,
    include_targets: bool,
    include_signals: bool,
    prices: pd.DataFrame,
) -> tuple[list, list, list, dict]:
    """Run cycle_fn path. Returns (all_orders, all_targets, all_signals_list, timings)."""
    all_orders: list = []
    all_targets: list = []
    all_signals_list: list = []
    timings: dict[str, Any] = {}
    decision_timings: list = []
    position_update_timings: list = []
    equity_values: list[float] = [start_capital]
    cash = start_capital
    profit_lock_state: dict[str, Any] | None = None
    current_positions = pd.DataFrame(columns=["symbol", "qty"])
    # last-known price per symbol (prevents fillna(0) gaps).
    # fmt: off — pre-commit ruff 0.8.6 and black 24.10.0 disagree on the
    # type-annotated empty-dict literal here, causing an unresolvable hook loop.
    _px_cache: dict[str, float] = {}
    # fmt: on

    for timestamp in timeline:
        if timestamp not in rebalance_timestamps_set:
            continue
        equity_curve_series = pd.Series(equity_values)
        equity_curve_index = len(equity_values) - 1
        with timed_step(f"decision_{timestamp}", timings, logger):
            cycle_result = cycle_fn(
                timestamp,
                current_positions,
                equity_curve=equity_curve_series,
                equity_curve_index=equity_curve_index,
                profit_lock_state=profit_lock_state,
            )
        if f"decision_{timestamp}" in timings:
            decision_timings.append(timings[f"decision_{timestamp}"]["duration_ms"])

        if cycle_result.status != "success":
            logger.warning(
                "Trading cycle failed for timestamp %s: %s",
                timestamp,
                cycle_result.error_message,
            )
            continue

        profit_lock_state = cycle_result.meta.get("profit_lock_state")
        orders = cycle_result.orders_filtered
        if not orders.empty and "qty" in orders.columns and (orders["qty"] < 0).any():
            orders = orders.copy()
            orders["qty"] = orders["qty"].abs()

        if not orders.empty:
            _sign = np.where(orders["side"] == "BUY", -1.0, 1.0)
            cash += float(
                (
                    orders["qty"].fillna(0).abs() * orders["price"].fillna(0) * _sign
                ).sum()
            )

        if not orders.empty:
            with timed_step(f"position_update_{timestamp}", timings, logger):
                current_positions = _update_positions_vectorized(
                    orders, current_positions, use_numba=use_numba
                )
            if f"position_update_{timestamp}" in timings:
                position_update_timings.append(
                    timings[f"position_update_{timestamp}"]["duration_ms"]
                )
            all_orders.append(orders)

        prices_at_ts = prices[prices["timestamp"] == timestamp]
        if not prices_at_ts.empty:
            _px_cache.update(prices_at_ts.set_index("symbol")["close"].to_dict())
        if not current_positions.empty:
            qty_series = current_positions.set_index("symbol")["qty"]
            # Use last known cached price; skip symbols with no price history entirely
            # (0.0 would silently understate equity for delisted/missing symbols)
            filled_px = pd.Series(
                {sym: _px_cache[sym] for sym in qty_series.index if sym in _px_cache}
            )
            mtm = float((qty_series.reindex(filled_px.index) * filled_px).sum())
        else:
            mtm = 0.0
        equity_values.append(cash + float(mtm))

        try:
            from src.assembled_core.strategies.multifactor_v2 import (
                update_drawdown_damper,
            )

            _as_of = timestamp.date() if hasattr(timestamp, "date") else None
            update_drawdown_damper(equity_values[-1], _as_of)
        except ImportError as _imp_exc:
            # multifactor_v2 missing — non-fatal, DD-damper just unavailable.
            logger.debug("[backtest] DD-damper import failed: %s", _imp_exc)
        except Exception as _exc:
            # B3-N4 R6 fix: previously bare except: pass — silent risk-control
            # failure. Now log at DEBUG so the failure is observable without
            # spamming on every bar.
            logger.debug(
                "[backtest] update_drawdown_damper raised %s at %s",
                _exc,
                timestamp,
            )

        if include_targets and not cycle_result.target_positions.empty:
            targets_with_timestamp = cycle_result.target_positions.copy()
            targets_with_timestamp["timestamp"] = timestamp
            all_targets.append(targets_with_timestamp)
        if include_signals and not cycle_result.signals.empty:
            all_signals_list.append(cycle_result.signals)

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
    return all_orders, all_targets, all_signals_list, timings


def _pb_run_legacy_loop(
    *,
    signals: pd.DataFrame,
    position_sizing_fn: Callable,
    start_capital: float,
    prices: pd.DataFrame,
    include_targets: bool,
    rebalance_schedule: str,
) -> tuple[list, list, dict]:
    """Run legacy signal/sizing path. Returns (all_orders, all_targets, timings)."""
    all_orders: list = []
    all_targets: list = []
    timings: dict[str, Any] = {}
    order_generation_timings: list = []
    _legacy_cash: float = start_capital
    current_positions = pd.DataFrame(columns=["symbol", "qty"])

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
        except Exception as _exc:
            logger.warning(
                "[backtest] MTM equity calc failed at %s: %s", timestamp, _exc
            )

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
        if f"order_generation_{timestamp}" in timings:
            order_generation_timings.append(
                timings[f"order_generation_{timestamp}"]["duration_ms"]
            )

        try:
            if not orders.empty:
                _sign = np.where(orders["side"] == "BUY", -1.0, 1.0)
                _legacy_cash += float(
                    (
                        orders["qty"].fillna(0).abs()
                        * orders["price"].fillna(0)
                        * _sign
                    ).sum()
                )
        except Exception as _exc:
            logger.warning(
                "[backtest] cash accounting failed at %s: %s", timestamp, _exc
            )

        current_positions = updated_positions
        if include_targets and not targets.empty:
            all_targets.append(targets)
        if not orders.empty:
            all_orders.append(orders)

    if order_generation_timings:
        timings["order_generation"] = {
            "total_duration_ms": sum(order_generation_timings),
            "avg_duration_ms": sum(order_generation_timings)
            / len(order_generation_timings),
            "count": len(order_generation_timings),
        }
    return all_orders, all_targets, timings


def _pb_simulate_equity(
    *,
    orders_df: pd.DataFrame,
    prices: pd.DataFrame,
    start_capital: float,
    cost_model: Any | None,
    commission_bps: float | None,
    spread_w: float | None,
    impact_w: float | None,
    include_costs: bool,
    include_trades: bool,
    rebalance_freq: str,
    strict_session_gate: bool,
    timings: dict[str, Any],
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Steps 4-4.6: equity simulation, fill pipeline, cost columns.

    Returns (equity, metrics, trades_df, orders_df).
    """
    with timed_step("fill_sim", timings, logger):
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

        # Normalize rebalance_freq to data-bar frequency for the fill pipeline.
        # "M"/"W"/"Q" are rebalancing cadences, not bar frequencies — the session
        # gate only recognises "1d" as EOD mode. Any non-intraday cadence maps to "1d"
        # so midnight-UTC EOD timestamps are accepted instead of rejected as OUTSIDE_SESSION.
        _INTRADAY_FREQS = {"1min", "5min", "15min", "30min", "1h", "2h", "4h"}
        fill_freq = rebalance_freq if rebalance_freq in _INTRADAY_FREQS else "1d"

        if include_costs:
            equity, metrics, trades_df = simulate_with_costs(
                orders=orders_df,
                start_capital=start_capital,
                commission_bps=commission_bps,
                spread_w=spread_w,
                impact_w=impact_w,
                freq=fill_freq,
                prices=prices,
                strict_session_gate=strict_session_gate,
            )
            metrics["trades"] = len(orders_df)
        else:
            # 2026-05-19 audit §9.6(d): route no-costs runs through the same
            # cash-aware simulator as with-costs, just with zero cost params.
            # The legacy simulate_equity path has no cash-constraint check
            # (src/assembled_core/pipeline/backtest.py:319), which lets
            # positions grow on negative cash and produces catastrophic
            # phantom losses in long backtests (mfv2 OOS 2025-01..2026-05-05
            # collapsed from $103k to $27k on 2026-03-24 with the legacy
            # path while the cash-aware path stayed at $92k on identical
            # trades). One source of truth = simulate_with_costs.
            equity, metrics, trades_df = simulate_with_costs(
                orders=orders_df,
                start_capital=start_capital,
                commission_bps=0.0,
                spread_w=0.0,
                impact_w=0.0,
                freq=fill_freq,
                prices=prices,
                strict_session_gate=strict_session_gate,
            )
            metrics["trades"] = len(orders_df)
            # F-9.6d-1 (Stage-2 MAJOR): simulate_with_costs returns only
            # {final_pf, sharpe, trades}. Legacy compute_metrics also exposed
            # {rows, first, last}. Re-add them for schema-compatibility so
            # external consumers reading result.metrics see the same keys
            # under include_costs=False as before.
            if not equity.empty and "equity" in equity.columns:
                metrics.setdefault("rows", int(len(equity)))
                metrics.setdefault("first", float(equity["equity"].iloc[0]))
                metrics.setdefault("last", float(equity["equity"].iloc[-1]))

    # Step 4.5: Apply fill model pipeline (session gate -> limit -> partial)
    # This must happen BEFORE cost calculation, as costs are based on fill_qty
    if not orders_df.empty:
        from src.assembled_core.execution.fill_model_pipeline import (
            apply_fill_model_pipeline,
        )

        orders_df = apply_fill_model_pipeline(
            orders_df,
            prices=prices,
            freq=fill_freq,
            partial_fill_model=None,
            strict_session_gate=strict_session_gate,
        )

    # Step 4.6: Add cost columns to orders (if include_trades=True)
    if include_trades and not orders_df.empty:
        if include_costs:
            if cost_model is not None:
                commission_model = commission_model_from_cost_params(
                    commission_bps=cost_model.commission_bps
                )
            else:
                commission_model = commission_model_from_cost_params(
                    commission_bps=commission_bps if commission_bps is not None else 0.0
                )
            spread_model = None
            if spread_w is not None and spread_w > 0.0:
                spread_model = SpreadModel(
                    adv_window=20,
                    buckets=None,
                    fallback_spread_bps=spread_w * 100.0,
                )
            slippage_model = None
            if impact_w is not None and impact_w > 0.0:
                slippage_model = SlippageModel(
                    vol_window=20,
                    k=impact_w,
                    min_bps=0.0,
                    max_bps=50.0,
                    fallback_slippage_bps=impact_w * 100.0,
                )
            orders_df = add_cost_columns_to_trades(
                orders_df,
                commission_model=commission_model,
                spread_model=spread_model,
                slippage_model=slippage_model,
                prices=prices if include_trades else None,
            )
        else:
            orders_df["commission_cash"] = 0.0
            orders_df["spread_cash"] = 0.0
            orders_df["slippage_cash"] = 0.0
            orders_df["total_cost_cash"] = 0.0

    return equity, metrics, trades_df, orders_df


def _pb_normalize_equity(equity: pd.DataFrame) -> pd.DataFrame:
    """Ensure equity DataFrame has date, timestamp, equity, daily_return columns."""
    if "timestamp" in equity.columns:
        equity = equity.copy()
        equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
        equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
        base_cols = ["date", "timestamp", "equity", "daily_return"]
        extra = [c for c in ["cash"] if c in equity.columns]
        return equity[base_cols + extra].copy()
    if "date" in equity.columns:
        equity = equity.copy()
        equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
        base = ["date", "equity", "daily_return"]
        if "timestamp" in equity.columns:
            base = ["date", "timestamp", "equity", "daily_return"]
        extra = [c for c in ["cash"] if c in equity.columns]
        return equity[base + extra].copy()
    equity = equity.copy()
    if equity.index.dtype == "datetime64[ns]":
        equity["date"] = equity.index.date
        equity["timestamp"] = equity.index
    else:
        if "timestamp" in equity.columns:
            equity["date"] = pd.to_datetime(equity["timestamp"]).dt.date
        else:
            equity["date"] = pd.date_range(
                start="2000-01-01", periods=len(equity), freq="D"
            ).date
            equity["timestamp"] = pd.to_datetime(equity["date"])
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
    base_cols = ["date", "timestamp", "equity", "daily_return"]
    extra = [c for c in ["cash"] if c in equity.columns]
    return equity[base_cols + extra].copy()


def _pb_build_ledger(
    *,
    orders_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    prices: pd.DataFrame,
    start_capital: float,
    include_ledger: bool,
    run_id: str | None,
    output_dir: Path | None,
    broker_snapshot_policy: str,
    write_broker_snapshot: bool,
    broker_snapshot_run_id: str | None,
    broker_snapshot_file: str | Path | None,
    broker_snapshot_date: str | None,
    include_costs: bool,
    write_evidence_pack: bool,
) -> dict | None:
    """Build ledger from trades (step 6). Returns ledger_result dict or None."""
    if not (include_ledger and run_id and output_dir):
        return None
    try:
        from src.assembled_core.accounting.ledger_integration import (
            build_ledger_from_trades,
        )

        trades_for_ledger = orders_df.copy()
        if include_costs and not trades_df.empty:
            trades_for_ledger = trades_df.copy()
        snapshot_run_id = (
            broker_snapshot_run_id if broker_snapshot_run_id is not None else run_id
        )
        if broker_snapshot_file:
            try:
                logger.info(
                    "Importing external broker snapshot from: %s", broker_snapshot_file
                )
                from src.assembled_core.accounting.broker_snapshot_importer import (
                    import_broker_snapshot,
                )

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
                import_result = import_broker_snapshot(
                    snapshot_path=Path(broker_snapshot_file),
                    run_id=snapshot_run_id,
                    snapshot_date=snapshot_date,
                    output_dir=output_dir,
                    qty_tol=1e-8,
                    store_parquet=True,
                )
                logger.info(
                    "Imported broker snapshot: %s, cash=%s",
                    import_result["broker_snapshot_path"],
                    import_result["cash"],
                )
            except Exception as e:
                logger.error("Failed to import broker snapshot: %s", e, exc_info=True)
                if broker_snapshot_policy == "require":
                    raise ValueError(
                        f"Broker snapshot import failed (policy=require): {e}"
                    ) from e
        ledger_result = build_ledger_from_trades(
            orders_df=orders_df,
            trades_df=trades_for_ledger,
            run_id=run_id,
            output_dir=output_dir,
            as_of_date=None,
            prices_df=prices,
            start_cash=start_capital,
            broker_snapshot_policy=broker_snapshot_policy,
            write_paper_broker_snapshot=write_broker_snapshot,
            broker_snapshot_run_id=snapshot_run_id,
            write_evidence_pack=write_evidence_pack,
        )
        logger.info(
            "Ledger integration completed: ledger_pack_path=%s, reconciliation_ok=%s",
            ledger_result.get("ledger_pack_path"),
            ledger_result.get("reconciliation_ok"),
        )
        return ledger_result
    except Exception as e:
        logger.warning("Ledger integration failed: %s", e, exc_info=True)
        # Distinguish "ledger not attempted" (None) from "attempted but failed".
        # A silent None after an exception has masked reconciliation gaps in prior
        # incidents — keep the failure explicit so downstream meta consumers see it.
        return {"reconciliation_ok": False, "ledger_error": str(e)}


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
    # A7: Corporate actions — splits/dividends adjustment (default: False for backtest to avoid data dep)
    enable_corporate_actions: bool = False,
    corporate_actions_path: str | None = None,
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
    # B3-N5 R6 fix: reset module-global _DD_DAMPER state at the start of every
    # backtest run. Previously only scripts/batch_runner.py called reset between
    # runs; programmatic multi-run sequences (notebooks, parameter sweeps) would
    # inherit the previous run's drawdown state, biasing results.
    try:
        from src.assembled_core.strategies.multifactor_v2 import reset_dd_damper

        reset_dd_damper()
    except Exception as _exc:
        # Defensive: if multifactor_v2 not importable, log and proceed.
        # Backtest doesn't strictly require DD-damper.
        logger.debug("[backtest] reset_dd_damper unavailable: %s", _exc)

    # Numba setup
    if use_numba is None:
        settings = get_settings()
        use_numba = settings.use_numba

    # Input validation
    if prices is None or prices.empty:
        raise ValueError("Missing required columns: prices DataFrame is None or empty")
    required_cols = ["timestamp", "symbol", "close"]
    missing = [c for c in required_cols if c not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()

    # Step 0: Corporate actions adjustment (splits/dividends)
    if not enable_corporate_actions:
        logger.warning(
            "[BACKTEST] Corporate actions DISABLED — splits/dividends ignored. "
            "Results may be misleading for multi-year backtests."
        )
    elif corporate_actions_path is not None:
        try:
            import pandas as _pd_ca
            from src.assembled_core.data.corporate_actions import (
                adjust_prices_for_splits,
            )

            splits = _pd_ca.read_csv(corporate_actions_path)
            prices = adjust_prices_for_splits(prices, splits)
            logger.info(
                "[BACKTEST] Corporate actions applied from %s", corporate_actions_path
            )
        except Exception as _ca_err:
            logger.warning(
                "[BACKTEST] Corporate actions adjustment failed: %s", _ca_err
            )
    else:
        logger.debug(
            "[BACKTEST] enable_corporate_actions=True but no corporate_actions_path provided"
            " — skipping adjustment"
        )

    # Step 1: Feature computation
    prices_with_features = _pb_compute_features(
        prices,
        compute_features,
        cycle_fn,
        feature_config,
        use_factor_store,
        factor_store_root,
        factor_group,
        rebalance_freq,
    )

    # Step 2: Signal generation + meta-model ensemble
    signals = _pb_generate_signals(
        prices_with_features,
        signal_fn,
        cycle_fn,
        use_meta_model=use_meta_model,
        meta_model=meta_model,
        meta_model_path=meta_model_path,
        meta_min_confidence=meta_min_confidence,
        meta_ensemble_mode=meta_ensemble_mode,
    )

    # Step 3: Build timeline and rebalance set
    raw_timeline = sorted(prices["timestamp"].unique())
    timeline: list[pd.Timestamp] = []
    for ts in raw_timeline:
        ts_pd = pd.to_datetime(ts)
        if ts_pd.tzinfo is None or ts_pd.tz is None:
            ts_pd = ts_pd.tz_localize("UTC")
        else:
            ts_pd = ts_pd.tz_convert("UTC")
        timeline.append(ts_pd)

    if rebalance_timestamps is not None:
        normalized_rebalance: list[pd.Timestamp] = []
        for ts in rebalance_timestamps:
            ts_pd = pd.to_datetime(ts)
            if ts_pd.tzinfo is None or ts_pd.tz is None:
                ts_pd = ts_pd.tz_localize("UTC")
            else:
                ts_pd = ts_pd.tz_convert("UTC")
            normalized_rebalance.append(ts_pd)
        rebalance_timestamps_set: set = set(normalized_rebalance)
    elif rebalance_schedule == "weekly":
        rebalance_timestamps_set = set(timeline[i] for i in range(0, len(timeline), 5))
    elif rebalance_schedule == "monthly":
        # First trading day of each calendar month: keep ts if previous ts has a different month.
        rebalance_timestamps_set = set()
        prev_month: int | None = None
        for ts in timeline:
            if ts.month != prev_month:
                rebalance_timestamps_set.add(ts)
            prev_month = ts.month
    else:
        rebalance_timestamps_set = set(timeline)

    # Step 3 (continued): Execute per-timestamp loop
    all_signals_list: list = []
    if cycle_fn is not None:
        all_orders, all_targets, all_signals_list, timings = _pb_run_cycle_fn_loop(
            cycle_fn=cycle_fn,
            timeline=timeline,
            rebalance_timestamps_set=rebalance_timestamps_set,
            start_capital=start_capital,
            use_numba=use_numba,
            include_targets=include_targets,
            include_signals=include_signals,
            prices=prices,
        )
    else:
        if signal_fn is None or position_sizing_fn is None:
            raise ValueError(
                "signal_fn and position_sizing_fn are required when cycle_fn is not provided"
            )
        all_orders, all_targets, timings = _pb_run_legacy_loop(
            signals=signals,
            position_sizing_fn=position_sizing_fn,
            start_capital=start_capital,
            prices=prices,
            include_targets=include_targets,
            rebalance_schedule=rebalance_schedule,
        )

    # Combine all orders
    if all_orders:
        orders_df = pd.concat(all_orders, ignore_index=True)
        orders_df = orders_df.sort_values("timestamp").reset_index(drop=True)
    else:
        orders_df = pd.DataFrame(
            columns=["timestamp", "symbol", "side", "qty", "price"]
        )
    _validate_order_notional_guard(orders_df, start_capital)
    if not orders_df.empty:
        orders_df.attrs["qty_unit"] = "shares"

    # Steps 4-4.6: Equity simulation + fill pipeline + cost columns
    equity, metrics, trades_df, orders_df = _pb_simulate_equity(
        orders_df=orders_df,
        prices=prices,
        start_capital=start_capital,
        cost_model=cost_model,
        commission_bps=commission_bps,
        spread_w=spread_w,
        impact_w=impact_w,
        include_costs=include_costs,
        include_trades=include_trades,
        rebalance_freq=rebalance_freq,
        strict_session_gate=strict_session_gate,
        timings=timings,
    )

    # Step 5: Normalize equity DataFrame
    equity = _pb_normalize_equity(equity)

    # Step 6: Ledger integration
    ledger_result = _pb_build_ledger(
        orders_df=orders_df,
        trades_df=trades_df,
        prices=prices,
        start_capital=start_capital,
        include_ledger=include_ledger,
        run_id=run_id,
        output_dir=output_dir,
        broker_snapshot_policy=broker_snapshot_policy,
        write_broker_snapshot=write_broker_snapshot,
        broker_snapshot_run_id=broker_snapshot_run_id,
        broker_snapshot_file=broker_snapshot_file,
        broker_snapshot_date=broker_snapshot_date,
        include_costs=include_costs,
        write_evidence_pack=write_evidence_pack,
    )

    # Step 7: Build result
    signals_result = None
    if include_signals:
        if cycle_fn is not None and all_signals_list:
            signals_result = pd.concat(all_signals_list, ignore_index=True)
            signals_result = signals_result.sort_values(
                ["symbol", "timestamp"]
            ).reset_index(drop=True)
        elif cycle_fn is None:
            signals_result = signals

    meta_dict: dict[str, Any] = {}
    if timings:
        meta_dict["timings"] = timings
    if ledger_result:
        meta_dict["ledger_pack_path"] = ledger_result.get("ledger_pack_path")
        meta_dict["reconcile_report_path"] = ledger_result.get("reconcile_report_path")
        meta_dict["reconciliation_ok"] = ledger_result.get("reconciliation_ok")
        meta_dict["broker_snapshot_path"] = ledger_result.get("broker_snapshot_path")
        if ledger_result.get("ledger_error"):
            meta_dict["ledger_error"] = ledger_result.get("ledger_error")
        meta_dict["evidence_index_path"] = ledger_result.get("evidence_index_path")
        meta_dict["evidence_pack_path"] = ledger_result.get("evidence_pack_path")
        meta_dict["evidence_pack_manifest_path"] = ledger_result.get(
            "evidence_pack_manifest_path"
        )

    trades_for_result = None
    if include_trades:
        # F-9.6d-3 (Stage-2 follow-up, 2026-05-19):
        # Since 3357fc9 the no-costs branch also routes through
        # simulate_with_costs(0,0,0), so trades_df is the cash-gated
        # fill representation that matches the cash-gated equity curve.
        # The non-empty check is the primary safety; the schema check
        # ("status" + "fill_qty") is secondary defense against partial
        # schemas (e.g. the empty-orders shortcut in portfolio.py
        # returns a trades_df without status — empty check catches it).
        # Pre-3357fc9 the no-costs path used simulate_equity (unbounded,
        # no cash gate), so trades==orders was locally consistent. The
        # old include_costs gate would now keep that un-gated request
        # list under no-costs, breaking consistency with the cash-gated
        # equity curve — hence dropped.
        trades_for_result = (
            trades_df
            if (
                not trades_df.empty
                and "status" in trades_df.columns
                and "fill_qty" in trades_df.columns
            )
            else orders_df
        )

    return BacktestResult(
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
