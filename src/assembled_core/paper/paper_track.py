"""Paper Track Runner - Orchestrator + State IO.

This module provides the core orchestrator for paper trading track execution.
It manages state persistence and coordinates signal generation, position sizing,
order generation, and fill simulation without duplicating financial logic.

Key principles:
- No duplication of financial logic (reuse existing signal/sizing/backtest modules)
- Stateful: Persist portfolio state (positions, cash, equity) between runs
- PIT-safe: Pass as_of timestamps through, optional strict checks via env
- Deterministic: Optional seed for reproducibility
- Read-only: No external API calls, only local file access
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.assembled_core.config.constants import (
    DEFAULT_ATR_WINDOW,
    DEFAULT_MA_WINDOWS,
    DEFAULT_RSI_WINDOW,
    DEFAULT_SEED_CAPITAL,
    PAPER_TRACK_STATE_VERSION,
)
from src.assembled_core.data.prices_ingest import (
    load_eod_prices_for_universe,
)
from src.assembled_core.execution.order_generation import (
    generate_orders_from_targets,
)
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)
from src.assembled_core.qa.backtest_engine import _update_positions_vectorized
from src.assembled_core.qa.point_in_time_checks import (
    check_features_pit_safe,
)
from src.assembled_core.signals.rules_trend import (
    generate_trend_signals_from_prices,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperTrackConfig:
    """Configuration for paper track runner.

    Attributes:
        strategy_name: Name of the strategy (e.g., "core_ai_tech")
        strategy_type: Strategy type ("trend_baseline" or "multifactor_long_short")
        universe_file: Path to universe file (symbol list)
        freq: Trading frequency ("1d" or "5min")
        seed_capital: Starting capital (default: 100000.0)
        commission_bps: Commission in basis points (default: 0.5)
        spread_w: Spread weight (default: 0.25)
        impact_w: Market impact weight (default: 0.5)
        strategy_params: Strategy-specific parameters (dict, e.g., ma_fast, ma_slow)
        enable_pit_checks: Enable PIT-safety checks (default: True, can be overridden by env)
        random_seed: Optional random seed for reproducibility (default: None)
        output_root: Output root directory (default: None, uses strategy_name in default location)
    """

    strategy_name: str
    strategy_type: Literal["trend_baseline", "multifactor_long_short"]
    universe_file: Path
    freq: Literal["1d", "5min"]
    seed_capital: float = DEFAULT_SEED_CAPITAL
    commission_bps: float = 0.5
    spread_w: float = 0.25
    impact_w: float = 0.5
    strategy_params: dict[str, Any] = field(default_factory=dict)
    enable_pit_checks: bool = True
    random_seed: int | None = None
    output_root: Path | None = None


@dataclass
class PaperTrackState:
    """State of paper track portfolio.

    Attributes:
        strategy_name: Name of the strategy (for validation)
        last_run_date: Last date when paper track was executed (pd.Timestamp, UTC, None if never run)
        version: Version of state format (default: "1.0")
        positions: DataFrame with columns: symbol, qty (positive = long, negative = short)
        cash: Current cash balance (float)
        equity: Current portfolio equity (cash + mark-to-market positions)
        seed_capital: Original seed capital (for reference)
        created_at: Timestamp when state was first created (pd.Timestamp, UTC)
        updated_at: Timestamp when state was last updated (pd.Timestamp, UTC)
        total_trades: Total number of trades executed since start (default: 0)
        total_pnl: Cumulative PnL (equity - seed_capital, default: 0.0)
    """

    strategy_name: str
    last_run_date: pd.Timestamp | None
    version: str = PAPER_TRACK_STATE_VERSION
    positions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["symbol", "qty"])
    )
    cash: float = 0.0
    equity: float = 0.0
    seed_capital: float = 100000.0
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    updated_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    total_trades: int = 0
    total_pnl: float = 0.0


@dataclass
class PaperTrackDayResult:
    """Result of a single paper track day execution.

    Attributes:
        date: Run date (pd.Timestamp, UTC)
        config: PaperTrackConfig used
        state_before: State before run
        state_after: State after run
        orders: Orders generated and filled (DataFrame: timestamp, symbol, side, qty, price, fill_price, costs)
        daily_return_pct: Daily return percentage (float)
        daily_pnl: Daily PnL (float)
        trades_count: Number of trades executed (int)
        buy_count: Number of buy orders (int)
        sell_count: Number of sell orders (int)
        status: "success" or "error"
        error_message: Error message if status == "error" (None otherwise)
    """

    date: pd.Timestamp
    config: PaperTrackConfig
    state_before: PaperTrackState
    state_after: PaperTrackState
    orders: pd.DataFrame
    daily_return_pct: float
    daily_pnl: float
    trades_count: int
    buy_count: int
    sell_count: int
    status: Literal["success", "error"] = "success"
    error_message: str | None = None


def _set_random_seed(seed: int | None) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed (None = no seed set)
    """
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed} for reproducibility")


def _should_enable_pit_checks(config: PaperTrackConfig) -> bool:
    """Determine if PIT checks should be enabled (env override support).

    Args:
        config: PaperTrackConfig

    Returns:
        True if PIT checks should be enabled, False otherwise
    """
    # Check environment variable first (allows override)
    env_var = os.environ.get("PAPER_TRACK_STRICT_PIT", "").lower()
    if env_var in ("1", "true", "yes", "on"):
        return True
    elif env_var in ("0", "false", "no", "off"):
        return False
    # Otherwise use config value
    return config.enable_pit_checks


def _filter_prices_for_date(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Filter prices to last available data <= as_of (per symbol).

    Args:
        prices: DataFrame with columns: timestamp, symbol, close, ...
        as_of: Maximum allowed timestamp (pd.Timestamp, UTC)

    Returns:
        Filtered DataFrame (one row per symbol: last available <= as_of)
    """
    if prices.empty:
        return prices

    # Ensure timestamp is timezone-aware UTC
    if prices["timestamp"].dt.tz is None:
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    # Filter to dates <= as_of, then group by symbol and take last row
    # This is more efficient than iterating over symbols
    filtered = prices[prices["timestamp"] <= as_of].copy()

    if filtered.empty:
        return pd.DataFrame(columns=prices.columns)

    # Group by symbol and take last row (most recent timestamp per symbol)
    # Use dropna=False to preserve all columns
    filtered = (
        filtered.groupby("symbol", group_keys=False, dropna=False)
        .last()
        .reset_index(drop=True)
    )

    return filtered


def _compute_features_for_strategy(
    config: PaperTrackConfig, prices: pd.DataFrame
) -> pd.DataFrame:
    """Compute features for a strategy configuration.

    Args:
        config: PaperTrackConfig with strategy parameters
        prices: DataFrame with price data (columns: timestamp, symbol, close)

    Returns:
        DataFrame with computed features added

    Raises:
        ValueError: If strategy_type is unsupported
    """
    if config.strategy_type == "trend_baseline":
        ma_fast = config.strategy_params.get("ma_fast", DEFAULT_MA_WINDOWS[0])
        ma_slow = config.strategy_params.get("ma_slow", DEFAULT_MA_WINDOWS[1])
        return add_all_features(
            prices,
            ma_windows=(ma_fast, ma_slow),
            atr_window=DEFAULT_ATR_WINDOW,
            rsi_window=DEFAULT_RSI_WINDOW,
            include_rsi=True,
        )
    else:
        # For other strategy types, use basic features with defaults
        return add_all_features(
            prices,
            ma_windows=DEFAULT_MA_WINDOWS,
            atr_window=DEFAULT_ATR_WINDOW,
            rsi_window=DEFAULT_RSI_WINDOW,
            include_rsi=True,
        )


def _compute_position_value(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
) -> float:
    """Compute mark-to-market value of positions.

    Args:
        positions: DataFrame with columns: symbol, qty
        prices: DataFrame with columns: symbol, close

    Returns:
        Total position value (float)
    """
    if positions.empty or prices.empty:
        return 0.0

    # Merge positions with prices
    merged = positions.merge(
        prices[["symbol", "close"]],
        on="symbol",
        how="left",
    )

    # Compute value: qty * price
    merged["value"] = merged["qty"] * merged["close"].fillna(0.0)
    return float(merged["value"].sum())


def _simulate_order_fills(
    orders: pd.DataFrame,
    current_cash: float,
    commission_bps: float,
    spread_w: float,
    impact_w: float,
) -> tuple[pd.DataFrame, float]:
    """Simulate order fills and update cash.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        current_cash: Current cash balance
        commission_bps: Commission in basis points
        spread_w: Spread weight
        impact_w: Impact weight

    Returns:
        Tuple of (filled_orders DataFrame with fill_price and costs columns, new_cash balance)
    """
    if orders.empty:
        return orders.copy(), current_cash

    filled = orders.copy()

    # Cost parameters
    k = commission_bps * 1e-4
    s = spread_w * 1e-4
    im = impact_w * 1e-4

    # Compute notional
    filled["notional"] = filled["qty"] * filled["price"]

    # Compute fill price (with spread and impact) - vectorized
    # BUY pays: price * (1 + s + im)
    # SELL receives: price * (1 - s - im)
    filled["fill_price"] = np.where(
        filled["side"] == "BUY",
        filled["price"] * (1.0 + s + im),
        filled["price"] * (1.0 - s - im),
    )

    # Compute costs (commission on notional)
    filled["costs"] = k * filled["notional"]

    # Compute cash delta (vectorized for better performance)
    # BUY: cash decreases by (qty * fill_price + costs)
    # SELL: cash increases by (qty * fill_price - costs)
    filled["cash_delta"] = np.where(
        filled["side"] == "BUY",
        -(filled["qty"] * filled["fill_price"] + filled["costs"]),
        +(filled["qty"] * filled["fill_price"] - filled["costs"]),
    )

    # Update cash
    new_cash = current_cash + filled["cash_delta"].sum()

    return filled, float(new_cash)


def load_paper_state(state_path: Path, strategy_name: str) -> PaperTrackState | None:
    """Load paper track state from file.

    Args:
        state_path: Path to state file (JSON)
        strategy_name: Expected strategy name (for validation)

    Returns:
        PaperTrackState if file exists, None otherwise

    Raises:
        ValueError: If state is invalid or strategy_name mismatch
        FileNotFoundError: If state file doesn't exist (returns None instead)
    """
    if not state_path.exists():
        logger.debug(f"State file does not exist, will create new state: {state_path}")
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate strategy name
        if data.get("strategy_name") != strategy_name:
            raise ValueError(
                f"State strategy_name mismatch for {state_path}: "
                f"expected '{strategy_name}', got '{data.get('strategy_name')}'. "
                f"This usually indicates the state file belongs to a different strategy."
            )

        # Parse timestamps
        last_run_date = (
            pd.to_datetime(data["last_run_date"], utc=True)
            if data.get("last_run_date")
            else None
        )
        created_at = pd.to_datetime(data["created_at"], utc=True)
        updated_at = pd.to_datetime(data["updated_at"], utc=True)

        # Parse positions DataFrame
        if data.get("positions"):
            positions = pd.DataFrame(data["positions"])
        else:
            positions = pd.DataFrame(columns=["symbol", "qty"])

        # Create state
        state = PaperTrackState(
            strategy_name=data["strategy_name"],
            last_run_date=last_run_date,
            version=data.get("version", "1.0"),
            positions=positions,
            cash=float(data["cash"]),
            equity=float(data["equity"]),
            seed_capital=float(data.get("seed_capital", 100000.0)),
            created_at=created_at,
            updated_at=updated_at,
            total_trades=int(data.get("total_trades", 0)),
            total_pnl=float(data.get("total_pnl", 0.0)),
        )

        logger.info(f"Loaded paper state from {state_path}")
        return state

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in state file {state_path}: {e}") from e
    except KeyError as e:
        raise ValueError(
            f"Missing required field in state file {state_path}: {e}"
        ) from e


def save_paper_state(state: PaperTrackState, state_path: Path) -> None:
    """Save paper track state to file (atomic write with backup).

    Args:
        state: PaperTrackState to save
        state_path: Path to state file

    Raises:
        IOError: If write fails
    """
    # Ensure directory exists
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if file exists
    backup_path = state_path.with_suffix(state_path.suffix + ".backup")
    if state_path.exists():
        shutil.copy2(state_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")

    # Prepare data for JSON serialization
    data = {
        "strategy_name": state.strategy_name,
        "last_run_date": state.last_run_date.isoformat()
        if state.last_run_date
        else None,
        "version": state.version,
        "positions": state.positions.to_dict(orient="records")
        if not state.positions.empty
        else [],
        "cash": state.cash,
        "equity": state.equity,
        "seed_capital": state.seed_capital,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "total_trades": state.total_trades,
        "total_pnl": state.total_pnl,
    }

    # Atomic write: write to temp file, then rename
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)

        # Rename (atomic on most filesystems)
        temp_path.replace(state_path)
        logger.info(f"Saved paper state to {state_path}")

    except (OSError, PermissionError, IOError) as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to save state to {state_path}: {e}") from e


def run_paper_day(
    config: PaperTrackConfig,
    as_of: pd.Timestamp,
    state_path: Path | None = None,
) -> PaperTrackDayResult:
    """Run paper track for a single day.

    This function orchestrates the complete daily flow:
    1. Load or initialize state
    2. Load prices (filtered to as_of)
    3. Compute features (PIT-safe)
    4. Generate signals
    5. Compute target positions
    6. Generate orders
    7. Simulate fills
    8. Update state

    Args:
        config: PaperTrackConfig
        as_of: Date to run for (pd.Timestamp, UTC)
        state_path: Optional path to state file (default: None, uses in-memory state only)

    Returns:
        PaperTrackDayResult

    Raises:
        ValueError: If already run for date (and not forced), or if config parameters are invalid
        FileNotFoundError: If required data files not found
        PointInTimeViolationError: If PIT-checks enabled and violation detected
    """
    # Validate config parameters
    if config.seed_capital <= 0 or not math.isfinite(config.seed_capital):
        raise ValueError(
            f"seed_capital must be > 0 and finite, got {config.seed_capital}"
        )
    if config.commission_bps < 0 or not math.isfinite(config.commission_bps):
        raise ValueError(
            f"commission_bps must be >= 0 and finite, got {config.commission_bps}"
        )
    if config.spread_w < 0 or not math.isfinite(config.spread_w):
        raise ValueError(
            f"spread_w must be >= 0 and finite, got {config.spread_w}"
        )
    if config.impact_w < 0 or not math.isfinite(config.impact_w):
        raise ValueError(
            f"impact_w must be >= 0 and finite, got {config.impact_w}"
        )

    # Validate as_of
    now = pd.Timestamp.utcnow()
    if as_of > now:
        raise ValueError(
            f"as_of ({as_of.date()}) cannot be in the future (current: {now.date()})"
        )

    # Validate state_path if provided
    if state_path is not None and not state_path.parent.exists():
        raise ValueError(
            f"state_path parent directory does not exist: {state_path.parent}"
        )

    # Set random seed if provided
    _set_random_seed(config.random_seed)

    # Load or initialize state
    state_before: PaperTrackState | None = None
    if state_path and state_path.exists():
        state_before = load_paper_state(state_path, config.strategy_name)
        if state_before and state_before.last_run_date is not None:
            if state_before.last_run_date.date() == as_of.date():
                raise ValueError(
                    f"Paper track already run for date {as_of.date()}. "
                    "Use force=True or delete state file to re-run."
                )
            if state_before.last_run_date > as_of:
                raise ValueError(
                    f"State last_run_date ({state_before.last_run_date.date()}) is in the future "
                    f"compared to as_of ({as_of.date()}). Possible clock skew."
                )

    if state_before is None:
        # Initialize new state
        now = pd.Timestamp.utcnow()
        state_before = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=None,
            cash=config.seed_capital,
            equity=config.seed_capital,
            seed_capital=config.seed_capital,
            created_at=now,
            updated_at=now,
        )
        logger.info(f"Initialized new paper state (seed_capital={config.seed_capital})")

    try:
        # Step 1: Load prices (filtered to as_of)
        logger.info(f"Loading prices for {as_of.date()}")
        prices = load_eod_prices_for_universe(
            universe_file=config.universe_file,
            freq=config.freq,
        )
        prices_filtered = _filter_prices_for_date(prices, as_of)

        if prices_filtered.empty:
            raise FileNotFoundError(f"No price data available for date {as_of.date()}")

        # Step 2: Compute features (PIT-safe)
        logger.debug("Computing features")
        # Strategy-specific feature computation
        prices_with_features = _compute_features_for_strategy(config, prices_filtered)

        # PIT check (if enabled)
        enable_pit = _should_enable_pit_checks(config)
        if enable_pit:
            check_features_pit_safe(
                prices_with_features,
                as_of=as_of,
                timestamp_col="timestamp",
                strict=True,
                feature_source="paper_track",
            )
            logger.debug("PIT checks passed")

        # Step 3: Generate signals
        logger.debug("Generating signals")
        if config.strategy_type == "trend_baseline":
            ma_fast = config.strategy_params.get("ma_fast", 20)
            ma_slow = config.strategy_params.get("ma_slow", 50)
            # Generate trend signals (function expects DataFrame with ma_fast, ma_slow columns)
            signals = generate_trend_signals_from_prices(
                prices_with_features,
                ma_fast=ma_fast,
                ma_slow=ma_slow,
            )
        else:
            raise ValueError(f"Unsupported strategy_type: {config.strategy_type}")

        # Step 4: Compute target positions
        logger.debug("Computing target positions")
        # Use current equity (not seed_capital) for position sizing
        target_positions = compute_target_positions_from_trend_signals(
            signals,
            total_capital=state_before.equity,  # Use current equity
            top_n=config.strategy_params.get("top_n"),
            min_score=config.strategy_params.get("min_score", 0.0),
        )

        # Step 5: Generate orders
        logger.debug("Generating orders")
        current_positions = state_before.positions.copy()
        orders = generate_orders_from_targets(
            target_positions,
            current_positions=current_positions
            if not current_positions.empty
            else None,
            timestamp=as_of,
            prices=prices_filtered,
        )

        # Step 6: Simulate fills
        logger.debug("Simulating order fills")
        filled_orders, new_cash = _simulate_order_fills(
            orders,
            state_before.cash,
            config.commission_bps,
            config.spread_w,
            config.impact_w,
        )

        # Step 7: Update positions
        logger.debug("Updating positions")
        updated_positions = _update_positions_vectorized(
            filled_orders[["timestamp", "symbol", "side", "qty", "price"]],
            current_positions,
            use_numba=True,
        )

        # Step 8: Compute new equity (mark-to-market)
        position_value = _compute_position_value(updated_positions, prices_filtered)
        new_equity = new_cash + position_value

        # Step 9: Create updated state
        now = pd.Timestamp.utcnow()
        state_after = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=as_of,
            version=state_before.version,
            positions=updated_positions,
            cash=new_cash,
            equity=new_equity,
            seed_capital=state_before.seed_capital,
            created_at=state_before.created_at,
            updated_at=now,
            total_trades=state_before.total_trades + len(filled_orders),
            total_pnl=new_equity - state_before.seed_capital,
        )

        # Compute daily metrics
        daily_pnl = new_equity - state_before.equity
        daily_return_pct = (
            (daily_pnl / state_before.equity * 100.0)
            if state_before.equity > 0
            else 0.0
        )
        buy_count = (
            len(filled_orders[filled_orders["side"] == "BUY"])
            if not filled_orders.empty
            else 0
        )
        sell_count = (
            len(filled_orders[filled_orders["side"] == "SELL"])
            if not filled_orders.empty
            else 0
        )

        # Create result
        result = PaperTrackDayResult(
            date=as_of,
            config=config,
            state_before=state_before,
            state_after=state_after,
            orders=filled_orders,
            daily_return_pct=float(daily_return_pct),
            daily_pnl=float(daily_pnl),
            trades_count=len(filled_orders),
            buy_count=buy_count,
            sell_count=sell_count,
            status="success",
            error_message=None,
        )

        logger.info(
            f"Paper day completed: equity={new_equity:.2f}, daily_return={daily_return_pct:.2f}%, "
            f"trades={len(filled_orders)}"
        )
        return result

    except Exception as e:
        # Return error result
        logger.error(f"Paper day failed: {e}", exc_info=True)
        return PaperTrackDayResult(
            date=as_of,
            config=config,
            state_before=state_before
            or PaperTrackState(strategy_name=config.strategy_name, last_run_date=None),
            state_after=state_before
            or PaperTrackState(strategy_name=config.strategy_name, last_run_date=None),
            orders=pd.DataFrame(
                columns=["timestamp", "symbol", "side", "qty", "price"]
            ),
            daily_return_pct=0.0,
            daily_pnl=0.0,
            trades_count=0,
            buy_count=0,
            sell_count=0,
            status="error",
            error_message=str(e),
        )


def write_paper_day_outputs(result: PaperTrackDayResult, output_dir: Path) -> None:
    """Write daily outputs for paper track run.

    Args:
        result: PaperTrackDayResult
        output_dir: Output directory (will create runs/{YYYYMMDD}/ subdirectory)

    Side effects:
        Creates output directory and writes files:
        - equity_snapshot.json
        - positions.csv
        - orders_today.csv
        - trades_today.csv
        - daily_summary.json
        - daily_summary.md
    """
    # Create run directory
    run_date_str = result.date.strftime("%Y%m%d")
    run_dir = output_dir / "runs" / run_date_str
    run_dir.mkdir(parents=True, exist_ok=True)

    state = result.state_after
    positions_value = state.equity - state.cash

    # 1. Equity snapshot (JSON)
    equity_snapshot = {
        "timestamp": result.date.isoformat(),
        "equity": state.equity,
        "cash": state.cash,
        "positions_value": positions_value,
        "seed_capital": state.seed_capital,
        "total_pnl": state.total_pnl,
        "total_return_pct": (state.total_pnl / state.seed_capital * 100.0)
        if state.seed_capital > 0
        else 0.0,
    }
    with open(run_dir / "equity_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(equity_snapshot, f, indent=2, ensure_ascii=True)

    # 2. Positions (CSV)
    if not state.positions.empty:
        state.positions.to_csv(run_dir / "positions.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "qty"]).to_csv(
            run_dir / "positions.csv", index=False
        )

    # 3. Orders/Trades (CSV)
    if not result.orders.empty:
        # Orders with fill info
        result.orders.to_csv(run_dir / "orders_today.csv", index=False)
        # Trades (same as orders, for compatibility)
        result.orders.to_csv(run_dir / "trades_today.csv", index=False)
    else:
        empty_df = pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "qty",
                "price",
                "fill_price",
                "costs",
                "cash_delta",
            ]
        )
        empty_df.to_csv(run_dir / "orders_today.csv", index=False)
        empty_df.to_csv(run_dir / "trades_today.csv", index=False)

    # 4. Daily summary (JSON)
    daily_summary = {
        "date": result.date.strftime("%Y-%m-%d"),
        "equity": state.equity,
        "cash": state.cash,
        "positions_value": positions_value,
        "daily_return_pct": result.daily_return_pct,
        "daily_pnl": result.daily_pnl,
        "trades_count": result.trades_count,
        "buy_count": result.buy_count,
        "sell_count": result.sell_count,
        "turnover": 0.0,  # TODO(#future): Compute from orders (see qa.metrics module)
        "sharpe_daily": None,  # TODO(#future): Compute from equity curve history
        "max_drawdown": None,  # TODO(#future): Compute from equity curve history
        "positions_count": len(state.positions),
        "status": result.status,
        "error_message": result.error_message,
    }
    with open(run_dir / "daily_summary.json", "w", encoding="utf-8") as f:
        json.dump(daily_summary, f, indent=2, ensure_ascii=True)

    # 5. Daily summary (Markdown)
    with open(run_dir / "daily_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Daily Summary - {result.date.strftime('%Y-%m-%d')}\n\n")
        f.write("## Portfolio\n")
        f.write(f"- Equity: ${state.equity:,.2f}\n")
        f.write(f"- Cash: ${state.cash:,.2f}\n")
        f.write(f"- Positions Value: ${positions_value:,.2f}\n")
        f.write(f"- Daily Return: {result.daily_return_pct:+.2f}%\n")
        f.write(f"- Daily PnL: ${result.daily_pnl:+,.2f}\n\n")
        f.write("## Trading\n")
        f.write(
            f"- Trades: {result.trades_count} ({result.buy_count} BUY, {result.sell_count} SELL)\n"
        )
        f.write(f"- Positions: {len(state.positions)}\n\n")
        f.write("## Performance\n")
        total_return_pct = daily_summary.get(
            "total_return_pct",
            (state.total_pnl / state.seed_capital * 100.0)
            if state.seed_capital > 0
            else 0.0,
        )
        f.write(f"- Total Return: {total_return_pct:+.2f}%\n")
        f.write(f"- Total PnL: ${state.total_pnl:+,.2f}\n")
        if result.status == "error":
            f.write("\n## Error\n")
            f.write(f"- Status: {result.status}\n")
            f.write(f"- Error: {result.error_message}\n")

    logger.info(f"Wrote paper day outputs to {run_dir}")
