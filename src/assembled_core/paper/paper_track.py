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

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import warnings
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
from src.assembled_core.qa.backtest_engine import _update_positions_vectorized
from src.assembled_core.qa.metrics import (
    compute_drawdown,
    compute_turnover,
    deflated_sharpe_ratio_from_returns,
    compute_sharpe_ratio,
)
from src.assembled_core.qa.point_in_time_checks import (
    check_features_pit_safe,
)
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.paper.strategy_adapters import (
    generate_signals_and_targets_for_day,
)
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    run_trading_cycle,
)

logger = logging.getLogger(__name__)


def _get_git_commit_hash() -> str | None:
    """Try to get current git commit hash (optional, returns None if git unavailable).

    Returns:
        Commit hash (short, 7 chars) if git is available, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _compute_config_hash(config: PaperTrackConfig) -> str:
    """Compute a hash of the config for versioning/tracking.

    Args:
        config: PaperTrackConfig instance

    Returns:
        Short hash string (first 8 chars of SHA256)
    """
    # Serialize key config fields (exclude paths that might vary)
    config_dict = {
        "strategy_name": config.strategy_name,
        "strategy_type": config.strategy_type,
        "freq": config.freq,
        "seed_capital": config.seed_capital,
        "commission_bps": config.commission_bps,
        "spread_w": config.spread_w,
        "impact_w": config.impact_w,
        "strategy_params": sorted(config.strategy_params.items())
        if isinstance(config.strategy_params, dict)
        else config.strategy_params,
        "random_seed": config.random_seed,
        "output_format": config.output_format,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


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
        output_format: Format for aggregated artifacts ("csv" or "parquet", default: "csv")
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
    output_format: Literal["csv", "parquet"] = "csv"


@dataclass
class PaperTrackState:
    """State of paper track portfolio.

    Attributes:
        strategy_name: Name of the strategy (for validation)
        last_run_date: Last date when paper track was executed (pd.Timestamp, UTC, None if never run)
        version: Version of state format (default: "2.0")
        positions: DataFrame with columns: symbol, qty (positive = long, negative = short)
        cash: Current cash balance (float)
        equity: Current portfolio equity (cash + mark-to-market positions)
        seed_capital: Original seed capital (for reference)
        created_at: Timestamp when state was first created (pd.Timestamp, UTC)
        updated_at: Timestamp when state was last updated (pd.Timestamp, UTC)
        total_trades: Total number of trades executed since start (default: 0)
        total_pnl: Cumulative PnL (equity - seed_capital, default: 0.0)
        last_equity: Previous day's equity value (for tracking, default: None)
        last_positions_value: Previous day's positions value (for tracking, default: None)
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
    last_equity: float | None = None  # Previous day's equity (v2.0+)
    last_positions_value: float | None = None  # Previous day's positions value (v2.0+)


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
    daily_turnover: float = 0.0  # Daily turnover (sum(abs(trade_notional)) / equity_prev)
    gross_exposure: float = 0.0  # Gross exposure in currency
    net_exposure: float = 0.0  # Net exposure in currency
    gross_exposure_pct: float = 0.0  # Gross exposure as % of equity
    net_exposure_pct: float = 0.0  # Net exposure as % of equity
    n_symbols_requested: int = 0  # Number of symbols in universe
    n_tradeable: int = 0  # Number of symbols with valid data at as_of
    n_missing: int = 0  # Number of symbols without data or with NaN prices
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

    Supports unified env var `PAPER_TRACK_STRICT_PIT_CHECKS` (preferred) and
    legacy `PAPER_TRACK_STRICT_PIT` (deprecated).

    Args:
        config: PaperTrackConfig

    Returns:
        True if PIT checks should be enabled, False otherwise
    """
    # Check unified env var first (PAPER_TRACK_STRICT_PIT_CHECKS)
    unified_env = os.environ.get("PAPER_TRACK_STRICT_PIT_CHECKS", "").lower()
    if unified_env in ("1", "true", "yes", "on"):
        return True
    elif unified_env in ("0", "false", "no", "off"):
        return False

    # Check legacy env var (PAPER_TRACK_STRICT_PIT) with deprecation warning
    legacy_env = os.environ.get("PAPER_TRACK_STRICT_PIT", "").lower()
    if legacy_env:
        warnings.warn(
            "Deprecated env var PAPER_TRACK_STRICT_PIT is set. "
            "Please use PAPER_TRACK_STRICT_PIT_CHECKS instead. "
            "Support for PAPER_TRACK_STRICT_PIT will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if legacy_env in ("1", "true", "yes", "on"):
            return True
        elif legacy_env in ("0", "false", "no", "off"):
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
    # Note: groupby puts 'symbol' in the index, so reset_index() (without drop=True) 
    # makes it a column again
    filtered = filtered.groupby("symbol", group_keys=False, dropna=False).last()
    filtered = filtered.reset_index()  # Keep 'symbol' as column (don't drop index)

    return filtered


def filter_tradeable_universe(
    prices_filtered: pd.DataFrame,
    universe_symbols: list[str],
    min_history_days: int = 0,
) -> tuple[pd.DataFrame, int, int, int]:
    """Filter prices to tradeable symbols (exclude NaNs, missing data).

    Args:
        prices_filtered: DataFrame with columns: timestamp, symbol, close, ... (one row per symbol)
        universe_symbols: List of all symbols in universe (requested symbols)
        min_history_days: Minimum number of historical data points required (default: 0, disabled)

    Returns:
        Tuple of:
        - tradeable_prices: DataFrame with valid prices (no NaNs, no missing close)
        - n_symbols_requested: Total number of symbols in universe
        - n_tradeable: Number of symbols with valid data
        - n_missing: Number of symbols without data or with NaN prices
    """
    n_symbols_requested = len(universe_symbols)

    if prices_filtered.empty:
        return (
            pd.DataFrame(columns=prices_filtered.columns if not prices_filtered.empty else ["symbol", "close"]),
            n_symbols_requested,
            0,
            n_symbols_requested,
        )

    # Filter out rows with NaN close prices
    # Also check for missing required columns
    required_cols = ["symbol", "close"]
    if not all(col in prices_filtered.columns for col in required_cols):
        logger.warning(
            f"Missing required columns in prices_filtered: {required_cols}. "
            f"Available: {list(prices_filtered.columns)}"
        )
        return (
            pd.DataFrame(columns=required_cols),
            n_symbols_requested,
            0,
            n_symbols_requested,
        )

    # Filter out NaN close prices
    tradeable = prices_filtered[
        prices_filtered["close"].notna() & np.isfinite(prices_filtered["close"])
    ].copy()

    # Check for symbols in universe that are missing from prices_filtered
    symbols_in_prices = set(tradeable["symbol"].unique())
    symbols_in_universe = set(s.upper().strip() for s in universe_symbols)
    missing_symbols = symbols_in_universe - symbols_in_prices

    n_tradeable = len(tradeable)
    n_missing = n_symbols_requested - n_tradeable

    if n_missing > 0:
        logger.warning(
            f"Missing or invalid data for {n_missing} symbol(s): {sorted(missing_symbols)[:10]}"
            + (f" (showing first 10 of {len(missing_symbols)})" if len(missing_symbols) > 10 else "")
        )

    return tradeable, n_symbols_requested, n_tradeable, n_missing


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


def _generate_signals_and_targets_for_day(
    config: PaperTrackConfig,
    state_before: PaperTrackState,
    prices_full: pd.DataFrame,
    prices_filtered: pd.DataFrame,
    prices_with_features: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper around strategy adapters for use inside paper_track.

    This keeps the core orchestrator thin and reuses existing strategy modules.
    """
    return generate_signals_and_targets_for_day(
        config=config,
        state_before=state_before,
        prices_full=prices_full,
        prices_filtered=prices_filtered,
        prices_with_features=prices_with_features,
        as_of=as_of,
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


def compute_paper_performance_panel(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame | None,
    config: PaperTrackConfig,
    windows: list[int] | None = None,
    n_tests: int | None = None,
) -> pd.DataFrame:
    """Compute rolling performance metrics panel for paper track.

    The panel is computed over rolling windows on the equity curve and includes:
    - rolling Sharpe ratio (annualized)
    - rolling max drawdown (in percent)
    - rolling annualized turnover
    - deflated Sharpe ratio (DSR)

    Args:
        equity_curve: DataFrame with columns: date, timestamp, equity, ...
        trades: Optional trades DataFrame (timestamp, symbol, side, qty, price, ...)
        config: PaperTrackConfig (for freq, seed_capital, strategy_params)
        windows: Rolling window sizes in number of observations (e.g. [63, 252]).
                 If None, defaults to [63, 252] for "1d" and [252, 1260] for "5min".
        n_tests: Effective number of strategy tests for DSR (defaults from config.strategy_params["dsr_n_tests"] or 1).

    Returns:
        DataFrame with columns:
        - date: End date of the rolling window (string YYYY-MM-DD)
        - timestamp: End timestamp (ISO)
        - window: Window size (int)
        - n_obs: Number of return observations in window
        - sharpe: Annualized Sharpe ratio
        - max_drawdown_pct: Max drawdown in percent (negative)
        - turnover_annualized: Annualized turnover (may be None if no trades)
        - deflated_sharpe: Deflated Sharpe ratio (DSR)
    """
    if equity_curve.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "timestamp",
                "window",
                "n_obs",
                "sharpe",
                "max_drawdown_pct",
                "turnover_annualized",
                "deflated_sharpe",
            ]
        )

    df = equity_curve.copy()

    # Ensure required columns
    if "equity" not in df.columns:
        raise ValueError("equity_curve must contain 'equity' column")
    if "timestamp" not in df.columns:
        # Fallback: construct timestamp from date if available
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        else:
            raise ValueError("equity_curve must contain 'timestamp' or 'date' column")

    # Sort by timestamp and reset index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute returns
    equity_series = pd.to_numeric(df["equity"], errors="coerce")
    returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan)

    # Default rolling windows by frequency
    if windows is None:
        if config.freq == "1d":
            windows = [63, 252]  # ~3 months, 1 year
        else:
            # For intraday, use larger windows in terms of bars
            windows = [252, 1260]  # Approx daily equivalents

    # Default n_tests for DSR
    if n_tests is None:
        n_tests = int(config.strategy_params.get("dsr_n_tests", 1) or 1)

    rows: list[dict[str, Any]] = []

    for window in windows:
        if window <= 1:
            continue

        for end_idx in range(window - 1, len(df)):
            start_idx = end_idx - window + 1

            window_returns = returns.iloc[start_idx : end_idx + 1].dropna()
            n_obs = int(len(window_returns))
            if n_obs < max(10, window // 4):
                # Require a minimum number of observations for stable metrics
                continue

            window_equity = equity_series.iloc[start_idx : end_idx + 1]

            # Sharpe ratio (annualized) using daily/5min freq
            sharpe = compute_sharpe_ratio(
                window_returns,
                freq=config.freq,
                risk_free_rate=0.0,
            )

            # Drawdown metrics
            _, _, max_dd_pct, _ = compute_drawdown(window_equity)

            # Turnover (annualized) using trades within window
            turnover_annualized: float | None
            if trades is not None and not trades.empty and "timestamp" in trades.columns:
                start_ts = df.loc[start_idx, "timestamp"]
                end_ts = df.loc[end_idx, "timestamp"]
                trades_window = trades[
                    (trades["timestamp"] >= start_ts)
                    & (trades["timestamp"] <= end_ts)
                ].copy()
                if not trades_window.empty:
                    equity_df = pd.DataFrame(
                        {
                            "timestamp": df.loc[
                                start_idx : end_idx, "timestamp"
                            ].values,
                            "equity": window_equity.values,
                        }
                    )
                    turnover_annualized = compute_turnover(
                        trades_window,
                        equity_df,
                        start_capital=config.seed_capital,
                        freq=config.freq,
                    )
                else:
                    turnover_annualized = None
            else:
                turnover_annualized = None

            # Deflated Sharpe Ratio (DSR)
            dsr = deflated_sharpe_ratio_from_returns(
                window_returns,
                n_tests=n_tests,
                scale="daily",
                risk_free_rate=0.0,
            )

            # Build row
            end_ts = df.loc[end_idx, "timestamp"]
            date_str = (
                df.loc[end_idx, "date"]
                if "date" in df.columns
                else end_ts.strftime("%Y-%m-%d")
            )

            rows.append(
                {
                    "date": date_str,
                    "timestamp": end_ts.isoformat(),
                    "window": int(window),
                    "n_obs": n_obs,
                    "sharpe": float(sharpe) if sharpe is not None else None,
                    "max_drawdown_pct": float(max_dd_pct),
                    "turnover_annualized": float(turnover_annualized)
                    if turnover_annualized is not None
                    else None,
                    "deflated_sharpe": float(dsr)
                    if not (isinstance(dsr, float) and math.isnan(dsr))
                    else None,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "timestamp",
                "window",
                "n_obs",
                "sharpe",
                "max_drawdown_pct",
                "turnover_annualized",
                "deflated_sharpe",
            ]
        )

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["timestamp", "window"]).reset_index(drop=True)
    return panel


def _simulate_order_fills(
    orders: pd.DataFrame,
    current_cash: float,
    commission_bps: float,
    spread_w: float,
    impact_w: float,
) -> tuple[pd.DataFrame, float]:
    """Simulate order fills and update cash.

    Cost model:
    - Spread: Applied to fill price (buy pays more, sell receives less)
      - BUY fill_price = mid_price * (1 + spread_w * 1e-4 + impact_w * 1e-4)
      - SELL fill_price = mid_price * (1 - spread_w * 1e-4 - impact_w * 1e-4)
    - Commission: Applied to notional (qty * mid_price) as basis points
      - costs = commission_bps * 1e-4 * notional
    - Cash delta:
      - BUY: cash decreases by (qty * fill_price + costs)
      - SELL: cash increases by (qty * fill_price - costs)

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        current_cash: Current cash balance
        commission_bps: Commission in basis points (e.g., 0.5 = 0.5 bps)
        spread_w: Spread weight in basis points (e.g., 0.25 = 0.25 bps)
        impact_w: Market impact weight in basis points (e.g., 0.5 = 0.5 bps)

    Returns:
        Tuple of (filled_orders DataFrame with fill_price, costs, cash_delta columns, new_cash balance)
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


def _migrate_state_dict(state_dict: dict[str, Any], from_version: str) -> dict[str, Any]:
    """Migrate state dictionary from older version to current version.

    Args:
        state_dict: State dictionary loaded from JSON
        from_version: Version string (e.g., "1.0")

    Returns:
        Migrated state dictionary (version updated to current)
    """
    current_version = PAPER_TRACK_STATE_VERSION
    migrated = state_dict.copy()

    # Migration from v1.0 to v2.0
    if from_version == "1.0" and current_version == "2.0":
        logger.info(f"Migrating state from v{from_version} to v{current_version}")
        # Add new fields with None defaults
        migrated["last_equity"] = None
        migrated["last_positions_value"] = None
        migrated["version"] = current_version
        logger.debug("State migration v1.0 -> v2.0: Added last_equity, last_positions_value")

    # Future migrations can be added here:
    # if from_version == "2.0" and current_version == "3.0":
    #     ...

    # If already at current version, no migration needed
    if from_version == current_version:
        return migrated

    # If version is newer than current, warn but don't fail
    try:
        from_major = int(from_version.split(".")[0])
        current_major = int(current_version.split(".")[0])
        if from_major > current_major:
            logger.warning(
                f"State file version {from_version} is newer than current {current_version}. "
                f"Some features may not be available."
            )
    except (ValueError, IndexError):
        logger.warning(f"Could not parse version '{from_version}', assuming compatible")

    return migrated


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

        # Get version and migrate if needed
        file_version = data.get("version", "1.0")  # Default to 1.0 for old files
        current_version = PAPER_TRACK_STATE_VERSION

        if file_version != current_version:
            logger.info(
                f"State file version {file_version} differs from current {current_version}, "
                f"applying migration"
            )
            data = _migrate_state_dict(data, file_version)

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

        # Create state (with migrated version and new fields)
        state = PaperTrackState(
            strategy_name=data["strategy_name"],
            last_run_date=last_run_date,
            version=data.get("version", current_version),  # Use migrated version
            positions=positions,
            cash=float(data["cash"]),
            equity=float(data["equity"]),
            seed_capital=float(data.get("seed_capital", 100000.0)),
            created_at=created_at,
            updated_at=updated_at,
            total_trades=int(data.get("total_trades", 0)),
            total_pnl=float(data.get("total_pnl", 0.0)),
            last_equity=data.get("last_equity"),  # v2.0+ field
            last_positions_value=data.get("last_positions_value"),  # v2.0+ field
        )

        logger.info(f"Loaded paper state from {state_path} (version: {state.version})")
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

    # Ensure state version is current (migrate if needed)
    if state.version != PAPER_TRACK_STATE_VERSION:
        logger.warning(
            f"State version {state.version} differs from current {PAPER_TRACK_STATE_VERSION}, "
            f"updating to current version"
        )
        state.version = PAPER_TRACK_STATE_VERSION

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
        "last_equity": state.last_equity,  # v2.0+ field
        "last_positions_value": state.last_positions_value,  # v2.0+ field
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
    # Note: parent directory will be created by save_paper_state if needed
    if state_path is not None and state_path.exists():
        # Only validate if file exists but is invalid
        if not state_path.is_file():
            raise ValueError(f"state_path exists but is not a file: {state_path}")

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
        # Step 1: Load universe symbols
        universe_symbols: list[str] = []
        universe_file_path = Path(config.universe_file) if isinstance(config.universe_file, str) else config.universe_file
        if universe_file_path.exists():
            with open(universe_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        universe_symbols.append(line.upper())
        else:
            logger.warning(f"Universe file not found: {universe_file_path}")
            universe_symbols = []

        # Step 2: Load prices (filtered to as_of)
        logger.info(f"Loading prices for {as_of.date()}")
        prices = load_eod_prices_for_universe(
            universe_file=config.universe_file,
            freq=config.freq,
        )
        prices_filtered = _filter_prices_for_date(prices, as_of)

        # Step 3: Filter to tradeable universe (exclude NaNs, missing data)
        prices_tradeable, n_symbols_requested, n_tradeable, n_missing = filter_tradeable_universe(
            prices_filtered=prices_filtered,
            universe_symbols=universe_symbols,
            min_history_days=0,  # TODO: Make configurable via config
        )

        if prices_tradeable.empty:
            logger.warning(
                f"No tradeable symbols for date {as_of.date()} "
                f"(requested: {n_symbols_requested}, tradeable: {n_tradeable}, missing: {n_missing})"
            )
            # Continue with empty prices_tradeable - will result in no orders

        # Step 4-7: Run unified trading cycle (features/signals/positions/orders)
        logger.debug("Running trading cycle (features/signals/positions/orders)")
        
        # Build signal and sizing functions from strategy adapter
        # We need to wrap the strategy adapter to work with trading_cycle
        def signal_fn(df_with_features: pd.DataFrame) -> pd.DataFrame:
            """Signal function wrapper for trading_cycle."""
            # The strategy adapter expects full context, but trading_cycle only provides
            # prices_with_features. We'll use a simplified version that extracts signals.
            # For now, delegate to strategy adapter with minimal context.
            signals, _ = _generate_signals_and_targets_for_day(
                config=config,
                state_before=state_before,
                prices_full=prices,
                prices_filtered=prices_tradeable,
                prices_with_features=df_with_features,
                as_of=as_of,
            )
            return signals
        
        def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
            """Position sizing function wrapper for trading_cycle."""
            # For trend_baseline, we can compute target positions directly from signals
            # For other strategies, we'll need to call the strategy adapter with features
            if config.strategy_type == "trend_baseline":
                from src.assembled_core.portfolio.position_sizing import (
                    compute_target_positions_from_trend_signals,
                )
                params = config.strategy_params or {}
                return compute_target_positions_from_trend_signals(
                    signals,
                    total_capital=capital,
                    top_n=params.get("top_n"),
                    min_score=params.get("min_score", 0.0),
                )
            else:
                # For other strategies (e.g., multifactor), we need to call the adapter
                # We'll need prices_with_features, so we compute them again (cached in cycle)
                # This is a limitation, but acceptable for now
                prices_with_features_temp = _compute_features_for_strategy(config, prices_tradeable)
                _, target_positions = _generate_signals_and_targets_for_day(
                    config=config,
                    state_before=state_before,
                    prices_full=prices,
                    prices_filtered=prices_tradeable,
                    prices_with_features=prices_with_features_temp,
                    as_of=as_of,
                )
                # Filter to signals that are in the input
                if not signals.empty and not target_positions.empty:
                    signal_symbols = set(signals["symbol"].unique())
                    target_positions = target_positions[
                        target_positions["symbol"].isin(signal_symbols)
                    ].copy()
                return target_positions
        
        # Build TradingContext
        current_positions = state_before.positions.copy()
        ctx = TradingContext(
            prices=prices_tradeable,  # Tradeable prices (already filtered)
            as_of=as_of,
            freq=config.freq,
            universe=universe_symbols if universe_symbols else None,
            use_factor_store=False,  # Paper track doesn't use factor store yet
            factor_store_root=None,
            factor_group="core_ta",
            feature_config={
                "ma_windows": (
                    config.strategy_params.get("ma_fast", DEFAULT_MA_WINDOWS[0]),
                    config.strategy_params.get("ma_slow", DEFAULT_MA_WINDOWS[1]),
                ),
                "atr_window": DEFAULT_ATR_WINDOW,
                "rsi_window": DEFAULT_RSI_WINDOW,
                "include_rsi": True,
            },
            signal_fn=signal_fn,
            signal_config=config.strategy_params,
            position_sizing_fn=sizing_fn,
            capital=state_before.equity,  # Use current equity as capital
            current_positions=current_positions
            if not current_positions.empty
            else None,
            order_timestamp=as_of,
            enable_risk_controls=False,  # Paper track handles risk separately
            risk_config={},
            output_dir=None,  # No outputs from trading_cycle (we handle outputs)
            output_format="none",
            write_outputs=False,  # Pure function
            run_id=None,
            strategy_name=config.strategy_name,
            logger=logger,
            timings=None,
        )
        
        # Run trading cycle
        cycle_result = run_trading_cycle(ctx)
        
        if cycle_result.status != "success":
            raise ValueError(f"Trading cycle failed: {cycle_result.error_message}")
        
        # PIT check (if enabled) - now on cycle_result.prices_with_features
        enable_pit = _should_enable_pit_checks(config)
        if enable_pit:
            check_features_pit_safe(
                cycle_result.prices_with_features,
                as_of=as_of,
                timestamp_col="timestamp",
                strict=True,
                feature_source="paper_track",
            )
            logger.debug("PIT checks passed")
        
        # Extract results (cycle_result.orders already generated from targets)
        orders = cycle_result.orders

        # Step 8: Simulate fills
        logger.debug("Simulating order fills")
        filled_orders, new_cash = _simulate_order_fills(
            orders,
            state_before.cash,
            config.commission_bps,
            config.spread_w,
            config.impact_w,
        )

        # Step 9: Update positions
        logger.debug("Updating positions")
        updated_positions = _update_positions_vectorized(
            filled_orders[["timestamp", "symbol", "side", "qty", "price"]],
            current_positions,
            use_numba=True,
        )

        # Step 10: Compute new equity (mark-to-market)
        position_value = _compute_position_value(updated_positions, prices_tradeable)
        new_equity = new_cash + position_value

        # Step 9: Create updated state
        now = pd.Timestamp.utcnow()
        # Store previous day's values for tracking (v2.0+)
        last_equity = state_before.equity
        last_positions_value = state_before.equity - state_before.cash

        state_after = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=as_of,
            version=state_before.version,  # Will be updated to current on save
            positions=updated_positions,
            cash=new_cash,
            equity=new_equity,
            seed_capital=state_before.seed_capital,
            created_at=state_before.created_at,
            updated_at=now,
            total_trades=state_before.total_trades + len(filled_orders),
            total_pnl=new_equity - state_before.seed_capital,
            last_equity=last_equity,  # v2.0+ field
            last_positions_value=last_positions_value,  # v2.0+ field
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

        # Compute turnover: sum(abs(trade_notional)) / previous equity
        if not filled_orders.empty:
            trade_notionals = (
                filled_orders["qty"].abs() * filled_orders["fill_price"].abs()
            )
            total_trade_notional = float(trade_notionals.sum())
            daily_turnover = (
                total_trade_notional / state_before.equity
                if state_before.equity > 0
                else 0.0
            )
        else:
            daily_turnover = 0.0

        # Compute gross/net exposure from positions
        if not updated_positions.empty and not prices_tradeable.empty:
            # Get latest prices for each symbol
            latest_prices = (
                prices_tradeable.groupby("symbol")["close"].last().to_dict()
            )
            # Calculate position values
            positions_with_prices = updated_positions.copy()
            positions_with_prices["price"] = positions_with_prices["symbol"].map(
                latest_prices
            )
            positions_with_prices["value"] = (
                positions_with_prices["qty"] * positions_with_prices["price"]
            )

            # Gross exposure = sum of absolute position values
            gross_exposure = float(positions_with_prices["value"].abs().sum())
            # Net exposure = sum of position values (can be negative)
            net_exposure = float(positions_with_prices["value"].sum())
            # Normalize by equity
            gross_exposure_pct = (
                (gross_exposure / new_equity * 100.0) if new_equity > 0 else 0.0
            )
            net_exposure_pct = (
                (net_exposure / new_equity * 100.0) if new_equity > 0 else 0.0
            )
        else:
            gross_exposure = 0.0
            net_exposure = 0.0
            gross_exposure_pct = 0.0
            net_exposure_pct = 0.0

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
            daily_turnover=float(daily_turnover),
            gross_exposure=float(gross_exposure),
            net_exposure=float(net_exposure),
            gross_exposure_pct=float(gross_exposure_pct),
            net_exposure_pct=float(net_exposure_pct),
            n_symbols_requested=n_symbols_requested,
            n_tradeable=n_tradeable,
            n_missing=n_missing,
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
            daily_turnover=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            gross_exposure_pct=0.0,
            net_exposure_pct=0.0,
            n_symbols_requested=0,
            n_tradeable=0,
            n_missing=0,
            status="error",
            error_message=str(e),
        )


def write_paper_day_outputs(
    result: PaperTrackDayResult, output_dir: Path, config: PaperTrackConfig | None = None, run_id: str | None = None
) -> None:
    """Write daily outputs for paper track run.

    Args:
        result: PaperTrackDayResult
        output_dir: Output directory (will create runs/{YYYYMMDD}/ subdirectory)
        config: Optional PaperTrackConfig (for output_format, defaults to "csv" if None)

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
        "turnover": result.daily_turnover,
        "gross_exposure": result.gross_exposure,
        "net_exposure": result.net_exposure,
        "gross_exposure_pct": result.gross_exposure_pct,
        "net_exposure_pct": result.net_exposure_pct,
        "n_symbols_requested": result.n_symbols_requested,
        "n_tradeable": result.n_tradeable,
        "n_missing": result.n_missing,
        "sharpe_daily": None,  # TODO(#future): Compute from equity curve history
        "max_drawdown": None,  # TODO(#future): Compute from equity curve history
        "positions_count": len(state.positions),
        "status": result.status,
        "error_message": result.error_message,
    }
    # Add manifest reference to daily_summary.json (for convenience)
    daily_summary["manifest_path"] = "manifest.json"
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
        f.write(f"- Positions: {len(state.positions)}\n")
        f.write(f"- Daily Turnover: {result.daily_turnover:.4f}\n")
        f.write(f"- Gross Exposure: ${result.gross_exposure:,.2f} ({result.gross_exposure_pct:.2f}%)\n")
        f.write(f"- Net Exposure: ${result.net_exposure:,.2f} ({result.net_exposure_pct:.2f}%)\n\n")
        f.write("## Universe\n")
        f.write(f"- Symbols Requested: {result.n_symbols_requested}\n")
        f.write(f"- Tradeable: {result.n_tradeable}\n")
        f.write(f"- Missing/Invalid: {result.n_missing}\n")
        if result.n_missing > 0:
            f.write(f"  ⚠️  Warning: {result.n_missing} symbol(s) have missing data or NaN prices\n")
        f.write("\n")
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
        f.write("\n## Metadata\n")
        f.write("- See `manifest.json` for run metadata, config hash, and git commit hash.\n")

    logger.info(f"Wrote paper day outputs to {run_dir}")

    # 6. Write run manifest (metadata for this run)
    _write_run_manifest(result, run_dir, config, run_id=run_id)

    # 7. Write aggregated artifacts (only if run was successful)
    if result.status == "success":
        output_format = config.output_format if config else "csv"
        _write_aggregated_artifacts(result, output_dir, output_format=output_format)


def _write_run_manifest(
    result: PaperTrackDayResult, run_dir: Path, config: PaperTrackConfig | None, run_id: str | None = None
) -> None:
    """Write run manifest.json with metadata for this paper track run.

    Args:
        result: PaperTrackDayResult
        run_dir: Run directory (runs/{YYYYMMDD}/)
        config: Optional PaperTrackConfig (for config hash)

    Side effects:
        Creates manifest.json in run_dir with metadata about the run.
    """
    state_before = result.state_before
    state_after = result.state_after

    # Build artifacts list
    artifacts = [
        "equity_snapshot.json",
        "positions.csv",
        "orders_today.csv",
        "trades_today.csv",
        "daily_summary.json",
        "daily_summary.md",
        "manifest.json",
    ]

    # Compute config hash if config available
    config_hash = None
    config_path = None
    if config:
        config_hash = _compute_config_hash(config)
        # Try to find config file path (might be set via env or passed)
        # For now, just note that config was used
        config_path = "config_from_api"

    # Try to get git commit hash (optional)
    git_hash = _get_git_commit_hash()

    # Build manifest
    manifest = {
        "date": result.date.strftime("%Y-%m-%d"),
        "timestamp": result.date.isoformat(),
        "run_id": run_id,  # Run-ID from runner execution (if provided)
        "strategy_name": state_after.strategy_name if state_after else (config.strategy_name if config else "unknown"),
        "run_directory": str(run_dir.name),  # Just the YYYYMMDD part
        "config_hash": config_hash,
        "config_path": config_path,
        "git_commit_hash": git_hash,
        "state_before": {
            "equity": float(state_before.equity) if state_before else None,
            "cash": float(state_before.cash) if state_before else None,
            "positions_count": len(state_before.positions) if state_before and not state_before.positions.empty else 0,
            "last_run_date": state_before.last_run_date.isoformat() if state_before and state_before.last_run_date else None,
        },
        "state_after": {
            "equity": float(state_after.equity) if state_after else None,
            "cash": float(state_after.cash) if state_after else None,
            "positions_count": len(state_after.positions) if state_after and not state_after.positions.empty else 0,
            "last_run_date": state_after.last_run_date.isoformat() if state_after and state_after.last_run_date else None,
        },
        "run_summary": {
            "status": result.status,
            "daily_return_pct": result.daily_return_pct,
            "daily_pnl": result.daily_pnl,
            "trades_count": result.trades_count,
            "buy_count": result.buy_count,
            "sell_count": result.sell_count,
        },
        "artifacts": artifacts,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }

    # Add error message if present
    if result.error_message:
        manifest["error_message"] = result.error_message

    # Write manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    logger.debug(f"Wrote run manifest to {manifest_path}")


def _write_aggregated_artifacts(
    result: PaperTrackDayResult, output_dir: Path, output_format: Literal["csv", "parquet"] = "csv"
) -> None:
    """Write aggregated artifacts (equity_curve, trades_all, positions_history).

    This function appends/merges daily data into aggregated files:
    - equity_curve.{csv|parquet}: Complete equity curve over time
    - trades_all.{csv|parquet}: All trades across all days
    - positions_history.{csv|parquet}: Historical positions snapshots

    Args:
        result: PaperTrackDayResult (must have status == "success")
        output_dir: Base output directory (e.g., output/paper_track/{strategy_name})
        output_format: Output format for aggregated files ("csv" or "parquet", default: "csv")

    Side effects:
        Creates/updates files in output_dir/aggregates/:
        - equity_curve.{csv|parquet}
        - trades_all.{csv|parquet}
        - positions_history.{csv|parquet}
    """
    import logging

    logger = logging.getLogger(__name__)
    aggregates_dir = output_dir / "aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    state = result.state_after
    date = result.date
    date_str = date.strftime("%Y-%m-%d")

    # 1. Equity curve (append new row, dedup by date)
    equity_curve_path = aggregates_dir / f"equity_curve.{output_format}"
    positions_value = state.equity - state.cash

    new_equity_row = pd.DataFrame(
        [
            {
                "date": date_str,
                "timestamp": date.isoformat(),
                "equity": state.equity,
                "cash": state.cash,
                "positions_value": positions_value,
                "total_pnl": state.total_pnl,
                "total_return_pct": (
                    (state.total_pnl / state.seed_capital * 100.0)
                    if state.seed_capital > 0
                    else 0.0
                ),
                "daily_return_pct": result.daily_return_pct,
                "daily_pnl": result.daily_pnl,
            }
        ]
    )

    if equity_curve_path.exists():
        # Load existing file based on format
        if output_format == "parquet":
            existing = pd.read_parquet(equity_curve_path)
        else:
            existing = pd.read_csv(equity_curve_path)
        # Remove duplicate for this date (if rerun)
        existing = existing[existing["date"] != date_str]
        # Append new row
        equity_curve = pd.concat([existing, new_equity_row], ignore_index=True)
    else:
        equity_curve = new_equity_row

    # Sort by date and write (atomic)
    equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
    temp_path = equity_curve_path.with_suffix(f".tmp.{output_format}")
    if output_format == "parquet":
        equity_curve.to_parquet(temp_path, index=False)
    else:
        equity_curve.to_csv(temp_path, index=False)
    temp_path.replace(equity_curve_path)

    # 2. Trades all (append trades from today, dedup by date)
    trades_all_path = aggregates_dir / f"trades_all.{output_format}"

    if not result.orders.empty:
        trades_today = result.orders.copy()
        trades_today["date"] = date_str
        # Reorder columns: date first, then original columns
        cols = ["date"] + [c for c in trades_today.columns if c != "date"]
        trades_today = trades_today[cols]
    else:
        trades_today = pd.DataFrame(columns=["date"])

    if trades_all_path.exists():
        # Load existing file based on format
        if output_format == "parquet":
            existing = pd.read_parquet(trades_all_path)
        else:
            existing = pd.read_csv(trades_all_path)
        # Remove duplicates for this date (if rerun)
        if "date" in existing.columns:
            existing = existing[existing["date"] != date_str]
        # Append new trades
        if not trades_today.empty:
            trades_all = pd.concat([existing, trades_today], ignore_index=True)
        else:
            trades_all = existing
    else:
        trades_all = trades_today

    # Sort by date, timestamp and write (atomic)
    if not trades_all.empty:
        if "timestamp" in trades_all.columns:
            trades_all = trades_all.sort_values(["date", "timestamp"]).reset_index(
                drop=True
            )
        else:
            trades_all = trades_all.sort_values("date").reset_index(drop=True)
    temp_path = trades_all_path.with_suffix(f".tmp.{output_format}")
    if output_format == "parquet":
        trades_all.to_parquet(temp_path, index=False)
    else:
        trades_all.to_csv(temp_path, index=False)
    temp_path.replace(trades_all_path)

    # 3. Positions history (append positions snapshot, dedup by date)
    # Always write at least one row per day (even if no positions)
    positions_history_path = aggregates_dir / f"positions_history.{output_format}"

    if not state.positions.empty:
        positions_today = state.positions.copy()
        positions_today["date"] = date_str
        # Reorder columns: date first, then symbol, qty
        cols = ["date", "symbol", "qty"]
        positions_today = positions_today[cols]
    else:
        # Write empty row with date to track that day was processed
        positions_today = pd.DataFrame([{"date": date_str, "symbol": None, "qty": 0.0}])

    if positions_history_path.exists():
        # Load existing file based on format
        if output_format == "parquet":
            existing = pd.read_parquet(positions_history_path)
        else:
            existing = pd.read_csv(positions_history_path)
        # Remove duplicates for this date (if rerun)
        if "date" in existing.columns:
            existing = existing[existing["date"] != date_str]
        # Append new positions
        if not positions_today.empty:
            positions_history = pd.concat([existing, positions_today], ignore_index=True)
        else:
            positions_history = existing
    else:
        positions_history = positions_today

    # Sort by date, symbol and write (atomic)
    if not positions_history.empty:
        positions_history = positions_history.sort_values(
            ["date", "symbol"]
        ).reset_index(drop=True)
    temp_path = positions_history_path.with_suffix(f".tmp.{output_format}")
    if output_format == "parquet":
        positions_history.to_parquet(temp_path, index=False)
    else:
        positions_history.to_csv(temp_path, index=False)
    temp_path.replace(positions_history_path)

    # 4. Performance metrics panel (rolling metrics over equity_curve/trades_all)
    try:
        performance_panel = compute_paper_performance_panel(
            equity_curve=equity_curve,
            trades=trades_all,
            config=result.config,
        )

        if not performance_panel.empty:
            perf_path = aggregates_dir / f"performance_metrics.{output_format}"

            if perf_path.exists():
                if output_format == "parquet":
                    existing_perf = pd.read_parquet(perf_path)
                else:
                    existing_perf = pd.read_csv(perf_path)
                # Deduplicate by (timestamp, window)
                key_cols = ["timestamp", "window"]
                merged = pd.concat(
                    [existing_perf, performance_panel], ignore_index=True
                )
                merged = merged.drop_duplicates(subset=key_cols, keep="last")
                performance_panel = merged.sort_values(key_cols).reset_index(
                    drop=True
                )

            temp_perf_path = perf_path.with_suffix(f".tmp.{output_format}")
            if output_format == "parquet":
                performance_panel.to_parquet(temp_perf_path, index=False)
            else:
                performance_panel.to_csv(temp_perf_path, index=False)
            temp_perf_path.replace(perf_path)
    except Exception as e:  # pragma: no cover - defensive logging only
        logger.warning(
            f"Failed to update performance_metrics aggregated artifact: {e}",
            exc_info=True,
        )

    logger.debug(f"Updated aggregated artifacts in {aggregates_dir}")

