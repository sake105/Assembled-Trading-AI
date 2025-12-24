# src/assembled_core/pipeline/backtest.py
"""Backtest simulation (equity curve without costs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.config import OUTPUT_DIR


def _simulate_fills_per_order(
    orders_at_timestamp: pd.DataFrame,
    cash: float,
    positions: dict[str, float],
    spread_w: float = 0.0,
    impact_w: float = 0.0,
    commission_bps: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Simulate order fills for a single timestamp using vectorized numpy operations.

    This function executes all orders at a given timestamp using vectorized numpy
    operations instead of per-order loops. It computes fill prices, fees, and notional
    values vectorially.

    Args:
        orders_at_timestamp: DataFrame with columns: symbol, side, qty, price
            Orders to execute at this timestamp
        cash: Current cash balance
        positions: Dictionary mapping symbol -> quantity (current positions)
        spread_w: Spread weight (multiplier for bid/ask spread, default: 0.0 = no costs)
        impact_w: Market impact weight (multiplier for price impact, default: 0.0 = no costs)
        commission_bps: Commission in basis points (default: 0.0 = no costs)

    Returns:
        Tuple of (updated_cash, updated_positions)
        - updated_cash: Cash balance after executing all orders
        - updated_positions: Positions dictionary after executing all orders

    Note:
        This function uses vectorized numpy operations for performance.
        Edge cases handled: buy/sell sign, NaNs, qty=0, invalid symbols.
    """
    if orders_at_timestamp.empty:
        return cash, positions.copy()

    # Extract numpy arrays from DataFrame columns
    symbols = orders_at_timestamp["symbol"].values
    sides = orders_at_timestamp["side"].values
    qtys = orders_at_timestamp["qty"].values.astype(np.float64)
    prices = orders_at_timestamp["price"].values.astype(np.float64)

    # Edge case: Filter out invalid orders (NaN prices, zero qty, empty symbols)
    valid_mask = (
        ~np.isnan(prices)
        & (qtys != 0.0)
        & (qtys > 0.0)  # qty should be positive
        & (symbols != "")
        & (symbols != None)  # noqa: E711
    )

    if not np.any(valid_mask):
        return cash, positions.copy()

    # Filter to valid orders only
    symbols = symbols[valid_mask]
    sides = sides[valid_mask]
    qtys = qtys[valid_mask]
    prices = prices[valid_mask]

    # Convert side strings to numeric signs (BUY=+1, SELL=-1, else=0)
    # Using vectorized string comparison
    is_buy = sides == "BUY"
    is_sell = sides == "SELL"
    signs = np.where(is_buy, 1.0, np.where(is_sell, -1.0, 0.0))

    # Filter out orders with invalid sides (sign=0 means neither BUY nor SELL)
    valid_side_mask = signs != 0.0
    if not np.any(valid_side_mask):
        return cash, positions.copy()

    # Apply side filter
    symbols = symbols[valid_side_mask]
    sides = sides[valid_side_mask]
    qtys = qtys[valid_side_mask]
    prices = prices[valid_side_mask]
    signs = signs[valid_side_mask]
    is_buy = is_buy[valid_side_mask]
    is_sell = is_sell[valid_side_mask]

    # Compute notional (qty * price) for each order
    notionals = qtys * prices

    # Compute fill prices with costs (if costs are enabled)
    # BUY pays: price * (1 + spread_w + impact_w)
    # SELL receives: price * (1 - spread_w - impact_w)
    spread_factor = spread_w * 1e-4  # Convert to decimal
    impact_factor = impact_w * 1e-4
    commission_factor = commission_bps * 1e-4

    fill_prices = np.where(
        is_buy,
        prices * (1.0 + spread_factor + impact_factor),
        prices * (1.0 - spread_factor - impact_factor),
    )

    # Compute fees (commission on notional)
    fees = notionals * commission_factor

    # Compute cash deltas vectorially
    # BUY: cash decreases by (qty * fill_price + fee)
    # SELL: cash increases by (qty * fill_price - fee)
    cash_deltas = np.where(
        is_buy,
        -(qtys * fill_prices + fees),
        +(qtys * fill_prices - fees),
    )

    # Aggregate cash delta (sum of all orders)
    total_cash_delta = np.sum(cash_deltas)
    updated_cash = cash + total_cash_delta

    # Compute position deltas vectorially
    # BUY: position increases by qty
    # SELL: position decreases by qty
    position_deltas = signs * qtys

    # Aggregate position deltas by symbol (using pandas groupby for efficiency)
    # Convert to DataFrame for groupby, then back to dict
    position_df = pd.DataFrame(
        {"symbol": symbols, "qty_delta": position_deltas}
    )
    position_agg = position_df.groupby("symbol")["qty_delta"].sum().to_dict()

    # Update positions dictionary
    updated_positions = positions.copy()
    for symbol, qty_delta in position_agg.items():
        updated_positions[symbol] = updated_positions.get(symbol, 0.0) + qty_delta

    return updated_cash, updated_positions


def _update_equity_mark_to_market(
    timestamp: pd.Timestamp,
    cash: float,
    positions: dict[str, float],
    price_pivot: pd.DataFrame,
    symbols: list[str],
    use_numba: bool = True,
) -> float:
    """Update equity via mark-to-market for a single timestamp using vectorized operations.

    This function computes the portfolio equity by marking all positions to market prices.
    Uses vectorized numpy operations: equity = cash + sum(position_shares * price).
    Optionally uses Numba-accelerated kernel if available.

    Args:
        timestamp: Current timestamp
        cash: Current cash balance
        positions: Dictionary mapping symbol -> quantity (current positions)
        price_pivot: Pivoted DataFrame with timestamp index and symbol columns (close prices)
        symbols: List of all symbols in the portfolio (must be sorted and stable)
        use_numba: If True, attempt to use Numba-accelerated path (default: True)
            Falls back to pure NumPy if Numba is not available

    Returns:
        Equity value (cash + mark-to-market value of positions)

    Note:
        This function uses vectorized numpy operations for performance.
        The calculation is: equity = cash + sum(position_shares * price) for all symbols.
        If Numba is available and use_numba=True, uses Numba-accelerated kernel.
    """
    if timestamp not in price_pivot.index:
        return cash  # Falls Lücke in Preisdaten

    # Extract price row as numpy array (aligned with symbols)
    row = price_pivot.loc[timestamp]
    prices_array = np.array([float(row.get(sym, np.nan)) for sym in symbols], dtype=np.float64)

    # Extract position quantities as numpy array (aligned with symbols)
    positions_array = np.array([positions.get(sym, 0.0) for sym in symbols], dtype=np.float64)

    # Try Numba-accelerated path if available and requested
    if use_numba:
        try:
            from src.assembled_core.qa.numba_kernels import (
                NUMBA_AVAILABLE,
                compute_mark_to_market_numba,
            )

            if NUMBA_AVAILABLE:
                mtm = compute_mark_to_market_numba(positions_array, prices_array)
                equity = cash + mtm
                return equity
        except (ImportError, AttributeError):
            # Fall through to pure NumPy implementation
            pass

    # Pure NumPy implementation (fallback or if use_numba=False)
    # Vectorized mark-to-market: sum(position_shares * price)
    # Filter out NaN prices (missing data)
    valid_mask = ~np.isnan(prices_array)
    mtm = np.sum(positions_array[valid_mask] * prices_array[valid_mask])

    equity = cash + mtm
    return equity


def simulate_equity(
    prices: pd.DataFrame, orders: pd.DataFrame, start_capital: float
) -> pd.DataFrame:
    """Simulate equity curve from prices and orders (without costs).

    Args:
        prices: DataFrame with columns: timestamp, symbol, close
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        start_capital: Starting capital

    Returns:
        DataFrame with columns: timestamp, equity
        Sorted by timestamp

    Side effects:
        None (pure function)
    """
    # Timeline & Price pivot
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    timeline = prices["timestamp"].sort_values().drop_duplicates().to_list()
    symbols = sorted(prices["symbol"].unique().tolist())
    px = prices.pivot(index="timestamp", columns="symbol", values="close").sort_index()

    # Orders nach Timestamp gruppieren
    orders = orders.sort_values("timestamp").reset_index(drop=True)
    orders_by_ts = {}
    if len(orders):
        for ts, group in orders.groupby("timestamp"):
            orders_by_ts[ts] = group

    # Simulationszustand
    cash = float(start_capital)
    pos = {s: 0.0 for s in symbols}
    equity_series = []

    # Per-timestamp loop: execute orders and update equity
    # Note: This loop must remain because positions change over time (orders update positions).
    # However, the equity calculation within the loop is fully vectorized.
    for ts in timeline:
        # Execute orders at this timestamp (vectorized per-order processing)
        if ts in orders_by_ts:
            cash, pos = _simulate_fills_per_order(orders_by_ts[ts], cash, pos)

        # Mark-to-market (vectorized: equity = cash + sum(position_shares * price))
        equity = _update_equity_mark_to_market(ts, cash, pos, px, symbols)
        equity_series.append((ts, float(equity)))

    eq = pd.DataFrame(equity_series, columns=["timestamp", "equity"])
    # Sanitizing
    s = pd.Series(eq["equity"].values, index=eq["timestamp"])
    s = s.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_capital)
    eq["equity"] = s.values
    return eq


def compute_metrics(equity: pd.DataFrame) -> dict[str, float | int]:
    """Compute performance metrics from equity curve.

    Args:
        equity: DataFrame with columns: timestamp, equity

    Returns:
        Dictionary with keys: final_pf, sharpe, rows, first, last
        final_pf: Final performance factor (equity[-1] / equity[0])
        sharpe: Sharpe ratio (simple calculation)
        rows: Number of rows
        first: First timestamp
        last: Last timestamp
    """
    pf = float(equity["equity"].iloc[-1] / max(equity["equity"].iloc[0], 1e-12))
    ret = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(ret.mean() / (ret.std() + 1e-12)) if not ret.empty else float("nan")

    return {
        "final_pf": pf,
        "sharpe": sharpe,
        "rows": len(equity),
        "first": equity["timestamp"].iloc[0],
        "last": equity["timestamp"].iloc[-1],
    }


def write_backtest_report(
    equity: pd.DataFrame,
    metrics: dict[str, float | int],
    freq: str,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write backtest results to CSV and Markdown report.

    Args:
        equity: DataFrame with columns: timestamp, equity
        metrics: Dictionary with performance metrics
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)

    Returns:
        Tuple of (equity_curve_path, report_path)

    Side effects:
        Creates output directory if it doesn't exist
        Writes CSV file: output_dir/equity_curve_{freq}.csv
        Writes Markdown file: output_dir/performance_report_{freq}.md
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_path = out_dir / f"equity_curve_{freq}.csv"
    rep_path = out_dir / f"performance_report_{freq}.md"

    equity.to_csv(curve_path, index=False)

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"# Performance Report ({freq})\n\n")
        f.write(f"- Final PF: {metrics['final_pf']:.4f}\n")
        f.write(f"- Sharpe: {metrics['sharpe']:.4f}\n")
        f.write(f"- Rows: {metrics['rows']}\n")
        f.write(f"- First: {metrics['first']}\n")
        f.write(f"- Last:  {metrics['last']}\n")

    return curve_path, rep_path
