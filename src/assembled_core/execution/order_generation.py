"""Order generation module.

This module generates orders by comparing current positions to target positions.
It extends the basic order generation from pipeline.orders.

Zukünftige Integration:
- Nutzt pipeline.orders.signals_to_orders als Basis
- Erweitert um Position-Sizing (portfolio.position_sizing)
- Bietet verschiedene Order-Typen (Market, Limit, Stop-Loss)
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.assembled_core.portfolio.position_sizing import compute_target_positions

logger = logging.getLogger(__name__)


def generate_orders_from_targets_fast(
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    timestamp: datetime | None = None,
    prices_latest: pd.DataFrame | None = None,
    min_trade_value: float = 0.0,
) -> pd.DataFrame:
    """Fast-path order generation when target and current positions are already aligned.

    This function avoids expensive pandas merge operations by assuming that:
    - target_positions and current_positions have the same symbols in the same order
    - Both DataFrames are sorted by symbol
    - Columns are minimal: symbol, target_qty (or qty for current)

    This is optimized for performance-critical backtest loops where positions
    are already aligned (e.g., from TradingCycle result).

    Args:
        target_positions: DataFrame with columns: symbol, target_qty (or qty)
            Must be sorted by symbol
        current_positions: Optional DataFrame with columns: symbol, qty
            Must have same symbols in same order as target_positions (or empty)
            If None, assumes all positions are zero
        timestamp: Order timestamp (default: current UTC time)
        prices_latest: Optional DataFrame with columns: symbol, close (one row per symbol)
            For fast price lookup (already aligned/latest per symbol)

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
        side: "BUY" or "SELL"
        qty: Quantity (always positive)
        price: Order price (from prices_latest if available, else 0.0)
        Sorted by symbol

    Raises:
        ValueError: If alignment assumption is violated (symbols don't match)
    """
    if target_positions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Extract columns (handle both "target_qty" and "qty" in target)
    if "target_qty" in target_positions.columns:
        target_qty_col = "target_qty"
    elif "qty" in target_positions.columns:
        target_qty_col = "qty"
    else:
        raise ValueError("target_positions must have 'target_qty' or 'qty' column")

    # Ensure target is sorted by symbol (required for alignment)
    if not target_positions["symbol"].is_monotonic_increasing:
        target_positions = target_positions.sort_values("symbol").reset_index(drop=True)

    # Use current timestamp if not provided
    if timestamp is None:
        timestamp = pd.Timestamp.now("UTC")

    # Extract numpy arrays directly (no merge, no sort)
    symbols = target_positions["symbol"].values
    target_qty_raw = target_positions[target_qty_col].values.astype(np.float64)

    # Get prices for order price field (not for qty conversion —
    # fast-path treats target_qty as SHARES, not notional)
    if (
        prices_latest is not None
        and "close" in prices_latest.columns
        and "symbol" in prices_latest.columns
    ):
        price_map = dict(
            zip(prices_latest["symbol"].values, prices_latest["close"].values)
        )
        prices_array = np.array(
            [price_map.get(sym, 0.0) for sym in symbols], dtype=np.float64
        )
    else:
        prices_array = np.zeros(len(symbols), dtype=np.float64)

    # Fast-path: target_qty is already in SHARES (caller is responsible for conversion)
    target_qty = target_qty_raw  # values are shares

    # Get current quantities in shares (assume aligned if current_positions provided)
    if current_positions is None or current_positions.empty:
        current_qty = np.zeros(len(symbols), dtype=np.float64)
    else:
        # Ensure current is sorted by symbol (required for alignment check)
        if not current_positions["symbol"].is_monotonic_increasing:
            current_positions = current_positions.sort_values("symbol").reset_index(
                drop=True
            )
        # Fast-path: assume same symbols in same order
        if len(current_positions) != len(target_positions):
            raise ValueError(
                "Fast-path requires current_positions to have same length as target_positions"
            )
        current_symbols = current_positions["symbol"].values
        if not np.array_equal(current_symbols, symbols):
            raise ValueError(
                "Fast-path requires current_positions to have same symbols in same order as target_positions"
            )
        current_qty = current_positions["qty"].values.astype(np.float64)

    # Delta in SHARES (target_shares - current_shares)
    qty_delta = target_qty - current_qty

    # Filter for non-zero deltas (vectorized)
    abs_delta = np.abs(qty_delta)
    non_zero_mask = abs_delta > 1e-6

    # Min-trade-value filter: skip trades below threshold (cost > expected benefit)
    if min_trade_value > 0:
        trade_values = abs_delta * prices_array
        non_zero_mask = non_zero_mask & (trade_values >= min_trade_value)

    if not np.any(non_zero_mask):
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Extract non-zero deltas (all in shares now)
    symbols_filtered = symbols[non_zero_mask]
    qty_delta_filtered = qty_delta[non_zero_mask]
    sides = np.where(qty_delta_filtered > 0, "BUY", "SELL")
    qtys = np.abs(qty_delta_filtered)
    prices_filtered = prices_array[non_zero_mask]

    # Build DataFrame directly (no pandas operations except construction)
    # arrival_price (Sprint 2 / C11): snapshot the price at decision time
    # so downstream TCA can compute implementation shortfall.
    result = pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": symbols_filtered,
            "side": sides,
            "qty": qtys,
            "price": prices_filtered,
            "arrival_price": prices_filtered,
        }
    )

    # Ensure columns are in correct order
    result = result[
        ["timestamp", "symbol", "side", "qty", "price", "arrival_price"]
    ]
    result.attrs["qty_unit"] = "shares"
    return result


def generate_orders_from_targets(
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    timestamp: datetime | None = None,
    prices: pd.DataFrame | None = None,
    min_trade_value: float = 0.0,
) -> pd.DataFrame:
    """Generate orders to transition from current to target positions.

    This function compares current positions to target positions and generates
    orders to achieve the target portfolio.

    Args:
        target_positions: DataFrame with columns: symbol, target_weight, target_qty
            (from portfolio.position_sizing.compute_target_positions)
        current_positions: Optional DataFrame with columns: symbol, qty
            If None, assumes all positions are zero (starting from scratch)
        timestamp: Order timestamp (default: current UTC time)
        prices: Optional DataFrame with columns: symbol, close (for price lookup)
            If None, price will be set to 0.0 (must be filled later)

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
        side: "BUY" or "SELL"
        qty: Quantity (always positive)
        price: Order price (from prices if available, else 0.0)
        Sorted by symbol

    Raises:
        ValueError: If required columns are missing
    """
    if target_positions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Ensure required columns
    required = ["symbol", "target_qty"]
    missing = [c for c in required if c not in target_positions.columns]
    if missing:
        raise ValueError(f"Missing required columns in target_positions: {missing}")

    # Use current timestamp if not provided
    if timestamp is None:
        timestamp = pd.Timestamp.now("UTC")

    # Get current positions (default to zero if not provided)
    if current_positions is None:
        current_positions = pd.DataFrame(columns=["symbol", "qty"])

    # Ensure current_positions has required columns
    if "symbol" not in current_positions.columns:
        current_positions = pd.DataFrame(columns=["symbol", "qty"])
    if "qty" not in current_positions.columns:
        current_positions["qty"] = 0.0

    # Try fast-path if positions are already aligned
    # Fast-path condition: same symbols in same order, no missing symbols
    target_symbols = set(target_positions["symbol"].unique())
    if current_positions is not None and not current_positions.empty:
        current_symbols = set(current_positions["symbol"].unique())
        # Fast-path: exact same symbols, can use aligned arrays
        if target_symbols == current_symbols:
            target_sorted = (
                target_positions[["symbol", "target_qty"]]
                .sort_values("symbol")
                .reset_index(drop=True)
            )
            current_sorted = (
                current_positions[["symbol", "qty"]]
                .sort_values("symbol")
                .reset_index(drop=True)
            )
            # Check if symbols match exactly (same order after sort)
            if (
                target_sorted["symbol"].values == current_sorted["symbol"].values
            ).all():
                # Use fast-path with prices_latest (extract latest per symbol if prices provided)
                prices_latest = None
                if (
                    prices is not None
                    and "close" in prices.columns
                    and "symbol" in prices.columns
                ):
                    prices_latest = (
                        prices.groupby("symbol", group_keys=False)["close"]
                        .last()
                        .reset_index()
                    )
                    prices_latest = (
                        prices_latest[prices_latest["symbol"].isin(target_symbols)]
                        .sort_values("symbol")
                        .reset_index(drop=True)
                    )
                try:
                    # Convert target_qty (notional) to shares before fast-path
                    # Fast-path expects qty in shares, but compute_target_positions
                    # outputs notional (weight * capital).
                    fp_target = target_sorted.copy()
                    if prices_latest is not None and not prices_latest.empty:
                        price_map = dict(
                            zip(
                                prices_latest["symbol"].values,
                                prices_latest["close"].values,
                            )
                        )
                        fp_target["target_qty"] = fp_target.apply(
                            lambda r: (
                                r["target_qty"] / price_map[r["symbol"]]
                                if r["symbol"] in price_map
                                and price_map[r["symbol"]] > 1e-10
                                else 0.0
                            ),
                            axis=1,
                        )
                    return generate_orders_from_targets_fast(
                        fp_target,
                        current_positions=current_sorted,
                        timestamp=timestamp,
                        prices_latest=prices_latest,
                        min_trade_value=min_trade_value,
                    )
                except (ValueError, KeyError) as exc:
                    # Fallback to merge-based path if fast-path fails
                    logger.error("[OrderGeneration] fast-path order gen failed, falling back to merge: %s", exc)

    # Fallback to merge-based path (handles misaligned or missing symbols)
    # Ensure both DataFrames are sorted by symbol for stable alignment
    target_sorted = (
        target_positions[["symbol", "target_qty"]]
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    current_sorted = (
        current_positions[["symbol", "qty"]]
        .sort_values("symbol")
        .reset_index(drop=True)
        if current_positions is not None and not current_positions.empty
        else pd.DataFrame(columns=["symbol", "qty"])
    )

    # Merge target and current positions (outer join to include all symbols)
    merged = target_sorted.merge(
        current_sorted,
        on="symbol",
        how="outer",
        suffixes=("_target", "_current"),
    )

    # Ensure stable symbol order (sorted)
    merged = merged.sort_values("symbol").reset_index(drop=True)

    # Fill NaN with 0.0 (symbols not in current or target)
    # Ensure columns are float type before filling to avoid FutureWarning
    if "target_qty" in merged.columns:
        merged["target_qty"] = merged["target_qty"].astype(float).fillna(0.0)
    if "qty" in merged.columns:
        merged["qty"] = merged["qty"].astype(float).fillna(0.0)

    # target_qty from position_sizing is NOTIONAL; current qty is SHARES -> convert target to shares for delta
    # Get prices if available (needed to convert target notional to shares)
    if prices is not None and "close" in prices.columns and "symbol" in prices.columns:
        latest_prices = prices.groupby("symbol")["close"].last()
        merged["price"] = merged["symbol"].map(latest_prices).fillna(0.0)
    else:
        merged["price"] = 0.0
    # Warn for symbols with missing price but non-zero target notional
    for _, row in merged[
        (merged["price"] == 0.0) & (merged["target_qty"].abs() > 1e-10)
    ].iterrows():
        logger.warning(
            "[WARN] order_generation: missing price for symbol %s, "
            "target_notional=%.2f — order skipped",
            row["symbol"],
            row["target_qty"],
        )
    price_vals = merged["price"].values.astype(np.float64)
    safe_price = np.where(price_vals > 1e-10, price_vals, np.nan)
    target_shares = np.where(
        np.isfinite(safe_price),
        merged["target_qty"].values.astype(np.float64) / safe_price,
        0.0,
    )
    current_shares = merged["qty"].fillna(0.0).values.astype(np.float64)
    merged["qty_delta"] = target_shares - current_shares

    # Filter for non-zero deltas (vectorized)
    abs_delta = np.abs(merged["qty_delta"])
    non_zero_mask = abs_delta > 1e-6
    orders = merged[non_zero_mask].copy()

    if orders.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # Determine side and quantity (qty in SHARES)
    qty_delta_filtered = orders["qty_delta"].values
    orders["side"] = np.where(qty_delta_filtered > 0, "BUY", "SELL")
    orders["qty"] = np.abs(qty_delta_filtered)

    # Select output columns
    result = orders[["symbol", "side", "qty", "price"]].copy()
    result["timestamp"] = timestamp
    # arrival_price (Sprint 2 / C11): snapshot decision-time price
    result["arrival_price"] = result["price"]

    # Reorder columns
    result = result[["timestamp", "symbol", "side", "qty", "price", "arrival_price"]]
    result = result.sort_values("symbol").reset_index(drop=True)
    result.attrs["qty_unit"] = "shares"
    return result


def generate_orders_from_signals(
    signals: pd.DataFrame,
    total_capital: float = 1.0,
    top_n: int | None = None,
    current_positions: pd.DataFrame | None = None,
    timestamp: datetime | None = None,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate orders directly from signals (convenience function).

    This function combines position sizing and order generation in one step.

    Args:
        signals: DataFrame with columns: symbol, direction (and optionally score)
        total_capital: Total capital available (default: 1.0)
        top_n: Optional maximum number of positions (default: None)
        current_positions: Optional current positions DataFrame
        timestamp: Order timestamp (default: current UTC time)
        prices: Optional prices DataFrame for price lookup

    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
    """
    # Compute target positions
    targets = compute_target_positions(
        signals, total_capital=total_capital, top_n=top_n, equal_weight=True
    )

    # Generate orders
    return generate_orders_from_targets(
        targets, current_positions=current_positions, timestamp=timestamp, prices=prices
    )


# ---------------------------------------------------------------------------
# Order Netting (Plan 6.5)
# ---------------------------------------------------------------------------


def net_orders(
    orders: "pd.DataFrame",
    symbol_col: str = "symbol",
    qty_col: str = "qty",
    side_col: str = "side",
) -> "pd.DataFrame":
    """Net opposing orders per symbol.

    Aggregates and eliminates offsetting buy/sell orders for the same symbol.

    Two input conventions are supported:

    1. Signed ``qty`` (negative = SELL): sums directly.
    2. Unsigned ``qty`` with a ``side`` column (``BUY``/``SELL``), as produced
       by ``generate_orders_from_targets``: signs ``qty`` via ``side``, sums,
       then reconstructs ``side`` and unsigned ``qty``. This prevents the
       previous bug where BUY 100 + SELL 100 summed to **200** instead of
       netting to zero.

    Args:
        orders: DataFrame with ``symbol_col`` and ``qty_col`` (optionally
            ``side_col``).
        symbol_col: Symbol column name.
        qty_col: Quantity column name.
        side_col: Side column name (used only if present).

    Returns:
        Netted orders (only symbols with non-zero net quantity). If
        ``side_col`` was present in input, it is present in output.
    """

    if orders.empty:
        return orders

    has_side = side_col in orders.columns
    work = orders.copy()

    if has_side:
        # Sign the qty via side so netting truly offsets opposing orders.
        side_sign = work[side_col].astype(str).str.upper().map(
            {"BUY": 1, "SELL": -1}
        ).fillna(1)
        work["__signed_qty__"] = work[qty_col].astype(float) * side_sign
        netted = work.groupby(symbol_col, as_index=False)["__signed_qty__"].sum()
        netted = netted[netted["__signed_qty__"].abs() > 1e-10].reset_index(drop=True)
        netted[side_col] = np.where(netted["__signed_qty__"] >= 0, "BUY", "SELL")
        netted[qty_col] = netted["__signed_qty__"].abs()
        return netted[[symbol_col, side_col, qty_col]]

    netted = work.groupby(symbol_col, as_index=False)[qty_col].sum()
    return netted[netted[qty_col].abs() > 1e-10].reset_index(drop=True)
