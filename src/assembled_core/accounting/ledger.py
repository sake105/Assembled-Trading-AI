"""Ledger event generation and contract (Sprint 13).

This module provides functions to generate ledger events from orders and trades,
ensuring deterministic event IDs and proper cash/position tracking.
"""

from __future__ import annotations

import hashlib
import logging
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

logger = logging.getLogger(__name__)


def _canonical_float_str(value: float, precision: int = 10) -> str:
    """Format float to canonical string representation (avoids rounding issues).

    Uses Decimal quantization to ensure stable string representation.
    This prevents issues like 0.1 + 0.2 != 0.3 in floating-point arithmetic
    from affecting hash generation.

    Args:
        value: Float value to format
        precision: Number of decimal places (default: 10)

    Returns:
        Canonical string representation of float

    Example:
        >>> _canonical_float_str(150.123456789)
        '150.1234567890'
        >>> _canonical_float_str(0.1 + 0.2)
        '0.3000000000'
    """
    # Convert to Decimal, quantize to specified precision
    # This ensures stable representation regardless of floating-point quirks
    decimal_value = Decimal(str(value))
    quantized = decimal_value.quantize(
        Decimal(10) ** -precision,
        rounding=ROUND_HALF_UP,
    )
    # Format with fixed precision (no scientific notation)
    return f"{quantized:.{precision}f}"


# Required columns for ledger events
REQUIRED_COLUMNS = [
    "event_ts",
    "event_type",
    "symbol",
    "qty",
    "price",
    "cash_delta",
    "run_id",
    "event_id",
]

# Optional columns (preserved if present)
OPTIONAL_COLUMNS = [
    "order_type",
    "limit_price",
    "commission_cash",
    "spread_cash",
    "slippage_cash",
    "total_cost_cash",
    "metadata_json",
    "source",
]

# Event types (ENUM-like)
EVENT_TYPE_ORDER_SUBMIT = "ORDER_SUBMIT"
EVENT_TYPE_ACK = "ACK"
EVENT_TYPE_FILL = "FILL"
EVENT_TYPE_CANCEL = "CANCEL"
EVENT_TYPE_REJECT = "REJECT"
EVENT_TYPE_CASH_MOVEMENT = "CASH_MOVEMENT"

VALID_EVENT_TYPES = [
    EVENT_TYPE_ORDER_SUBMIT,
    EVENT_TYPE_ACK,
    EVENT_TYPE_FILL,
    EVENT_TYPE_CANCEL,
    EVENT_TYPE_REJECT,
    EVENT_TYPE_CASH_MOVEMENT,
]


def generate_event_id(
    event_type: str,
    event_ts: pd.Timestamp,
    symbol: str | None,
    qty: float,
    price: float | None,
    row_index: int | None = None,
) -> str:
    """Generate deterministic event ID from stable fields.

    Args:
        event_type: Event type (e.g., "FILL", "ORDER_SUBMIT")
        event_ts: Event timestamp (UTC, tz-aware)
        symbol: Symbol (None for cash-only events)
        qty: Quantity (signed: BUY +, SELL -)
        price: Price (None/NaN if not applicable)
        row_index: Optional row index for uniqueness (default: None)

    Returns:
        Deterministic event ID string (format: "{event_type}_{hash}")
    """
    # Normalize timestamp to ISO string (UTC)
    if pd.isna(event_ts):
        ts_str = "NaT"
    else:
        # Ensure UTC and format as ISO
        if event_ts.tz is None:
            ts_str = pd.Timestamp(event_ts).tz_localize("UTC").isoformat()
        else:
            ts_str = event_ts.tz_convert("UTC").isoformat()

    # Normalize symbol (None -> empty string)
    symbol_str = str(symbol).strip() if symbol is not None and pd.notna(symbol) else ""

    # Normalize qty (canonical float formatting to avoid rounding issues)
    if pd.notna(qty):
        qty_str = _canonical_float_str(qty)
    else:
        qty_str = "0.0"

    # Normalize price (None/NaN -> empty string)
    if price is None or pd.isna(price):
        price_str = ""
    else:
        price_str = _canonical_float_str(price)

    # Build stable string for hashing
    stable_str = f"{event_type}|{ts_str}|{symbol_str}|{qty_str}|{price_str}"
    if row_index is not None:
        stable_str += f"|{row_index}"

    # Generate hash (SHA256, first 16 chars)
    hash_str = hashlib.sha256(stable_str.encode("utf-8")).hexdigest()[:16]

    # Return event ID: "{event_type}_{hash}"
    return f"{event_type}_{hash_str}"


def events_from_orders(
    orders_df: pd.DataFrame,
    run_id: str,
    *,
    source: str = "backtest",
) -> pd.DataFrame:
    """Generate ledger events from orders DataFrame.

    Creates ORDER_SUBMIT events for each order. For fills, use events_from_trades().

    Args:
        orders_df: DataFrame with columns: timestamp, symbol, side, qty, price
            Optional: order_type, limit_price
        run_id: Run identifier (str)
        source: Source identifier (default: "backtest")

    Returns:
        DataFrame with ledger events (REQUIRED_COLUMNS + optional columns)
    """
    if orders_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    # Validate required columns
    required_order_cols = ["timestamp", "symbol", "side", "qty", "price"]
    missing_cols = [col for col in required_order_cols if col not in orders_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in orders_df: {missing_cols}")

    events_list = []

    for idx, row in orders_df.iterrows():
        # Normalize timestamp to UTC
        event_ts = pd.to_datetime(row["timestamp"], utc=True)
        if event_ts.tz is None:
            event_ts = event_ts.tz_localize("UTC")

        # Normalize symbol (trim)
        symbol = str(row["symbol"]).strip() if pd.notna(row["symbol"]) else None

        # Normalize qty (signed: BUY +, SELL -)
        qty = float(row["qty"])
        if row["side"] == "SELL":
            qty = -abs(qty)  # SELL is negative
        elif row["side"] == "BUY":
            qty = abs(qty)  # BUY is positive
        else:
            qty = 0.0

        # Normalize price
        price = float(row["price"]) if pd.notna(row["price"]) else None

        # For ORDER_SUBMIT, cash_delta is 0 (no cash movement yet)
        cash_delta = 0.0

        # Generate event ID
        event_id = generate_event_id(
            EVENT_TYPE_ORDER_SUBMIT,
            event_ts,
            symbol,
            qty,
            price,
            row_index=idx,
        )

        # Build event dict
        event_dict = {
            "event_ts": event_ts,
            "event_type": EVENT_TYPE_ORDER_SUBMIT,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "cash_delta": cash_delta,
            "run_id": run_id,
            "event_id": event_id,
            "source": source,
        }

        # Add optional columns if present
        if "order_type" in orders_df.columns:
            event_dict["order_type"] = row.get("order_type")
        if "limit_price" in orders_df.columns:
            event_dict["limit_price"] = row.get("limit_price")

        events_list.append(event_dict)

    # Build DataFrame
    events_df = pd.DataFrame(events_list)

    # Ensure deterministic sorting (before final event_id generation if needed)
    events_df = events_df.sort_values(
        ["event_ts", "symbol", "event_type", "qty", "price"],
        kind="mergesort",
        ignore_index=True,
    )

    return events_df


def events_from_trades(
    trades_df: pd.DataFrame,
    run_id: str,
    *,
    source: str = "backtest",
) -> pd.DataFrame:
    """Generate ledger events from trades DataFrame (with fills).

    Creates FILL events for each fill (including partial fills).
    Rejected fills create REJECT events with cash_delta=0.

    Args:
        trades_df: DataFrame with columns: timestamp, symbol, side, qty, price
            Required for fills: fill_qty, fill_price, status
            Optional: commission_cash, spread_cash, slippage_cash, total_cost_cash
        run_id: Run identifier (str)
        source: Source identifier (default: "backtest")

    Returns:
        DataFrame with ledger events (REQUIRED_COLUMNS + optional columns)
    """
    if trades_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    # Validate required columns
    required_trade_cols = ["timestamp", "symbol", "side", "qty", "price"]
    missing_cols = [col for col in required_trade_cols if col not in trades_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trades_df: {missing_cols}")

    # Ensure fill schema (add fill_qty, fill_price, status if missing)
    from src.assembled_core.execution.fill_model import ensure_fill_schema

    trades_with_fills = ensure_fill_schema(trades_df, default_full_fill=True)

    events_list = []

    for idx, row in trades_with_fills.iterrows():
        # Normalize timestamp to UTC
        event_ts = pd.to_datetime(row["timestamp"], utc=True)
        if event_ts.tz is None:
            event_ts = event_ts.tz_localize("UTC")

        # Normalize symbol (trim)
        symbol = str(row["symbol"]).strip() if pd.notna(row["symbol"]) else None

        # Get fill info
        fill_qty_raw = row.get("fill_qty", row["qty"])
        fill_qty = float(fill_qty_raw) if pd.notna(fill_qty_raw) else 0.0

        fill_price_raw = row.get("fill_price", row["price"])
        fill_price = float(fill_price_raw) if pd.notna(fill_price_raw) else None

        status_raw = row.get("status", "filled")
        status = str(status_raw).lower() if pd.notna(status_raw) else "filled"

        # Determine event type
        if status == "rejected" or fill_qty == 0.0:
            event_type = EVENT_TYPE_REJECT
            # Rejected: no cash movement, no position change
            qty = 0.0
            price = None
            cash_delta = 0.0
            # Costs should be 0 for rejected fills
            commission_cash = 0.0
            spread_cash = 0.0
            slippage_cash = 0.0
            total_cost_cash = 0.0
        else:
            event_type = EVENT_TYPE_FILL
            # Normalize qty (signed: BUY +, SELL -)
            qty = fill_qty
            if row["side"] == "SELL":
                qty = -abs(qty)  # SELL is negative
            elif row["side"] == "BUY":
                qty = abs(qty)  # BUY is positive
            else:
                qty = 0.0

            price = fill_price

            # Get costs (default to 0.0 if missing)
            commission_cash = (
                float(row.get("commission_cash", 0.0))
                if pd.notna(row.get("commission_cash"))
                else 0.0
            )
            spread_cash = (
                float(row.get("spread_cash", 0.0))
                if pd.notna(row.get("spread_cash"))
                else 0.0
            )
            slippage_cash = (
                float(row.get("slippage_cash", 0.0))
                if pd.notna(row.get("slippage_cash"))
                else 0.0
            )
            total_cost_cash = (
                float(row.get("total_cost_cash", 0.0))
                if pd.notna(row.get("total_cost_cash"))
                else 0.0
            )

            # Calculate cash_delta
            # BUY: -(fill_qty * fill_price + total_cost_cash)
            # SELL: +(abs(fill_qty) * fill_price - total_cost_cash)
            if fill_price is None or pd.isna(fill_price):
                # Fallback: use order price if fill_price is missing (shouldn't happen for FILL)
                logger.warning(
                    "fill_price is None for FILL event, using order price as fallback"
                )
                fill_price = float(row["price"]) if pd.notna(row["price"]) else 0.0

            if row["side"] == "BUY":
                cash_delta = -(abs(fill_qty) * fill_price + total_cost_cash)
            elif row["side"] == "SELL":
                cash_delta = abs(fill_qty) * fill_price - total_cost_cash
            else:
                cash_delta = 0.0

        # Generate event ID
        event_id = generate_event_id(
            event_type,
            event_ts,
            symbol,
            qty,
            price,
            row_index=idx,
        )

        # Build event dict
        event_dict = {
            "event_ts": event_ts,
            "event_type": event_type,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "cash_delta": cash_delta,
            "run_id": run_id,
            "event_id": event_id,
            "source": source,
        }

        # Add cost columns
        if event_type == EVENT_TYPE_FILL:
            event_dict["commission_cash"] = commission_cash
            event_dict["spread_cash"] = spread_cash
            event_dict["slippage_cash"] = slippage_cash
            event_dict["total_cost_cash"] = total_cost_cash
        else:
            event_dict["commission_cash"] = 0.0
            event_dict["spread_cash"] = 0.0
            event_dict["slippage_cash"] = 0.0
            event_dict["total_cost_cash"] = 0.0

        # Add optional columns if present
        if "order_type" in trades_with_fills.columns:
            event_dict["order_type"] = row.get("order_type")
        if "limit_price" in trades_with_fills.columns:
            event_dict["limit_price"] = row.get("limit_price")

        events_list.append(event_dict)

    # Build DataFrame
    events_df = pd.DataFrame(events_list)

    # Ensure deterministic sorting (before final event_id generation if needed)
    events_df = events_df.sort_values(
        ["event_ts", "symbol", "event_type", "qty", "price"],
        kind="mergesort",
        ignore_index=True,
    )

    return events_df
