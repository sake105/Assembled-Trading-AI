"""Position engine: Build positions from ledger events (Sprint 13 L2).

This module provides functions to compute positions, average cost basis,
realized/unrealized PnL from ledger events using average-cost accounting.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from src.assembled_core.accounting.ledger import (
    EVENT_TYPE_CASH_MOVEMENT,
    EVENT_TYPE_FILL,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


def build_positions_from_ledger(
    events_df: pd.DataFrame,
    *,
    prices_df: pd.DataFrame | None = None,
    mark_ts: pd.Timestamp | None = None,
    start_cash: float = 0.0,
    missing_price_policy: Literal["raise", "zero"] = "zero",
    currency_map: dict[str, str] | None = None,
    fx_rates: dict[str, float] | None = None,
) -> dict:
    """Build positions from ledger events using average-cost accounting.

    This function:
    1. Processes events in deterministic order (event_ts, event_type, symbol, event_id)
    2. Tracks positions with average cost basis per symbol
    3. Calculates realized PnL on position reduction or flip
    4. Calculates unrealized PnL using mark prices (if prices_df provided)
    5. Tracks cash balance from cash_delta

    Args:
        events_df: Ledger events DataFrame (must have REQUIRED_COLUMNS)
        prices_df: Optional prices DataFrame with columns: timestamp, symbol, close
            Used for mark-to-market (unrealized PnL)
        mark_ts: Optional mark timestamp (default: latest event_ts)
            If None, uses max(event_ts) from events_df
        start_cash: Starting cash balance (default: 0.0)
        missing_price_policy: How to handle missing prices ("raise" or "zero", default: "zero")
            - "raise": Raise ValueError if price missing for position
            - "zero": Set unrealized_pnl=0, last_price=NaN

    Returns:
        Dictionary with keys:
        - positions_df: DataFrame with columns:
            - symbol: str
            - qty: float (positive=long, negative=short)
            - avg_price: float (average cost basis)
            - realized_pnl: float (cumulative realized PnL for this symbol)
            - unrealized_pnl: float (mark-to-market PnL)
            - notional: float (abs(qty) * last_price)
            - last_price: float (mark price at mark_ts, or NaN if missing)
        - cash_balance: float (final cash balance)
        - summary: dict with keys:
            - gross_exposure: float (sum of abs(notional))
            - net_exposure: float (sum of notional, signed)
            - n_positions: int (number of non-zero positions)
            - total_realized_pnl: float (sum of realized_pnl)
            - total_unrealized_pnl: float (sum of unrealized_pnl)
            - total_pnl: float (total_realized_pnl + total_unrealized_pnl)

    Raises:
        ValueError: If events_df is missing required columns
        ValueError: If missing_price_policy="raise" and price missing for position
    """
    if events_df.empty:
        # Return empty positions
        positions_df = pd.DataFrame(
            columns=[
                "symbol",
                "qty",
                "avg_price",
                "realized_pnl",
                "unrealized_pnl",
                "notional",
                "last_price",
            ]
        )
        return {
            "positions_df": positions_df,
            "cash_balance": start_cash,
            "summary": {
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "n_positions": 0,
                "total_realized_pnl": 0.0,
                "total_unrealized_pnl": 0.0,
                "total_pnl": 0.0,
            },
        }

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in events_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in events_df: {missing_cols}")

    # Normalize events: ensure UTC timestamps, deterministic sort
    events_normalized = events_df.copy()

    # Ensure event_ts is UTC-aware
    events_normalized["event_ts"] = pd.to_datetime(
        events_normalized["event_ts"], utc=True
    )
    if events_normalized["event_ts"].dt.tz is None:
        events_normalized["event_ts"] = events_normalized["event_ts"].dt.tz_localize(
            "UTC"
        )

    # Trim symbol strings
    if "symbol" in events_normalized.columns:
        events_normalized["symbol"] = (
            events_normalized["symbol"].astype(str).str.strip()
        )
        # Replace empty strings with None
        events_normalized["symbol"] = events_normalized["symbol"].replace("", None)

    # A NaN cash_delta on a FILL event produced by an upstream schema drift
    # or a corrupted parquet round-trip used to be silently coerced to 0.0 —
    # the position leg updated but the cash leg evaporated, creating an
    # untraceable reconciliation break. Refuse to silently zero; surface
    # the ids so the operator can chase the source.
    if "cash_delta" in events_normalized.columns:
        nan_mask = events_normalized["cash_delta"].isna()
        if bool(nan_mask.any()):
            bad_ids = (
                events_normalized.loc[nan_mask, "event_id"].astype(str).head(5).tolist()
                if "event_id" in events_normalized.columns
                else []
            )
            raise ValueError(
                f"[POSITION] {int(nan_mask.sum())} event(s) have NaN cash_delta; "
                f"refusing to silently zero. first event_ids={bad_ids}"
            )
        events_normalized["cash_delta"] = events_normalized["cash_delta"].astype(float)

    # Deterministic sort: event_ts, event_type, symbol, event_id
    events_normalized = events_normalized.sort_values(
        ["event_ts", "event_type", "symbol", "event_id"],
        kind="mergesort",
        ignore_index=True,
    )

    # Determine mark_ts (default: latest event_ts)
    if mark_ts is None:
        mark_ts = events_normalized["event_ts"].max()
    else:
        mark_ts = pd.to_datetime(mark_ts, utc=True)
        if mark_ts.tz is None:
            mark_ts = mark_ts.tz_localize("UTC")

    # Initialize position tracking
    positions: dict[str, dict] = {}  # symbol -> {qty, cost_basis, realized_pnl}
    cash_balance = float(start_cash)

    # Process events
    for _, event in events_normalized.iterrows():
        event_type = str(event["event_type"])
        symbol = event["symbol"] if pd.notna(event["symbol"]) else None
        qty = float(event["qty"]) if pd.notna(event["qty"]) else 0.0
        price = float(event["price"]) if pd.notna(event["price"]) else None
        cash_delta = (
            float(event["cash_delta"]) if pd.notna(event["cash_delta"]) else 0.0
        )

        # Handle cash movements
        if event_type == EVENT_TYPE_CASH_MOVEMENT:
            cash_balance += cash_delta
            continue

        # Skip non-FILL events for position tracking (ORDER_SUBMIT, ACK, etc. don't change positions)
        if event_type != EVENT_TYPE_FILL:
            # Still update cash if cash_delta is non-zero
            if cash_delta != 0.0:
                cash_balance += cash_delta
            continue

        # Skip if no symbol or qty=0
        if symbol is None or qty == 0.0:
            if cash_delta != 0.0:
                cash_balance += cash_delta
            continue

        # Get current position (default: 0.0)
        if symbol not in positions:
            positions[symbol] = {
                "qty": 0.0,
                "cost_basis": 0.0,  # Total cost basis (qty * avg_price)
                "realized_pnl": 0.0,
            }

        current_qty = positions[symbol]["qty"]
        current_cost_basis = positions[symbol]["cost_basis"]
        current_realized_pnl = positions[symbol]["realized_pnl"]

        # Determine new position and realized PnL
        # new_qty = current_qty + qty (qty is already signed: BUY +, SELL -)
        new_qty = current_qty + qty

        # Calculate realized PnL and update cost basis
        # Note: cost_basis should have same sign as qty (positive for long, negative for short)
        if current_qty == 0.0:
            # Opening new position
            if price is not None:
                # cost_basis = qty * price (qty already signed, so cost_basis has correct sign)
                new_cost_basis = qty * price
                new_realized_pnl = current_realized_pnl
            else:
                # No price: can't calculate cost basis
                new_cost_basis = 0.0
                new_realized_pnl = current_realized_pnl
        elif (current_qty > 0 and qty > 0) or (current_qty < 0 and qty < 0):
            # Same direction: increasing position
            # Average cost basis: (old_cost_basis + new_qty * price) / (old_qty + new_qty)
            if price is not None:
                # qty already signed, so cost_basis update has correct sign
                new_cost_basis = current_cost_basis + qty * price
                new_realized_pnl = current_realized_pnl
            else:
                new_cost_basis = current_cost_basis
                new_realized_pnl = current_realized_pnl
        elif (current_qty > 0 and qty < 0) or (current_qty < 0 and qty > 0):
            # Opposite direction: reducing or flipping position
            if price is not None:
                # Calculate average cost basis for current position
                # cost_basis and qty have same sign, so cost_basis / qty is always positive
                if current_qty != 0.0 and current_cost_basis != 0.0:
                    avg_price = abs(current_cost_basis / current_qty)
                else:
                    avg_price = 0.0

                # Determine if we're reducing or flipping
                abs_current = abs(current_qty)
                abs_qty = abs(qty)

                if abs_qty <= abs_current:
                    # Reducing position (no flip)
                    # Realized PnL: (price - avg_price) * abs(qty) for long, (avg_price - price) * abs(qty) for short
                    if current_qty > 0:
                        # Long: selling
                        realized_pnl_delta = (price - avg_price) * abs_qty
                    else:
                        # Short: covering
                        realized_pnl_delta = (avg_price - price) * abs_qty

                    new_realized_pnl = current_realized_pnl + realized_pnl_delta
                    # Update cost basis: reduce proportionally
                    new_cost_basis = current_cost_basis * (1.0 - abs_qty / abs_current)
                else:
                    # Flipping position (abs_qty > abs_current)
                    # Split into: close old position + open new position
                    # Close old position: realized PnL on entire old position
                    if current_qty > 0:
                        # Long -> Short: selling entire long, then shorting remainder
                        realized_pnl_close = (price - avg_price) * abs_current
                        # New short position: (abs_qty - abs_current) at price
                        # cost_basis should have same sign as qty (negative for short)
                        new_cost_basis = -(abs_qty - abs_current) * price
                    else:
                        # Short -> Long: covering entire short, then buying remainder
                        realized_pnl_close = (avg_price - price) * abs_current
                        # New long position: (abs_qty - abs_current) at price
                        # cost_basis should have same sign as qty (positive for long)
                        new_cost_basis = (abs_qty - abs_current) * price

                    new_realized_pnl = current_realized_pnl + realized_pnl_close
            else:
                # No price: can't calculate realized PnL
                new_cost_basis = current_cost_basis
                new_realized_pnl = current_realized_pnl
        else:
            # Edge case: shouldn't happen
            new_cost_basis = current_cost_basis
            new_realized_pnl = current_realized_pnl

        # Update position
        positions[symbol]["qty"] = new_qty
        positions[symbol]["cost_basis"] = new_cost_basis
        positions[symbol]["realized_pnl"] = new_realized_pnl

        # Update cash balance
        cash_balance += cash_delta

    # Build positions DataFrame
    positions_list = []
    for symbol, pos_data in sorted(positions.items()):  # Deterministic sort
        qty = pos_data["qty"]
        cost_basis = pos_data["cost_basis"]
        realized_pnl = pos_data["realized_pnl"]

        # Calculate average price
        # cost_basis and qty should have same sign, so cost_basis / qty is always positive
        if qty != 0.0 and cost_basis != 0.0:
            avg_price = abs(cost_basis / qty)
        else:
            avg_price = 0.0

        # Get mark price (for unrealized PnL)
        last_price = None
        if prices_df is not None and not prices_df.empty:
            # Filter prices to symbol and mark_ts
            symbol_prices = prices_df[prices_df["symbol"] == symbol].copy()
            if not symbol_prices.empty:
                # Ensure timestamp is UTC-aware
                symbol_prices["timestamp"] = pd.to_datetime(
                    symbol_prices["timestamp"], utc=True
                )
                if symbol_prices["timestamp"].dt.tz is None:
                    symbol_prices["timestamp"] = symbol_prices[
                        "timestamp"
                    ].dt.tz_localize("UTC")

                # Filter to prices <= mark_ts
                symbol_prices = symbol_prices[symbol_prices["timestamp"] <= mark_ts]

                if not symbol_prices.empty:
                    # Get latest price (backward merge_asof equivalent)
                    symbol_prices = symbol_prices.sort_values(
                        "timestamp", kind="mergesort"
                    )
                    last_price = (
                        float(symbol_prices.iloc[-1]["close"])
                        if pd.notna(symbol_prices.iloc[-1]["close"])
                        else None
                    )

        # Calculate unrealized PnL
        if last_price is not None and qty != 0.0:
            unrealized_pnl = qty * (last_price - avg_price)
            notional = abs(qty) * last_price
        else:
            if missing_price_policy == "raise" and qty != 0.0:
                raise ValueError(
                    f"Missing price for symbol {symbol} at mark_ts {mark_ts}"
                )
            unrealized_pnl = 0.0
            notional = 0.0
            if last_price is None:
                last_price = np.nan

        positions_list.append(
            {
                "symbol": symbol,
                "qty": qty,
                "avg_price": avg_price,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "notional": notional,
                "last_price": last_price,
            }
        )

    # Create DataFrame with expected columns (even if empty)
    if positions_list:
        positions_df = pd.DataFrame(positions_list)
        # Remove zero positions (threshold: 1e-6)
        positions_df = positions_df[positions_df["qty"].abs() > 1e-6].reset_index(
            drop=True
        )
    else:
        # Empty positions list: create empty DataFrame with expected columns
        positions_df = pd.DataFrame(
            columns=[
                "symbol",
                "qty",
                "avg_price",
                "realized_pnl",
                "unrealized_pnl",
                "notional",
                "last_price",
            ]
        )

    # Tier-1 wiring: optional FX-aware mark-to-market (accounting/currency).
    # Passive — only adds ``currency`` and ``usd_notional`` columns. The
    # default ``notional`` column is unchanged so existing callers see no
    # behavior shift unless they read the new columns.
    if currency_map and not positions_df.empty:
        try:
            from src.assembled_core.accounting.currency import FXConverter

            fx = FXConverter(rates=dict(fx_rates)) if fx_rates else FXConverter()
            positions_df["currency"] = (
                positions_df["symbol"].map(currency_map).fillna("USD")
            )
            positions_df["usd_notional"] = [
                fx.to_usd(float(row["notional"]), str(row["currency"]))
                for _, row in positions_df.iterrows()
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FX] usd_notional enrichment skipped: %s", exc)

    # Calculate summary
    total_realized_pnl = (
        float(positions_df["realized_pnl"].sum()) if not positions_df.empty else 0.0
    )
    total_unrealized_pnl = (
        float(positions_df["unrealized_pnl"].sum()) if not positions_df.empty else 0.0
    )
    gross_exposure = (
        float(positions_df["notional"].abs().sum()) if not positions_df.empty else 0.0
    )
    net_exposure = (
        float((positions_df["qty"] * positions_df["last_price"]).sum())
        if not positions_df.empty
        else 0.0
    )
    n_positions = len(positions_df)

    summary = {
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "n_positions": n_positions,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_pnl": total_realized_pnl + total_unrealized_pnl,
    }

    return {
        "positions_df": positions_df,
        "cash_balance": cash_balance,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 8.6  Corporate Action Adjustments
# ---------------------------------------------------------------------------


def adjust_for_stock_split(
    positions: dict[str, dict],
    symbol: str,
    split_ratio: float,
) -> dict[str, dict]:
    """Adjust positions for a stock split.

    Args:
        positions: Symbol → {qty, cost_basis, ...} dict.
        symbol: Symbol being split.
        split_ratio: e.g. 4.0 for a 4:1 split.

    Returns:
        Updated positions dict.
    """
    if split_ratio <= 0:
        # A 4:1 forward split has ratio 4.0; a 1:4 reverse has ratio 0.25.
        # Zero/negative would silently multiply qty to 0 or flip its sign
        # without any audit trail — always caller error.
        raise ValueError(
            f"split_ratio must be positive, got {split_ratio} for {symbol}"
        )

    if symbol not in positions:
        return positions

    pos = positions[symbol]
    old_qty = pos["qty"]

    # qty × ratio, cost stays same (per-share cost / ratio)
    pos["qty"] = old_qty * split_ratio
    # cost_basis stays the same in total dollars
    # avg_price = cost_basis / qty → new avg_price = old_avg_price / ratio
    positions[symbol] = pos
    return positions


def adjust_for_spinoff(
    positions: dict[str, dict],
    parent_symbol: str,
    child_symbol: str,
    parent_cost_fraction: float = 0.85,
    shares_ratio: float = 0.1,
) -> dict[str, dict]:
    """Adjust positions for a spinoff.

    Args:
        positions: Symbol → {qty, cost_basis, ...} dict.
        parent_symbol: Parent company symbol.
        child_symbol: New spinoff symbol.
        parent_cost_fraction: Fraction of cost basis remaining with parent.
        shares_ratio: Ratio of child shares per parent share.

    Returns:
        Updated positions dict.
    """
    if parent_symbol not in positions:
        return positions

    parent = positions[parent_symbol]
    old_cost_basis = parent["cost_basis"]

    # Split cost basis
    parent["cost_basis"] = old_cost_basis * parent_cost_fraction

    # Create child position
    child_qty = parent["qty"] * shares_ratio
    child_cost = old_cost_basis * (1.0 - parent_cost_fraction)
    positions[child_symbol] = {
        "qty": child_qty,
        "cost_basis": child_cost,
        "realized_pnl": 0.0,
    }

    return positions
