# src/assembled_core/risk/exposure_engine.py
"""Exposure Engine: Compute target portfolio state and exposure metrics.

This module provides deterministic functions to compute target positions from
current positions and orders, and calculate exposure metrics (weights, gross/net exposure).

**Layering:** This module belongs to the risk layer and does NOT import from
pipeline or qa layers to maintain independence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExposureSummary:
    """Summary of portfolio exposure metrics.

    Attributes:
        gross_exposure: Gross exposure (sum of absolute notional values)
        net_exposure: Net exposure (sum of signed notional values)
        gross_exposure_pct: Gross exposure as percentage of equity
        net_exposure_pct: Net exposure as percentage of equity
        n_positions: Number of positions (symbols with non-zero target_qty)
    """

    gross_exposure: float
    net_exposure: float
    gross_exposure_pct: float
    net_exposure_pct: float
    n_positions: int


def compute_target_positions(
    current_positions: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Compute target positions from current positions and orders.

    Target positions = current positions + order deltas (BUY adds, SELL subtracts).

    Args:
        current_positions: DataFrame with columns: symbol, qty
            Current portfolio positions (may be empty)
        orders: DataFrame with columns: symbol, side, qty
            Proposed orders (side: "BUY" or "SELL", qty: always positive)

    Returns:
        DataFrame with columns: symbol, target_qty
        - target_qty: Target quantity after executing orders (can be negative for short positions)
        - Sorted by symbol (deterministic)

    Example:
        >>> current = pd.DataFrame({"symbol": ["AAPL"], "qty": [100.0]})
        >>> orders = pd.DataFrame({
        ...     "symbol": ["AAPL", "MSFT"],
        ...     "side": ["SELL", "BUY"],
        ...     "qty": [50.0, 200.0]
        ... })
        >>> target = compute_target_positions(current, orders)
        >>> # AAPL: 100 - 50 = 50, MSFT: 0 + 200 = 200
    """
    # Handle empty inputs
    if current_positions.empty and orders.empty:
        return pd.DataFrame(columns=["symbol", "target_qty"])

    # Normalize current positions: ensure symbol column and qty
    if current_positions.empty:
        current_df = pd.DataFrame(columns=["symbol", "qty"])
    else:
        current_df = current_positions[["symbol", "qty"]].copy()

    # Normalize orders: ensure symbol, side, qty columns
    if orders.empty:
        orders_df = pd.DataFrame(columns=["symbol", "side", "qty"])
    else:
        orders_df = orders[["symbol", "side", "qty"]].copy()

    # Compute signed order deltas: BUY = +qty, SELL = -qty
    orders_df = orders_df.copy()
    orders_df["signed_qty"] = np.where(
        orders_df["side"] == "BUY",
        orders_df["qty"].values,
        -orders_df["qty"].values,
    )

    # Aggregate order deltas by symbol (multiple orders for same symbol are summed)
    order_deltas = (
        orders_df.groupby("symbol", as_index=False)["signed_qty"]
        .sum()
        .rename(columns={"signed_qty": "order_delta"})
    )

    # Merge current positions with order deltas
    # Use outer join to include symbols that appear only in current or only in orders
    target_df = current_df.merge(
        order_deltas,
        on="symbol",
        how="outer",
        suffixes=("", "_order"),
    )

    # Fill NaN values (symbols only in orders have NaN current_qty, symbols only in current have NaN order_delta)
    target_df["qty"] = target_df["qty"].fillna(0.0)
    target_df["order_delta"] = target_df["order_delta"].fillna(0.0)

    # Compute target_qty = current_qty + order_delta
    target_df["target_qty"] = target_df["qty"] + target_df["order_delta"]

    # Select and sort by symbol (deterministic)
    result = target_df[["symbol", "target_qty"]].copy()
    result = result.sort_values("symbol", ignore_index=True)

    return result


def compute_exposures(
    target_positions: pd.DataFrame,
    prices_latest: pd.DataFrame,
    equity: float,
    *,
    missing_price_handling: str = "raise",
) -> tuple[pd.DataFrame, ExposureSummary]:
    """Compute exposure metrics from target positions and prices.

    Computes per-symbol exposures (notional, weight) and portfolio-level summary
    (gross/net exposure).

    Args:
        target_positions: DataFrame with columns: symbol, target_qty
            Target portfolio positions (from compute_target_positions)
        prices_latest: DataFrame with columns: symbol, close (or price)
            Latest prices for each symbol (one row per symbol)
        equity: Portfolio equity (cash + mark-to-market positions)
            Used for weight calculation
        missing_price_handling: How to handle missing prices (default: "raise")
            - "raise": Raise ValueError if price is missing for any symbol
            - "zero": Use 0.0 as price (deterministic fallback)

    Returns:
        Tuple of (exposures_df, summary):
        - exposures_df: DataFrame with columns: symbol, target_qty, price, notional, weight
          Sorted by symbol (deterministic)
        - summary: ExposureSummary with portfolio-level metrics

    Raises:
        ValueError: If missing_price_handling="raise" and price is missing for any symbol
        ValueError: If equity <= 0 (cannot compute weights)

    Example:
        >>> target = pd.DataFrame({
        ...     "symbol": ["AAPL", "MSFT"],
        ...     "target_qty": [100.0, -50.0]
        ... })
        >>> prices = pd.DataFrame({
        ...     "symbol": ["AAPL", "MSFT"],
        ...     "close": [150.0, 200.0]
        ... })
        >>> exposures, summary = compute_exposures(target, prices, equity=10000.0)
        >>> # AAPL: notional=15000, weight=1.5, MSFT: notional=-10000, weight=-1.0
    """
    if target_positions.empty:
        # Return empty exposures with zero summary
        empty_exposures = pd.DataFrame(
            columns=["symbol", "target_qty", "price", "notional", "weight"]
        )
        summary = ExposureSummary(
            gross_exposure=0.0,
            net_exposure=0.0,
            gross_exposure_pct=0.0,
            net_exposure_pct=0.0,
            n_positions=0,
        )
        return empty_exposures, summary

    if equity <= 0.0:
        raise ValueError(f"equity must be > 0, got {equity}")

    # Normalize target positions
    target_df = target_positions[["symbol", "target_qty"]].copy()

    # Normalize prices: use "close" if available, else "price"
    prices_df = prices_latest.copy()
    if "close" in prices_df.columns:
        price_col = "close"
    elif "price" in prices_df.columns:
        price_col = "price"
    else:
        raise ValueError(
            "prices_latest must have 'close' or 'price' column"
        )

    # Merge target positions with prices
    exposures_df = target_df.merge(
        prices_df[["symbol", price_col]],
        on="symbol",
        how="left",
    )

    # Handle missing prices
    missing_prices = exposures_df[price_col].isna()
    if missing_prices.any():
        if missing_price_handling == "raise":
            missing_symbols = exposures_df.loc[missing_prices, "symbol"].tolist()
            raise ValueError(
                f"Missing price for symbols: {missing_symbols}. "
                "Provide prices_latest with all symbols or use missing_price_handling='zero'"
            )
        elif missing_price_handling == "zero":
            exposures_df[price_col] = exposures_df[price_col].fillna(0.0)
        else:
            raise ValueError(
                f"Invalid missing_price_handling: {missing_price_handling}. "
                "Must be 'raise' or 'zero'"
            )

    # Rename price column to "price" for consistency
    exposures_df = exposures_df.rename(columns={price_col: "price"})

    # Compute notional = target_qty * price
    exposures_df["notional"] = exposures_df["target_qty"] * exposures_df["price"]

    # Compute weight = notional / equity
    exposures_df["weight"] = exposures_df["notional"] / equity

    # Ensure deterministic sorting by symbol
    exposures_df = exposures_df.sort_values("symbol", ignore_index=True)

    # Compute summary metrics
    gross_exposure = float(exposures_df["notional"].abs().sum())
    net_exposure = float(exposures_df["notional"].sum())
    gross_exposure_pct = (gross_exposure / equity) * 100.0 if equity > 0.0 else 0.0
    net_exposure_pct = (net_exposure / equity) * 100.0 if equity > 0.0 else 0.0
    n_positions = int((exposures_df["target_qty"].abs() > 1e-10).sum())

    summary = ExposureSummary(
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        gross_exposure_pct=gross_exposure_pct,
        net_exposure_pct=net_exposure_pct,
        n_positions=n_positions,
    )

    return exposures_df, summary
