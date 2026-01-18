# src/assembled_core/qa/tca.py
"""Transaction Cost Analysis (TCA) Reporting.

This module provides TCA report generation from trades/fills with cost breakdowns.
Reports aggregate costs per day, symbol, and overall strategy.

**Layering:** This module belongs to the qa layer (reporting/analysis).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_tca_report(
    trades_df: pd.DataFrame,
    *,
    freq: str,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """Build TCA report from trades DataFrame.

    Aggregates costs per day+symbol and computes cost_bps metrics.
    Required input columns: timestamp (UTC), symbol, qty, price, commission_cash,
    spread_cash, slippage_cash, total_cost_cash.

    Args:
        trades_df: DataFrame with trades and cost columns:
            - timestamp: UTC timestamps
            - symbol: Symbol names
            - qty: Trade quantities
            - price: Trade prices
            - commission_cash: Commission costs per trade
            - spread_cash: Spread costs per trade
            - slippage_cash: Slippage costs per trade
            - total_cost_cash: Total costs per trade
        freq: Trading frequency ("1d" or "5min") for date aggregation
        strategy_name: Optional strategy name for reporting (default: None)

    Returns:
        DataFrame with TCA report columns:
        - date: Date (date object, UTC-normalized)
        - symbol: Symbol name
        - notional: Sum of notional values (abs(qty) * price) for day+symbol
        - commission_cash: Sum of commission costs
        - spread_cash: Sum of spread costs
        - slippage_cash: Sum of slippage costs
        - total_cost_cash: Sum of total costs
        - cost_bps: Total cost in basis points (total_cost_cash / notional * 10000)
        - n_trades: Number of trades for day+symbol
        Sorted by (date, symbol)

    Raises:
        ValueError: If required columns are missing in trades_df
    """
    if trades_df.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "notional",
                "commission_cash",
                "spread_cash",
                "slippage_cash",
                "total_cost_cash",
                "cost_bps",
                "n_trades",
            ]
        )

    # Validate required columns
    required_cols = [
        "timestamp",
        "symbol",
        "qty",
        "price",
        "commission_cash",
        "spread_cash",
        "slippage_cash",
        "total_cost_cash",
    ]
    missing_cols = [col for col in required_cols if col not in trades_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trades_df: {missing_cols}")

    # Make a copy to avoid modifying original
    trades = trades_df.copy()

    # Ensure timestamp is UTC-aware datetime
    if not pd.api.types.is_datetime64_any_dtype(trades["timestamp"]):
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    elif trades["timestamp"].dt.tz is None:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    
    # Extract date (UTC-normalized)
    trades["date"] = trades["timestamp"].dt.date

    # Compute notional based on fill_qty if available, else use original qty
    # For partial fills, TCA should use filled notional (fill_qty * fill_price)
    if "fill_qty" in trades.columns and "fill_price" in trades.columns:
        trades["notional"] = (trades["fill_qty"].abs() * trades["fill_price"].abs()).astype(np.float64)
    else:
        # Fallback: use original qty * price (for backward compatibility)
        trades["notional"] = (trades["qty"].abs() * trades["price"].abs()).astype(np.float64)

    # Aggregate per day+symbol
    agg_dict = {
        "notional": "sum",
        "commission_cash": "sum",
        "spread_cash": "sum",
        "slippage_cash": "sum",
        "total_cost_cash": "sum",
    }

    tca_report = trades.groupby(["date", "symbol"], as_index=False).agg(agg_dict)

    # Count trades per day+symbol
    trade_counts = trades.groupby(["date", "symbol"], as_index=False).size()
    trade_counts = trade_counts.rename(columns={"size": "n_trades"})
    tca_report = tca_report.merge(trade_counts, on=["date", "symbol"], how="left")
    tca_report["n_trades"] = tca_report["n_trades"].fillna(0).astype(int)

    # Compute cost_bps: total_cost_cash / notional * 10000
    # Handle division by zero (notional = 0)
    tca_report["cost_bps"] = np.where(
        tca_report["notional"] > 0.0,
        (tca_report["total_cost_cash"] / tca_report["notional"]) * 10000.0,
        0.0,
    ).astype(np.float64)

    # Ensure no NaNs in key columns (fill with 0.0)
    cost_cols = ["commission_cash", "spread_cash", "slippage_cash", "total_cost_cash", "cost_bps"]
    for col in cost_cols:
        tca_report[col] = tca_report[col].fillna(0.0).astype(np.float64)

    # Ensure deterministic sorting (by date, symbol)
    tca_report = tca_report.sort_values(["date", "symbol"], ignore_index=True)

    # Add strategy_name if provided (for metadata, not in aggregation)
    if strategy_name is not None:
        tca_report["strategy_name"] = strategy_name

    return tca_report


def write_tca_report_csv(
    tca_report: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Write TCA report to CSV file.

    Args:
        tca_report: DataFrame from build_tca_report()
        output_path: Path to output CSV file

    Returns:
        Path to written CSV file

    Raises:
        OSError: If file write fails
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tca_report.to_csv(output_path, index=False)
        logger.info(f"TCA report written to {output_path}")
    except (IOError, OSError) as e:
        raise OSError(f"Failed to write TCA report to {output_path}: {e}") from e

    return output_path


def write_tca_report_md(
    tca_report: pd.DataFrame,
    output_path: Path | str,
    *,
    strategy_name: str | None = None,
) -> Path:
    """Write TCA report to Markdown file.

    Args:
        tca_report: DataFrame from build_tca_report()
        output_path: Path to output Markdown file
        strategy_name: Optional strategy name for header (default: None)

    Returns:
        Path to written Markdown file

    Raises:
        OSError: If file write fails
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            # Header
            f.write("# Transaction Cost Analysis (TCA) Report\n\n")
            if strategy_name:
                f.write(f"**Strategy:** {strategy_name}\n\n")
            f.write("## Cost Breakdown\n\n")

            # Summary statistics
            if not tca_report.empty:
                total_notional = tca_report["notional"].sum()
                total_commission = tca_report["commission_cash"].sum()
                total_spread = tca_report["spread_cash"].sum()
                total_slippage = tca_report["slippage_cash"].sum()
                total_cost = tca_report["total_cost_cash"].sum()
                total_trades = tca_report["n_trades"].sum()
                avg_cost_bps = (
                    (total_cost / total_notional * 10000.0) if total_notional > 0.0 else 0.0
                )

                f.write("### Summary\n\n")
                f.write(f"- Total Notional: ${total_notional:,.2f}\n")
                f.write(f"- Total Commission: ${total_commission:,.2f}\n")
                f.write(f"- Total Spread: ${total_spread:,.2f}\n")
                f.write(f"- Total Slippage: ${total_slippage:,.2f}\n")
                f.write(f"- Total Cost: ${total_cost:,.2f}\n")
                f.write(f"- Average Cost (bps): {avg_cost_bps:.2f}\n")
                f.write(f"- Total Trades: {total_trades}\n\n")

                # Per-symbol summary
                symbol_summary = (
                    tca_report.groupby("symbol")
                    .agg(
                        {
                            "notional": "sum",
                            "commission_cash": "sum",
                            "spread_cash": "sum",
                            "slippage_cash": "sum",
                            "total_cost_cash": "sum",
                            "n_trades": "sum",
                        }
                    )
                    .reset_index()
                )
                symbol_summary["cost_bps"] = np.where(
                    symbol_summary["notional"] > 0.0,
                    (symbol_summary["total_cost_cash"] / symbol_summary["notional"]) * 10000.0,
                    0.0,
                )

                f.write("### Per-Symbol Summary\n\n")
                f.write("| Symbol | Notional | Commission | Spread | Slippage | Total Cost | Cost (bps) | Trades |\n")
                f.write("|--------|----------|------------|--------|----------|------------|------------|--------|\n")
                for _, row in symbol_summary.iterrows():
                    f.write(
                        f"| {row['symbol']} | ${row['notional']:,.2f} | "
                        f"${row['commission_cash']:,.2f} | ${row['spread_cash']:,.2f} | "
                        f"${row['slippage_cash']:,.2f} | ${row['total_cost_cash']:,.2f} | "
                        f"{row['cost_bps']:.2f} | {row['n_trades']} |\n"
                    )
                f.write("\n")

                # Per-day summary
                day_summary = (
                    tca_report.groupby("date")
                    .agg(
                        {
                            "notional": "sum",
                            "commission_cash": "sum",
                            "spread_cash": "sum",
                            "slippage_cash": "sum",
                            "total_cost_cash": "sum",
                            "n_trades": "sum",
                        }
                    )
                    .reset_index()
                )
                day_summary["cost_bps"] = np.where(
                    day_summary["notional"] > 0.0,
                    (day_summary["total_cost_cash"] / day_summary["notional"]) * 10000.0,
                    0.0,
                )

                f.write("### Per-Day Summary\n\n")
                f.write("| Date | Notional | Commission | Spread | Slippage | Total Cost | Cost (bps) | Trades |\n")
                f.write("|------|----------|------------|--------|----------|------------|------------|--------|\n")
                for _, row in day_summary.iterrows():
                    f.write(
                        f"| {row['date']} | ${row['notional']:,.2f} | "
                        f"${row['commission_cash']:,.2f} | ${row['spread_cash']:,.2f} | "
                        f"${row['slippage_cash']:,.2f} | ${row['total_cost_cash']:,.2f} | "
                        f"{row['cost_bps']:.2f} | {row['n_trades']} |\n"
                    )
                f.write("\n")

            else:
                f.write("No trades in report.\n")

        logger.info(f"TCA report (Markdown) written to {output_path}")
    except (IOError, OSError) as e:
        raise OSError(f"Failed to write TCA report (Markdown) to {output_path}: {e}") from e

    return output_path
