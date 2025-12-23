# src/assembled_core/api/routers/portfolio.py
"""Portfolio endpoints."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import (
    EquityCurveResponse,
    EquityPoint,
    Frequency,
    PortfolioSnapshot,
)
from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.pipeline.io import load_orders

router = APIRouter()


def _parse_portfolio_report(report_path: Path) -> dict[str, float | int]:
    """Parse portfolio report markdown file.

    Args:
        report_path: Path to portfolio_report_{freq}.md (or legacy portfolio_report.md)

    Returns:
        Dictionary with parsed metrics (final_pf, sharpe, trades)
    """
    if not report_path.exists():
        return {"final_pf": 1.0, "sharpe": None, "trades": 0}

    try:
        content = report_path.read_text(encoding="utf-8")

        # Extract metrics using regex
        pf_match = re.search(r"Final PF:\s*([\d.]+)", content)
        sharpe_match = re.search(r"Sharpe:\s*([\d.]+|nan)", content)
        trades_match = re.search(r"Trades:\s*(\d+)", content)

        final_pf = float(pf_match.group(1)) if pf_match else 1.0

        sharpe_str = sharpe_match.group(1) if sharpe_match else None
        sharpe = (
            float(sharpe_str) if sharpe_str and sharpe_str.lower() != "nan" else None
        )

        trades = int(trades_match.group(1)) if trades_match else 0

        return {"final_pf": final_pf, "sharpe": sharpe, "trades": trades}
    except Exception:
        # Return defaults if parsing fails
        return {"final_pf": 1.0, "sharpe": None, "trades": 0}


def _calculate_positions_from_orders(orders: pd.DataFrame) -> dict[str, float]:
    """Calculate current positions from orders.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price

    Returns:
        Dictionary mapping symbol to quantity (positive = long, negative = short)
    """
    if orders.empty:
        return {}

    positions = {}
    for _, row in orders.iterrows():
        symbol = str(row["symbol"])
        side = str(row["side"]).upper()
        qty = float(row["qty"])

        if symbol not in positions:
            positions[symbol] = 0.0

        if side == "BUY":
            positions[symbol] += qty
        elif side == "SELL":
            positions[symbol] -= qty

    # Remove zero positions
    return {k: v for k, v in positions.items() if v != 0.0}


def _estimate_cash_from_equity_and_positions(
    equity: float,
    positions: dict[str, float],
    orders: pd.DataFrame,
    last_price_timestamp: pd.Timestamp,
) -> float:
    """Estimate cash position from equity and positions.

    This is a simplified calculation. In a real system, we'd track cash explicitly.

    Args:
        equity: Current equity value
        positions: Current positions dict
        orders: Orders DataFrame (for price reference)
        last_price_timestamp: Last timestamp in equity curve

    Returns:
        Estimated cash position
    """
    # If no positions, cash equals equity
    if not positions:
        return equity

    # Try to get latest prices from orders for position valuation
    # This is a simplification - ideally we'd have current market prices
    position_value = 0.0
    for symbol, qty in positions.items():
        # Get latest order price for this symbol as proxy
        symbol_orders = orders[orders["symbol"] == symbol]
        if not symbol_orders.empty:
            latest_price = float(symbol_orders.iloc[-1]["price"])
            position_value += qty * latest_price

    # Cash = Equity - Position Value
    cash = equity - position_value
    return cash


@router.get("/portfolio/{freq}/current", response_model=PortfolioSnapshot)
def get_portfolio_current(freq: Frequency) -> PortfolioSnapshot:
    """Get current portfolio snapshot for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        PortfolioSnapshot with current portfolio state

    Raises:
        HTTPException: 404 if portfolio data not found, 500 if data is malformed
    """
    # Validate frequency
    if freq.value not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq.value}. Supported: {SUPPORTED_FREQS}",
        )

    # Load portfolio equity curve
    equity_file = OUTPUT_DIR / f"portfolio_equity_{freq.value}.csv"

    if not equity_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio data for freq {freq.value} not found. Run portfolio simulation first.",
        )

    try:
        equity_df = pd.read_csv(equity_file)

        if "timestamp" not in equity_df.columns or "equity" not in equity_df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed portfolio equity: missing required columns. Found: {list(equity_df.columns)}",
            )

        # Convert timestamps and equity
        equity_df["timestamp"] = pd.to_datetime(
            equity_df["timestamp"], utc=True, errors="coerce"
        )
        equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")

        # Drop NaNs and sort
        equity_df = equity_df.dropna(subset=["timestamp", "equity"]).sort_values(
            "timestamp"
        )

        if equity_df.empty:
            raise HTTPException(
                status_code=500,
                detail="Portfolio equity file is empty or contains only invalid data",
            )

        # Get last row (current state)
        last_row = equity_df.iloc[-1]
        current_timestamp = last_row["timestamp"]
        current_equity = float(last_row["equity"])

        # Get first row for start_capital estimation
        first_equity = float(equity_df.iloc[0]["equity"])

        # Parse portfolio report (prefer freq-specific, fallback to legacy)
        report_path_freq = OUTPUT_DIR / f"portfolio_report_{freq.value}.md"
        report_path_legacy = OUTPUT_DIR / "portfolio_report.md"

        if report_path_freq.exists():
            report_path = report_path_freq
        elif report_path_legacy.exists():
            # Backwards compatibility: use legacy file but log deprecation
            import warnings

            warnings.warn(
                f"Using deprecated portfolio_report.md. Please use portfolio_report_{freq.value}.md instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            report_path = report_path_legacy
        else:
            # No report file found, use defaults
            report_path = report_path_freq

        metrics = _parse_portfolio_report(report_path)

        # Load orders to calculate positions
        try:
            orders = load_orders(freq.value, output_dir=OUTPUT_DIR, strict=False)
            positions = _calculate_positions_from_orders(orders)

            # Estimate cash
            cash = _estimate_cash_from_equity_and_positions(
                current_equity, positions, orders, current_timestamp
            )
        except Exception:
            # If orders can't be loaded, use defaults
            positions = {}
            cash = current_equity  # Assume all equity is cash if no positions

        # Build PortfolioSnapshot
        return PortfolioSnapshot(
            timestamp=current_timestamp,
            equity=current_equity,
            cash=cash,
            positions=positions,
            performance_factor=metrics["final_pf"],
            sharpe=metrics["sharpe"],
            total_trades=metrics["trades"],
            start_capital=first_equity,  # Use first equity as proxy for start capital
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading portfolio data: {e}"
        )


@router.get("/portfolio/{freq}/equity-curve", response_model=EquityCurveResponse)
def get_portfolio_equity_curve(freq: Frequency) -> EquityCurveResponse:
    """Get portfolio equity curve for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        EquityCurveResponse with portfolio equity curve points

    Raises:
        HTTPException: 404 if portfolio equity file not found, 500 if data is malformed
    """
    equity_file = OUTPUT_DIR / f"portfolio_equity_{freq.value}.csv"

    if not equity_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio equity file not found: {equity_file}. Run portfolio simulation first.",
        )

    try:
        df = pd.read_csv(equity_file)

        if "timestamp" not in df.columns or "equity" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed portfolio equity: missing required columns. Found: {list(df.columns)}",
            )

        # Convert timestamps and equity
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

        # Drop NaNs
        df = df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Portfolio equity file is empty or contains only invalid data",
            )

        # Convert to EquityPoint models
        points = [
            EquityPoint(timestamp=row["timestamp"], equity=float(row["equity"]))
            for _, row in df.iterrows()
        ]

        start_equity = float(df["equity"].iloc[0])
        end_equity = float(df["equity"].iloc[-1])

        return EquityCurveResponse(
            frequency=freq,
            points=points,
            count=len(points),
            start_equity=start_equity,
            end_equity=end_equity,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading portfolio equity curve: {e}"
        )
