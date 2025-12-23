# src/assembled_core/api/routers/performance.py
"""Performance endpoints."""

from __future__ import annotations


import pandas as pd
from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import EquityCurveResponse, EquityPoint, Frequency
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.pipeline.backtest import compute_metrics

router = APIRouter()


@router.get("/performance/{freq}/backtest-curve", response_model=EquityCurveResponse)
def get_backtest_curve(freq: Frequency) -> EquityCurveResponse:
    """Get backtest equity curve for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        EquityCurveResponse with equity curve points

    Raises:
        HTTPException: 404 if equity curve file not found, 500 if data is malformed
    """
    curve_file = OUTPUT_DIR / f"equity_curve_{freq.value}.csv"

    if not curve_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Equity curve file not found: {curve_file}. Run backtest first.",
        )

    try:
        df = pd.read_csv(curve_file)

        if "timestamp" not in df.columns or "equity" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed equity curve: missing required columns. Found: {list(df.columns)}",
            )

        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

        # Drop NaNs
        df = df.dropna(subset=["timestamp", "equity"])

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Equity curve file is empty or contains only invalid data",
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
        raise HTTPException(status_code=500, detail=f"Error reading equity curve: {e}")


@router.get("/performance/{freq}/metrics")
def get_performance_metrics(freq: Frequency) -> dict:
    """Get performance metrics for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        Dictionary with performance metrics (final_pf, sharpe, rows, first, last)

    Raises:
        HTTPException: 404 if equity curve file not found, 500 if data is malformed
    """
    curve_file = OUTPUT_DIR / f"equity_curve_{freq.value}.csv"

    if not curve_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Equity curve file not found: {curve_file}. Run backtest first.",
        )

    try:
        df = pd.read_csv(curve_file)

        if "timestamp" not in df.columns or "equity" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed equity curve: missing required columns. Found: {list(df.columns)}",
            )

        # Convert timestamps and equity
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

        # Drop NaNs
        df = df.dropna(subset=["timestamp", "equity"])

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Equity curve file is empty or contains only invalid data",
            )

        # Compute metrics using pipeline function
        metrics = compute_metrics(df)

        # Convert timestamps to ISO format strings for JSON serialization
        return {
            "freq": freq.value,
            "final_pf": metrics["final_pf"],
            "sharpe": metrics["sharpe"],
            "rows": metrics["rows"],
            "first": metrics["first"].isoformat()
            if hasattr(metrics["first"], "isoformat")
            else str(metrics["first"]),
            "last": metrics["last"].isoformat()
            if hasattr(metrics["last"], "isoformat")
            else str(metrics["last"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {e}")
