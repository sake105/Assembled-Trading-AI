# src/assembled_core/api/routers/risk.py
"""Risk endpoints."""

from __future__ import annotations


import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import Frequency, RiskMetrics
from src.assembled_core.config import OUTPUT_DIR

router = APIRouter()


@router.get("/risk/{freq}/summary", response_model=RiskMetrics)
def get_risk_summary(freq: Frequency) -> RiskMetrics:
    """Get risk metrics summary for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        RiskMetrics with sharpe_ratio, max_drawdown, volatility, current_drawdown, var_95

    Raises:
        HTTPException: 404 if portfolio equity file not found, 500 if data is malformed
    """
    portfolio_file = OUTPUT_DIR / f"portfolio_equity_{freq.value}.csv"

    if not portfolio_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio equity file not found: {portfolio_file}. Run portfolio simulation first.",
        )

    try:
        df = pd.read_csv(portfolio_file)

        if "timestamp" not in df.columns or "equity" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Malformed portfolio equity: missing required columns. Found: {list(df.columns)}",
            )

        # Convert timestamps and equity
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")

        # Drop NaNs
        df = df.dropna(subset=["timestamp", "equity"])

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Portfolio equity file is empty or contains only invalid data",
            )

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        equity_series = df["equity"].values

        # Compute Sharpe ratio from returns
        returns = pd.Series(equity_series).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            # Annualize based on frequency
            periods_per_year = (
                252 if freq.value == "1d" else 252 * 78
            )  # 78 trading sessions per day (5min)
            sharpe_ratio = float(
                returns.mean() / returns.std() * np.sqrt(periods_per_year)
            )
        else:
            sharpe_ratio = float("nan")

        # Compute drawdown
        rolling_max = pd.Series(equity_series).expanding().max()
        drawdown = equity_series - rolling_max.values
        max_drawdown = float(drawdown.min())
        max_drawdown_pct = (
            float((max_drawdown / rolling_max.max()) * 100)
            if rolling_max.max() > 0
            else 0.0
        )
        current_drawdown = float(drawdown.iloc[-1])

        # Compute volatility (annualized)
        if len(returns) > 1:
            volatility = float(returns.std() * np.sqrt(periods_per_year))
        else:
            volatility = float("nan")

        # Compute VaR (95% confidence, historical)
        if len(returns) > 5:
            var_95 = float(np.percentile(returns, 5) * equity_series[-1])
        else:
            var_95 = float("nan")

        return RiskMetrics(
            sharpe_ratio=sharpe_ratio if not np.isnan(sharpe_ratio) else None,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            volatility=volatility if not np.isnan(volatility) else None,
            current_drawdown=current_drawdown,
            var_95=var_95 if not np.isnan(var_95) else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error computing risk metrics: {e}"
        )
