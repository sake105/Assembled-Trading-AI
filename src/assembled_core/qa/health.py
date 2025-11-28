# src/assembled_core/qa/health.py
"""Health check and QA functions for pipeline outputs."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR


@dataclass
class QaCheckResult:
    """Result of a single QA check.
    
    Attributes:
        name: Name of the check (e.g., "prices", "orders", "portfolio")
        status: Check status ("ok", "warning", "error")
        message: Human-readable message describing the check result
        details: Optional dictionary with additional check details
    """
    name: str
    status: Literal["ok", "warning", "error"]
    message: str
    details: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def check_prices(freq: str, output_dir: Path | None = None) -> QaCheckResult:
    """Check price data file for a given frequency.
    
    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
    
    Returns:
        QaCheckResult with status and message
    """
    base = output_dir if output_dir else OUTPUT_DIR
    
    # Determine expected file path
    if freq == "1d":
        price_file = base / "aggregates" / "daily.parquet"
    elif freq == "5min":
        price_file = base / "aggregates" / "5min.parquet"
    else:
        return QaCheckResult(
            name="prices",
            status="error",
            message=f"Unsupported frequency: {freq}",
            details={"freq": freq}
        )
    
    # Check file exists
    if not price_file.exists():
        return QaCheckResult(
            name="prices",
            status="error",
            message=f"Price file not found: {price_file}",
            details={"file": str(price_file), "freq": freq}
        )
    
    try:
        # Load and check DataFrame
        df = pd.read_parquet(price_file)
        
        if df.empty:
            return QaCheckResult(
                name="prices",
                status="warning",
                message=f"Price file is empty: {price_file}",
                details={"file": str(price_file), "rows": 0}
            )
        
        # Check required columns
        required_cols = ["timestamp", "symbol", "close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            return QaCheckResult(
                name="prices",
                status="error",
                message=f"Missing required columns: {missing_cols}",
                details={
                    "file": str(price_file),
                    "missing": missing_cols,
                    "available": list(df.columns)
                }
            )
        
        # Check for NaNs in critical columns
        nan_counts = {}
        for col in required_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = int(nan_count)
        
        if nan_counts:
            return QaCheckResult(
                name="prices",
                status="warning",
                message=f"Found NaNs in price data: {nan_counts}",
                details={
                    "file": str(price_file),
                    "rows": len(df),
                    "nan_counts": nan_counts
                }
            )
        
        # Success
        return QaCheckResult(
            name="prices",
            status="ok",
            message=f"Price file OK: {len(df)} rows, {df['symbol'].nunique()} symbols",
            details={
                "file": str(price_file),
                "rows": len(df),
                "symbols": int(df["symbol"].nunique()),
                "first": str(df["timestamp"].min()),
                "last": str(df["timestamp"].max())
            }
        )
    
    except Exception as e:
        return QaCheckResult(
            name="prices",
            status="error",
            message=f"Error reading price file: {e}",
            details={"file": str(price_file), "error": str(e)}
        )


def check_orders(freq: str, output_dir: Path | None = None) -> QaCheckResult:
    """Check orders file for a given frequency.
    
    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
    
    Returns:
        QaCheckResult with status and message
    """
    base = output_dir if output_dir else OUTPUT_DIR
    orders_file = base / f"orders_{freq}.csv"
    
    # Check file exists
    if not orders_file.exists():
        return QaCheckResult(
            name="orders",
            status="error",
            message=f"Orders file not found: {orders_file}",
            details={"file": str(orders_file), "freq": freq}
        )
    
    try:
        # Load and check DataFrame
        df = pd.read_csv(orders_file)
        
        if df.empty:
            return QaCheckResult(
                name="orders",
                status="warning",
                message=f"Orders file is empty: {orders_file}",
                details={"file": str(orders_file), "rows": 0}
            )
        
        # Check required columns
        required_cols = ["timestamp", "symbol", "side", "qty", "price"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            return QaCheckResult(
                name="orders",
                status="error",
                message=f"Missing required columns: {missing_cols}",
                details={
                    "file": str(orders_file),
                    "missing": missing_cols,
                    "available": list(df.columns)
                }
            )
        
        # Check for invalid side values
        if "side" in df.columns:
            invalid_sides = df[~df["side"].isin(["BUY", "SELL"])]
            if len(invalid_sides) > 0:
                return QaCheckResult(
                    name="orders",
                    status="warning",
                    message=f"Found invalid side values: {invalid_sides['side'].unique().tolist()}",
                    details={
                        "file": str(orders_file),
                        "rows": len(df),
                        "invalid_sides": invalid_sides["side"].unique().tolist()
                    }
                )
        
        # Success
        return QaCheckResult(
            name="orders",
            status="ok",
            message=f"Orders file OK: {len(df)} orders",
            details={
                "file": str(orders_file),
                "rows": len(df),
                "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
                "first": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
                "last": str(df["timestamp"].max()) if "timestamp" in df.columns else None
            }
        )
    
    except Exception as e:
        return QaCheckResult(
            name="orders",
            status="error",
            message=f"Error reading orders file: {e}",
            details={"file": str(orders_file), "error": str(e)}
        )


def check_portfolio(freq: str, output_dir: Path | None = None) -> QaCheckResult:
    """Check portfolio equity file for a given frequency.
    
    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
    
    Returns:
        QaCheckResult with status and message
    """
    base = output_dir if output_dir else OUTPUT_DIR
    portfolio_file = base / f"portfolio_equity_{freq}.csv"
    
    # Check file exists
    if not portfolio_file.exists():
        return QaCheckResult(
            name="portfolio",
            status="error",
            message=f"Portfolio equity file not found: {portfolio_file}",
            details={"file": str(portfolio_file), "freq": freq}
        )
    
    try:
        # Load and check DataFrame
        df = pd.read_csv(portfolio_file)
        
        # Check required columns
        if "equity" not in df.columns:
            return QaCheckResult(
                name="portfolio",
                status="error",
                message=f"Missing 'equity' column in portfolio file",
                details={
                    "file": str(portfolio_file),
                    "available": list(df.columns)
                }
            )
        
        # Check minimum rows
        min_rows = 5
        if len(df) < min_rows:
            return QaCheckResult(
                name="portfolio",
                status="warning",
                message=f"Portfolio file has too few rows: {len(df)} < {min_rows}",
                details={
                    "file": str(portfolio_file),
                    "rows": len(df),
                    "min_required": min_rows
                }
            )
        
        # Check for NaNs in equity column
        nan_count = df["equity"].isna().sum()
        if nan_count > 0:
            return QaCheckResult(
                name="portfolio",
                status="error",
                message=f"Found {nan_count} NaNs in equity column",
                details={
                    "file": str(portfolio_file),
                    "rows": len(df),
                    "nan_count": int(nan_count)
                }
            )
        
        # Check for negative or zero equity (warning, not error)
        negative_count = (df["equity"] <= 0).sum()
        if negative_count > 0:
            return QaCheckResult(
                name="portfolio",
                status="warning",
                message=f"Found {negative_count} rows with non-positive equity",
                details={
                    "file": str(portfolio_file),
                    "rows": len(df),
                    "negative_count": int(negative_count)
                }
            )
        
        # Success
        return QaCheckResult(
            name="portfolio",
            status="ok",
            message=f"Portfolio file OK: {len(df)} rows, equity range [{df['equity'].min():.2f}, {df['equity'].max():.2f}]",
            details={
                "file": str(portfolio_file),
                "rows": len(df),
                "equity_min": float(df["equity"].min()),
                "equity_max": float(df["equity"].max()),
                "first": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
                "last": str(df["timestamp"].max()) if "timestamp" in df.columns else None
            }
        )
    
    except Exception as e:
        return QaCheckResult(
            name="portfolio",
            status="error",
            message=f"Error reading portfolio file: {e}",
            details={"file": str(portfolio_file), "error": str(e)}
        )


def aggregate_qa_status(freq: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Run all QA checks and aggregate results.
    
    Args:
        freq: Frequency string ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
    
    Returns:
        Dictionary with:
            - freq: Frequency string
            - overall_status: "ok", "warning", or "error"
            - checks: List of QaCheckResult dictionaries
    """
    checks = [
        check_prices(freq, output_dir),
        check_orders(freq, output_dir),
        check_portfolio(freq, output_dir),
    ]
    
    # Determine overall status
    statuses = [c.status for c in checks]
    
    if "error" in statuses:
        overall_status = "error"
    elif "warning" in statuses:
        overall_status = "warning"
    else:
        overall_status = "ok"
    
    return {
        "freq": freq,
        "overall_status": overall_status,
        "checks": [c.to_dict() for c in checks]
    }

