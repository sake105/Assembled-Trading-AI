# src/assembled_core/api/routers/monitoring.py
"""Monitoring endpoints for QA, Risk, and Drift status.

This module provides simplified monitoring endpoints that aggregate key status
information for dashboards and operational monitoring.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.assembled_core.api.models import (
    DriftStatusSummary,
    FeatureDriftItem,
    QAStatusSummary,
    RiskStatusSummary,
)
from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.logging_utils import get_logger
from src.assembled_core.pipeline.io import load_orders
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.qa_gates import evaluate_all_gates

router = APIRouter()
logger = get_logger(__name__)


@router.get("/monitoring/qa_status", response_model=QAStatusSummary)
def get_qa_status_summary(
    freq: str = Query(default="1d", description="Trading frequency"),
) -> QAStatusSummary:
    """Get simplified QA status summary for monitoring.

    Returns a quick overview of QA gate results and key performance metrics.
    Uses the most recent QA evaluation available (from run_manifest or computed on-the-fly).

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"

    Returns:
        QAStatusSummary with overall result, gate counts, and key metrics

    Raises:
        HTTPException: 400 if freq is not supported, 404 if no data found, 500 for errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Try to load from run manifest first (has most recent evaluation)
        manifest_path = OUTPUT_DIR / f"run_manifest_{freq}.json"
        last_updated = None

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                # Get gate results if available
                gate_counts = {"ok": 0, "warning": 0, "block": 0}
                overall_result = "UNKNOWN"
                key_metrics = {}

                if "qa_gate_result" in manifest and manifest["qa_gate_result"]:
                    gate_dict = manifest["qa_gate_result"]
                    overall_result = str(
                        gate_dict.get("overall_result", "UNKNOWN")
                    ).upper()
                    gate_counts = {
                        "ok": gate_dict.get("passed_gates", 0),
                        "warning": gate_dict.get("warning_gates", 0),
                        "block": gate_dict.get("blocked_gates", 0),
                    }

                # Get metrics if available
                if "qa_metrics" in manifest and manifest["qa_metrics"]:
                    metrics_dict = manifest["qa_metrics"]
                    key_metrics = {
                        "sharpe_ratio": metrics_dict.get("sharpe_ratio"),
                        "max_drawdown_pct": metrics_dict.get("max_drawdown_pct"),
                        "total_return": metrics_dict.get("total_return"),
                        "cagr": metrics_dict.get("cagr"),
                    }

                # Get timestamp from manifest if available
                if "timestamp" in manifest:
                    try:
                        last_updated = datetime.fromisoformat(
                            str(manifest["timestamp"]).replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                return QAStatusSummary(
                    overall_result=overall_result,
                    gate_counts=gate_counts,
                    key_metrics=key_metrics
                    if key_metrics
                    else {
                        "sharpe_ratio": None,
                        "max_drawdown_pct": None,
                        "total_return": None,
                        "cagr": None,
                    },
                    last_updated=last_updated,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load from manifest: {e}, computing on-the-fly"
                )

        # Fallback: Compute on-the-fly from equity/trades
        logger.info(f"Computing QA status from equity/trades for freq={freq}")

        # Load equity
        portfolio_equity_file = OUTPUT_DIR / f"portfolio_equity_{freq}.csv"
        backtest_equity_file = OUTPUT_DIR / f"equity_curve_{freq}.csv"

        equity_df = None
        if portfolio_equity_file.exists():
            equity_df = pd.read_csv(portfolio_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()

        if equity_df is None or equity_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No equity file found for freq={freq}. Cannot compute QA status.",
            )

        # Load trades (optional)
        trades_df = None
        try:
            trades_df = load_orders(freq, output_dir=OUTPUT_DIR, strict=False)
            if trades_df.empty:
                trades_df = None
        except Exception:
            pass

        # Compute metrics
        start_capital = (
            equity_df["equity"].iloc[0] if "equity" in equity_df.columns else 10000.0
        )
        metrics = compute_all_metrics(
            equity=equity_df,
            trades=trades_df,
            start_capital=start_capital,
            freq=freq,
            risk_free_rate=0.0,
        )

        # Evaluate gates
        gate_summary = evaluate_all_gates(metrics)

        return QAStatusSummary(
            overall_result=gate_summary.overall_result.value.upper(),
            gate_counts={
                "ok": gate_summary.passed_gates,
                "warning": gate_summary.warning_gates,
                "block": gate_summary.blocked_gates,
            },
            key_metrics={
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
            },
            last_updated=last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing QA status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing QA status: {e}")


@router.get("/monitoring/risk_status", response_model=RiskStatusSummary)
def get_risk_status_summary(
    freq: str = Query(default="1d", description="Trading frequency"),
) -> RiskStatusSummary:
    """Get simplified risk status summary for monitoring.

    Returns key risk metrics from the last portfolio evaluation.
    Uses portfolio equity curve if available, falls back to backtest equity.

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"

    Returns:
        RiskStatusSummary with sharpe_ratio, max_drawdown_pct, volatility, var_95, current_drawdown

    Raises:
        HTTPException: 400 if freq is not supported, 404 if no data found, 500 for errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Try to load from portfolio report or risk metrics
        portfolio_equity_file = OUTPUT_DIR / f"portfolio_equity_{freq}.csv"
        backtest_equity_file = OUTPUT_DIR / f"equity_curve_{freq}.csv"

        equity_df = None
        last_updated = None

        if portfolio_equity_file.exists():
            equity_df = pd.read_csv(portfolio_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            last_updated = equity_df["timestamp"].max().to_pydatetime()

        if equity_df is None or equity_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No equity file found for freq={freq}. Cannot compute risk status.",
            )

        # Compute risk metrics using qa.risk_metrics module
        from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics

        equity_series = equity_df.set_index("timestamp")["equity"].sort_index()
        risk_metrics = compute_portfolio_risk_metrics(equity_series, freq=freq)

        # Compute current drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = equity_series - rolling_max
        current_drawdown = float(drawdown.iloc[-1])

        return RiskStatusSummary(
            sharpe_ratio=risk_metrics.get("ann_sharpe"),
            max_drawdown_pct=risk_metrics.get("max_drawdown_pct"),
            volatility=risk_metrics.get("ann_vol"),
            var_95=risk_metrics.get("var_95"),
            current_drawdown=current_drawdown,
            last_updated=last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing risk status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing risk status: {e}")


@router.get("/monitoring/drift_status", response_model=DriftStatusSummary)
def get_drift_status_summary(
    freq: str = Query(default="1d", description="Trading frequency"),
    top_n: int = Query(
        default=10, ge=1, le=50, description="Number of top features to return"
    ),
) -> DriftStatusSummary:
    """Get drift status summary for monitoring.

    Returns the status of the last feature drift analysis, showing which features
    have drifted and their severity. Currently returns dummy/example data as drift
    analysis persistence is not yet implemented.

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"
        top_n: Number of top features with drift to return (default: 10, max: 50)

    Returns:
        DriftStatusSummary with overall severity, top features with drift, and total features checked

    Raises:
        HTTPException: 400 if freq is not supported, 500 for errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # TODO: In the future, load from persisted drift analysis results
        # For now, return example/dummy data indicating no drift detected

        # Try to load from a hypothetical drift results file (future implementation)
        drift_results_file = OUTPUT_DIR / f"drift_analysis_{freq}.parquet"

        if drift_results_file.exists():
            # Load drift results
            drift_df = pd.read_parquet(drift_results_file)

            if (
                not drift_df.empty
                and "feature" in drift_df.columns
                and "psi" in drift_df.columns
            ):
                # Sort by PSI descending and take top N
                drift_df_sorted = drift_df.sort_values("psi", ascending=False).head(
                    top_n
                )

                features_with_drift = [
                    FeatureDriftItem(
                        feature=row["feature"],
                        psi=float(row["psi"]),
                        drift_flag=str(row.get("drift_flag", "NONE")),
                    )
                    for _, row in drift_df_sorted.iterrows()
                ]

                # Determine overall severity (worst case)
                if "drift_flag" in drift_df.columns:
                    if (drift_df["drift_flag"] == "SEVERE").any():
                        overall_severity = "SEVERE"
                    elif (drift_df["drift_flag"] == "MODERATE").any():
                        overall_severity = "MODERATE"
                    else:
                        overall_severity = "NONE"
                else:
                    # Fallback: use PSI thresholds
                    max_psi = drift_df["psi"].max()
                    if max_psi >= 0.3:
                        overall_severity = "SEVERE"
                    elif max_psi >= 0.2:
                        overall_severity = "MODERATE"
                    else:
                        overall_severity = "NONE"

                # Get last updated from file modification time
                last_updated = datetime.fromtimestamp(
                    drift_results_file.stat().st_mtime
                )

                return DriftStatusSummary(
                    overall_severity=overall_severity,
                    features_with_drift=features_with_drift,
                    total_features_checked=len(drift_df),
                    last_updated=last_updated,
                )

        # Fallback: Return example data indicating no drift (no analysis available)
        logger.info(
            f"No drift analysis file found for freq={freq}, returning example data"
        )
        return DriftStatusSummary(
            overall_severity="NONE",
            features_with_drift=[],
            total_features_checked=0,
            last_updated=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing drift status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error computing drift status: {e}"
        )
