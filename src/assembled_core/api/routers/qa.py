# src/assembled_core/api/routers/qa.py
"""QA/Health check endpoints."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.assembled_core.api.models import (
    QAGateResultModel,
    QAGatesSummaryResponse,
    QaCheck,
    QaStatus,
    QaStatusEnum,
    PerformanceMetricsResponse,
)
from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.logging_utils import get_logger
from src.assembled_core.pipeline.io import load_orders
from src.assembled_core.qa.health import aggregate_qa_status
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.qa_gates import evaluate_all_gates

router = APIRouter()
logger = get_logger(__name__)


@router.get("/qa/status", response_model=QaStatus)
def get_qa_status(
    freq: str = Query(default="1d", description="Trading frequency"),
) -> QaStatus:
    """Get QA/Health check status for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min"), default "1d"

    Returns:
        QaStatus with overall status and list of checks

    Raises:
        HTTPException: 400 if freq is not supported, 500 for unexpected errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Call aggregate_qa_status
        result = aggregate_qa_status(freq, output_dir=OUTPUT_DIR)

        # Map to QaStatus Pydantic model
        checks = []
        for check_dict in result["checks"]:
            checks.append(
                QaCheck(
                    check_name=check_dict["name"],
                    status=QaStatusEnum(check_dict["status"]),
                    message=check_dict["message"],
                    details=check_dict.get("details"),
                )
            )

        # Map overall_status
        overall_status = QaStatusEnum(result["overall_status"])

        # Build summary
        summary = {
            "ok": sum(1 for c in checks if c.status == QaStatusEnum.OK),
            "warning": sum(1 for c in checks if c.status == QaStatusEnum.WARNING),
            "error": sum(1 for c in checks if c.status == QaStatusEnum.ERROR),
        }

        return QaStatus(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            checks=checks,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing QA status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing QA status: {e}")


@router.get("/qa/metrics/{freq}", response_model=PerformanceMetricsResponse)
def get_qa_metrics(freq: str) -> PerformanceMetricsResponse:
    """Get performance metrics for a given frequency.

    Tries to load metrics from run_manifest_{freq}.json first (if available).
    Falls back to computing metrics from equity/trades files if manifest is missing.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        PerformanceMetricsResponse with all performance metrics

    Raises:
        HTTPException: 400 if freq is not supported, 404 if no data found, 500 for unexpected errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Try to load from run manifest first
        manifest_path = OUTPUT_DIR / f"run_manifest_{freq}.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                if "qa_metrics" in manifest and manifest["qa_metrics"]:
                    logger.info(f"Loading metrics from manifest: {manifest_path}")
                    metrics_dict = manifest["qa_metrics"]

                    # Convert dict to PerformanceMetricsResponse
                    # Handle timestamp conversion (from ISO string to datetime)
                    if "start_date" in metrics_dict:
                        if isinstance(metrics_dict["start_date"], str):
                            metrics_dict["start_date"] = pd.to_datetime(
                                metrics_dict["start_date"], utc=True
                            )
                        elif isinstance(metrics_dict["start_date"], pd.Timestamp):
                            metrics_dict["start_date"] = metrics_dict[
                                "start_date"
                            ].to_pydatetime()
                    if "end_date" in metrics_dict:
                        if isinstance(metrics_dict["end_date"], str):
                            metrics_dict["end_date"] = pd.to_datetime(
                                metrics_dict["end_date"], utc=True
                            )
                        elif isinstance(metrics_dict["end_date"], pd.Timestamp):
                            metrics_dict["end_date"] = metrics_dict[
                                "end_date"
                            ].to_pydatetime()

                    return PerformanceMetricsResponse(**metrics_dict)
            except Exception as e:
                logger.warning(
                    f"Failed to load metrics from manifest: {e}, falling back to computation"
                )

        # Fallback: Compute metrics from equity/trades files
        logger.info(f"Computing metrics from equity/trades files for freq={freq}")

        # Try portfolio equity first, then backtest equity
        portfolio_equity_file = OUTPUT_DIR / f"portfolio_equity_{freq}.csv"
        backtest_equity_file = OUTPUT_DIR / f"equity_curve_{freq}.csv"

        equity_df = None
        if portfolio_equity_file.exists():
            equity_df = pd.read_csv(portfolio_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            logger.info(f"Using portfolio equity: {len(equity_df)} rows")
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            logger.info(f"Using backtest equity: {len(equity_df)} rows")

        if equity_df is None or equity_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No equity file found for freq={freq}. Expected: portfolio_equity_{freq}.csv or equity_curve_{freq}.csv",
            )

        # Try to load trades (optional)
        trades_df = None
        try:
            trades_df = load_orders(freq, output_dir=OUTPUT_DIR, strict=False)
            if trades_df.empty:
                trades_df = None
        except Exception:
            pass  # Trades are optional

        # Compute metrics
        # Extract start_capital from equity (first value) or use default
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

        # Convert PerformanceMetrics to PerformanceMetricsResponse
        return PerformanceMetricsResponse(
            final_pf=metrics.final_pf,
            total_return=metrics.total_return,
            cagr=metrics.cagr,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            max_drawdown=metrics.max_drawdown,
            max_drawdown_pct=metrics.max_drawdown_pct,
            current_drawdown=metrics.current_drawdown,
            volatility=metrics.volatility,
            var_95=metrics.var_95,
            hit_rate=metrics.hit_rate,
            profit_factor=metrics.profit_factor,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss,
            turnover=metrics.turnover,
            total_trades=metrics.total_trades,
            start_date=metrics.start_date,
            end_date=metrics.end_date,
            periods=metrics.periods,
            start_capital=metrics.start_capital,
            end_equity=metrics.end_equity,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing QA metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing QA metrics: {e}")


@router.get("/qa/gates/{freq}", response_model=QAGatesSummaryResponse)
def get_qa_gates(freq: str) -> QAGatesSummaryResponse:
    """Get QA gates summary for a given frequency.

    Tries to load gate results from run_manifest_{freq}.json first (if available).
    Falls back to computing gates from metrics if manifest is missing.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        QAGatesSummaryResponse with overall result and gate details

    Raises:
        HTTPException: 400 if freq is not supported, 404 if no data found, 500 for unexpected errors
    """
    # Validate frequency
    if freq not in SUPPORTED_FREQS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}",
        )

    try:
        # Try to load from run manifest first
        manifest_path = OUTPUT_DIR / f"run_manifest_{freq}.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                if "qa_gate_result" in manifest and manifest["qa_gate_result"]:
                    logger.info(f"Loading gate results from manifest: {manifest_path}")
                    gate_dict = manifest["qa_gate_result"]

                    # Convert dict to QAGatesSummaryResponse
                    gate_results = []
                    for gate_result_dict in gate_dict.get("gate_results", []):
                        # Handle result value (could be "ok", "OK", or QAResult enum value)
                        result_value = gate_result_dict.get("result", "")
                        if isinstance(result_value, str):
                            result_value = result_value.upper()  # "ok" -> "OK"
                        else:
                            result_value = str(result_value).upper()

                        gate_results.append(
                            QAGateResultModel(
                                gate_name=gate_result_dict["gate_name"],
                                result=result_value,
                                reason=gate_result_dict["reason"],
                                details=gate_result_dict.get("details"),
                            )
                        )

                    # Handle overall_result (could be "ok", "OK", or QAResult enum value)
                    overall_result_value = gate_dict.get("overall_result", "")
                    if isinstance(overall_result_value, str):
                        overall_result_value = overall_result_value.upper()
                    else:
                        overall_result_value = str(overall_result_value).upper()

                    return QAGatesSummaryResponse(
                        overall_result=overall_result_value,
                        counts={
                            "ok": gate_dict.get("passed_gates", 0),
                            "warning": gate_dict.get("warning_gates", 0),
                            "block": gate_dict.get("blocked_gates", 0),
                        },
                        gate_results=gate_results,
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load gate results from manifest: {e}, falling back to computation"
                )

        # Fallback: Compute gates from metrics
        logger.info(f"Computing gates from metrics for freq={freq}")

        # Load equity/trades and compute metrics (reuse logic from get_qa_metrics)
        portfolio_equity_file = OUTPUT_DIR / f"portfolio_equity_{freq}.csv"
        backtest_equity_file = OUTPUT_DIR / f"equity_curve_{freq}.csv"

        equity_df = None
        if portfolio_equity_file.exists():
            equity_df = pd.read_csv(portfolio_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)

        if equity_df is None or equity_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No equity file found for freq={freq}. Cannot compute gates without metrics.",
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

        # Convert QAGatesSummary to QAGatesSummaryResponse
        gate_results = []
        for gate_result in gate_summary.gate_results:
            gate_results.append(
                QAGateResultModel(
                    gate_name=gate_result.gate_name,
                    result=gate_result.result.value.upper(),  # "ok" -> "OK"
                    reason=gate_result.reason,
                    details=gate_result.details,
                )
            )

        return QAGatesSummaryResponse(
            overall_result=gate_summary.overall_result.value.upper(),
            counts={
                "ok": gate_summary.passed_gates,
                "warning": gate_summary.warning_gates,
                "block": gate_summary.blocked_gates,
            },
            gate_results=gate_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing QA gates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing QA gates: {e}")
