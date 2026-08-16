# src/assembled_core/api/routers/qa.py
"""QA/Health check endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import cast

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from src.assembled_core.api.models import (
    PerformanceMetricsResponse,
    QaCheck,
    QAGateResultModel,
    QAGatesSummaryResponse,
    QaStatus,
    QaStatusEnum,
    SharpeDistributionResponse,
    StressTestsResponse,
    StressTestWindow,
    WalkForwardWindow,
    WalkForwardWindowsResponse,
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
            timestamp=datetime.now(tz=timezone.utc),
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
            try:
                equity_df = pd.read_csv(
                    portfolio_equity_file, dtype={"timestamp": "string"}
                )
            except (IOError, OSError) as exc:
                logger.warning(
                    f"Failed to read portfolio equity file {portfolio_equity_file}: {exc}"
                )
                equity_df = None
            else:
                equity_df["timestamp"] = pd.to_datetime(
                    equity_df["timestamp"], utc=True
                )
                logger.info(f"Using portfolio equity: {len(equity_df)} rows")
        if equity_df is None and backtest_equity_file.exists():
            try:
                equity_df = pd.read_csv(
                    backtest_equity_file, dtype={"timestamp": "string"}
                )
            except (IOError, OSError) as exc:
                logger.warning(
                    f"Failed to read backtest equity file {backtest_equity_file}: {exc}"
                )
                equity_df = None
            else:
                equity_df["timestamp"] = pd.to_datetime(
                    equity_df["timestamp"], utc=True
                )
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
        except Exception as exc:
            logger.warning("[QA] failed to load trades for metrics: %s", exc)

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
                            # E-066: a gate that was never CHECKED must not be
                            # invisible. Without this key a SKIPPED gate falls
                            # into no bucket and the caller reads "3 ok, 0 block"
                            # as "everything verified" when one gate never ran.
                            "skipped": gate_dict.get("skipped_gates", 0),
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
            equity_df = pd.read_csv(portfolio_equity_file, dtype={"timestamp": str})
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
        elif backtest_equity_file.exists():
            equity_df = pd.read_csv(backtest_equity_file, dtype={"timestamp": str})
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
        except Exception as exc:
            logger.warning("[QA] failed to load trades for gates: %s", exc)

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
                # E-066: "not checked" is its own state, never a pass.
                "skipped": gate_summary.skipped_gates,
            },
            gate_results=gate_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing QA gates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error computing QA gates: {e}")


# ---------------------------------------------------------------------------
# Walk-Forward endpoints (V2)
# ---------------------------------------------------------------------------


def _load_latest_walk_forward() -> dict:
    """Load the most recent walk_forward JSON from output/qa/release_gate/."""
    import glob as _glob

    pattern = str(OUTPUT_DIR / "qa" / "release_gate" / "walk_forward_*.json")
    files = sorted(_glob.glob(pattern))
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as fh:
        return cast(dict, json.load(fh))


@router.get(
    "/qa/walk_forward/{freq}/windows", response_model=WalkForwardWindowsResponse
)
def get_walk_forward_windows(freq: str) -> WalkForwardWindowsResponse:
    """Walk-forward split results: aggregated metrics + per-split summary."""
    try:
        data = _load_latest_walk_forward()
        wf = data.get("walk_forward", {}) if data else {}
        agg = wf.get("aggregated_metrics", {})
        n_splits = int(wf.get("n_splits", 0))
        # EHRLICHKEIT (Audit-Plan 5.2; erweitert F-senior-3/E-164, 2026-08-17):
        # 503 nicht nur bei fehlender Datei, sondern auch bei leerem/teil-
        # geschriebenem Artefakt — "lief nie" und "brach ab" duerfen beide
        # nicht wie "lief mit 0 Splits" aussehen.
        if not data or not wf or not agg or n_splits <= 0:
            raise HTTPException(
                status_code=503,
                detail="no usable walk_forward artifact under output/qa/release_gate/ (missing, empty or partial) — walk-forward has not run",
            )
        n_ok = int(wf.get("n_successful_splits", 0))
        generated_at = data.get("generated_at")

        # Synthetic per-split rows from aggregated stats (no per-split storage
        # yet). Jede Zeile traegt den Marker synthetic_from_aggregate=1.0,
        # damit kein Konsument identische Zeilen fuer echte Splits haelt
        # (Audit-Plan 5.2).
        windows: list[WalkForwardWindow] = []
        for i in range(n_splits):
            windows.append(
                WalkForwardWindow(
                    split=i + 1,
                    metrics={
                        "sharpe": agg.get("mean_sharpe", 0.0),
                        "total_return": agg.get("mean_total_return", 0.0),
                        "synthetic_from_aggregate": 1.0,
                    },
                )
            )

        return WalkForwardWindowsResponse(
            freq=freq,
            n_splits=n_splits,
            n_successful_splits=n_ok,
            aggregated_metrics={k: float(v) for k, v in agg.items()},
            windows=windows,
            generated_at=generated_at,
        )
    except HTTPException:
        # Der bewusste 503 (fehlendes Artefakt) darf nicht zum 500 werden.
        raise
    except Exception as exc:
        logger.error("walk_forward/windows error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/qa/walk_forward/{freq}/sharpe-distribution",
    response_model=SharpeDistributionResponse,
)
def get_walk_forward_sharpe_distribution(freq: str) -> SharpeDistributionResponse:
    """Walk-forward Sharpe distribution derived from aggregated split stats."""
    try:
        data = _load_latest_walk_forward()
        wf = data.get("walk_forward", {}) if data else {}
        agg = wf.get("aggregated_metrics", {})
        # EHRLICHKEIT (Audit-Plan 5.2; erweitert F-senior-3/E-164): keine
        # Fantasie-Perzentile aus mean=0/std=1 — weder bei fehlender Datei
        # noch bei leerem/teilgeschriebenem Artefakt.
        if not data or not wf or not agg or int(agg.get("n_splits", 0)) <= 0:
            raise HTTPException(
                status_code=503,
                detail="no usable walk_forward artifact under output/qa/release_gate/ (missing, empty or partial) — walk-forward has not run",
            )
        mean_s = float(agg.get("mean_sharpe", 0.0))
        std_s = float(agg.get("std_sharpe", 1.0))
        min_s = float(agg.get("min_sharpe", mean_s - 2 * std_s))
        max_s = float(agg.get("max_sharpe", mean_s + 2 * std_s))
        n = int(agg.get("n_splits", 0))

        # Perzentile sind PARAMETRISCH aus mean/std unter Normalannahme
        # abgeleitet, keine empirischen Quantile — der source-String sagt das
        # jetzt explizit (Audit-Plan 5.2).
        return SharpeDistributionResponse(
            freq=freq,
            source="walk_forward_parametric_normal",
            n_samples=n,
            p10=round(min_s, 4),
            p25=round(mean_s - 0.67 * std_s, 4),
            p50=round(mean_s, 4),
            p75=round(mean_s + 0.67 * std_s, 4),
            p90=round(max_s, 4),
            mean=round(mean_s, 4),
            std=round(std_s, 4),
        )
    except HTTPException:
        # Der bewusste 503 (fehlendes Artefakt) darf nicht zum 500 werden.
        raise
    except Exception as exc:
        logger.error("walk_forward/sharpe-distribution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/qa/monte_carlo/{freq}/sharpe-distribution",
    response_model=SharpeDistributionResponse,
)
def get_monte_carlo_sharpe_distribution(freq: str) -> SharpeDistributionResponse:
    """Monte Carlo Sharpe distribution from permuted trade paths.

    §6.5.3 Phase 2 migration (2026-05-17): switched from legacy
    qa.monte_carlo_paths.monte_carlo_trade_paths to risk.monte_carlo.permute_trades
    + pnl_to_returns conversion. Reads ShuffleResult.sharpe_distribution directly
    (fixes legacy bug where mc.get("sharpe", [0.0]) wrapped a dict-of-quantiles
    into _np.array, producing 0-element arrays for downstream percentiles).
    """
    try:
        import numpy as _np
        from src.assembled_core.pipeline.io import load_orders
        from src.assembled_core.risk.monte_carlo import (
            permute_trades,
            pnl_to_returns,
        )

        orders = load_orders(freq=freq)
        if orders.empty or "pnl" not in orders.columns:
            raise HTTPException(
                status_code=404, detail="No trade PnL data available for freq"
            )

        _pnl_series = orders["pnl"].dropna()
        if len(_pnl_series) < 5:
            raise HTTPException(
                status_code=404,
                detail=f"Too few trades for Monte Carlo (got {len(_pnl_series)})",
            )
        # API has no per-request initial_capital; use legacy default 100_000.
        _returns = pnl_to_returns(_pnl_series, initial_capital=100_000.0)
        _result = permute_trades(_returns, n_iterations=2000, seed=42)
        sharpes = _result.sharpe_distribution
        sharpes = sharpes[_np.isfinite(sharpes)]

        if len(sharpes) == 0:
            raise HTTPException(
                status_code=404, detail="Monte Carlo returned no valid Sharpe values"
            )

        return SharpeDistributionResponse(
            freq=freq,
            source="monte_carlo",
            n_samples=len(sharpes),
            p10=round(float(_np.percentile(sharpes, 10)), 4),
            p25=round(float(_np.percentile(sharpes, 25)), 4),
            p50=round(float(_np.percentile(sharpes, 50)), 4),
            p75=round(float(_np.percentile(sharpes, 75)), 4),
            p90=round(float(_np.percentile(sharpes, 90)), 4),
            mean=round(float(sharpes.mean()), 4),
            std=round(float(sharpes.std()), 4),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("monte_carlo/sharpe-distribution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa/stress_tests/{freq}", response_model=StressTestsResponse)
def get_stress_tests(freq: str) -> StressTestsResponse:
    """Latest stress test results from output/stress/aggregate.json."""
    try:
        agg_path = OUTPUT_DIR / "stress" / "aggregate.json"
        if not agg_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No stress test results found — run scripts/run_stress_test.py",
            )

        with open(agg_path, encoding="utf-8") as fh:
            data = json.load(fh)

        windows_raw = data.get("windows", [])
        windows = [
            StressTestWindow(
                window=w.get("window", ""),
                description=w.get("description", ""),
                start=w.get("start", ""),
                end=w.get("end", ""),
                cagr=w.get("cagr"),
                sharpe=w.get("sharpe"),
                mdd=w.get("mdd"),
                n_trades=w.get("n_trades"),
                total_return=w.get("total_return"),
                worst_day=w.get("worst_day"),
            )
            for w in windows_raw
        ]

        checks_raw = data.get("threshold_checks", {})
        verdict = data.get("live_activation_verdict", "UNKNOWN")
        policy_field = data.get("policy")
        generated_at = data.get("generated_at") or (
            policy_field.get("generated_at") if isinstance(policy_field, dict) else None
        )

        return StressTestsResponse(
            freq=freq,
            verdict=verdict,
            windows=windows,
            threshold_checks={k: bool(v) for k, v in checks_raw.items()},
            generated_at=generated_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("stress_tests error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
