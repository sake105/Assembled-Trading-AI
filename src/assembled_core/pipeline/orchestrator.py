# src/assembled_core/pipeline/orchestrator.py
"""Pipeline orchestration for EOD runs."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.ema_config import get_default_ema_config
from src.assembled_core.logging_utils import get_logger
from src.assembled_core.pipeline.backtest import compute_metrics, simulate_equity, write_backtest_report
from src.assembled_core.pipeline.io import load_orders, load_prices, load_prices_with_fallback
from src.assembled_core.pipeline.orders import signals_to_orders, write_orders
from src.assembled_core.pipeline.portfolio import simulate_with_costs, write_portfolio_report
from src.assembled_core.pipeline.signals import compute_ema_signals
from src.assembled_core.qa.health import aggregate_qa_status
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.qa_gates import QAResult, evaluate_all_gates
from src.assembled_core.reports.daily_qa_report import generate_qa_report

# Get logger (will use default logging if not configured)
logger = get_logger("assembled_core.pipeline")


def _metrics_to_dict(metrics) -> dict[str, Any] | None:
    """Convert PerformanceMetrics to dictionary for JSON serialization.
    
    Args:
        metrics: PerformanceMetrics instance or None
    
    Returns:
        Dictionary representation or None
    """
    if metrics is None:
        return None
    
    return {
        "final_pf": metrics.final_pf,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "current_drawdown": metrics.current_drawdown,
        "volatility": metrics.volatility,
        "var_95": metrics.var_95,
        "hit_rate": metrics.hit_rate,
        "profit_factor": metrics.profit_factor,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "turnover": metrics.turnover,
        "total_trades": metrics.total_trades,
        "start_date": metrics.start_date.isoformat() if metrics.start_date else None,
        "end_date": metrics.end_date.isoformat() if metrics.end_date else None,
        "periods": metrics.periods,
        "start_capital": metrics.start_capital,
        "end_equity": metrics.end_equity
    }


def _gate_result_to_dict(gate_result) -> dict[str, Any] | None:
    """Convert QAGatesSummary to dictionary for JSON serialization.
    
    Args:
        gate_result: QAGatesSummary instance or None
    
    Returns:
        Dictionary representation or None
    """
    if gate_result is None:
        return None
    
    passed = sum(1 for r in gate_result.gate_results if r.result.value == "ok")
    warnings = sum(1 for r in gate_result.gate_results if r.result.value == "warning")
    blocked = sum(1 for r in gate_result.gate_results if r.result.value == "block")
    
    return {
        "overall_result": gate_result.overall_result.value,
        "passed_gates": passed,
        "warning_gates": warnings,
        "blocked_gates": blocked,
        "gate_results": [
            {
                "gate_name": r.gate_name,
                "result": r.result.value,
                "reason": r.reason,
                "details": r.details
            }
            for r in gate_result.gate_results
        ]
    }


def run_execute_step(
    freq: str,
    output_dir: Path | None = None,
    price_file: str | None = None
) -> tuple[Path, pd.DataFrame]:
    """Run execution step: generate orders from EMA signals.
    
    Args:
        freq: Trading frequency ("1d" or "5min")
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        price_file: Optional explicit path to price file
    
    Returns:
        Tuple of (orders_file_path, orders DataFrame)
    
    Side effects:
        Writes orders CSV file: output_dir/orders_{freq}.csv
    """
    base = output_dir if output_dir else OUTPUT_DIR
    
    # Get EMA defaults
    ema_config = get_default_ema_config(freq)
    
    # Load prices
    prices = load_prices(freq, price_file=price_file, output_dir=base)
    
    # Compute signals
    signals = compute_ema_signals(prices, ema_config.fast, ema_config.slow)
    
    # Generate orders
    orders = signals_to_orders(signals)
    
    # Write orders
    orders_path = write_orders(orders, freq, output_dir=base)
    
    return orders_path, orders


def run_backtest_step(
    freq: str,
    start_capital: float,
    output_dir: Path | None = None,
    price_file: str | None = None
) -> tuple[Path, Path]:
    """Run backtest step: simulate equity without costs.
    
    Args:
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        price_file: Optional explicit path to price file
    
    Returns:
        Tuple of (equity_curve_path, report_path)
    
    Side effects:
        Writes equity_curve_{freq}.csv and performance_report_{freq}.md
    """
    base = output_dir if output_dir else OUTPUT_DIR
    
    # Load data
    if price_file:
        prices = load_prices(freq, price_file=price_file, output_dir=base)
    else:
        prices = load_prices_with_fallback(freq, output_dir=base)
    
    orders = load_orders(freq, output_dir=base, strict=False)
    
    # Simulate
    equity = simulate_equity(prices, orders, start_capital=start_capital)
    metrics = compute_metrics(equity)
    
    # Write results
    curve_path, rep_path = write_backtest_report(equity, metrics, freq, output_dir=base)
    
    return curve_path, rep_path


def run_portfolio_step(
    freq: str,
    start_capital: float,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None,
    output_dir: Path | None = None
) -> tuple[Path, Path]:
    """Run portfolio step: simulate equity with costs.
    
    Args:
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        commission_bps: Commission in basis points (default: from cost model)
        spread_w: Spread weight (default: from cost model)
        impact_w: Impact weight (default: from cost model)
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
    
    Returns:
        Tuple of (equity_path, report_path)
    
    Side effects:
        Writes portfolio_equity_{freq}.csv and portfolio_report_{freq}.md
    """
    base = output_dir if output_dir else OUTPUT_DIR
    
    # Get default cost model if not provided
    if commission_bps is None or spread_w is None or impact_w is None:
        cost_model = get_default_cost_model()
        commission_bps = commission_bps if commission_bps is not None else cost_model.commission_bps
        spread_w = spread_w if spread_w is not None else cost_model.spread_w
        impact_w = impact_w if impact_w is not None else cost_model.impact_w
    
    # Load orders
    orders = load_orders(freq, output_dir=base, strict=True)
    
    # Simulate with costs
    equity, metrics = simulate_with_costs(
        orders,
        start_capital,
        commission_bps,
        spread_w,
        impact_w,
        freq
    )
    
    # Write results
    eq_path, rep_path = write_portfolio_report(equity, metrics, freq, output_dir=base)
    
    return eq_path, rep_path


def run_eod_pipeline(
    freq: str,
    start_capital: float = 10000.0,
    skip_backtest: bool = False,
    skip_portfolio: bool = False,
    skip_qa: bool = False,
    output_dir: Path | None = None,
    price_file: str | None = None,
    commission_bps: float | None = None,
    spread_w: float | None = None,
    impact_w: float | None = None
) -> dict[str, Any]:
    """Run full EOD pipeline for a given frequency.
    
    Args:
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        skip_backtest: If True, skip backtest step
        skip_portfolio: If True, skip portfolio step
        skip_qa: If True, skip QA step
        output_dir: Base output directory (default: None, uses config.OUTPUT_DIR)
        price_file: Optional explicit path to price file
        commission_bps: Commission in basis points (default: from cost model)
        spread_w: Spread weight (default: from cost model)
        impact_w: Impact weight (default: from cost model)
    
    Returns:
        Dictionary with run manifest data
    
    Side effects:
        Executes pipeline steps and writes run_manifest_{freq}.json
    """
    if freq not in SUPPORTED_FREQS:
        raise ValueError(f"Unsupported frequency: {freq}. Supported: {SUPPORTED_FREQS}")
    
    base = output_dir if output_dir else OUTPUT_DIR
    started_at = datetime.utcnow()
    
    completed_steps = []
    failure_flag = False
    
    # Step 1: Check price data exists
    try:
        if price_file:
            prices = load_prices(freq, price_file=price_file, output_dir=base)
        else:
            prices = load_prices_with_fallback(freq, output_dir=base)
        logger.info(f"Price data OK: {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except FileNotFoundError as e:
        logger.error(f"Price data not found: {e}")
        failure_flag = True
    
    # Step 2: Execute
    try:
        logger.info("Step 2: Execute")
        orders_path, orders = run_execute_step(freq, output_dir=base, price_file=price_file)
        logger.info(f"Orders written: {orders_path} | rows={len(orders)}")
        completed_steps.append("execute")
    except Exception as e:
        logger.error(f"ERROR in execute step: {e}", exc_info=True)
        failure_flag = True
    
    # Step 3: Backtest
    if not skip_backtest:
        try:
            logger.info("Step 3: Backtest")
            curve_path, rep_path = run_backtest_step(freq, start_capital, output_dir=base, price_file=price_file)
            logger.info(f"Backtest written: {curve_path}, {rep_path}")
            completed_steps.append("backtest")
        except Exception as e:
            logger.error(f"ERROR in backtest step: {e}", exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 3: Backtest (SKIPPED)")
    
    # Step 4: Portfolio
    if not skip_portfolio:
        try:
            logger.info("Step 4: Portfolio")
            eq_path, rep_path = run_portfolio_step(
                freq,
                start_capital,
                commission_bps=commission_bps,
                spread_w=spread_w,
                impact_w=impact_w,
                output_dir=base
            )
            logger.info(f"Portfolio written: {eq_path}, {rep_path}")
            completed_steps.append("portfolio")
        except Exception as e:
            logger.error(f"ERROR in portfolio step: {e}", exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 4: Portfolio (SKIPPED)")
    
    # Step 5: QA
    qa_result = None
    qa_metrics = None
    qa_gate_result = None
    qa_report_path_rel = None
    if not skip_qa:
        try:
            logger.info("Step 5: QA")
            
            # 5a: Health checks (existing)
            qa_result = aggregate_qa_status(freq, output_dir=base)
            qa_status = qa_result.get("overall_status", "unknown")
            logger.info(f"QA health checks completed: overall_status={qa_status}")
            
            if qa_status == "error":
                logger.error("QA overall_status is 'error' - some checks failed")
            elif qa_status == "warning":
                logger.warning("QA overall_status is 'warning' - some checks have warnings")
            
            # 5b: Performance metrics (new)
            try:
                logger.info("Step 5b: Computing performance metrics")
                # Load portfolio equity (preferred) or backtest equity
                portfolio_equity_file = base / f"portfolio_equity_{freq}.csv"
                backtest_equity_file = base / f"equity_curve_{freq}.csv"
                
                if portfolio_equity_file.exists():
                    equity_df = pd.read_csv(portfolio_equity_file)
                    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
                    logger.info(f"Using portfolio equity: {len(equity_df)} rows")
                elif backtest_equity_file.exists():
                    equity_df = pd.read_csv(backtest_equity_file)
                    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
                    logger.info(f"Using backtest equity: {len(equity_df)} rows")
                else:
                    logger.warning("No equity file found for metrics computation")
                    equity_df = None
                
                # Load trades if available
                orders_df = None
                try:
                    orders_df = load_orders(freq, output_dir=base, strict=False)
                    if orders_df.empty:
                        orders_df = None
                except Exception:
                    pass  # Orders optional for metrics
                
                if equity_df is not None and not equity_df.empty:
                    qa_metrics = compute_all_metrics(
                        equity=equity_df,
                        trades=orders_df,
                        start_capital=start_capital,
                        freq=freq,
                        risk_free_rate=0.0
                    )
                    logger.info(f"Performance metrics computed: PF={qa_metrics.final_pf:.4f}, Sharpe={qa_metrics.sharpe_ratio}, CAGR={qa_metrics.cagr}")
                    
                    # 5c: QA gates (new)
                    logger.info("Step 5c: Evaluating QA gates")
                    qa_gate_result = evaluate_all_gates(qa_metrics)
                    gate_status = qa_gate_result.overall_result.value
                    passed = sum(1 for r in qa_gate_result.gate_results if r.result.value == "ok")
                    warnings = sum(1 for r in qa_gate_result.gate_results if r.result.value == "warning")
                    blocked = sum(1 for r in qa_gate_result.gate_results if r.result.value == "block")
                    logger.info(f"QA gates completed: overall_result={gate_status} (passed={passed}, warnings={warnings}, blocked={blocked})")
                    
                    if qa_gate_result.overall_result == QAResult.BLOCK:
                        logger.error("QA gates BLOCKED - strategy does not meet quality thresholds")
                        # Don't set failure_flag here - gates are informational, not blocking
                    elif qa_gate_result.overall_result == QAResult.WARNING:
                        logger.warning("QA gates WARNING - some quality thresholds not met")
                    
                    # 5d: Generate QA report
                    try:
                        logger.info("Step 5d: Generating QA report")
                        
                        # Build config info for report
                        # Get actual cost parameters used (CLI overrides or defaults)
                        cost_model = get_default_cost_model()
                        final_commission_bps = commission_bps if commission_bps is not None else cost_model.commission_bps
                        final_spread_w = spread_w if spread_w is not None else cost_model.spread_w
                        final_impact_w = impact_w if impact_w is not None else cost_model.impact_w
                        
                        ema_config = get_default_ema_config(freq)
                        config_info = {
                            "strategy": "eod_pipeline_core",
                            "freq": freq,
                            "start_capital": start_capital,
                            "ema_fast": ema_config.fast,
                            "ema_slow": ema_config.slow,
                            "commission_bps": final_commission_bps,
                            "spread_w": final_spread_w,
                            "impact_w": final_impact_w
                        }
                        
                        # Determine equity curve path for report
                        equity_curve_path = None
                        if portfolio_equity_file.exists():
                            equity_curve_path = portfolio_equity_file
                        elif backtest_equity_file.exists():
                            equity_curve_path = backtest_equity_file
                        
                        # Generate report
                        qa_report_path = generate_qa_report(
                            metrics=qa_metrics,
                            gate_result=qa_gate_result,
                            strategy_name="eod_pipeline_core",
                            freq=freq,
                            equity_curve_path=equity_curve_path,
                            data_start_date=qa_metrics.start_date,
                            data_end_date=qa_metrics.end_date,
                            config_info=config_info,
                            output_dir=base / "reports"
                        )
                        
                        # Convert to relative path for manifest (relative to base output dir)
                        qa_report_path_rel = qa_report_path.relative_to(base)
                        logger.info(f"QA report written: {qa_report_path}")
                        
                    except Exception as e:
                        logger.warning(f"QA report generation failed: {e}", exc_info=True)
                        qa_report_path_rel = None
                        # Don't fail the pipeline if report generation fails - it's optional
                else:
                    logger.warning("Cannot compute QA metrics: no equity data available")
                    qa_report_path_rel = None
                    
            except Exception as e:
                logger.warning(f"QA metrics/gates computation failed: {e}", exc_info=True)
                # Don't fail the pipeline if metrics/gates fail - they're optional
                qa_report_path_rel = None
            
            completed_steps.append("qa")
        except Exception as e:
            logger.error(f"ERROR in QA step: {e}", exc_info=True)
            failure_flag = True
    else:
        logger.info("Step 5: QA (SKIPPED)")
    
    finished_at = datetime.utcnow()
    
    # Build manifest
    manifest = {
        "freq": freq,
        "start_capital": start_capital,
        "completed_steps": completed_steps,
        "qa_overall_status": qa_result["overall_status"] if qa_result else None,
        "qa_checks": qa_result["checks"] if qa_result else [],
        "qa_metrics": _metrics_to_dict(qa_metrics) if qa_metrics else None,
        "qa_gate_result": _gate_result_to_dict(qa_gate_result) if qa_gate_result else None,
        "qa_report_path": str(qa_report_path_rel) if qa_report_path_rel else None,
        "timestamps": {
            "started": started_at.isoformat(),
            "finished": finished_at.isoformat()
        },
        "failure": failure_flag
    }
    
    # Write manifest
    manifest_path = base / f"run_manifest_{freq}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest written: {manifest_path}")
    
    return manifest

