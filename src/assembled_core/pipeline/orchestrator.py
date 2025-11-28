# src/assembled_core/pipeline/orchestrator.py
"""Pipeline orchestration for EOD runs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.ema_config import get_default_ema_config
from src.assembled_core.pipeline.backtest import compute_metrics, simulate_equity, write_backtest_report
from src.assembled_core.pipeline.io import load_orders, load_prices, load_prices_with_fallback
from src.assembled_core.pipeline.orders import signals_to_orders, write_orders
from src.assembled_core.pipeline.portfolio import simulate_with_costs, write_portfolio_report
from src.assembled_core.pipeline.signals import compute_ema_signals
from src.assembled_core.qa.health import aggregate_qa_status


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
        print(f"[EOD] Price data OK: {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except FileNotFoundError as e:
        print(f"[EOD] ERROR: Price data not found: {e}")
        failure_flag = True
    
    # Step 2: Execute
    try:
        print(f"[EOD] Step 2: Execute")
        orders_path, orders = run_execute_step(freq, output_dir=base, price_file=price_file)
        print(f"[EOD] [OK] Orders written: {orders_path} | rows={len(orders)}")
        completed_steps.append("execute")
    except Exception as e:
        print(f"[EOD] ERROR in execute step: {e}")
        failure_flag = True
    
    # Step 3: Backtest
    if not skip_backtest:
        try:
            print(f"[EOD] Step 3: Backtest")
            curve_path, rep_path = run_backtest_step(freq, start_capital, output_dir=base, price_file=price_file)
            print(f"[EOD] [OK] Backtest written: {curve_path}, {rep_path}")
            completed_steps.append("backtest")
        except Exception as e:
            print(f"[EOD] ERROR in backtest step: {e}")
            failure_flag = True
    else:
        print(f"[EOD] Step 3: Backtest (SKIPPED)")
    
    # Step 4: Portfolio
    if not skip_portfolio:
        try:
            print(f"[EOD] Step 4: Portfolio")
            eq_path, rep_path = run_portfolio_step(
                freq,
                start_capital,
                commission_bps=commission_bps,
                spread_w=spread_w,
                impact_w=impact_w,
                output_dir=base
            )
            print(f"[EOD] [OK] Portfolio written: {eq_path}, {rep_path}")
            completed_steps.append("portfolio")
        except Exception as e:
            print(f"[EOD] ERROR in portfolio step: {e}")
            failure_flag = True
    else:
        print(f"[EOD] Step 4: Portfolio (SKIPPED)")
    
    # Step 5: QA
    qa_result = None
    if not skip_qa:
        try:
            print(f"[EOD] Step 5: QA")
            qa_result = aggregate_qa_status(freq, output_dir=base)
            print(f"[EOD] [OK] QA status: {qa_result['overall_status']}")
            completed_steps.append("qa")
        except Exception as e:
            print(f"[EOD] ERROR in QA step: {e}")
            failure_flag = True
    else:
        print(f"[EOD] Step 5: QA (SKIPPED)")
    
    finished_at = datetime.utcnow()
    
    # Build manifest
    manifest = {
        "freq": freq,
        "start_capital": start_capital,
        "completed_steps": completed_steps,
        "qa_overall_status": qa_result["overall_status"] if qa_result else None,
        "qa_checks": qa_result["checks"] if qa_result else [],
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
    
    print(f"[EOD] [OK] Manifest written: {manifest_path}")
    
    return manifest

