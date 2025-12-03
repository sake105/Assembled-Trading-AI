# notebooks/operator_overview_example.py
"""Operator Overview Example - System Health Check.

This script provides a comprehensive overview of the trading system's health by:
1. Running backtests with trend_baseline strategy (with QA report)
2. Running backtests with event_insider_shipping strategy
3. Building ML datasets from backtest results
4. Running validation and drift checks on the datasets
5. Printing a structured text summary

Usage:
    python notebooks/operator_overview_example.py

This script is designed for operators to quickly assess system health and
verify that all components are working correctly.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print("")
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print("")
    print(f"--- {title} ---")


def run_trend_baseline_backtest(output_dir: Path, price_file: Path | None = None) -> dict:
    """Run backtest with trend_baseline strategy.
    
    Args:
        output_dir: Output directory for results
        price_file: Optional explicit price file path
    
    Returns:
        Dictionary with backtest results and metrics
    """
    print_section("STEP 1: Trend Baseline Backtest")
    
    from scripts.run_backtest_strategy import run_backtest_from_args
    import argparse
    
    # Create args namespace
    args = argparse.Namespace()
    args.freq = "1d"
    args.strategy = "trend_baseline"
    args.price_file = price_file
    args.universe = None
    args.start_capital = 10000.0
    args.with_costs = True
    args.commission_bps = None
    args.spread_w = None
    args.impact_w = None
    args.generate_report = True
    args.out = output_dir
    
    print(f"Strategy: {args.strategy}")
    print(f"Frequency: {args.freq}")
    print(f"Start Capital: ${args.start_capital:,.2f}")
    print(f"Generate Report: {args.generate_report}")
    
    try:
        exit_code = run_backtest_from_args(args)
        if exit_code != 0:
            print(f"[WARNING] Backtest exited with code {exit_code}")
            return {"success": False, "exit_code": exit_code}
        
        # Load metrics from report or compute
        from src.assembled_core.qa.metrics import compute_all_metrics
        from src.assembled_core.pipeline.io import load_orders
        import pandas as pd
        
        # Try to load equity curve
        equity_file = output_dir / "equity_curve_1d.csv"
        if equity_file.exists():
            equity_df = pd.read_csv(equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            
            # Try to load trades
            try:
                trades_df = load_orders("1d", output_dir=output_dir, strict=False)
            except Exception:
                trades_df = None
            
            metrics = compute_all_metrics(
                equity=equity_df,
                trades=trades_df,
                start_capital=args.start_capital,
                freq=args.freq,
                risk_free_rate=0.0
            )
            
            result = {
                "success": True,
                "strategy": "trend_baseline",
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_trades": metrics.total_trades,
                "final_pf": metrics.final_pf
            }
            
            print_subsection("Results Summary")
            print(f"Total Return: {metrics.total_return:.2%}")
            if metrics.cagr:
                print(f"CAGR: {metrics.cagr:.2%}")
            if metrics.sharpe_ratio:
                print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
            print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
            print(f"Total Trades: {metrics.total_trades if metrics.total_trades else 0}")
            print(f"Final PF: {metrics.final_pf:.4f}")
            
            return result
        else:
            print("[WARNING] Equity curve file not found")
            return {"success": False, "error": "Equity curve not found"}
    
    except Exception as e:
        logger.error(f"Error in trend_baseline backtest: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_event_insider_shipping_backtest(output_dir: Path, price_file: Path | None = None) -> dict:
    """Run backtest with event_insider_shipping strategy.
    
    Args:
        output_dir: Output directory for results
        price_file: Optional explicit price file path
    
    Returns:
        Dictionary with backtest results and metrics
    """
    print_section("STEP 2: Event Insider Shipping Backtest")
    
    from scripts.run_backtest_strategy import run_backtest_from_args
    import argparse
    
    # Create args namespace
    args = argparse.Namespace()
    args.freq = "1d"
    args.strategy = "event_insider_shipping"
    args.price_file = price_file
    args.universe = None
    args.start_capital = 10000.0
    args.with_costs = True
    args.commission_bps = None
    args.spread_w = None
    args.impact_w = None
    args.generate_report = True
    args.out = output_dir
    
    print(f"Strategy: {args.strategy}")
    print(f"Frequency: {args.freq}")
    print(f"Start Capital: ${args.start_capital:,.2f}")
    print(f"Generate Report: {args.generate_report}")
    
    try:
        exit_code = run_backtest_from_args(args)
        if exit_code != 0:
            print(f"[WARNING] Backtest exited with code {exit_code}")
            return {"success": False, "exit_code": exit_code}
        
        # Load metrics similar to trend_baseline
        from src.assembled_core.qa.metrics import compute_all_metrics
        from src.assembled_core.pipeline.io import load_orders
        import pandas as pd
        
        equity_file = output_dir / "equity_curve_1d.csv"
        if equity_file.exists():
            equity_df = pd.read_csv(equity_file)
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            
            try:
                trades_df = load_orders("1d", output_dir=output_dir, strict=False)
            except Exception:
                trades_df = None
            
            metrics = compute_all_metrics(
                equity=equity_df,
                trades=trades_df,
                start_capital=args.start_capital,
                freq=args.freq,
                risk_free_rate=0.0
            )
            
            result = {
                "success": True,
                "strategy": "event_insider_shipping",
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "total_trades": metrics.total_trades,
                "final_pf": metrics.final_pf
            }
            
            print_subsection("Results Summary")
            print(f"Total Return: {metrics.total_return:.2%}")
            if metrics.cagr:
                print(f"CAGR: {metrics.cagr:.2%}")
            if metrics.sharpe_ratio:
                print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
            print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
            print(f"Total Trades: {metrics.total_trades if metrics.total_trades else 0}")
            print(f"Final PF: {metrics.final_pf:.4f}")
            
            return result
        else:
            print("[WARNING] Equity curve file not found")
            return {"success": False, "error": "Equity curve not found"}
    
    except Exception as e:
        logger.error(f"Error in event_insider_shipping backtest: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def build_ml_dataset(output_dir: Path, strategy: str, price_file: Path | None = None) -> Path | None:
    """Build ML dataset from backtest results.
    
    Args:
        output_dir: Output directory
        strategy: Strategy name ("trend_baseline" or "event_insider_shipping")
        price_file: Optional explicit price file path
    
    Returns:
        Path to created ML dataset, or None if failed
    """
    print_section(f"STEP 3: Build ML Dataset - {strategy}")
    
    from scripts.cli import _run_backtest_for_ml_dataset
    from src.assembled_core.qa.dataset_builder import build_ml_dataset_from_backtest, save_ml_dataset
    
    ml_datasets_dir = output_dir / "ml_datasets"
    ml_datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ml_datasets_dir / f"{strategy}_1d.parquet"
    
    print(f"Strategy: {strategy}")
    print(f"Frequency: 1d")
    print(f"Output: {dataset_path}")
    
    try:
        # Run backtest to get prices_with_features and trades
        print("Running backtest for ML dataset...")
        prices_with_features, trades = _run_backtest_for_ml_dataset(
            strategy=strategy,
            freq="1d",
            price_file=price_file,
            universe=None,
            start_capital=10000.0,
            with_costs=True,
            output_dir=output_dir
        )
        
        if prices_with_features.empty:
            print("[ERROR] No price data with features available")
            return None
        
        if trades.empty:
            print("[ERROR] No trades generated from backtest")
            return None
        
        print(f"Prices with features: {len(prices_with_features)} rows, {prices_with_features['symbol'].nunique()} symbols")
        print(f"Trades: {len(trades)} trades")
        
        # Build ML dataset
        print("Building ML dataset...")
        ml_dataset = build_ml_dataset_from_backtest(
            prices_with_features=prices_with_features,
            trades=trades,
            label_horizon_days=10,
            success_threshold=0.02,
            feature_prefixes=("ta_", "insider_", "congress_", "shipping_", "news_")
        )
        
        if ml_dataset.empty:
            print("[ERROR] ML dataset is empty")
            return None
        
        # Count labels
        label_counts = {}
        if "label" in ml_dataset.columns:
            label_counts = ml_dataset["label"].value_counts().to_dict()
        
        # Save dataset
        print("Saving ML dataset...")
        save_ml_dataset(ml_dataset, dataset_path)
        
        print_subsection("Dataset Summary")
        print(f"Records: {len(ml_dataset)}")
        print(f"Features: {len([c for c in ml_dataset.columns if c not in ['label', 'open_time', 'symbol', 'open_price', 'close_time', 'pnl_pct', 'horizon_days']])}")
        if label_counts:
            print(f"Label Distribution: {label_counts}")
        print(f"Saved to: {dataset_path}")
        
        return dataset_path
    
    except Exception as e:
        logger.error(f"Error building ML dataset: {e}", exc_info=True)
        print(f"[ERROR] Failed to build ML dataset: {e}")
        return None


def run_validation_and_drift_checks(dataset_path: Path, output_dir: Path) -> dict:
    """Run validation and drift checks on ML dataset.
    
    Args:
        dataset_path: Path to ML dataset
        output_dir: Output directory for reports
    
    Returns:
        Dictionary with validation and drift results
    """
    print_section("STEP 4: Validation & Drift Checks")
    
    from scripts.run_validation_and_drift_checks import run_validation_and_drift_checks as run_checks
    
    print(f"Dataset: {dataset_path}")
    
    try:
        monitoring_dir = output_dir / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        report_path = monitoring_dir / "validation_drift_summary.md"
        
        # Run checks (without reference dataset for simplicity)
        report_path = run_checks(
            current_dataset_path=dataset_path,
            reference_dataset_path=None,
            output_path=report_path,
            model_name=dataset_path.stem
        )
        
        print(f"Report generated: {report_path}")
        
        # Read report and extract key information
        if report_path.exists():
            report_content = report_path.read_text(encoding="utf-8")
            
            # Extract validation status
            validation_ok = "[PASSED]" in report_content or "Status: [PASSED]" in report_content
            
            # Extract drift severity
            drift_severity = "NONE"
            if "Feature Drift: [SEVERE]" in report_content:
                drift_severity = "SEVERE"
            elif "Feature Drift: [MODERATE]" in report_content:
                drift_severity = "MODERATE"
            
            print_subsection("Validation & Drift Summary")
            print(f"Validation Status: {'[PASSED]' if validation_ok else '[FAILED]'}")
            print(f"Drift Severity: {drift_severity}")
            print(f"Full report: {report_path}")
            
            return {
                "success": True,
                "validation_ok": validation_ok,
                "drift_severity": drift_severity,
                "report_path": report_path
            }
        else:
            print("[WARNING] Report file was not created")
            return {"success": False, "error": "Report not created"}
    
    except Exception as e:
        logger.error(f"Error in validation and drift checks: {e}", exc_info=True)
        print(f"[ERROR] Failed validation/drift checks: {e}")
        return {"success": False, "error": str(e)}


def print_final_summary(all_results: dict) -> None:
    """Print final system health summary.
    
    Args:
        all_results: Dictionary with all step results
    """
    print_section("FINAL SYSTEM HEALTH SUMMARY", width=70)
    
    # Trend Baseline Results
    trend_result = all_results.get("trend_baseline", {})
    if trend_result.get("success"):
        print("[OK] Trend Baseline Backtest: SUCCESS")
        if trend_result.get("sharpe_ratio"):
            print(f"    Sharpe Ratio: {trend_result['sharpe_ratio']:.4f}")
        print(f"    Total Return: {trend_result.get('total_return', 0):.2%}")
    else:
        print("[FAILED] Trend Baseline Backtest: FAILED")
        if "error" in trend_result:
            print(f"    Error: {trend_result['error']}")
    
    # Event Strategy Results
    event_result = all_results.get("event_insider_shipping", {})
    if event_result.get("success"):
        print("[OK] Event Insider Shipping Backtest: SUCCESS")
        if event_result.get("sharpe_ratio"):
            print(f"    Sharpe Ratio: {event_result['sharpe_ratio']:.4f}")
        print(f"    Total Return: {event_result.get('total_return', 0):.2%}")
    else:
        print("[FAILED] Event Insider Shipping Backtest: FAILED")
        if "error" in event_result:
            print(f"    Error: {event_result['error']}")
    
    # ML Dataset Results
    ml_dataset_path = all_results.get("ml_dataset_path")
    if ml_dataset_path:
        print(f"[OK] ML Dataset Created: {ml_dataset_path.name}")
    else:
        print("[FAILED] ML Dataset Creation: FAILED")
    
    # Validation & Drift Results
    validation_result = all_results.get("validation", {})
    if validation_result.get("success"):
        validation_ok = validation_result.get("validation_ok", False)
        drift_severity = validation_result.get("drift_severity", "NONE")
        
        status = "[OK]" if validation_ok else "[WARNING]"
        print(f"{status} Validation & Drift Checks: {'PASSED' if validation_ok else 'FAILED'}")
        print(f"    Drift Severity: {drift_severity}")
    else:
        print("[FAILED] Validation & Drift Checks: FAILED")
        if "error" in validation_result:
            print(f"    Error: {validation_result['error']}")
    
    print("")
    print("=" * 70)
    print(f"System Health Check completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)


def main() -> int:
    """Main entry point for operator overview example."""
    settings = get_settings()
    output_dir = settings.output_dir
    
    # Check if sample price file exists
    price_file = settings.sample_eod_file if settings.sample_eod_file.exists() else None
    
    if not price_file:
        logger.warning("Sample price file not found, backtests may fail if no data is available")
    
    print("")
    print("=" * 70)
    print("  ASSEMBLED TRADING AI - OPERATOR OVERVIEW / SYSTEM HEALTH CHECK")
    print("=" * 70)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Output Directory: {output_dir}")
    if price_file:
        print(f"Price File: {price_file}")
    else:
        print("Price File: Using default data sources")
    
    all_results = {}
    
    # Step 1: Trend Baseline Backtest
    trend_result = run_trend_baseline_backtest(output_dir, price_file)
    all_results["trend_baseline"] = trend_result
    
    # Step 2: Event Insider Shipping Backtest
    event_result = run_event_insider_shipping_backtest(output_dir, price_file)
    all_results["event_insider_shipping"] = event_result
    
    # Step 3: Build ML Dataset (use trend_baseline results)
    ml_dataset_path = build_ml_dataset(output_dir, "trend_baseline", price_file)
    all_results["ml_dataset_path"] = ml_dataset_path
    
    # Step 4: Validation & Drift Checks (only if dataset was created)
    if ml_dataset_path and ml_dataset_path.exists():
        validation_result = run_validation_and_drift_checks(ml_dataset_path, output_dir)
        all_results["validation"] = validation_result
    else:
        all_results["validation"] = {"success": False, "error": "ML dataset not available"}
        print_section("STEP 4: Validation & Drift Checks")
        print("[SKIPPED] ML dataset not available, skipping validation checks")
    
    # Final Summary
    print_final_summary(all_results)
    
    # Determine overall success
    overall_success = (
        trend_result.get("success", False) and
        event_result.get("success", False) and
        ml_dataset_path is not None and
        all_results.get("validation", {}).get("success", False)
    )
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())

