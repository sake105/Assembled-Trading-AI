"""Generate Risk Report from Backtest Results.

This script generates comprehensive risk reports from backtest outputs, including:
- Basic risk metrics (Sharpe, Sortino, Volatility, Max Drawdown, Skewness, Kurtosis, VaR/ES)
- Exposure time-series (Gross/Net Exposure, HHI Concentration)
- Risk segmentation by regime (if regime data available)
- Performance attribution by factor groups (if factor data available)

Usage:
    python scripts/generate_risk_report.py --backtest-dir output/backtests/experiment_123/
    python scripts/generate_risk_report.py --backtest-dir output/backtests/experiment_123/ --regime-file output/regime/regime_state.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.qa.metrics import _compute_returns
from src.assembled_core.risk.risk_metrics import (
    compute_basic_risk_metrics,
    compute_exposure_timeseries,
    compute_risk_by_factor_group,
    compute_risk_by_regime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dataframe(file_path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame | None:
    """Load DataFrame from CSV or Parquet file.
    
    Args:
        file_path: Path to file (can be .csv or .parquet)
        expected_columns: Optional list of expected column names (for validation)
    
    Returns:
        DataFrame if file exists and can be loaded, None otherwise
    """
    if not file_path.exists():
        logger.debug(f"File not found: {file_path}")
        return None
    
    try:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            # Try to parse timestamp column if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            logger.warning(f"Unknown file format: {file_path.suffix}. Skipping.")
            return None
        
        if expected_columns:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                logger.warning(f"Missing expected columns in {file_path.name}: {', '.join(missing)}")
                return None
        
        logger.info(f"Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def find_backtest_files(backtest_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Find and load backtest files from directory.
    
    Args:
        backtest_dir: Directory containing backtest outputs
    
    Returns:
        Tuple of (equity_df, positions_df, trades_df)
        Each can be None if file not found or cannot be loaded
    """
    equity_df = None
    positions_df = None
    trades_df = None
    
    # Try various filename patterns
    equity_patterns = [
        "equity_curve.parquet",
        "equity_curve.csv",
        "equity*.parquet",
        "equity*.csv",
        "portfolio_equity*.parquet",
        "portfolio_equity*.csv",
    ]
    
    positions_patterns = [
        "positions.parquet",
        "positions.csv",
        "target_positions*.parquet",
        "target_positions*.csv",
    ]
    
    trades_patterns = [
        "trades.parquet",
        "trades.csv",
        "orders*.parquet",
        "orders*.csv",
    ]
    
    # Try to find equity curve
    for pattern in equity_patterns:
        if "*" in pattern:
            matches = list(backtest_dir.glob(pattern))
            if matches:
                equity_df = load_dataframe(matches[0], expected_columns=["timestamp", "equity"])
                if equity_df is not None:
                    break
        else:
            equity_path = backtest_dir / pattern
            equity_df = load_dataframe(equity_path, expected_columns=["timestamp", "equity"])
            if equity_df is not None:
                break
    
    # Try to find positions
    for pattern in positions_patterns:
        if "*" in pattern:
            matches = list(backtest_dir.glob(pattern))
            if matches:
                positions_df = load_dataframe(matches[0], expected_columns=["timestamp", "symbol"])
                if positions_df is not None:
                    break
        else:
            positions_path = backtest_dir / pattern
            positions_df = load_dataframe(positions_path, expected_columns=["timestamp", "symbol"])
            if positions_df is not None:
                break
    
    # Try to find trades
    for pattern in trades_patterns:
        if "*" in pattern:
            matches = list(backtest_dir.glob(pattern))
            if matches:
                trades_df = load_dataframe(matches[0])
                if trades_df is not None:
                    break
        else:
            trades_path = backtest_dir / pattern
            trades_df = load_dataframe(trades_path)
            if trades_df is not None:
                break
    
    return equity_df, positions_df, trades_df


def infer_freq_from_equity(equity_df: pd.DataFrame) -> Literal["1d", "5min"]:
    """Infer trading frequency from equity curve timestamps.
    
    Args:
        equity_df: DataFrame with timestamp column
    
    Returns:
        Frequency string ("1d" or "5min")
    """
    if "timestamp" not in equity_df.columns:
        logger.warning("Cannot infer freq: no timestamp column. Defaulting to '1d'.")
        return "1d"
    
    timestamps = pd.to_datetime(equity_df["timestamp"], utc=True)
    if len(timestamps) < 2:
        return "1d"
    
    # Compute median time difference
    time_diffs = timestamps.sort_values().diff().dropna()
    median_diff = time_diffs.median()
    
    # If median is less than 1 hour, assume 5min; otherwise 1d
    if median_diff < pd.Timedelta(hours=1):
        return "5min"
    else:
        return "1d"


def write_risk_summary_csv(
    metrics: dict[str, float | None | int],
    output_path: Path,
) -> None:
    """Write risk summary to CSV.
    
    Args:
        metrics: Dictionary with risk metrics
        output_path: Output file path
    """
    # Convert to DataFrame (single row)
    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved risk summary to {output_path}")


def write_risk_report_markdown(
    metrics: dict[str, float | None | int],
    exposure_timeseries: pd.DataFrame | None = None,
    risk_by_regime: pd.DataFrame | None = None,
    risk_by_factor_group: pd.DataFrame | None = None,
    output_path: Path | None = None,
    backtest_dir: Path | None = None,
) -> None:
    """Write comprehensive risk report as Markdown.
    
    Args:
        metrics: Dictionary with risk metrics
        exposure_timeseries: Optional DataFrame with exposure time-series
        risk_by_regime: Optional DataFrame with risk by regime
        risk_by_factor_group: Optional DataFrame with risk by factor group
        output_path: Output file path
        backtest_dir: Optional backtest directory (for metadata)
    """
    if output_path is None:
        raise ValueError("output_path is required")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Risk Report\n\n")
        
        if backtest_dir:
            f.write(f"**Backtest Directory:** `{backtest_dir}`\n\n")
        
        # Global Risk Metrics
        f.write("## Global Risk Metrics\n\n")
        
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        # Format metrics
        metric_display_names = {
            "mean_return_annualized": "Mean Return (Annualized)",
            "vol_annualized": "Volatility (Annualized)",
            "sharpe": "Sharpe Ratio",
            "sortino": "Sortino Ratio",
            "max_drawdown": "Max Drawdown (%)",
            "calmar": "Calmar Ratio",
            "skew": "Skewness",
            "kurtosis": "Kurtosis",
            "var_95": "VaR (95%)",
            "cvar_95": "CVaR / ES (95%)",
            "n_periods": "Number of Periods",
        }
        
        for key, display_name in metric_display_names.items():
            value = metrics.get(key)
            if value is None:
                value_str = "N/A"
            elif isinstance(value, float):
                # Format based on metric type
                if key in ["sharpe", "sortino", "calmar"]:
                    value_str = f"{value:.4f}"
                elif key in ["max_drawdown", "var_95", "cvar_95"]:
                    value_str = f"{value:.2f}%"
                elif key in ["mean_return_annualized", "vol_annualized"]:
                    value_str = f"{value:.2%}"
                elif key in ["skew", "kurtosis"]:
                    value_str = f"{value:.3f}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            
            f.write(f"| {display_name} | {value_str} |\n")
        
        f.write("\n")
        
        # Exposure Analysis (if available)
        if exposure_timeseries is not None and not exposure_timeseries.empty:
            f.write("## Exposure Analysis\n\n")
            
            avg_gross = exposure_timeseries["gross_exposure"].mean()
            avg_net = exposure_timeseries["net_exposure"].mean()
            avg_positions = exposure_timeseries["n_positions"].mean()
            avg_hhi = exposure_timeseries["hhi_concentration"].mean()
            
            f.write("| Metric | Average Value |\n")
            f.write("|--------|---------------|\n")
            f.write(f"| Average Gross Exposure | {avg_gross:.2%} |\n")
            f.write(f"| Average Net Exposure | {avg_net:.2%} |\n")
            f.write(f"| Average Number of Positions | {avg_positions:.1f} |\n")
            f.write(f"| Average HHI Concentration | {avg_hhi:.4f} |\n")
            f.write("\n")
        
        # Risk by Regime (if available)
        if risk_by_regime is not None and not risk_by_regime.empty:
            f.write("## Risk by Regime\n\n")
            
            f.write("| Regime | Periods | Mean Return | Volatility | Sharpe | MaxDD | Total Return |\n")
            f.write("|--------|---------|-------------|------------|--------|-------|--------------|\n")
            
            for _, row in risk_by_regime.iterrows():
                regime = row["regime"]
                n_periods = int(row["n_periods"])
                mean_ret = row.get("mean_return_annualized")
                vol = row.get("vol_annualized")
                sharpe = row.get("sharpe")
                max_dd = row.get("max_drawdown")
                total_ret = row.get("total_return")
                
                mean_ret_str = f"{mean_ret:.2%}" if mean_ret is not None else "N/A"
                vol_str = f"{vol:.2%}" if vol is not None else "N/A"
                sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
                max_dd_str = f"{max_dd:.2f}%" if max_dd is not None else "N/A"
                total_ret_str = f"{total_ret:.2%}" if total_ret is not None else "N/A"
                
                f.write(f"| {regime} | {n_periods} | {mean_ret_str} | {vol_str} | {sharpe_str} | {max_dd_str} | {total_ret_str} |\n")
            
            f.write("\n")
        
        # Risk by Factor Group (if available)
        if risk_by_factor_group is not None and not risk_by_factor_group.empty:
            f.write("## Performance Attribution by Factor Group\n\n")
            
            f.write("| Factor Group | Factors | Correlation | Avg Exposure | Periods |\n")
            f.write("|--------------|---------|-------------|--------------|----------|\n")
            
            for _, row in risk_by_factor_group.iterrows():
                group = row["factor_group"]
                factors = row.get("factors", "N/A")
                corr = row.get("correlation_with_returns")
                exposure = row.get("avg_exposure")
                n_periods = int(row.get("n_periods", 0))
                
                corr_str = f"{corr:.4f}" if corr is not None else "N/A"
                exposure_str = f"{exposure:.4f}" if exposure is not None else "N/A"
                
                f.write(f"| {group} | {factors} | {corr_str} | {exposure_str} | {n_periods} |\n")
            
            f.write("\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        
        f.write("### Risk Metrics\n")
        f.write("- **Sharpe Ratio**: Risk-adjusted return (higher is better, > 1.0 is good)\n")
        f.write("- **Sortino Ratio**: Downside-risk-adjusted return (penalizes only negative volatility)\n")
        f.write("- **Max Drawdown**: Largest peak-to-trough decline (negative value, smaller magnitude is better)\n")
        f.write("- **Calmar Ratio**: CAGR / |Max Drawdown| (higher is better)\n")
        f.write("- **Skewness**: Distribution asymmetry (negative = left-skewed, frequent large losses)\n")
        f.write("- **Kurtosis**: Tail heaviness (high = frequent extreme events)\n")
        f.write("- **VaR (95%)**: Value at Risk (worst 5% of outcomes, as return percentile)\n")
        f.write("- **CVaR / ES (95%)**: Expected Shortfall (average loss in worst 5% scenarios)\n\n")
        
        if risk_by_regime is not None and not risk_by_regime.empty:
            f.write("### Risk by Regime\n")
            f.write("- Risk metrics segmented by identified market regimes.\n")
            f.write("- Compare Sharpe, Volatility, and Max Drawdown across regimes to identify regime-specific risks.\n\n")
        
        if risk_by_factor_group is not None and not risk_by_factor_group.empty:
            f.write("### Factor Attribution\n")
            f.write("- **Correlation**: Correlation between portfolio factor scores and portfolio returns.\n")
            f.write("- **Avg Exposure**: Average portfolio exposure to factors in this group.\n")
            f.write("- Positive correlation suggests the factor group contributes positively to returns.\n\n")


def generate_risk_report(
    backtest_dir: Path,
    regime_file: Path | None = None,
    factor_panel_file: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    """Generate comprehensive risk report from backtest results.
    
    Args:
        backtest_dir: Directory containing backtest outputs
        regime_file: Optional path to regime state file
        factor_panel_file: Optional path to factor panel file
        output_dir: Output directory (default: same as backtest_dir)
    
    Returns:
        Exit code (0 = success, 1 = error)
    """
    if not backtest_dir.exists():
        logger.error(f"Backtest directory does not exist: {backtest_dir}")
        return 1
    
    if output_dir is None:
        output_dir = backtest_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating risk report from: {backtest_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load backtest files
    logger.info("Loading backtest files...")
    equity_df, positions_df, trades_df = find_backtest_files(backtest_dir)
    
    if equity_df is None:
        logger.error("Could not find equity curve file. Required for risk report.")
        return 1
    
    # Ensure timestamp is datetime
    if "timestamp" in equity_df.columns:
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
        equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)
    
    # Step 2: Compute returns
    logger.info("Computing returns from equity curve...")
    if "returns" not in equity_df.columns:
        equity_series = equity_df["equity"].copy()
        returns = _compute_returns(equity_series)
        # Align returns with equity timestamps (shift by 1, since returns[i] = (equity[i] / equity[i-1]) - 1)
        if len(equity_df) > 1:
            returns_series = pd.Series(
                returns.values,
                index=equity_df["timestamp"].iloc[1:],
                name="return"
            )
        else:
            logger.error("Equity curve has less than 2 data points. Cannot compute returns.")
            return 1
    else:
        returns_series = pd.Series(
            equity_df["returns"].values,
            index=equity_df["timestamp"],
            name="return"
        ).dropna()
    
    # Infer frequency
    freq = infer_freq_from_equity(equity_df)
    logger.info(f"Inferred trading frequency: {freq}")
    
    # Step 3: Compute basic risk metrics
    logger.info("Computing basic risk metrics...")
    try:
        risk_metrics = compute_basic_risk_metrics(
            returns=returns_series,
            freq=freq,
            risk_free_rate=0.0,
        )
        logger.info(f"Computed {len(risk_metrics)} risk metrics")
    except Exception as e:
        logger.error(f"Failed to compute risk metrics: {e}", exc_info=True)
        return 1
    
    # Step 4: Compute exposure time-series (if positions available)
    exposure_timeseries = None
    if positions_df is not None:
        logger.info("Computing exposure time-series...")
        try:
            # Check if positions have weight column
            if "weight" not in positions_df.columns and "qty" in positions_df.columns:
                logger.warning("Positions have 'qty' but no 'weight'. Cannot compute exposure without weights.")
            elif "weight" in positions_df.columns:
                exposure_timeseries = compute_exposure_timeseries(
                    positions=positions_df,
                    trades=trades_df,
                    equity=equity_df,
                    freq=freq,
                )
                logger.info(f"Computed exposure time-series: {len(exposure_timeseries)} timestamps")
        except Exception as e:
            logger.warning(f"Failed to compute exposure time-series: {e}")
    
    # Step 5: Compute risk by regime (if regime file available)
    risk_by_regime = None
    if regime_file is not None and regime_file.exists():
        logger.info(f"Loading regime state from {regime_file}...")
        try:
            regime_state_df = load_dataframe(regime_file, expected_columns=["timestamp", "regime_label"])
            if regime_state_df is not None:
                risk_by_regime = compute_risk_by_regime(
                    returns=returns_series,
                    regime_state_df=regime_state_df,
                    trades=trades_df,
                    freq=freq,
                    risk_free_rate=0.0,
                )
                logger.info(f"Computed risk by regime: {len(risk_by_regime)} regimes")
        except Exception as e:
            logger.warning(f"Failed to compute risk by regime: {e}")
    
    # Step 6: Compute risk by factor group (if factor file available)
    risk_by_factor_group = None
    if factor_panel_file is not None and factor_panel_file.exists() and positions_df is not None:
        logger.info(f"Loading factor panel from {factor_panel_file}...")
        try:
            factor_panel_df = load_dataframe(factor_panel_file)
            if factor_panel_df is not None and "symbol" in factor_panel_df.columns:
                # Default factor groups
                factor_groups = {
                    "Trend": ["returns_12m", "trend_strength_50", "trend_strength_200"],
                    "Vol/Liq": ["rv_20", "vov_20_60", "turnover_20d"],
                    "Earnings": ["earnings_eps_surprise_last", "post_earnings_drift_20d"],
                    "Insider": ["insider_net_notional_60d", "insider_buy_ratio_60d"],
                    "News/Macro": ["news_sentiment_trend_20d", "macro_growth_regime"],
                }
                
                risk_by_factor_group = compute_risk_by_factor_group(
                    returns=returns_series,
                    factor_panel_df=factor_panel_df,
                    positions_df=positions_df,
                    factor_groups=factor_groups,
                )
                logger.info(f"Computed risk by factor group: {len(risk_by_factor_group)} groups")
        except Exception as e:
            logger.warning(f"Failed to compute risk by factor group: {e}")
    
    # Step 7: Write outputs
    logger.info("Writing risk report outputs...")
    
    # Risk summary CSV
    summary_csv_path = output_dir / "risk_summary.csv"
    write_risk_summary_csv(risk_metrics, summary_csv_path)
    
    # Risk by regime CSV (if available)
    if risk_by_regime is not None:
        regime_csv_path = output_dir / "risk_by_regime.csv"
        risk_by_regime.to_csv(regime_csv_path, index=False)
        logger.info(f"Saved risk by regime to {regime_csv_path}")
    
    # Risk by factor group CSV (if available)
    if risk_by_factor_group is not None:
        factor_csv_path = output_dir / "risk_by_factor_group.csv"
        risk_by_factor_group.to_csv(factor_csv_path, index=False)
        logger.info(f"Saved risk by factor group to {factor_csv_path}")
    
    # Exposure time-series CSV (if available)
    if exposure_timeseries is not None:
        exposure_csv_path = output_dir / "exposure_timeseries.csv"
        exposure_timeseries.to_csv(exposure_csv_path, index=False)
        logger.info(f"Saved exposure time-series to {exposure_csv_path}")
    
    # Markdown report
    report_md_path = output_dir / "risk_report.md"
    write_risk_report_markdown(
        metrics=risk_metrics,
        exposure_timeseries=exposure_timeseries,
        risk_by_regime=risk_by_regime,
        risk_by_factor_group=risk_by_factor_group,
        output_path=report_md_path,
        backtest_dir=backtest_dir,
    )
    logger.info(f"Saved risk report to {report_md_path}")
    
    logger.info("Risk report generation completed successfully")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive risk report from backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic risk report from backtest directory
  python scripts/generate_risk_report.py --backtest-dir output/backtests/experiment_123/
  
  # With regime data
  python scripts/generate_risk_report.py --backtest-dir output/backtests/experiment_123/ --regime-file output/regime/regime_state.parquet
  
  # With factor attribution
  python scripts/generate_risk_report.py --backtest-dir output/backtests/experiment_123/ --factor-panel-file output/factor_analysis/factors.parquet
        """
    )
    
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        required=True,
        help="Path to backtest output directory (should contain equity_curve.csv/parquet, etc.)"
    )
    
    parser.add_argument(
        "--regime-file",
        type=Path,
        default=None,
        help="Optional path to regime state file (parquet or csv)"
    )
    
    parser.add_argument(
        "--factor-panel-file",
        type=Path,
        default=None,
        help="Optional path to factor panel file (parquet or csv) for attribution"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --backtest-dir)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    backtest_dir = args.backtest_dir.resolve() if args.backtest_dir.is_absolute() else (ROOT / args.backtest_dir).resolve()
    
    regime_file = None
    if args.regime_file:
        regime_file = args.regime_file.resolve() if args.regime_file.is_absolute() else (ROOT / args.regime_file).resolve()
    
    factor_panel_file = None
    if args.factor_panel_file:
        factor_panel_file = args.factor_panel_file.resolve() if args.factor_panel_file.is_absolute() else (ROOT / args.factor_panel_file).resolve()
    
    output_dir = None
    if args.output_dir:
        output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    
    return generate_risk_report(
        backtest_dir=backtest_dir,
        regime_file=regime_file,
        factor_panel_file=factor_panel_file,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())

