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

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.qa.metrics import _compute_returns
from src.assembled_core.risk.regime_analysis import (
    RegimeConfig,
    classify_regimes_from_index,
    summarize_metrics_by_regime,
)
from src.assembled_core.risk.risk_metrics import (
    compute_basic_risk_metrics,
    compute_exposure_timeseries,
    compute_risk_by_factor_group,
    compute_risk_by_regime,
)
from src.assembled_core.risk.factor_exposures import (
    FactorExposureConfig,
    compute_factor_exposures,
    summarize_factor_exposures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_dataframe(
    file_path: Path, expected_columns: list[str] | None = None
) -> pd.DataFrame | None:
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
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], utc=True, errors="coerce"
                )
        else:
            logger.warning(f"Unknown file format: {file_path.suffix}. Skipping.")
            return None

        if expected_columns:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                logger.warning(
                    f"Missing expected columns in {file_path.name}: {', '.join(missing)}"
                )
                return None

        logger.info(
            f"Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def find_backtest_files(
    backtest_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
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
                equity_df = load_dataframe(
                    matches[0], expected_columns=["timestamp", "equity"]
                )
                if equity_df is not None:
                    break
        else:
            equity_path = backtest_dir / pattern
            equity_df = load_dataframe(
                equity_path, expected_columns=["timestamp", "equity"]
            )
            if equity_df is not None:
                break

    # Try to find positions
    for pattern in positions_patterns:
        if "*" in pattern:
            matches = list(backtest_dir.glob(pattern))
            if matches:
                positions_df = load_dataframe(
                    matches[0], expected_columns=["timestamp", "symbol"]
                )
                if positions_df is not None:
                    break
        else:
            positions_path = backtest_dir / pattern
            positions_df = load_dataframe(
                positions_path, expected_columns=["timestamp", "symbol"]
            )
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
    regime_metrics: pd.DataFrame | None = None,
    factor_exposures_summary: pd.DataFrame | None = None,
    paper_track_mode: bool = False,
    equity_df: pd.DataFrame | None = None,
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

        # Paper Track: Performance Since Inception Section
        if paper_track_mode and equity_df is not None and not equity_df.empty:
            f.write("## Performance Since Inception\n\n")
            try:
                # Get first and last date
                if "date" in equity_df.columns:
                    equity_df_sorted = equity_df.sort_values("date")
                    dates = equity_df_sorted["date"]
                    first_date = pd.to_datetime(dates.iloc[0])
                    last_date = pd.to_datetime(dates.iloc[-1])
                    equity_series = equity_df_sorted["equity"].values
                elif "timestamp" in equity_df.columns:
                    equity_df_sorted = equity_df.sort_values("timestamp")
                    dates = equity_df_sorted["timestamp"]
                    first_date = pd.to_datetime(dates.iloc[0], utc=True)
                    last_date = pd.to_datetime(dates.iloc[-1], utc=True)
                    equity_series = equity_df_sorted["equity"].values
                else:
                    first_date = None
                    last_date = None
                    equity_series = equity_df["equity"].values if "equity" in equity_df.columns else None

                if first_date is not None and last_date is not None and equity_series is not None and len(equity_series) >= 2:
                    # Compute since inception metrics
                    start_equity = float(equity_series[0])
                    end_equity = float(equity_series[-1])
                    total_return = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0 else 0.0

                    # Compute returns for volatility and Sharpe
                    returns = pd.Series(equity_series).pct_change().dropna()
                    vol_annualized = float(returns.std() * (252**0.5)) * 100.0 if len(returns) > 1 else None
                    sharpe_since_inception = (
                        float(returns.mean() / returns.std() * (252**0.5))
                        if len(returns) > 1 and returns.std() > 0
                        else None
                    )

                    # Compute max drawdown since inception
                    cumulative = pd.Series(equity_series) / equity_series[0]
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max * 100.0
                    max_dd_since_inception = float(drawdown.min()) if len(drawdown) > 0 else None

                    # Compute days since inception
                    days_since_inception = (last_date - first_date).days

                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| **Inception Date** | {first_date.strftime('%Y-%m-%d')} |\n")
                    f.write(f"| **Days Since Inception** | {days_since_inception} |\n")
                    f.write(f"| **Total Return** | {total_return:.2f}% |\n")
                    if vol_annualized is not None:
                        f.write(f"| **Volatility (Annualized)** | {vol_annualized:.2f}% |\n")
                    if sharpe_since_inception is not None:
                        f.write(f"| **Sharpe Ratio** | {sharpe_since_inception:.4f} |\n")
                    if max_dd_since_inception is not None:
                        f.write(f"| **Max Drawdown** | {max_dd_since_inception:.2f}% |\n")
                    f.write("\n")
            except Exception as e:
                logger.warning(f"Failed to compute 'since inception' metrics: {e}", exc_info=True)
                f.write("*Could not compute 'since inception' metrics.*\n\n")

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

        # Performance by Regime (if available, from extended regime analysis)
        if regime_metrics is not None and not regime_metrics.empty:
            f.write("## Performance by Regime\n\n")

            f.write(
                "| Regime | Periods | Sharpe | Volatility | MaxDD | CAGR | Total Return | Sortino |\n"
            )
            f.write(
                "|--------|---------|--------|------------|-------|------|--------------|----------|\n"
            )

            for _, row in regime_metrics.iterrows():
                regime = row["regime_label"]
                n_periods = int(row["n_periods"])
                sharpe = row.get("sharpe")
                vol = row.get("volatility")
                max_dd = row.get("max_drawdown")
                cagr = row.get("cagr")
                total_ret = row.get("total_return")
                sortino = row.get("sortino")

                sharpe_str = (
                    f"{sharpe:.4f}"
                    if sharpe is not None and not pd.isna(sharpe)
                    else "N/A"
                )
                vol_str = (
                    f"{vol:.2%}" if vol is not None and not pd.isna(vol) else "N/A"
                )
                max_dd_str = (
                    f"{max_dd:.2f}%"
                    if max_dd is not None and not pd.isna(max_dd)
                    else "N/A"
                )
                cagr_str = (
                    f"{cagr:.2%}" if cagr is not None and not pd.isna(cagr) else "N/A"
                )
                total_ret_str = (
                    f"{total_ret:.2%}"
                    if total_ret is not None and not pd.isna(total_ret)
                    else "N/A"
                )
                sortino_str = (
                    f"{sortino:.4f}"
                    if sortino is not None and not pd.isna(sortino)
                    else "N/A"
                )

                f.write(
                    f"| {regime} | {n_periods} | {sharpe_str} | {vol_str} | {max_dd_str} | {cagr_str} | {total_ret_str} | {sortino_str} |\n"
                )

            f.write("\n")

            # Add verbal summary
            f.write("### Regime Performance Summary\n\n")

            # Find best and worst performing regimes
            sharpe_col = "sharpe"
            if sharpe_col in regime_metrics.columns:
                regime_metrics_clean = regime_metrics.dropna(subset=[sharpe_col])
                if not regime_metrics_clean.empty:
                    best_regime = regime_metrics_clean.loc[
                        regime_metrics_clean[sharpe_col].idxmax()
                    ]
                    worst_regime = regime_metrics_clean.loc[
                        regime_metrics_clean[sharpe_col].idxmin()
                    ]

                    best_name = best_regime["regime_label"]
                    worst_name = worst_regime["regime_label"]
                    best_sharpe = best_regime[sharpe_col]
                    worst_sharpe = worst_regime[sharpe_col]

                    f.write(
                        f"- **Best Performing Regime**: {best_name} (Sharpe: {best_sharpe:.4f})\n"
                    )
                    f.write(
                        f"- **Worst Performing Regime**: {worst_name} (Sharpe: {worst_sharpe:.4f})\n"
                    )

                    if worst_name.lower() in ["crisis", "bear"]:
                        f.write(
                            f"- **Note**: Strategy performs significantly worse in {worst_name} regimes.\n"
                        )
                    elif best_name.lower() in ["bull", "reflation"]:
                        f.write(
                            f"- **Note**: Strategy performs best in {best_name} regimes.\n"
                        )

            f.write("\n")

        # Risk by Regime (if available, legacy format)
        elif risk_by_regime is not None and not risk_by_regime.empty:
            f.write("## Risk by Regime\n\n")

            f.write(
                "| Regime | Periods | Mean Return | Volatility | Sharpe | MaxDD | Total Return |\n"
            )
            f.write(
                "|--------|---------|-------------|------------|--------|-------|--------------|\n"
            )

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

                f.write(
                    f"| {regime} | {n_periods} | {mean_ret_str} | {vol_str} | {sharpe_str} | {max_dd_str} | {total_ret_str} |\n"
                )

            f.write("\n")

        # Risk by Factor Group (if available)
        if risk_by_factor_group is not None and not risk_by_factor_group.empty:
            f.write("## Performance Attribution by Factor Group\n\n")

            f.write(
                "| Factor Group | Factors | Correlation | Avg Exposure | Periods |\n"
            )
            f.write(
                "|--------------|---------|-------------|--------------|----------|\n"
            )

            for _, row in risk_by_factor_group.iterrows():
                group = row["factor_group"]
                factors = row.get("factors", "N/A")
                corr = row.get("correlation_with_returns")
                exposure = row.get("avg_exposure")
                n_periods = int(row.get("n_periods", 0))

                corr_str = f"{corr:.4f}" if corr is not None else "N/A"
                exposure_str = f"{exposure:.4f}" if exposure is not None else "N/A"

                f.write(
                    f"| {group} | {factors} | {corr_str} | {exposure_str} | {n_periods} |\n"
                )

            f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        f.write("### Risk Metrics\n")
        f.write(
            "- **Sharpe Ratio**: Risk-adjusted return (higher is better, > 1.0 is good)\n"
        )
        f.write(
            "- **Sortino Ratio**: Downside-risk-adjusted return (penalizes only negative volatility)\n"
        )
        f.write(
            "- **Max Drawdown**: Largest peak-to-trough decline (negative value, smaller magnitude is better)\n"
        )
        f.write("- **Calmar Ratio**: CAGR / |Max Drawdown| (higher is better)\n")
        f.write(
            "- **Skewness**: Distribution asymmetry (negative = left-skewed, frequent large losses)\n"
        )
        f.write("- **Kurtosis**: Tail heaviness (high = frequent extreme events)\n")
        f.write(
            "- **VaR (95%)**: Value at Risk (worst 5% of outcomes, as return percentile)\n"
        )
        f.write(
            "- **CVaR / ES (95%)**: Expected Shortfall (average loss in worst 5% scenarios)\n\n"
        )

        if (
            regime_metrics is not None
            and not regime_metrics.empty
            or (risk_by_regime is not None and not risk_by_regime.empty)
        ):
            f.write("### Risk by Regime\n")
            f.write("- Risk metrics segmented by identified market regimes.\n")
            f.write(
                "- Compare Sharpe, Volatility, and Max Drawdown across regimes to identify regime-specific risks.\n"
            )
            f.write("- **Bull**: Positive trend, moderate volatility\n")
            f.write("- **Bear**: Negative trend, elevated volatility\n")
            f.write("- **Crisis**: Deep drawdown or very high volatility\n")
            f.write("- **Sideways**: Moderate trend, moderate volatility\n")
            f.write("- **Reflation**: Strong recovery after crisis\n\n")

        if risk_by_factor_group is not None and not risk_by_factor_group.empty:
            f.write("### Factor Attribution\n")
            f.write(
                "- **Correlation**: Correlation between portfolio factor scores and portfolio returns.\n"
            )
            f.write(
                "- **Avg Exposure**: Average portfolio exposure to factors in this group.\n"
            )
            f.write(
                "- Positive correlation suggests the factor group contributes positively to returns.\n\n"
            )

        # Factor Exposures (if available)
        if factor_exposures_summary is not None and not factor_exposures_summary.empty:
            f.write("## Factor Exposures\n\n")

            # Show top 5 factors by |mean_beta|
            top_factors = factor_exposures_summary.head(5).copy()

            f.write(
                "| Factor | Mean Beta | Std Beta | Mean R² | Mean Residual Vol | Windows |\n"
            )
            f.write(
                "|--------|-----------|----------|---------|-------------------|----------|\n"
            )

            for _, row in top_factors.iterrows():
                factor = row["factor"]
                mean_beta = row["mean_beta"]
                std_beta = row["std_beta"]
                mean_r2 = row["mean_r2"]
                mean_residual_vol = row["mean_residual_vol"]
                n_windows = int(row["n_windows"])

                mean_beta_str = f"{mean_beta:.4f}" if not pd.isna(mean_beta) else "N/A"
                std_beta_str = f"{std_beta:.4f}" if not pd.isna(std_beta) else "N/A"
                mean_r2_str = f"{mean_r2:.4f}" if not pd.isna(mean_r2) else "N/A"
                mean_residual_vol_str = (
                    f"{mean_residual_vol:.2%}"
                    if not pd.isna(mean_residual_vol)
                    else "N/A"
                )

                f.write(
                    f"| {factor} | {mean_beta_str} | {std_beta_str} | {mean_r2_str} | {mean_residual_vol_str} | {n_windows} |\n"
                )

            f.write("\n")

            # Verbal summary
            f.write("### Factor Exposure Summary\n\n")

            # Find strongest positive and negative exposures
            pos_exposures = factor_exposures_summary[
                factor_exposures_summary["mean_beta"] > 0
            ].copy()
            neg_exposures = factor_exposures_summary[
                factor_exposures_summary["mean_beta"] < 0
            ].copy()

            if not pos_exposures.empty:
                strongest_pos = pos_exposures.loc[pos_exposures["mean_beta"].idxmax()]
                f.write(
                    f"- **Strongest Positive Exposure**: {strongest_pos['factor']} (beta: {strongest_pos['mean_beta']:.4f}, R²: {strongest_pos['mean_r2']:.4f})\n"
                )

            if not neg_exposures.empty:
                strongest_neg = neg_exposures.loc[neg_exposures["mean_beta"].idxmin()]
                f.write(
                    f"- **Strongest Negative Exposure**: {strongest_neg['factor']} (beta: {strongest_neg['mean_beta']:.4f}, R²: {strongest_neg['mean_r2']:.4f})\n"
                )

            # Average R²
            avg_r2 = factor_exposures_summary["mean_r2"].mean()
            if not pd.isna(avg_r2):
                f.write(
                    f"- **Average R²**: {avg_r2:.4f} (explained variance across factors)\n"
                )

            f.write("\n")


def generate_risk_report(
    backtest_dir: Path,
    regime_file: Path | None = None,
    factor_panel_file: Path | None = None,
    output_dir: Path | None = None,
    benchmark_symbol: str | None = None,
    benchmark_file: Path | None = None,
    enable_regime_analysis: bool = False,
    enable_factor_exposures: bool = False,
    factor_returns_file: Path | None = None,
    factor_exposures_window: int = 252,
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
                returns.values, index=equity_df["timestamp"].iloc[1:], name="return"
            )
        else:
            logger.error(
                "Equity curve has less than 2 data points. Cannot compute returns."
            )
            return 1
    else:
        returns_series = pd.Series(
            equity_df["returns"].values, index=equity_df["timestamp"], name="return"
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
                logger.warning(
                    "Positions have 'qty' but no 'weight'. Cannot compute exposure without weights."
                )
            elif "weight" in positions_df.columns:
                exposure_timeseries = compute_exposure_timeseries(
                    positions=positions_df,
                    trades=trades_df,
                    equity=equity_df,
                    freq=freq,
                )
                logger.info(
                    f"Computed exposure time-series: {len(exposure_timeseries)} timestamps"
                )
        except Exception as e:
            logger.warning(f"Failed to compute exposure time-series: {e}")

    # Step 5: Compute risk by regime
    risk_by_regime = None  # Legacy format (from regime_file)
    regime_metrics = None  # Extended format (from classify_regimes_from_index)

    # Option 1: Load regime state from file (legacy)
    if regime_file is not None and regime_file.exists():
        logger.info(f"Loading regime state from {regime_file}...")
        try:
            regime_state_df = load_dataframe(
                regime_file, expected_columns=["timestamp", "regime_label"]
            )
            if regime_state_df is not None:
                risk_by_regime = compute_risk_by_regime(
                    returns=returns_series,
                    regime_state_df=regime_state_df,
                    trades=trades_df,
                    freq=freq,
                    risk_free_rate=0.0,
                )
                logger.info(
                    f"Computed risk by regime (legacy): {len(risk_by_regime)} regimes"
                )
        except Exception as e:
            logger.warning(f"Failed to compute risk by regime from file: {e}")

    # Option 2: Classify regimes from benchmark/index (extended analysis)
    if enable_regime_analysis and (
        benchmark_symbol is not None or benchmark_file is not None
    ):
        logger.info("Computing regime classification from benchmark...")
        try:
            benchmark_returns = None

            # Load benchmark returns
            if benchmark_file is not None and benchmark_file.exists():
                logger.info(f"Loading benchmark from file: {benchmark_file}")
                benchmark_df = load_dataframe(benchmark_file)
                if benchmark_df is not None:
                    if "returns" in benchmark_df.columns:
                        benchmark_returns = pd.Series(
                            benchmark_df["returns"].values,
                            index=pd.to_datetime(benchmark_df["timestamp"], utc=True),
                        )
                    elif "close" in benchmark_df.columns:
                        # Compute returns from prices
                        benchmark_prices = pd.Series(
                            benchmark_df["close"].values,
                            index=pd.to_datetime(benchmark_df["timestamp"], utc=True),
                        ).sort_index()
                        benchmark_returns = benchmark_prices.pct_change().dropna()

            elif benchmark_symbol is not None:
                logger.info(f"Loading benchmark prices for symbol: {benchmark_symbol}")
                try:
                    settings = get_settings()
                    # Try to load benchmark prices (assuming daily frequency for regime classification)
                    benchmark_prices_df = load_eod_prices(
                        freq="1d",
                        symbols=[benchmark_symbol],
                        data_source="local",
                        start_date=equity_df["timestamp"].min(),
                        end_date=equity_df["timestamp"].max(),
                        settings=settings,
                    )
                    if (
                        benchmark_prices_df is not None
                        and not benchmark_prices_df.empty
                    ):
                        # Filter to benchmark symbol and compute returns
                        benchmark_symbol_data = benchmark_prices_df[
                            benchmark_prices_df["symbol"] == benchmark_symbol
                        ].sort_values("timestamp")
                        if not benchmark_symbol_data.empty:
                            benchmark_prices = pd.Series(
                                benchmark_symbol_data["close"].values,
                                index=pd.to_datetime(
                                    benchmark_symbol_data["timestamp"], utc=True
                                ),
                            )
                            benchmark_returns = benchmark_prices.pct_change().dropna()
                except Exception as e:
                    logger.warning(
                        f"Failed to load benchmark prices for {benchmark_symbol}: {e}"
                    )

            # Classify regimes if we have benchmark returns
            if benchmark_returns is not None and not benchmark_returns.empty:
                # Align benchmark returns with equity timestamps
                common_index = equity_df["timestamp"].intersection(
                    benchmark_returns.index
                )
                if len(common_index) > 0:
                    benchmark_returns_aligned = benchmark_returns.loc[common_index]

                    # Classify regimes
                    regime_config = RegimeConfig(
                        vol_window=20,
                        trend_ma_window=200,
                        drawdown_threshold=-0.20,
                        vol_threshold_high=0.30,
                        vol_threshold_low=0.15,
                    )
                    regimes = classify_regimes_from_index(
                        benchmark_returns_aligned, regime_config
                    )

                    # Compute extended metrics by regime
                    equity_series = pd.Series(
                        equity_df["equity"].values,
                        index=pd.to_datetime(equity_df["timestamp"], utc=True),
                    )

                    # Align equity with regimes
                    equity_aligned = equity_series.loc[
                        equity_series.index.intersection(regimes.index)
                    ]
                    regimes_aligned = regimes.loc[equity_aligned.index]

                    if len(equity_aligned) > 0 and len(regimes_aligned) > 0:
                        regime_metrics = summarize_metrics_by_regime(
                            equity=equity_aligned,
                            regimes=regimes_aligned,
                            trades=trades_df,
                            freq=freq,
                        )
                        logger.info(
                            f"Computed extended regime metrics: {len(regime_metrics)} regimes"
                        )
                    else:
                        logger.warning(
                            "Could not align equity with regimes for extended analysis"
                        )
                else:
                    logger.warning(
                        "No overlapping timestamps between equity and benchmark"
                    )
            else:
                logger.warning(
                    "Could not load benchmark returns for regime classification"
                )

        except Exception as e:
            logger.warning(
                f"Failed to compute regime analysis from benchmark: {e}", exc_info=True
            )

    # Step 6: Compute factor exposures (if enabled)
    factor_exposures_detail = None
    factor_exposures_summary = None
    if (
        enable_factor_exposures
        and factor_returns_file is not None
        and factor_returns_file.exists()
    ):
        logger.info(f"Loading factor returns from {factor_returns_file}...")
        try:
            factor_returns_df = load_dataframe(factor_returns_file)
            if factor_returns_df is not None:
                # Ensure timestamp is index or column
                if "timestamp" in factor_returns_df.columns and not isinstance(
                    factor_returns_df.index, pd.DatetimeIndex
                ):
                    factor_returns_df["timestamp"] = pd.to_datetime(
                        factor_returns_df["timestamp"], utc=True
                    )
                    factor_returns_df = factor_returns_df.set_index("timestamp")
                elif not isinstance(factor_returns_df.index, pd.DatetimeIndex):
                    logger.warning(
                        "Factor returns file must have timestamp as index or column"
                    )
                    factor_returns_df = None

                if factor_returns_df is not None and not factor_returns_df.empty:
                    # Get factor columns (exclude timestamp if it's a column)
                    factor_cols = [
                        col for col in factor_returns_df.columns if col != "timestamp"
                    ]
                    if len(factor_cols) > 0:
                        # Create config
                        # Map freq from "1d"/"5min" to "1d"/"1w"/"1m" (factor_exposures uses "1d", "1w", "1m")
                        freq_for_exposures = "1d"  # Default to daily
                        if freq == "1d":
                            freq_for_exposures = "1d"
                        elif freq == "5min":
                            # For intraday, use daily aggregation or assume daily factors
                            freq_for_exposures = "1d"

                        exposure_config = FactorExposureConfig(
                            freq=freq_for_exposures,
                            window_size=factor_exposures_window,
                            min_obs=60,
                            mode="rolling",
                            add_constant=True,
                            standardize_factors=True,
                            regression_method="ols",
                        )

                        # Compute exposures
                        factor_exposures_detail = compute_factor_exposures(
                            strategy_returns=returns_series,
                            factor_returns=factor_returns_df[factor_cols],
                            config=exposure_config,
                        )

                        if (
                            factor_exposures_detail is not None
                            and not factor_exposures_detail.empty
                        ):
                            logger.info(
                                f"Computed factor exposures: {len(factor_exposures_detail)} windows"
                            )

                            # Summarize
                            factor_exposures_summary = summarize_factor_exposures(
                                exposures=factor_exposures_detail,
                                config=exposure_config,
                            )
                            logger.info(
                                f"Summarized factor exposures: {len(factor_exposures_summary)} factors"
                            )
                        else:
                            logger.warning(
                                "Factor exposure computation returned empty result"
                            )
                    else:
                        logger.warning("No factor columns found in factor returns file")
        except Exception as e:
            logger.warning(f"Failed to compute factor exposures: {e}", exc_info=True)

    # Step 7: Compute risk by factor group (if factor file available)
    risk_by_factor_group = None
    if (
        factor_panel_file is not None
        and factor_panel_file.exists()
        and positions_df is not None
    ):
        logger.info(f"Loading factor panel from {factor_panel_file}...")
        try:
            factor_panel_df = load_dataframe(factor_panel_file)
            if factor_panel_df is not None and "symbol" in factor_panel_df.columns:
                # Default factor groups
                factor_groups = {
                    "Trend": ["returns_12m", "trend_strength_50", "trend_strength_200"],
                    "Vol/Liq": ["rv_20", "vov_20_60", "turnover_20d"],
                    "Earnings": [
                        "earnings_eps_surprise_last",
                        "post_earnings_drift_20d",
                    ],
                    "Insider": ["insider_net_notional_60d", "insider_buy_ratio_60d"],
                    "News/Macro": ["news_sentiment_trend_20d", "macro_growth_regime"],
                }

                risk_by_factor_group = compute_risk_by_factor_group(
                    returns=returns_series,
                    factor_panel_df=factor_panel_df,
                    positions_df=positions_df,
                    factor_groups=factor_groups,
                )
                logger.info(
                    f"Computed risk by factor group: {len(risk_by_factor_group)} groups"
                )
        except Exception as e:
            logger.warning(f"Failed to compute risk by factor group: {e}")

    # Step 8: Write outputs
    logger.info("Writing risk report outputs...")

    # Risk summary CSV
    summary_csv_path = output_dir / "risk_summary.csv"
    write_risk_summary_csv(risk_metrics, summary_csv_path)

    # Risk by regime CSV (if available, legacy format)
    if risk_by_regime is not None:
        regime_csv_path = output_dir / "risk_by_regime.csv"
        risk_by_regime.to_csv(regime_csv_path, index=False)
        logger.info(f"Saved risk by regime (legacy) to {regime_csv_path}")

    # Extended regime metrics CSV (if available)
    if regime_metrics is not None:
        regime_metrics_csv_path = output_dir / "risk_by_regime_extended.csv"
        regime_metrics.to_csv(regime_metrics_csv_path, index=False)
        logger.info(f"Saved extended regime metrics to {regime_metrics_csv_path}")

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

    # Factor exposures detail CSV (if available)
    if factor_exposures_detail is not None and not factor_exposures_detail.empty:
        factor_exposures_detail_csv_path = output_dir / "factor_exposures_detail.csv"
        factor_exposures_detail.to_csv(factor_exposures_detail_csv_path)
        logger.info(
            f"Saved factor exposures detail to {factor_exposures_detail_csv_path}"
        )

    # Factor exposures summary CSV (if available)
    if factor_exposures_summary is not None and not factor_exposures_summary.empty:
        factor_exposures_summary_csv_path = output_dir / "factor_exposures_summary.csv"
        factor_exposures_summary.to_csv(factor_exposures_summary_csv_path, index=False)
        logger.info(
            f"Saved factor exposures summary to {factor_exposures_summary_csv_path}"
        )

    # Detect if this is paper track mode (check if backtest_dir is aggregates dir)
    is_paper_track = "aggregates" in str(backtest_dir) or (
        backtest_dir.parent.name != "backtests" if backtest_dir.parent else False
    )

    # Markdown report
    report_md_path = output_dir / "risk_report.md"
    write_risk_report_markdown(
        metrics=risk_metrics,
        exposure_timeseries=exposure_timeseries,
        risk_by_regime=risk_by_regime,
        risk_by_factor_group=risk_by_factor_group,
        output_path=report_md_path,
        backtest_dir=backtest_dir,
        regime_metrics=regime_metrics,
        factor_exposures_summary=factor_exposures_summary,
        paper_track_mode=is_paper_track,
        equity_df=equity_df,
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
        """,
    )

    parser.add_argument(
        "--backtest-dir",
        type=Path,
        required=True,
        help="Path to backtest output directory (should contain equity_curve.csv/parquet, etc.)",
    )

    parser.add_argument(
        "--regime-file",
        type=Path,
        default=None,
        help="Optional path to regime state file (parquet or csv)",
    )

    parser.add_argument(
        "--factor-panel-file",
        type=Path,
        default=None,
        help="Optional path to factor panel file (parquet or csv) for attribution",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as --backtest-dir)",
    )

    parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        help="Benchmark symbol (e.g., 'SPY', 'QQQ') for regime classification. Requires --enable-regime-analysis.",
    )

    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Path to benchmark file (CSV/Parquet) with timestamp and returns/close columns. Requires --enable-regime-analysis.",
    )

    parser.add_argument(
        "--enable-regime-analysis",
        action="store_true",
        help="Enable extended regime analysis from benchmark/index. Classifies regimes and computes performance by regime.",
    )

    parser.add_argument(
        "--enable-factor-exposures",
        action="store_true",
        help="Enable factor exposure analysis. Requires --factor-returns-file.",
    )

    parser.add_argument(
        "--factor-returns-file",
        type=Path,
        default=None,
        help="Path to factor returns file (CSV/Parquet) with timestamp and factor columns. Required if --enable-factor-exposures is set.",
    )

    parser.add_argument(
        "--factor-exposures-window",
        type=int,
        default=252,
        help="Rolling window size for factor exposure regression (default: 252 periods)",
    )

    args = parser.parse_args()

    # Resolve paths
    backtest_dir = (
        args.backtest_dir.resolve()
        if args.backtest_dir.is_absolute()
        else (ROOT / args.backtest_dir).resolve()
    )

    regime_file = None
    if args.regime_file:
        regime_file = (
            args.regime_file.resolve()
            if args.regime_file.is_absolute()
            else (ROOT / args.regime_file).resolve()
        )

    factor_panel_file = None
    if args.factor_panel_file:
        factor_panel_file = (
            args.factor_panel_file.resolve()
            if args.factor_panel_file.is_absolute()
            else (ROOT / args.factor_panel_file).resolve()
        )

    output_dir = None
    if args.output_dir:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (ROOT / args.output_dir).resolve()
        )

    benchmark_file = None
    if args.benchmark_file:
        benchmark_file = (
            args.benchmark_file.resolve()
            if args.benchmark_file.is_absolute()
            else (ROOT / args.benchmark_file).resolve()
        )

    factor_returns_file = None
    if args.factor_returns_file:
        factor_returns_file = (
            args.factor_returns_file.resolve()
            if args.factor_returns_file.is_absolute()
            else (ROOT / args.factor_returns_file).resolve()
        )

    if args.enable_factor_exposures and factor_returns_file is None:
        logger.error("--enable-factor-exposures requires --factor-returns-file")
        return 1

    return generate_risk_report(
        backtest_dir=backtest_dir,
        regime_file=regime_file,
        factor_panel_file=factor_panel_file,
        output_dir=output_dir,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_file=benchmark_file,
        enable_regime_analysis=args.enable_regime_analysis,
        enable_factor_exposures=args.enable_factor_exposures,
        factor_returns_file=factor_returns_file,
        factor_exposures_window=args.factor_exposures_window,
    )


if __name__ == "__main__":
    sys.exit(main())
