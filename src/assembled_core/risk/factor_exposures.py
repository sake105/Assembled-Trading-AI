"""Factor Exposure Analysis Module.

This module provides factor exposure analysis by regressing strategy returns
against factor returns using rolling or expanding windows.

Key features:
- Rolling/expanding window regression (OLS or Ridge)
- Factor exposure betas over time
- Summary statistics (mean beta, std beta, R², etc.)
- Annualized residual volatility

Integration:
- Works with strategy returns from BacktestResult.equity["daily_return"]
- Factor returns from Factor Store, Factor Analysis, or ML predictions
- Integrated into Risk Reports (optional factor exposure section)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import sklearn for regression
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available. Factor exposure analysis requires sklearn.linear_model."
    )


def _get_periods_per_year(freq: Literal["1d", "1w", "1m"]) -> int:
    """Get periods per year for a given frequency.

    Args:
        freq: Frequency string ("1d", "1w", or "1m")

    Returns:
        Number of periods per year
    """
    if freq == "1d":
        return 252  # Trading days per year
    elif freq == "1w":
        return 52  # Weeks per year
    elif freq == "1m":
        return 12  # Months per year
    else:
        # Default to daily
        return 252


@dataclass
class FactorExposureConfig:
    """Configuration for factor exposure analysis.

    Attributes:
        freq: Trading frequency ("1d", "1w", or "1m") for annualization
        window_size: Rolling window size in periods (default: 252 for daily = 1 year)
        min_obs: Minimum observations required for regression (default: 60)
        mode: Window mode - "rolling" or "expanding" (default: "rolling")
        add_constant: If True, add intercept term to regression (default: True)
        standardize_factors: If True, standardize factor returns before regression (default: True)
        regression_method: Regression method - "ols" or "ridge" (default: "ols")
        ridge_alpha: Ridge shrinkage parameter (only used if regression_method="ridge", default: 1.0)
        min_r2_for_report: Minimum R² to include in summary report (default: 0.0)
    """

    freq: Literal["1d", "1w", "1m"] = "1d"
    window_size: int = 252
    min_obs: int = 60
    mode: Literal["rolling", "expanding"] = "rolling"
    add_constant: bool = True
    standardize_factors: bool = True
    regression_method: Literal["ols", "ridge"] = "ols"
    ridge_alpha: float = 1.0
    min_r2_for_report: float = 0.0


def compute_factor_exposures(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    config: FactorExposureConfig | None = None,
) -> pd.DataFrame:
    """Compute rolling factor exposures (betas) for strategy returns.

    Args:
        strategy_returns: Strategy returns time-series (index = timestamp)
        factor_returns: Factor returns DataFrame (index = timestamp, columns = factor names)
        config: FactorExposureConfig with regression parameters (default: FactorExposureConfig())

    Returns:
        DataFrame with columns:
        - timestamp: Window end timestamp (index)
        - beta_<factor>: Beta for each factor (one column per factor)
        - intercept: Intercept term (if add_constant=True)
        - r2: R-squared of regression
        - residual_vol: Residual volatility (annualized)
        - n_obs: Number of observations in window

    Raises:
        ImportError: If sklearn is not available
        ValueError: If inputs are empty or misaligned

    Note:
        - Rolling regression: For each timestamp, regress strategy_returns[t-window:t] on factor_returns[t-window:t]
        - Expanding regression: For each timestamp, regress strategy_returns[:t] on factor_returns[:t]
        - Missing values (NaN) in factor_returns or strategy_returns are dropped before regression
        - Returns NaN for windows with insufficient observations (< min_obs)
        - Factor returns are standardized per window if standardize_factors=True
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "sklearn.linear_model is required for factor exposure analysis. "
            "Install sklearn: pip install scikit-learn"
        )

    if config is None:
        config = FactorExposureConfig()

    # Align strategy_returns and factor_returns (inner join on index)
    aligned = pd.DataFrame({"strategy_return": strategy_returns}).join(
        factor_returns, how="inner"
    )

    if aligned.empty:
        logger.warning(
            "No overlapping timestamps between strategy_returns and factor_returns"
        )
        return pd.DataFrame()

    # Get factor column names
    factor_cols = list(factor_returns.columns)
    if not factor_cols:
        raise ValueError("factor_returns must have at least one factor column")

    # Get periods per year for annualization
    periods_per_year = _get_periods_per_year(config.freq)

    # Initialize result list
    results = []

    # Iterate over timestamps (rolling or expanding window)
    for i in range(len(aligned)):
        end_idx = i + 1

        if config.mode == "rolling":
            start_idx = max(0, end_idx - config.window_size)
        else:  # expanding
            start_idx = 0

        # Get window data
        window_data = aligned.iloc[start_idx:end_idx].copy()

        # Drop rows with any NaN (in strategy_returns or any factor)
        window_data = window_data.dropna()

        n_obs = len(window_data)

        # Check minimum observations
        if n_obs < config.min_obs:
            # Skip this window, add NaN row
            row = {"timestamp": aligned.index[i]}
            for factor in factor_cols:
                row[f"beta_{factor}"] = np.nan
            if config.add_constant:
                row["intercept"] = np.nan
            row["r2"] = np.nan
            row["residual_vol"] = np.nan
            row["n_obs"] = n_obs
            results.append(row)
            continue

        # Extract y (strategy returns) and X (factor returns)
        y = window_data["strategy_return"].values
        X = window_data[factor_cols].values

        # Standardize factors if requested
        if config.standardize_factors:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        # Add constant term if requested
        if config.add_constant:
            X_with_const = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        else:
            X_with_const = X_scaled

        # Fit regression model
        if config.regression_method == "ols":
            model = LinearRegression(
                fit_intercept=False
            )  # We already added constant if needed
        else:  # ridge
            model = Ridge(alpha=config.ridge_alpha, fit_intercept=False)

        model.fit(X_with_const, y)

        # Get coefficients
        coefs = model.coef_
        if config.add_constant:
            intercept = coefs[0]
            betas = coefs[1:]
        else:
            intercept = 0.0
            betas = coefs

        # Compute predictions and residuals
        y_pred = model.predict(X_with_const)
        residuals = y - y_pred

        # Compute R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot > 1e-10:
            r2 = 1.0 - (ss_res / ss_tot)
        else:
            r2 = np.nan

        # Compute residual volatility (annualized)
        if len(residuals) > 1:
            residual_vol = np.std(residuals, ddof=1) * np.sqrt(periods_per_year)
        else:
            residual_vol = np.nan

        # Build result row
        row = {"timestamp": aligned.index[i]}
        for j, factor in enumerate(factor_cols):
            row[f"beta_{factor}"] = betas[j]
        if config.add_constant:
            row["intercept"] = intercept
        row["r2"] = r2
        row["residual_vol"] = residual_vol
        row["n_obs"] = n_obs

        results.append(row)

    # Convert to DataFrame
    if not results:
        return pd.DataFrame()

    exposures_df = pd.DataFrame(results)
    exposures_df = exposures_df.set_index("timestamp")

    return exposures_df


def summarize_factor_exposures(
    exposures: pd.DataFrame,
    config: FactorExposureConfig | None = None,
) -> pd.DataFrame:
    """Summarize factor exposures over time.

    Args:
        exposures: Output from compute_factor_exposures()
        config: FactorExposureConfig (for min_r2_for_report, default: FactorExposureConfig())

    Returns:
        DataFrame with one row per factor:
        - factor: Factor name
        - mean_beta: Mean beta over all windows
        - std_beta: Standard deviation of beta over time
        - mean_r2: Mean R² across all windows
        - median_r2: Median R² across all windows
        - mean_residual_vol: Mean residual volatility (annualized)
        - n_windows: Number of windows with valid regression (non-NaN)
        - n_windows_total: Total number of windows

    Note:
        - Only includes factors with mean_r2 >= min_r2_for_report (if config provided)
        - Sorted by mean_beta (absolute value, descending)
    """
    if exposures.empty:
        return pd.DataFrame()

    if config is None:
        config = FactorExposureConfig()

    # Extract factor names from beta_* columns
    beta_cols = [col for col in exposures.columns if col.startswith("beta_")]
    factor_names = [col.replace("beta_", "") for col in beta_cols]

    if not factor_names:
        return pd.DataFrame()

    summary_rows = []

    for factor in factor_names:
        beta_col = f"beta_{factor}"
        if beta_col not in exposures.columns:
            continue

        beta_values = exposures[beta_col].dropna()
        r2_values = exposures["r2"].dropna()
        residual_vol_values = exposures["residual_vol"].dropna()

        n_windows = len(beta_values)
        n_windows_total = len(exposures)

        if n_windows == 0:
            continue

        mean_beta = beta_values.mean()
        std_beta = beta_values.std() if len(beta_values) > 1 else 0.0
        mean_r2 = r2_values.mean() if len(r2_values) > 0 else np.nan
        median_r2 = r2_values.median() if len(r2_values) > 0 else np.nan
        mean_residual_vol = (
            residual_vol_values.mean() if len(residual_vol_values) > 0 else np.nan
        )

        # Filter by min_r2_for_report
        if mean_r2 < config.min_r2_for_report:
            continue

        summary_rows.append(
            {
                "factor": factor,
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "mean_r2": mean_r2,
                "median_r2": median_r2,
                "mean_residual_vol": mean_residual_vol,
                "n_windows": n_windows,
                "n_windows_total": n_windows_total,
            }
        )

    if not summary_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    # Sort by mean_beta (absolute value, descending)
    summary_df = summary_df.reindex(
        summary_df["mean_beta"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    return summary_df
