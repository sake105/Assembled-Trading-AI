"""Advanced Risk Metrics and Attribution Module.

This module provides extended risk metrics and performance attribution capabilities
for backtests, including regime-based segmentation and factor-group attribution.

Key features:
- Extended risk metrics: Skewness, Kurtosis, Tail Ratio (beyond qa/metrics.py)
- Exposure time-series: Gross/Net Exposure, HHI Concentration, Turnover
- Risk by regime: Segment metrics by market regime (from D1)
- Risk by factor group: Performance attribution by factor categories

Note: This module builds on existing modules (qa/metrics.py, qa/risk_metrics.py)
and extends them with additional functionality rather than duplicating code.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.assembled_core.qa.metrics import (
    PERIODS_PER_YEAR_1D,
    PERIODS_PER_YEAR_5MIN,
    _compute_returns,
    compute_cagr,
    compute_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)

logger = logging.getLogger(__name__)


def _get_periods_per_year(freq: Literal["1d", "5min"]) -> int:
    """Get periods per year for a given frequency.
    
    Args:
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Number of periods per year
    """
    if freq == "1d":
        return PERIODS_PER_YEAR_1D
    elif freq == "5min":
        return PERIODS_PER_YEAR_5MIN
    else:
        # Default to daily
        return PERIODS_PER_YEAR_1D


def compute_basic_risk_metrics(
    returns: pd.Series,
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
) -> dict[str, float | None | int]:
    """
    Berechnet erweiterte Risk-Metriken aus Returns.
    
    Args:
        returns: Zeitreihe der täglichen Returns (pd.Series, Index = timestamp)
        freq: Trading-Frequenz für Annualisierung ("1d" oder "5min")
        risk_free_rate: Risk-free Rate (annualisiert, default: 0.0)
    
    Returns:
        Dictionary mit Metriken:
        - mean_return_annualized: Annualisierte mittlere Returns
        - vol_annualized: Annualisierte Volatilität
        - sharpe: Sharpe Ratio (annualisiert)
        - sortino: Sortino Ratio (annualisiert)
        - max_drawdown: Maximaler Drawdown (in Prozent, negativ)
        - calmar: Calmar Ratio (CAGR / |max_drawdown_pct|)
        - skew: Skewness der Returns
        - kurtosis: Kurtosis der Returns (Excess Kurtosis)
        - var_95: Value at Risk (95% Konfidenz, als Return-Perzentil)
        - cvar_95: Conditional VaR / Expected Shortfall (95% Konfidenz)
        - n_periods: Anzahl Perioden
    
    Note:
        - Nutzt bestehende Funktionen aus qa.metrics für Sharpe, Sortino, Drawdown
        - VaR/ES werden als Return-Perzentile zurückgegeben (nicht in absoluten Werten)
        - Max Drawdown wird aus kumulierten Returns berechnet (equity = cumprod(1 + returns))
    """
    # Clean returns
    returns = returns.copy().dropna()
    
    if len(returns) < 2:
        return {
            "mean_return_annualized": None,
            "vol_annualized": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "skew": None,
            "kurtosis": None,
            "var_95": None,
            "cvar_95": None,
            "n_periods": len(returns),
        }
    
    periods_per_year = _get_periods_per_year(freq)
    n_periods = len(returns)
    
    # Mean return (annualized)
    mean_return_daily = float(returns.mean())
    mean_return_annualized = mean_return_daily * periods_per_year
    
    # Volatility (annualized)
    vol_daily = float(returns.std())
    vol_annualized = vol_daily * np.sqrt(periods_per_year) if vol_daily > 0 else None
    
    # Sharpe Ratio
    sharpe = compute_sharpe_ratio(returns, freq=freq, risk_free_rate=risk_free_rate)
    
    # Sortino Ratio
    sortino = compute_sortino_ratio(returns, freq=freq, risk_free_rate=risk_free_rate)
    
    # Max Drawdown: Berechne aus kumulierten Returns (equity = cumprod(1 + returns))
    max_drawdown = None
    calmar = None
    
    equity_from_returns = (1.0 + returns).cumprod() * 100.0  # Start bei 100
    if len(equity_from_returns) >= 2:
        _, max_dd_abs, max_dd_pct, _ = compute_drawdown(equity_from_returns)
        max_drawdown = float(max_dd_pct)
        
        # Calmar Ratio (CAGR / |max_drawdown_pct|)
        if max_dd_pct < 0:
            start_value = float(equity_from_returns.iloc[0])
            end_value = float(equity_from_returns.iloc[-1])
            if start_value > 0:
                cagr_value = compute_cagr(start_value, end_value, len(equity_from_returns), freq)
                if cagr_value is not None and max_dd_pct != 0:
                    calmar = cagr_value / abs(max_dd_pct / 100.0)
    
    # Skewness
    skew = float(returns.skew()) if len(returns) >= 3 else None
    if skew is not None and np.isnan(skew):
        skew = None
    
    # Kurtosis (Excess Kurtosis, d.h. normalverteilt = 0)
    kurtosis = float(returns.kurtosis()) if len(returns) >= 4 else None
    if kurtosis is not None and np.isnan(kurtosis):
        kurtosis = None
    
    # VaR (95%): 5th percentile of returns
    var_95 = None
    if len(returns) >= 5:
        var_95 = float(np.percentile(returns, 5))
        if np.isnan(var_95):
            var_95 = None
    
    # CVaR / Expected Shortfall (95%): Mean of returns below VaR threshold
    cvar_95 = None
    if var_95 is not None and len(returns) >= 5:
        tail_returns = returns[returns <= var_95]
        if len(tail_returns) > 0:
            cvar_95 = float(tail_returns.mean())
            if np.isnan(cvar_95):
                cvar_95 = None
    
    return {
        "mean_return_annualized": mean_return_annualized,
        "vol_annualized": vol_annualized,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "skew": skew,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "n_periods": n_periods,
    }


def compute_exposure_timeseries(
    positions: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    equity: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    weight_col: str = "weight",
    freq: Literal["1d", "5min"] = "1d",
) -> pd.DataFrame:
    """
    Berechnet Exposure-Zeitreihen aus Positions-DataFrame.
    
    Args:
        positions: DataFrame mit Spalten: timestamp, symbol, weight (oder qty)
                  weight sollte Portfolio-Gewicht sein (kann positiv/negativ sein für Long/Short)
        trades: Optional, für Turnover-Berechnung (aktuell nicht implementiert)
        equity: Optional, für Turnover-Berechnung (aktuell nicht implementiert)
        timestamp_col: Name der Timestamp-Spalte (default: "timestamp")
        weight_col: Name der Weight-Spalte (default: "weight")
        freq: Trading-Frequenz für Annualisierung (default: "1d")
    
    Returns:
        DataFrame mit Spalten:
        - timestamp: Timestamp
        - gross_exposure: Summe der absoluten Gewichte
        - net_exposure: Summe der Gewichte (kann negativ sein)
        - n_positions: Anzahl nicht-null Positionen
        - hhi_concentration: Herfindahl-Hirschman Index (Summe der quadrierten absoluten Gewichte)
        - turnover: NaN (aktuell nicht implementiert)
    
    Raises:
        ValueError: Wenn required Spalten fehlen
    """
    required_cols = [timestamp_col, "symbol", weight_col]
    missing_cols = [col for col in required_cols if col not in positions.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in positions: {', '.join(missing_cols)}. "
            f"Available: {list(positions.columns)}"
        )
    
    if positions.empty:
        return pd.DataFrame(columns=[
            timestamp_col,
            "gross_exposure",
            "net_exposure",
            "n_positions",
            "hhi_concentration",
            "turnover",
        ])
    
    # Group by timestamp
    exposure_data = []
    
    for timestamp, group in positions.groupby(timestamp_col):
        weights = group[weight_col].dropna()
        
        if len(weights) == 0:
            # No positions at this timestamp
            exposure_data.append({
                timestamp_col: timestamp,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "n_positions": 0,
                "hhi_concentration": 0.0,
                "turnover": np.nan,
            })
        else:
            # Gross exposure: sum of absolute weights
            gross_exposure = float(weights.abs().sum())
            
            # Net exposure: sum of weights
            net_exposure = float(weights.sum())
            
            # Number of positions (non-zero)
            n_positions = int((weights.abs() > 1e-10).sum())
            
            # HHI Concentration: sum of squared absolute weights
            # HHI ranges from 0 (perfect diversification) to 1 (single position)
            hhi = float((weights.abs() ** 2).sum())
            
            # Normalize HHI: if gross_exposure > 0, divide by gross_exposure^2 to get [0, 1] range
            # Actually, HHI should be sum of squared weights relative to total
            # For normalized HHI, we divide by (sum of abs weights)^2
            if gross_exposure > 0:
                hhi_normalized = hhi / (gross_exposure ** 2)
            else:
                hhi_normalized = 0.0
            
            exposure_data.append({
                timestamp_col: timestamp,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "n_positions": n_positions,
                "hhi_concentration": hhi_normalized,
                "turnover": np.nan,  # Not implemented yet
            })
    
    exposure_df = pd.DataFrame(exposure_data)
    
    if not exposure_df.empty:
        exposure_df = exposure_df.sort_values(timestamp_col).reset_index(drop=True)
    
    logger.debug(f"Computed exposure timeseries for {len(exposure_df)} timestamps")
    
    return exposure_df


def compute_risk_by_regime(
    returns: pd.Series,
    regime_state_df: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Berechnet Risk-Metriken pro Regime.
    
    Args:
        returns: Portfolio-Returns (Zeitreihe, Index sollte timestamp sein)
        regime_state_df: DataFrame mit Spalten: timestamp, regime_label
        trades: Optional, für Win-Rate-Berechnung (aktuell nicht implementiert)
        timestamp_col: Name der Timestamp-Spalte (default: "timestamp")
        regime_col: Name der Regime-Spalte (default: "regime_label")
        freq: Trading-Frequenz für Annualisierung (default: "1d")
        risk_free_rate: Risk-free Rate (annualisiert, default: 0.0)
    
    Returns:
        DataFrame mit einer Zeile pro Regime:
        - regime: Regime-Label
        - n_periods: Anzahl Perioden
        - mean_return_annualized: Annualisierte mittlere Returns
        - vol_annualized: Annualisierte Volatilität
        - sharpe: Sharpe Ratio
        - max_drawdown: Maximaler Drawdown (in Prozent, negativ)
        - total_return: Total Return (kumulativ)
    
    Raises:
        ValueError: Wenn required Spalten fehlen oder keine Overlaps zwischen returns und regime
    """
    if regime_state_df.empty:
        logger.warning("regime_state_df is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "regime",
            "n_periods",
            "mean_return_annualized",
            "vol_annualized",
            "sharpe",
            "max_drawdown",
            "total_return",
        ])
    
    # Ensure returns is a Series with timestamp index
    if isinstance(returns, pd.Series):
        returns_series = returns.copy()
    else:
        raise ValueError("returns must be a pd.Series")
    
    # Convert regime_state_df timestamp to match returns index type
    regime_df = regime_state_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(regime_df[timestamp_col]):
        regime_df[timestamp_col] = pd.to_datetime(regime_df[timestamp_col], utc=True)
    
    # Merge returns with regime (if returns has index, use it; otherwise assume timestamp column)
    if returns_series.index.name == timestamp_col or isinstance(returns_series.index, pd.DatetimeIndex):
        # Returns has timestamp index
        returns_df = returns_series.reset_index(name="return")
        if returns_df.columns[0] != timestamp_col:
            returns_df.columns = [timestamp_col, "return"]
    else:
        # Returns doesn't have timestamp index, assume it's sequential
        # This is a fallback - ideally returns should have timestamp index
        logger.warning("returns Series doesn't have timestamp index. Using sequential index.")
        returns_df = pd.DataFrame({
            timestamp_col: range(len(returns_series)),
            "return": returns_series.values,
        })
    
    # Merge
    merged = pd.merge(
        returns_df,
        regime_df[[timestamp_col, regime_col]],
        on=timestamp_col,
        how="inner",
    )
    
    if merged.empty:
        logger.warning("No overlapping timestamps between returns and regime_state_df.")
        return pd.DataFrame(columns=[
            "regime",
            "n_periods",
            "mean_return_annualized",
            "vol_annualized",
            "sharpe",
            "max_drawdown",
            "total_return",
        ])
    
    # Compute metrics per regime
    regime_results = []
    
    for regime in merged[regime_col].unique():
        regime_returns = merged[merged[regime_col] == regime]["return"].dropna()
        
        if len(regime_returns) < 2:
            continue
        
        # Basic metrics using compute_basic_risk_metrics
        # We need to create equity series from returns for drawdown calculation
        equity_from_returns = (1.0 + regime_returns).cumprod() * 100.0
        
        metrics = compute_basic_risk_metrics(
            returns=regime_returns,
            equity=equity_from_returns,
            freq=freq,
            risk_free_rate=risk_free_rate,
        )
        
        # Total return (cumulative)
        total_return = float((1.0 + regime_returns).prod() - 1.0)
        
        regime_results.append({
            "regime": regime,
            "n_periods": metrics["n_periods"],
            "mean_return_annualized": metrics["mean_return_annualized"],
            "vol_annualized": metrics["vol_annualized"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "total_return": total_return,
        })
    
    result_df = pd.DataFrame(regime_results)
    
    if not result_df.empty:
        result_df = result_df.sort_values("regime").reset_index(drop=True)
    
    logger.info(
        f"Computed risk metrics by regime: {len(result_df)} regimes, "
        f"{len(merged)} total periods"
    )
    
    return result_df


def compute_risk_by_factor_group(
    returns: pd.Series,
    factor_panel_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    factor_groups: dict[str, list[str]] | None = None,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """
    Grobe Risiko-/Performance-Attribution per Faktorgruppe.
    
    Idee:
    - Für jeden Tag berechnet einen Portfolio-Factor-Score pro Gruppe:
      Gewichteter Durchschnitt der Faktorwerte (über Symbole mit Portfolio-Gewichten).
    - Schätzt pro Gruppe eine einfache Korrelation von Returns auf diesen Score.
    
    Args:
        returns: Portfolio-Returns (Zeitreihe, Index sollte timestamp sein)
        factor_panel_df: DataFrame mit Spalten: timestamp, symbol, factor_*
        positions_df: DataFrame mit Spalten: timestamp, symbol, weight
        factor_groups: Dictionary mapping Gruppe → Liste von Faktor-Namen
                      Default: Standard-Faktor-Gruppen
        timestamp_col: Name der Timestamp-Spalte (default: "timestamp")
        symbol_col: Name der Symbol-Spalte (default: "symbol")
        weight_col: Name der Weight-Spalte (default: "weight")
    
    Returns:
        DataFrame mit einer Zeile pro Faktor-Gruppe:
        - factor_group: Gruppen-Name
        - factors: Liste der Faktoren (als String, komma-separiert)
        - correlation_with_returns: Korrelation zwischen Portfolio-Factor-Score und Portfolio-Returns
        - avg_exposure: Durchschnittliche Exposure (Mittelwert der Scores über Zeit)
        - n_periods: Anzahl Perioden mit gültigen Daten
    
    Raises:
        ValueError: Wenn required Spalten fehlen
    """
    # Default factor groups
    if factor_groups is None:
        factor_groups = {
            "Trend": ["returns_12m", "trend_strength_50", "trend_strength_200"],
            "Vol/Liq": ["rv_20", "vov_20_60", "turnover_20d"],
            "Earnings": ["earnings_eps_surprise_last", "post_earnings_drift_20d"],
            "Insider": ["insider_net_notional_60d", "insider_buy_ratio_60d"],
            "News/Macro": ["news_sentiment_trend_20d", "macro_growth_regime"],
        }
    
    # Validate inputs
    if factor_panel_df.empty or positions_df.empty:
        logger.warning("Empty input DataFrames. Returning empty result.")
        return pd.DataFrame(columns=[
            "factor_group",
            "factors",
            "correlation_with_returns",
            "avg_exposure",
            "n_periods",
        ])
    
    required_factor_cols = [timestamp_col, symbol_col]
    missing_factor_cols = [col for col in required_factor_cols if col not in factor_panel_df.columns]
    if missing_factor_cols:
        raise ValueError(
            f"Missing required columns in factor_panel_df: {', '.join(missing_factor_cols)}"
        )
    
    required_pos_cols = [timestamp_col, symbol_col, weight_col]
    missing_pos_cols = [col for col in required_pos_cols if col not in positions_df.columns]
    if missing_pos_cols:
        raise ValueError(
            f"Missing required columns in positions_df: {', '.join(missing_pos_cols)}"
        )
    
    # Prepare returns (convert to DataFrame if Series)
    if isinstance(returns, pd.Series):
        if returns.index.name == timestamp_col or isinstance(returns.index, pd.DatetimeIndex):
            returns_df = returns.reset_index(name="portfolio_return")
            if returns_df.columns[0] != timestamp_col:
                returns_df.columns = [timestamp_col, "portfolio_return"]
        else:
            logger.warning("returns Series doesn't have timestamp index. Using sequential index.")
            returns_df = pd.DataFrame({
                timestamp_col: range(len(returns)),
                "portfolio_return": returns.values,
            })
    else:
        returns_df = returns.copy()
        if timestamp_col not in returns_df.columns:
            raise ValueError(f"returns must have '{timestamp_col}' column or be a Series with timestamp index")
    
    # Prepare DataFrames
    factor_df = factor_panel_df.copy()
    pos_df = positions_df.copy()
    
    # Ensure timestamps are datetime
    for df in [factor_df, pos_df, returns_df]:
        if timestamp_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    
    # Compute portfolio factor scores per group
    factor_group_results = []
    
    for group_name, factor_names in factor_groups.items():
        # Find available factors in this group
        available_factors = [f for f in factor_names if f in factor_df.columns]
        
        if not available_factors:
            # No factors available for this group
            factor_group_results.append({
                "factor_group": group_name,
                "factors": ",".join(factor_names),
                "correlation_with_returns": None,
                "avg_exposure": None,
                "n_periods": 0,
            })
            continue
        
        # For each timestamp, compute weighted average of factor values
        portfolio_scores = []
        
        for timestamp in sorted(factor_df[timestamp_col].unique()):
            # Get positions for this timestamp
            timestamp_positions = pos_df[pos_df[timestamp_col] == timestamp]
            
            if timestamp_positions.empty:
                continue
            
            # Get factors for this timestamp
            timestamp_factors = factor_df[factor_df[timestamp_col] == timestamp]
            
            if timestamp_factors.empty:
                continue
            
            # Merge positions with factors
            merged = pd.merge(
                timestamp_positions[[symbol_col, weight_col]],
                timestamp_factors[[symbol_col] + available_factors],
                on=symbol_col,
                how="inner",
            )
            
            if merged.empty:
                continue
            
            # Compute weighted average factor score for this group
            # Average across available factors, then weight by position weights
            group_scores = []
            
            for factor_name in available_factors:
                factor_values = merged[factor_name].dropna()
                weights = merged.loc[factor_values.index, weight_col]
                
                if len(factor_values) > 0:
                    # Weighted average of this factor across symbols
                    weighted_avg = float((factor_values * weights.abs()).sum() / weights.abs().sum())
                    group_scores.append(weighted_avg)
            
            if len(group_scores) > 0:
                # Average across factors in group (simple mean)
                portfolio_score = float(np.mean(group_scores))
                portfolio_scores.append({
                    timestamp_col: timestamp,
                    "portfolio_factor_score": portfolio_score,
                })
        
        if len(portfolio_scores) == 0:
            factor_group_results.append({
                "factor_group": group_name,
                "factors": ",".join(available_factors),
                "correlation_with_returns": None,
                "avg_exposure": None,
                "n_periods": 0,
            })
            continue
        
        scores_df = pd.DataFrame(portfolio_scores)
        
        # Merge with returns
        merged_with_returns = pd.merge(
            scores_df,
            returns_df[[timestamp_col, "portfolio_return"]],
            on=timestamp_col,
            how="inner",
        ).dropna()
        
        if len(merged_with_returns) < 2:
            factor_group_results.append({
                "factor_group": group_name,
                "factors": ",".join(available_factors),
                "correlation_with_returns": None,
                "avg_exposure": None,
                "n_periods": len(merged_with_returns),
            })
            continue
        
        # Compute correlation
        correlation = float(
            merged_with_returns["portfolio_factor_score"].corr(merged_with_returns["portfolio_return"])
        )
        if np.isnan(correlation):
            correlation = None
        
        # Average exposure (mean of portfolio scores)
        avg_exposure = float(merged_with_returns["portfolio_factor_score"].mean())
        if np.isnan(avg_exposure):
            avg_exposure = None
        
        factor_group_results.append({
            "factor_group": group_name,
            "factors": ",".join(available_factors),
            "correlation_with_returns": correlation,
            "avg_exposure": avg_exposure,
            "n_periods": len(merged_with_returns),
        })
    
    result_df = pd.DataFrame(factor_group_results)
    
    if not result_df.empty:
        result_df = result_df.sort_values("factor_group").reset_index(drop=True)
    
    logger.info(
        f"Computed risk by factor group: {len(result_df)} groups, "
        f"{len(factor_groups)} total factor groups analyzed"
    )
    
    return result_df

