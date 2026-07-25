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
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from src.assembled_core.qa.metrics import (
    PERIODS_PER_YEAR_1D,
    PERIODS_PER_YEAR_5MIN,
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
                cagr_value = compute_cagr(
                    start_value, end_value, len(equity_from_returns), freq
                )
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

    # VaR (95%): 5th percentile of returns (historical)
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

    # Parametric VaR + Cornish-Fisher VaR (Sprint 1 / C5a)
    var_95_parametric = compute_parametric_var(returns, alpha=0.95)
    var_99_parametric = compute_parametric_var(returns, alpha=0.99)
    var_95_cornish_fisher = compute_cornish_fisher_var(returns, alpha=0.95)
    var_99_cornish_fisher = compute_cornish_fisher_var(returns, alpha=0.99)

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
        "var_95_parametric": var_95_parametric,
        "var_99_parametric": var_99_parametric,
        "var_95_cornish_fisher": var_95_cornish_fisher,
        "var_99_cornish_fisher": var_99_cornish_fisher,
        "n_periods": n_periods,
    }


# ---------------------------------------------------------------------------
# Sprint 1 / C5a — Parametric + Cornish-Fisher VaR
# ---------------------------------------------------------------------------

# z-scores for standard-normal distribution (one-tailed lower quantile)
_Z_SCORES: dict[float, float] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.995: 2.5758,
    0.999: 3.0902,
}


def _z_score(alpha: float) -> float:
    """Return the standard-normal z-score for the given confidence level.

    Falls back to :func:`scipy.stats.norm.ppf` if ``alpha`` is not in the
    lookup table; if scipy is unavailable, raises ``ValueError``.
    """
    if alpha in _Z_SCORES:
        return _Z_SCORES[alpha]
    try:
        from scipy.stats import norm

        return float(norm.ppf(alpha))
    except Exception as exc:  # pragma: no cover - scipy missing
        raise ValueError(
            f"Unsupported alpha={alpha}; add to _Z_SCORES table or install scipy"
        ) from exc


def compute_parametric_var(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.95,
    horizon: int = 1,
) -> float | None:
    """Parametric (Gaussian) Value-at-Risk.

    ``VaR_α = -(μ - z_α · σ) · √horizon``

    Returned as a **positive loss magnitude** (e.g. ``0.023`` = 2.3 % loss).
    ``None`` when the input has fewer than 5 observations or is degenerate.

    Args:
        returns: Period returns (e.g. daily).
        alpha: Confidence level (default 0.95).
        horizon: Forecast horizon in periods (default 1). Uses √h scaling.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return None

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0.0:
        return None

    z = _z_score(alpha)
    var = -(mu - z * sigma) * np.sqrt(max(1, int(horizon)))
    if not np.isfinite(var):
        return None
    return float(var)


def compute_cornish_fisher_var(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.95,
    horizon: int = 1,
) -> float | None:
    """Cornish-Fisher adjusted VaR (accounts for skew & excess kurtosis).

    ``z_CF = z + (z² - 1)/6 · S + (z³ - 3z)/24 · K - (2z³ - 5z)/36 · S²``
    ``VaR_CF = -(μ - z_CF · σ) · √horizon``

    For fat-tailed, left-skewed distributions this yields a **larger** loss
    estimate than the plain parametric VaR. Returns ``None`` when the input
    has fewer than 4 observations or higher moments cannot be computed.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 4:
        return None

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0.0:
        return None

    s = float(r.skew())
    k = float(r.kurtosis())  # pandas returns excess kurtosis
    if not np.isfinite(s):
        s = 0.0
    if not np.isfinite(k):
        k = 0.0

    z = _z_score(alpha)
    z_cf = (
        z
        + (z**2 - 1) / 6.0 * s
        + (z**3 - 3 * z) / 24.0 * k
        - (2 * z**3 - 5 * z) / 36.0 * s**2
    )
    var = -(mu - z_cf * sigma) * np.sqrt(max(1, int(horizon)))
    if not np.isfinite(var):
        return None
    return float(var)


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
        return pd.DataFrame(
            columns=[
                timestamp_col,
                "gross_exposure",
                "net_exposure",
                "n_positions",
                "hhi_concentration",
                "turnover",
            ]
        )

    # Group by timestamp
    exposure_data = []

    for timestamp, group in positions.groupby(timestamp_col):
        weights = group[weight_col].dropna()

        if len(weights) == 0:
            # No positions at this timestamp
            exposure_data.append(
                {
                    timestamp_col: timestamp,
                    "gross_exposure": 0.0,
                    "net_exposure": 0.0,
                    "n_positions": 0,
                    "hhi_concentration": 0.0,
                    "turnover": np.nan,
                }
            )
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
                hhi_normalized = hhi / (gross_exposure**2)
            else:
                hhi_normalized = 0.0

            exposure_data.append(
                {
                    timestamp_col: timestamp,
                    "gross_exposure": gross_exposure,
                    "net_exposure": net_exposure,
                    "n_positions": n_positions,
                    "hhi_concentration": hhi_normalized,
                    "turnover": np.nan,  # Not implemented yet
                }
            )

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
        return pd.DataFrame(
            columns=[
                "regime",
                "n_periods",
                "mean_return_annualized",
                "vol_annualized",
                "sharpe",
                "max_drawdown",
                "total_return",
            ]
        )

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
    if returns_series.index.name == timestamp_col or isinstance(
        returns_series.index, pd.DatetimeIndex
    ):
        # Returns has timestamp index
        returns_df = returns_series.reset_index(name="return")
        if returns_df.columns[0] != timestamp_col:
            returns_df.columns = [timestamp_col, "return"]
    else:
        # Returns doesn't have timestamp index, assume it's sequential
        # This is a fallback - ideally returns should have timestamp index
        logger.warning(
            "returns Series doesn't have timestamp index. Using sequential index."
        )
        returns_df = pd.DataFrame(
            {
                timestamp_col: range(len(returns_series)),
                "return": returns_series.values,
            }
        )

    # Merge
    merged = pd.merge(
        returns_df,
        regime_df[[timestamp_col, regime_col]],
        on=timestamp_col,
        how="inner",
    )

    if merged.empty:
        logger.warning("No overlapping timestamps between returns and regime_state_df.")
        return pd.DataFrame(
            columns=[
                "regime",
                "n_periods",
                "mean_return_annualized",
                "vol_annualized",
                "sharpe",
                "max_drawdown",
                "total_return",
            ]
        )

    # Compute metrics per regime
    regime_results = []

    for regime in merged[regime_col].unique():
        regime_returns = merged[merged[regime_col] == regime]["return"].dropna()

        if len(regime_returns) < 2:
            continue

        # Basic metrics using compute_basic_risk_metrics
        # compute_basic_risk_metrics computes drawdown internally from returns
        metrics = compute_basic_risk_metrics(
            returns=regime_returns,
            freq=freq,
            risk_free_rate=risk_free_rate,
        )

        # Total return (cumulative)
        total_return = float((1.0 + regime_returns).prod() - 1.0)

        regime_results.append(
            {
                "regime": regime,
                "n_periods": metrics["n_periods"],
                "mean_return_annualized": metrics["mean_return_annualized"],
                "vol_annualized": metrics["vol_annualized"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "total_return": total_return,
            }
        )

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
        return pd.DataFrame(
            columns=[
                "factor_group",
                "factors",
                "correlation_with_returns",
                "avg_exposure",
                "n_periods",
            ]
        )

    required_factor_cols = [timestamp_col, symbol_col]
    missing_factor_cols = [
        col for col in required_factor_cols if col not in factor_panel_df.columns
    ]
    if missing_factor_cols:
        raise ValueError(
            f"Missing required columns in factor_panel_df: {', '.join(missing_factor_cols)}"
        )

    required_pos_cols = [timestamp_col, symbol_col, weight_col]
    missing_pos_cols = [
        col for col in required_pos_cols if col not in positions_df.columns
    ]
    if missing_pos_cols:
        raise ValueError(
            f"Missing required columns in positions_df: {', '.join(missing_pos_cols)}"
        )

    # Prepare returns (convert to DataFrame if Series)
    if isinstance(returns, pd.Series):
        if returns.index.name == timestamp_col or isinstance(
            returns.index, pd.DatetimeIndex
        ):
            returns_df = returns.reset_index(name="portfolio_return")
            if returns_df.columns[0] != timestamp_col:
                returns_df.columns = [timestamp_col, "portfolio_return"]
        else:
            logger.warning(
                "returns Series doesn't have timestamp index. Using sequential index."
            )
            returns_df = pd.DataFrame(
                {
                    timestamp_col: range(len(returns)),
                    "portfolio_return": returns.values,
                }
            )
    else:
        returns_df = returns.copy()
        if timestamp_col not in returns_df.columns:
            raise ValueError(
                f"returns must have '{timestamp_col}' column or be a Series with timestamp index"
            )

    # Prepare DataFrames
    factor_df = factor_panel_df.copy()
    pos_df = positions_df.copy()

    # Ensure timestamps are datetime
    for df in [factor_df, pos_df, returns_df]:
        if timestamp_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    # Compute portfolio factor scores per group
    factor_group_results: list[dict[str, Any]] = []

    for group_name, factor_names in factor_groups.items():
        # Find available factors in this group
        available_factors = [f for f in factor_names if f in factor_df.columns]

        if not available_factors:
            # No factors available for this group
            factor_group_results.append(
                {
                    "factor_group": group_name,
                    "factors": ",".join(factor_names),
                    "correlation_with_returns": None,
                    "avg_exposure": None,
                    "n_periods": 0,
                }
            )
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
                    weight_sum = weights.abs().sum()
                    weighted_avg = float(
                        (factor_values * weights.abs()).sum() / weight_sum
                        if weight_sum > 1e-12
                        else factor_values.mean()
                    )
                    group_scores.append(weighted_avg)

            if len(group_scores) > 0:
                # Average across factors in group (simple mean)
                portfolio_score = float(np.mean(group_scores))
                portfolio_scores.append(
                    {
                        timestamp_col: timestamp,
                        "portfolio_factor_score": portfolio_score,
                    }
                )

        if len(portfolio_scores) == 0:
            factor_group_results.append(
                {
                    "factor_group": group_name,
                    "factors": ",".join(available_factors),
                    "correlation_with_returns": None,
                    "avg_exposure": None,
                    "n_periods": 0,
                }
            )
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
            factor_group_results.append(
                {
                    "factor_group": group_name,
                    "factors": ",".join(available_factors),
                    "correlation_with_returns": None,
                    "avg_exposure": None,
                    "n_periods": len(merged_with_returns),
                }
            )
            continue

        # Compute correlation
        corr_val = float(
            merged_with_returns["portfolio_factor_score"].corr(
                merged_with_returns["portfolio_return"]
            )
        )
        correlation: float | None = corr_val
        if np.isnan(corr_val):
            correlation = None

        # Average exposure (mean of portfolio scores)
        avg_exposure_val = float(merged_with_returns["portfolio_factor_score"].mean())
        avg_exposure: float | None = avg_exposure_val
        if np.isnan(avg_exposure_val):
            avg_exposure = None

        factor_group_results.append(
            {
                "factor_group": group_name,
                "factors": ",".join(available_factors),
                "correlation_with_returns": correlation,
                "avg_exposure": avg_exposure,
                "n_periods": len(merged_with_returns),
            }
        )

    result_df = pd.DataFrame(factor_group_results)

    if not result_df.empty:
        result_df = result_df.sort_values("factor_group").reset_index(drop=True)

    logger.info(
        f"Computed risk by factor group: {len(result_df)} groups, "
        f"{len(factor_groups)} total factor groups analyzed"
    )

    return result_df


# ── Monte Carlo VaR with Cholesky decomposition (Plan 7.1) ───────────


def compute_monte_carlo_var(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray | None = None,
    *,
    n_simulations: int = 10_000,
    horizon_days: int = 1,
    confidence_levels: tuple[float, ...] = (0.95, 0.99, 0.999),
    seed: int | None = 42,
) -> dict[str, float]:
    """Compute portfolio VaR and CVaR via Monte Carlo simulation.

    Uses Cholesky decomposition to generate correlated return paths
    that respect the empirical covariance structure.

    Algorithm:
        1. Estimate mean (mu) and covariance (Sigma) from historical returns.
        2. Cholesky decomposition: L such that L @ L.T = Sigma.
        3. Generate z ~ N(0, I) random vectors.
        4. Simulated returns = mu + L @ z (preserves correlation structure).
        5. Portfolio return = w.T @ simulated_returns.
        6. VaR = quantile of portfolio return distribution.

    Args:
        returns: Wide-format returns (dates × symbols) or 2D array.
        weights: Portfolio weights (1D array, sums to 1).
            If None, equal weights are used.
        n_simulations: Number of Monte Carlo paths.
        horizon_days: Forecast horizon (multi-day uses sqrt-of-time scaling).
        confidence_levels: VaR confidence levels (e.g., 0.99 = 99%).
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``mc_var_{pct}``, ``mc_cvar_{pct}`` for each
        confidence level, plus ``mc_expected_return``,
        ``mc_expected_vol``, ``mc_worst_case``.
    """
    if isinstance(returns, pd.DataFrame):
        ret_arr = returns.dropna().values
        n_assets = ret_arr.shape[1]
    else:
        ret_arr = returns[~np.any(np.isnan(returns), axis=1)]
        n_assets = ret_arr.shape[1]

    if len(ret_arr) < 30 or n_assets < 1:
        return {f"mc_var_{int(cl * 100)}": 0.0 for cl in confidence_levels}

    if weights is None:
        weights = np.ones(n_assets) / n_assets

    # Mean and covariance
    mu = np.mean(ret_arr, axis=0)
    cov = np.cov(ret_arr, rowvar=False)

    # Handle single-asset case (cov returns scalar)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if mu.ndim == 0:
        mu = np.array([float(mu)])

    # Ensure positive semi-definite (eigenvalue fix)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        logger.warning("[MC-VaR] Cholesky failed — falling back to diagonal")
        L = np.diag(np.sqrt(np.diag(cov)))

    # Generate simulations
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_simulations, n_assets))
    simulated = mu + z @ L.T  # shape: (n_sim, n_assets)

    # Multi-day horizon scaling
    if horizon_days > 1:
        simulated = simulated * np.sqrt(horizon_days)

    # Portfolio returns
    port_returns = simulated @ weights  # shape: (n_sim,)

    results: dict[str, float] = {
        "mc_expected_return": round(float(np.mean(port_returns)) * 252, 6),
        "mc_expected_vol": round(float(np.std(port_returns)) * np.sqrt(252), 6),
        "mc_worst_case": round(float(np.min(port_returns)), 6),
    }

    for cl in confidence_levels:
        pct = int(cl * 100)
        var = float(np.quantile(port_returns, 1 - cl))
        # VaR as positive loss number
        results[f"mc_var_{pct}"] = round(abs(min(0, var)), 6)
        # CVaR: expected loss given loss exceeds VaR
        tail = port_returns[port_returns <= var]
        cvar = abs(float(np.mean(tail))) if len(tail) > 0 else abs(min(0, var))
        results[f"mc_cvar_{pct}"] = round(cvar, 6)

    return results


# ── Brinson-Fachler P&L Attribution (Plan 7.5) ───────────────────────


def compute_brinson_fachler_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Brinson-Fachler performance attribution by sector.

    Decomposes active return (portfolio − benchmark) into:
    - **Allocation Effect**: Over/underweight in sectors that outperformed
    - **Selection Effect**: Picking better stocks within each sector
    - **Interaction Effect**: Combined allocation × selection

    Total Active Return = sum(Allocation + Selection + Interaction)

    Args:
        portfolio_weights: Symbol → portfolio weight.
        benchmark_weights: Symbol → benchmark weight.
        portfolio_returns: Symbol → period return.
        benchmark_returns: Symbol → period return (benchmark constituents).
        sector_map: Symbol → sector name.

    Returns:
        DataFrame with columns: ``sector``, ``allocation_effect``,
        ``selection_effect``, ``interaction_effect``, ``total_effect``,
        ``portfolio_weight``, ``benchmark_weight``,
        ``portfolio_return``, ``benchmark_return``.
    """
    # Collect all sectors
    all_symbols = set(portfolio_weights) | set(benchmark_weights)
    sectors: dict[str, list[str]] = {}
    for sym in all_symbols:
        sec = sector_map.get(sym, "Other")
        sectors.setdefault(sec, []).append(sym)

    # Benchmark total return
    bm_total_return = sum(
        benchmark_weights.get(s, 0) * benchmark_returns.get(s, 0)
        for s in benchmark_weights
    )

    rows = []
    for sector, symbols in sorted(sectors.items()):
        # Sector-level weights and returns
        w_p = sum(portfolio_weights.get(s, 0) for s in symbols)
        w_b = sum(benchmark_weights.get(s, 0) for s in symbols)

        # Weighted-average returns within sector
        if w_p > 0:
            r_p = (
                sum(
                    portfolio_weights.get(s, 0) * portfolio_returns.get(s, 0)
                    for s in symbols
                )
                / w_p
            )
        else:
            r_p = 0.0

        if w_b > 0:
            r_b = (
                sum(
                    benchmark_weights.get(s, 0) * benchmark_returns.get(s, 0)
                    for s in symbols
                )
                / w_b
            )
        else:
            r_b = 0.0

        # Brinson-Fachler decomposition
        allocation = (w_p - w_b) * (r_b - bm_total_return)
        selection = w_b * (r_p - r_b)
        interaction = (w_p - w_b) * (r_p - r_b)

        rows.append(
            {
                "sector": sector,
                "allocation_effect": round(allocation, 6),
                "selection_effect": round(selection, 6),
                "interaction_effect": round(interaction, 6),
                "total_effect": round(allocation + selection + interaction, 6),
                "portfolio_weight": round(w_p, 6),
                "benchmark_weight": round(w_b, 6),
                "portfolio_return": round(r_p, 6),
                "benchmark_return": round(r_b, 6),
            }
        )

    result = pd.DataFrame(rows)

    # Add total row
    if not result.empty:
        total_row = {
            "sector": "TOTAL",
            "allocation_effect": result["allocation_effect"].sum(),
            "selection_effect": result["selection_effect"].sum(),
            "interaction_effect": result["interaction_effect"].sum(),
            "total_effect": result["total_effect"].sum(),
            "portfolio_weight": result["portfolio_weight"].sum(),
            "benchmark_weight": result["benchmark_weight"].sum(),
            "portfolio_return": np.nan,
            "benchmark_return": np.nan,
        }
        result = pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)

    return result


# ---------------------------------------------------------------------------
# Drawdown Duration Analysis (Plan 7.7)
# ---------------------------------------------------------------------------


def compute_drawdown_duration(equity_curve: pd.Series) -> dict:
    """Analyze drawdown duration statistics.

    Args:
        equity_curve: Equity/NAV series.

    Returns:
        Dict with max_dd_duration_days, avg_dd_duration_days, current_dd_days.
    """
    peak = equity_curve.cummax()
    in_dd = equity_curve < peak

    durations = []
    current_duration = 0
    for val in in_dd:
        if val:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0

    return {
        "max_dd_duration_days": max(durations) if durations else 0,
        "avg_dd_duration_days": round(np.mean(durations), 1) if durations else 0.0,
        "current_dd_days": current_duration,
        "n_drawdown_periods": len(durations),
    }


# ---------------------------------------------------------------------------
# CDaR - Conditional Drawdown at Risk (Plan 7.8)
# ---------------------------------------------------------------------------


def compute_cdar(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Compute Conditional Drawdown at Risk.

    Expected drawdown in the worst alpha-% of drawdown periods.

    Args:
        returns: Daily returns.
        alpha: Tail probability (default 5%).

    Returns:
        CDaR value (negative number).
    """
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdowns = (cum_returns - peak) / peak

    threshold_idx = int(len(drawdowns) * alpha)
    if threshold_idx < 1:
        return float(drawdowns.min())

    worst = drawdowns.nsmallest(threshold_idx)
    return round(float(worst.mean()), 6)


# ---------------------------------------------------------------------------
# Component VaR / Marginal VaR (Sprint 2 / C5b)
# ---------------------------------------------------------------------------


def compute_component_var(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray | pd.Series,
    *,
    alpha: float = 0.95,
) -> dict[str, np.ndarray | float]:
    """Compute parametric component VaR and marginal VaR per position.

    Decomposes portfolio VaR into additive per-asset contributions using
    the Euler / parametric formulation:

        sigma_p     = sqrt(w' Sigma w)
        VaR_p       = z_alpha * sigma_p
        mVaR_i      = z_alpha * (Sigma w)_i / sigma_p
        cVaR_i      = w_i * mVaR_i

    By construction sum(cVaR_i) == VaR_p (additivity).

    Args:
        returns: Wide-format returns (dates × assets) or 2D array.
        weights: Portfolio weights aligned with the columns of ``returns``.
        alpha: VaR confidence level (default 0.95).

    Returns:
        Dict with:
            - ``portfolio_var``: Parametric portfolio VaR at ``alpha``.
            - ``portfolio_vol``: Portfolio 1-period volatility.
            - ``marginal_var``: Array of marginal VaR per asset.
            - ``component_var``: Array of component VaR per asset.
            - ``pct_contribution``: cVaR_i / VaR_p (shares summing to 1).
    """
    if isinstance(returns, pd.DataFrame):
        ret_arr = returns.dropna().values
    else:
        ret_arr = np.asarray(returns)
        ret_arr = ret_arr[~np.any(np.isnan(ret_arr), axis=1)]

    if isinstance(weights, pd.Series):
        w = weights.values.astype(float)
    else:
        w = np.asarray(weights, dtype=float)

    n_assets = ret_arr.shape[1] if ret_arr.ndim == 2 else 1
    if len(ret_arr) < 10 or n_assets < 1 or len(w) != n_assets:
        zeros = np.zeros(n_assets, dtype=float)
        return {
            "portfolio_var": 0.0,
            "portfolio_vol": 0.0,
            "marginal_var": zeros,
            "component_var": zeros,
            "pct_contribution": zeros,
        }

    cov = np.cov(ret_arr, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    sigma_p = float(np.sqrt(max(w @ cov @ w, 0.0)))
    z = _z_score(alpha)
    var_p = z * sigma_p

    if sigma_p <= 1e-12:
        zeros = np.zeros(n_assets, dtype=float)
        return {
            "portfolio_var": 0.0,
            "portfolio_vol": 0.0,
            "marginal_var": zeros,
            "component_var": zeros,
            "pct_contribution": zeros,
        }

    marginal = z * (cov @ w) / sigma_p
    component = w * marginal
    # Normalise to shares; handle the (rare) case of a zero-sum division.
    total = float(np.sum(component))
    pct = component / total if abs(total) > 1e-12 else np.zeros_like(component)

    return {
        "portfolio_var": round(var_p, 8),
        "portfolio_vol": round(sigma_p, 8),
        "marginal_var": marginal,
        "component_var": component,
        "pct_contribution": pct,
    }


# ---------------------------------------------------------------------------
# EVT Tail-VaR (Sprint 3 / C9) — GPD Peaks-Over-Threshold wrapper
# ---------------------------------------------------------------------------


def compute_evt_tail_var(
    returns: pd.Series | np.ndarray,
    *,
    threshold_quantile: float = 0.95,
) -> dict[str, float]:
    """Compute EVT-based tail VaR/CVaR from a return series.

    Thin wrapper over ``ml.evt_models.compute_evt_risk_metrics`` that lives in
    the risk namespace so pipeline/portfolio code can import a single source of
    tail-risk numbers without reaching into the ML layer. Returns the standard
    flat metric dict with keys ``evt_var_95``, ``evt_var_99``, ``evt_var_999``,
    ``evt_cvar_95``, ``evt_cvar_99``, ``evt_shape_xi``, ``evt_return_period_100y``.

    Defensive: on any failure (scipy missing, insufficient data, fit error) the
    underlying module returns zeros rather than raising, so callers can blend
    EVT metrics into risk budgets without a hard dependency on scipy.
    """
    try:
        # NOTE (mypy-sweep 2026-07-25): ml/evt_models was deliberately
        # ARCHIVED (commit d2c3d093) — this import never succeeds anymore;
        # the function always returns the honest evt_status="unavailable"
        # fallback below. The ignore silences a missing MODULE, not missing
        # stubs. Restore from archive if EVT tail-VaR is ever needed again.
        from src.assembled_core.ml.evt_models import (  # type: ignore[import-not-found]
            compute_evt_risk_metrics,
        )
    except Exception as exc:  # pragma: no cover - optional import path
        # Zero tail-VaR is indistinguishable from "no tail risk" without a
        # status marker — a caller blending EVT into a tail-budget would treat
        # a degenerate ``scipy``-missing run as a calm market. The numeric
        # zeros are kept for backward compatibility with existing consumers
        # that blend-and-multiply, but we tag ``evt_status`` so new callers
        # (tail-risk gates, regime overlays) can distinguish "EVT says zero"
        # from "EVT is unavailable".
        logger.warning(
            "[RISK-EVT] compute_evt_tail_risk: module unavailable — "
            "returning zero-sentinel metrics with evt_status=unavailable (%s)",
            exc,
        )
        return {
            "evt_var_95": 0.0,
            "evt_var_99": 0.0,
            "evt_var_999": 0.0,
            "evt_cvar_95": 0.0,
            "evt_cvar_99": 0.0,
            "evt_shape_xi": 0.0,
            "evt_return_period_100y": 0.0,
            # Intentional schema deviation (see comment above): status marker is
            # a string in an otherwise float-valued dict.
            "evt_status": "unavailable",  # type: ignore[dict-item]
        }
    # NOTE: on the success path we do NOT add an "evt_status" field — the
    # frozen schema-stability test (tests/test_risk_metrics_evt.py) asserts the
    # exact 7-key schema. The status flag is emitted ONLY on the
    # module-unavailable fallback above, so callers that want to detect the
    # degraded state can check ``metrics.get("evt_status") == "unavailable"``.
    return cast(
        "dict[str, float]",
        compute_evt_risk_metrics(returns, threshold_quantile=threshold_quantile),
    )
