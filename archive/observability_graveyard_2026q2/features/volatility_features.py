"""GARCH-based volatility features for factor models.

Bridges the GARCH family models (ml/garch_models.py) into the feature layer
by producing per-symbol, per-day features:

- ``garch_vol_1d``: 1-step-ahead annualized GARCH volatility
- ``garch_vol_ratio``: GARCH vol / realized vol (>1 → vol rising)
- ``garch_asymmetry``: leverage/asymmetry parameter (how strongly negative
  returns drive volatility)
- ``garch_persistence``: alpha + beta (near 1 → long memory in volatility)
- ``garch_vol_zscore``: z-score of GARCH vol relative to rolling mean

These features complement the backward-looking realized volatility features
in ``ta_liquidity_vol_factors.py`` by adding forward-looking conditional
volatility information.

Requires: ``arch`` Python package.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from src.assembled_core.ml.garch_models import (
        ARCH_AVAILABLE,
        fit_best_garch,
    )
except ImportError:
    ARCH_AVAILABLE = False


def compute_garch_features(
    prices_df: pd.DataFrame,
    *,
    lookback_days: int = 252,
    refit_every: int = 5,
    realized_vol_window: int = 20,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute GARCH-based volatility features for all symbols.

    For each symbol, fits the best GARCH variant (GARCH/EGARCH/GJR by BIC)
    on a rolling window and produces 1-step-ahead features.

    Args:
        prices_df: Panel DataFrame with ``timestamp``, ``symbol``, ``close``.
        lookback_days: Rolling window for GARCH fitting.
        refit_every: Re-estimate model every N days (parameters change slowly).
        realized_vol_window: Window for realized vol (used for vol ratio).
        timestamp_col: Timestamp column name.
        symbol_col: Symbol column name.
        close_col: Close price column name.

    Returns:
        DataFrame with columns: ``timestamp``, ``symbol``,
        ``garch_vol_1d``, ``garch_vol_ratio``, ``garch_asymmetry``,
        ``garch_persistence``, ``garch_vol_zscore``.
        Empty DataFrame if ``arch`` is not installed.
    """
    output_cols = [
        timestamp_col, symbol_col,
        "garch_vol_1d", "garch_vol_ratio", "garch_asymmetry",
        "garch_persistence", "garch_vol_zscore",
    ]

    if not ARCH_AVAILABLE:
        logger.debug("[GARCH Features] arch package not installed — returning empty")
        return pd.DataFrame(columns=output_cols)

    if prices_df is None or prices_df.empty:
        return pd.DataFrame(columns=output_cols)

    all_rows: list[dict] = []
    symbols = prices_df[symbol_col].unique()

    for sym in symbols:
        sym_data = (
            prices_df[prices_df[symbol_col] == sym]
            .sort_values(timestamp_col)
            .reset_index(drop=True)
        )
        if len(sym_data) < lookback_days + 10:
            continue

        closes = sym_data[close_col].astype(float).values
        timestamps = sym_data[timestamp_col].values

        # Log returns
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(closes))

        # Realized vol (annualized)
        rv_series = pd.Series(log_ret).rolling(realized_vol_window).std() * np.sqrt(252)
        rv_values = rv_series.values

        last_result = None

        for i in range(lookback_days, len(log_ret)):
            # Refit periodically
            if last_result is None or (i - lookback_days) % refit_every == 0:
                window = log_ret[max(0, i - lookback_days):i]
                window = window[np.isfinite(window)]
                if len(window) >= 60:
                    result = fit_best_garch(window, sym)
                    if result is not None:
                        last_result = result

            if last_result is None:
                continue

            rv_current = rv_values[i] if i < len(rv_values) else np.nan
            garch_vol = last_result.vol_1d

            # Vol ratio: GARCH / realized (>1 means vol expanding)
            vol_ratio = (
                garch_vol / rv_current
                if rv_current is not None and np.isfinite(rv_current) and rv_current > 1e-10
                else np.nan
            )

            all_rows.append({
                timestamp_col: timestamps[i + 1],  # forecast for next day
                symbol_col: sym,
                "garch_vol_1d": round(garch_vol, 6),
                "garch_vol_ratio": round(vol_ratio, 4) if np.isfinite(vol_ratio) else np.nan,
                "garch_asymmetry": last_result.asymmetry,
                "garch_persistence": last_result.persistence,
            })

    if not all_rows:
        return pd.DataFrame(columns=output_cols)

    result_df = pd.DataFrame(all_rows)

    # Z-score of GARCH vol relative to rolling 60d mean per symbol
    result_df["garch_vol_zscore"] = np.nan
    for sym in result_df[symbol_col].unique():
        mask = result_df[symbol_col] == sym
        sym_vol = result_df.loc[mask, "garch_vol_1d"]
        if len(sym_vol) >= 20:
            rolling_mean = sym_vol.rolling(60, min_periods=20).mean()
            rolling_std = sym_vol.rolling(60, min_periods=20).std()
            zscore = (sym_vol - rolling_mean) / rolling_std.replace(0, np.nan)
            result_df.loc[mask, "garch_vol_zscore"] = zscore.round(4)

    return result_df


def compute_garch_features_snapshot(
    prices_df: pd.DataFrame,
    *,
    lookback_days: int = 252,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> dict[str, dict[str, float]]:
    """Single-point GARCH snapshot for each symbol (latest data only).

    Lighter-weight alternative to ``compute_garch_features`` when only
    the most recent GARCH estimates are needed (e.g., for current-day
    position sizing).

    Returns:
        Dict mapping symbol → feature dict.
    """
    if not ARCH_AVAILABLE:
        return {}

    if prices_df is None or prices_df.empty:
        return {}

    results: dict[str, dict[str, float]] = {}
    symbols = prices_df[symbol_col].unique()

    for sym in symbols:
        sym_data = (
            prices_df[prices_df[symbol_col] == sym]
            .sort_values(timestamp_col)
        )
        if len(sym_data) < 60:
            continue

        closes = sym_data[close_col].astype(float).values[-lookback_days:]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(closes))
        log_ret = log_ret[np.isfinite(log_ret)]

        if len(log_ret) < 60:
            continue

        r = fit_best_garch(log_ret, sym)
        if r is not None:
            results[sym] = {
                "garch_vol_1d": r.vol_1d,
                "garch_vol_5d": r.vol_5d,
                "garch_asymmetry": r.asymmetry,
                "garch_persistence": r.persistence,
                "garch_model_type": r.model_type,
                "garch_bic": r.bic,
            }

    return results


__all__ = [
    "compute_garch_features",
    "compute_garch_features_snapshot",
]
