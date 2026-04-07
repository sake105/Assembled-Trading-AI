"""Feature interaction terms (V10).

Computes domain-motivated cross-feature interactions that improve
linear model performance and reduce tree depth requirements.

Reference: QLib Alpha158 feature set; AQR "Value and Momentum Everywhere".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Default interaction definitions: (name, feature_a, feature_b, operation)
DEFAULT_INTERACTIONS = [
    ("ix_momentum_x_inv_vol", "momentum_12m_excl_1m", "rv_20", "divide_inv"),
    ("ix_trend_x_inv_vol", "trend_strength_50", "rv_20", "divide_inv"),
    ("ix_momentum_x_volume", "momentum_12m_excl_1m", "volume_ratio_20", "multiply"),
    ("ix_rsi_x_trend", "rsi_14", "trend_strength_200", "multiply"),
    ("ix_earnings_x_insider", "earnings_eps_surprise_last", "insider_net_notional_60d", "multiply"),
    ("ix_trend_short_x_long", "trend_strength_20", "trend_strength_200", "subtract"),
    ("ix_reversal_x_vol", "reversal_1d", "rv_20", "multiply"),
    ("ix_momentum_x_breadth", "momentum_12m_excl_1m", "fraction_above_ma50", "multiply"),
]


def _safe_op(a: pd.Series, b: pd.Series, operation: str) -> pd.Series:
    """Apply operation between two series with NaN safety."""
    if operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b.replace(0, np.nan)
    elif operation == "divide_inv":
        # a / b but b is something like vol that we want inverse of
        # So result = a * (1/b) = a / b
        return a / b.replace(0, np.nan)
    elif operation == "subtract":
        return a - b
    elif operation == "add":
        return a + b
    elif operation == "ratio":
        return a / (a.abs() + b.abs()).replace(0, np.nan)
    elif operation == "min":
        return pd.concat([a, b], axis=1).min(axis=1)
    elif operation == "max":
        return pd.concat([a, b], axis=1).max(axis=1)
    else:
        _log.warning("Unknown interaction operation: %s", operation)
        return a * b


def compute_interaction_features(
    df: pd.DataFrame,
    interactions: list[tuple[str, str, str, str]] | None = None,
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """Compute interaction features and add them to the DataFrame.

    Args:
        df: DataFrame with feature columns.
        interactions: List of (name, feature_a, feature_b, operation) tuples.
            If None, uses DEFAULT_INTERACTIONS.
        winsorize_pct: Winsorize percentile for output (0 = no winsorize).

    Returns:
        DataFrame with interaction columns added (ix_* prefix).
    """
    if interactions is None:
        interactions = DEFAULT_INTERACTIONS

    df = df.copy()
    added = 0

    for name, feat_a, feat_b, op in interactions:
        if feat_a not in df.columns or feat_b not in df.columns:
            continue

        result = _safe_op(df[feat_a], df[feat_b], op)

        # Winsorize
        if winsorize_pct > 0 and not result.dropna().empty:
            lo = result.quantile(winsorize_pct)
            hi = result.quantile(1 - winsorize_pct)
            result = result.clip(lo, hi)

        df[name] = result
        added += 1

    if added > 0:
        _log.info("Added %d interaction features", added)

    return df


__all__ = [
    "DEFAULT_INTERACTIONS",
    "compute_interaction_features",
]
