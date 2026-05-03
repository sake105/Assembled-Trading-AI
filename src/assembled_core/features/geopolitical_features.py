"""Geopolitical Risk (GPR) features for factor models.

Constructs a quantitative GPR proxy from available data sources:

- GDELT event counts and tone scores (from intel pipeline)
- VIX spike indicators (from market data)
- Conflict event counts (from intel/GDELT)

Output features:
- ``gpr_level``: 0-100 percentile-normalized GPR score
- ``gpr_zscore``: Short-term z-score for spike detection
- ``gpr_momentum``: 5-day change in GPR level
- ``gpr_regime``: Quartile-based regime (1=calm, 4=elevated)

If Caldara-Iacoviello GPR data is available via FRED (``GPRH``, ``GPRC``),
those are used directly.  Otherwise, a proxy is constructed from the
intel pipeline outputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_gpr_proxy(
    gdelt_event_counts: pd.Series | None = None,
    gdelt_tone_scores: pd.Series | None = None,
    conflict_counts: pd.Series | None = None,
    vix_series: pd.Series | None = None,
    *,
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15),
    rolling_window: int = 252,
) -> pd.DataFrame:
    """Build a composite GPR proxy index from available inputs.

    All inputs should be Series indexed by date. Missing inputs are
    handled gracefully (their weight is redistributed).

    Args:
        gdelt_event_counts: Daily GDELT conflict/instability event counts.
        gdelt_tone_scores: Daily average GDELT tone (inverted — more
            negative tone → higher risk).
        conflict_counts: Daily count of conflict events (ACLED or similar).
        vix_series: VIX close values.
        weights: Relative weights (gdelt_events, gdelt_tone, conflicts, vix).
        rolling_window: Window for percentile normalization.

    Returns:
        DataFrame with ``gpr_level``, ``gpr_zscore``, ``gpr_momentum``,
        ``gpr_regime`` columns.
    """
    components: list[tuple[pd.Series, float]] = []
    w_gdelt_evt, w_gdelt_tone, w_conflict, w_vix = weights

    if gdelt_event_counts is not None and not gdelt_event_counts.empty:
        components.append((gdelt_event_counts, w_gdelt_evt))

    if gdelt_tone_scores is not None and not gdelt_tone_scores.empty:
        # Invert tone: more negative tone → higher GPR
        components.append((-gdelt_tone_scores, w_gdelt_tone))

    if conflict_counts is not None and not conflict_counts.empty:
        components.append((conflict_counts, w_conflict))

    if vix_series is not None and not vix_series.empty:
        # VIX spike indicator: z-score > 1 contributes to GPR
        vix_z = (vix_series - vix_series.rolling(60, min_periods=20).mean()) / \
                vix_series.rolling(60, min_periods=20).std().replace(0, np.nan)
        components.append((vix_z.clip(lower=0), w_vix))

    if not components:
        logger.warning("[GPR] No input data available — returning empty")
        return pd.DataFrame(columns=["gpr_level", "gpr_zscore", "gpr_momentum", "gpr_regime"])

    # Align all series to common index
    all_series = [s for s, _ in components]
    common_idx = all_series[0].index
    for s in all_series[1:]:
        common_idx = common_idx.union(s.index)
    common_idx = common_idx.sort_values()

    # Normalize each component to z-scores, then weighted sum
    total_weight = sum(w for _, w in components)
    composite = pd.Series(0.0, index=common_idx)

    for series, weight in components:
        aligned = series.reindex(common_idx)
        # Rolling z-score normalization
        mean = aligned.rolling(rolling_window, min_periods=30).mean()
        std = aligned.rolling(rolling_window, min_periods=30).std().replace(0, np.nan)
        z = (aligned - mean) / std
        composite += (weight / total_weight) * z.fillna(0)

    # Percentile normalization to 0-100: proportion of window <= last value
    gpr_level = composite.rolling(rolling_window, min_periods=30).apply(
        lambda x: (x <= x[-1]).sum() / len(x) * 100,
        raw=True,
    )

    # Short-term z-score (20d)
    gpr_mean_20 = composite.rolling(20, min_periods=10).mean()
    gpr_std_20 = composite.rolling(20, min_periods=10).std().replace(0, np.nan)
    gpr_zscore = (composite - gpr_mean_20) / gpr_std_20

    # Momentum (5d change)
    gpr_momentum = gpr_level.diff(5)

    # Regime (quartile)
    gpr_regime = pd.cut(gpr_level, bins=[0, 25, 50, 75, 100], labels=[1, 2, 3, 4])

    result = pd.DataFrame({
        "gpr_level": gpr_level.round(2),
        "gpr_zscore": gpr_zscore.round(4),
        "gpr_momentum": gpr_momentum.round(2),
        "gpr_regime": gpr_regime,
    }, index=common_idx)

    return result


def compute_gpr_from_fred(
    gpr_series: pd.Series,
    *,
    rolling_window: int = 252,
) -> pd.DataFrame:
    """Compute GPR features from Caldara-Iacoviello GPR Index (via FRED).

    If the official GPR index is available, this produces more reliable
    features than the proxy.

    Args:
        gpr_series: Caldara-Iacoviello GPR index values (FRED: GPRH or GPRC).
        rolling_window: Window for percentile normalization.

    Returns:
        DataFrame with GPR features.
    """
    if gpr_series is None or gpr_series.empty:
        return pd.DataFrame(columns=["gpr_level", "gpr_zscore", "gpr_momentum", "gpr_regime"])

    # Percentile over rolling window: proportion of window <= last value
    gpr_level = gpr_series.rolling(rolling_window, min_periods=30).apply(
        lambda x: (x <= x[-1]).sum() / len(x) * 100,
        raw=True,
    )

    mean_20 = gpr_series.rolling(20, min_periods=10).mean()
    std_20 = gpr_series.rolling(20, min_periods=10).std().replace(0, np.nan)
    gpr_zscore = (gpr_series - mean_20) / std_20

    gpr_momentum = gpr_level.diff(5)
    gpr_regime = pd.cut(gpr_level, bins=[0, 25, 50, 75, 100], labels=[1, 2, 3, 4])

    return pd.DataFrame({
        "gpr_level": gpr_level.round(2),
        "gpr_zscore": gpr_zscore.round(4),
        "gpr_momentum": gpr_momentum.round(2),
        "gpr_regime": gpr_regime,
    }, index=gpr_series.index)


__all__ = [
    "compute_gpr_from_fred",
    "compute_gpr_proxy",
]
