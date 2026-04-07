"""Graceful degradation for missing data sources (Plan 11.2).

When a data source fails (API timeout, missing file, stale data), the pipeline
should degrade gracefully instead of crashing:
- Log the failure clearly
- Neutralize affected features (set to cross-sectional mean / neutral value)
- Continue pipeline with reduced signal quality
- Flag degraded components in output artifacts

Usage:
    from src.assembled_core.pipeline.graceful_degradation import (
        DegradationTracker,
        neutralize_missing_features,
    )

    tracker = DegradationTracker()
    try:
        fred_data = fetch_fred_macro()
    except Exception as exc:
        tracker.record_failure("fred_macro", str(exc))
        fred_data = None

    features_df = neutralize_missing_features(features_df, tracker.failed_sources)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# Source → features that depend on it
SOURCE_FEATURE_MAP: dict[str, list[str]] = {
    "fred_macro": ["yield_curve_slope", "fed_funds_rate", "unemployment", "cpi_yoy", "pce_inflation"],
    "vix": ["vix_level", "vix_zscore", "vix_term_structure", "vix_change"],
    "gdelt": ["gpr_level", "gpr_zscore", "gpr_momentum", "gdelt_event_count"],
    "congress": ["congress_net_buy_30d", "congress_net_buy_90d"],
    "earnings": ["earnings_surprise", "revision_momentum", "sue"],
    "insider": ["insider_net_buy", "insider_sentiment"],
    "options": ["skew_index", "implied_vol_spread", "put_call_ratio"],
    "bdi": ["baltic_dry_index", "copper_gold_ratio"],
}

# Default neutral values per feature type
NEUTRAL_VALUES: dict[str, float] = {
    "yield_curve_slope": 0.0,
    "fed_funds_rate": 0.0,
    "vix_level": 20.0,
    "vix_zscore": 0.0,
    "rsi_14": 50.0,
    "macd": 0.0,
    "bollinger_pctb": 0.5,
}


@dataclass
class DegradationTracker:
    """Track data source failures during a pipeline run."""

    failed_sources: dict[str, str] = field(default_factory=dict)
    degraded_features: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_failure(self, source: str, reason: str) -> None:
        """Record a data source failure."""
        self.failed_sources[source] = reason
        features = SOURCE_FEATURE_MAP.get(source, [])
        self.degraded_features.extend(features)
        logger.warning(
            "[Degradation] Source '%s' failed: %s — %d features degraded",
            source, reason, len(features),
        )

    @property
    def is_degraded(self) -> bool:
        return len(self.failed_sources) > 0

    @property
    def severity(self) -> str:
        n = len(self.failed_sources)
        if n == 0:
            return "none"
        if n <= 2:
            return "minor"
        if n <= 5:
            return "moderate"
        return "severe"

    def summary(self) -> dict:
        return {
            "is_degraded": self.is_degraded,
            "severity": self.severity,
            "n_failed_sources": len(self.failed_sources),
            "n_degraded_features": len(self.degraded_features),
            "failed_sources": self.failed_sources,
            "degraded_features": sorted(set(self.degraded_features)),
        }


def neutralize_missing_features(
    df,
    failed_sources: dict[str, str],
    neutral_values: dict[str, float] | None = None,
) -> "pd.DataFrame":
    """Neutralize features from failed sources.

    Sets affected columns to their neutral value (cross-sectional mean
    or predefined neutral). This prevents NaN propagation while clearly
    marking degraded data.

    Args:
        df: Features DataFrame.
        failed_sources: Source name → failure reason.
        neutral_values: Override neutral values per feature.

    Returns:
        DataFrame with neutralized features.
    """
    import pandas as pd

    if not failed_sources or df is None or df.empty:
        return df

    nv = neutral_values or NEUTRAL_VALUES
    affected_cols = set()

    for source in failed_sources:
        features = SOURCE_FEATURE_MAP.get(source, [])
        for feat in features:
            if feat in df.columns:
                affected_cols.add(feat)

    df = df.copy()
    for col in affected_cols:
        if col in nv:
            df[col] = nv[col]
        else:
            # Use cross-sectional mean as neutral
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0.0)

    if affected_cols:
        logger.info(
            "[Degradation] Neutralized %d features: %s",
            len(affected_cols), sorted(affected_cols),
        )

    return df


__all__ = [
    "DegradationTracker",
    "SOURCE_FEATURE_MAP",
    "neutralize_missing_features",
]
