"""Feature attribution for composite score decisions.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md.
"""
from assembled_core.attribution.schemas import CompositeAttribution
from assembled_core.attribution.storage import AttributionStore
from assembled_core.attribution.composite import (
    build_attribution,
    attribution_to_dict,
)
from assembled_core.attribution.time_series import (
    attributions_to_df,
    detect_dead_features,
    detect_attribution_drift,
    dead_feature_report,
    rolling_dimension_ic,
)

__all__ = [
    "CompositeAttribution",
    "AttributionStore",
    "build_attribution",
    "attribution_to_dict",
    "attributions_to_df",
    "detect_dead_features",
    "detect_attribution_drift",
    "dead_feature_report",
    "rolling_dimension_ic",
]
