"""Feature attribution for composite score decisions.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md.
"""
from src.assembled_core.attribution.schemas import CompositeAttribution
from src.assembled_core.attribution.storage import AttributionStore
from src.assembled_core.attribution.composite import (
    build_attribution,
    attribution_to_dict,
)

__all__ = [
    "CompositeAttribution",
    "AttributionStore",
    "build_attribution",
    "attribution_to_dict",
]
