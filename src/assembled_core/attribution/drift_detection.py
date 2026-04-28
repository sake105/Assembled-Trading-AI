"""Attribution-distribution drift detection — standalone entry point.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §6.1.

The full KS-test implementation lives in ``attribution.time_series``.
This module re-exports it under the canonical name from the spec
so that users can import directly from ``attribution.drift_detection``
without needing to know about the time_series module layout.
"""
from __future__ import annotations

from assembled_core.attribution.time_series import detect_attribution_drift

__all__ = ["detect_attribution_drift"]
