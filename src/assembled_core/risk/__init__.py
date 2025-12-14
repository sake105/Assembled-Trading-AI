"""Risk Management and Regime Detection Module.

This module provides regime detection, risk overlay, and advanced risk metrics functionality.
"""
from __future__ import annotations

from src.assembled_core.risk.regime_models import (
    RegimeStateConfig,
    build_regime_state,
    compute_regime_transition_stats,
    evaluate_factor_ic_by_regime,
)
from src.assembled_core.risk.risk_metrics import (
    compute_basic_risk_metrics,
    compute_exposure_timeseries,
    compute_risk_by_factor_group,
    compute_risk_by_regime,
)

__all__ = [
    # Regime Models (D1)
    "RegimeStateConfig",
    "build_regime_state",
    "compute_regime_transition_stats",
    "evaluate_factor_ic_by_regime",
    # Risk Metrics (D2)
    "compute_basic_risk_metrics",
    "compute_exposure_timeseries",
    "compute_risk_by_regime",
    "compute_risk_by_factor_group",
]

