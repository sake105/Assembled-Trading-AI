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
from src.assembled_core.risk.regime_analysis import (
    RegimeConfig,
    classify_regimes_from_index,
    compute_regime_transitions,
    summarize_factor_ic_by_regime,
    summarize_metrics_by_regime,
)
from src.assembled_core.risk.transaction_costs import (
    compute_cost_adjusted_risk_metrics,
    compute_tca_for_trades,
    estimate_per_trade_cost,
    summarize_tca,
)
from src.assembled_core.risk.factor_exposures import (
    FactorExposureConfig,
    compute_factor_exposures,
    summarize_factor_exposures,
)

__all__ = [
    # Regime Models (D1)
    "RegimeStateConfig",
    "build_regime_state",
    "compute_regime_transition_stats",
    "evaluate_factor_ic_by_regime",
    # Regime Analysis (B3)
    "RegimeConfig",
    "classify_regimes_from_index",
    "summarize_metrics_by_regime",
    "summarize_factor_ic_by_regime",
    "compute_regime_transitions",
    # Risk Metrics (D2)
    "compute_basic_risk_metrics",
    "compute_exposure_timeseries",
    "compute_risk_by_regime",
    "compute_risk_by_factor_group",
    # Transaction Costs (E4)
    "estimate_per_trade_cost",
    "compute_tca_for_trades",
    "summarize_tca",
    "compute_cost_adjusted_risk_metrics",
    # Factor Exposures (A2)
    "FactorExposureConfig",
    "compute_factor_exposures",
    "summarize_factor_exposures",
]
