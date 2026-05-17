"""Risk Management and Regime Detection Module.

This module provides regime detection, risk overlay, and advanced risk metrics functionality.
"""

from __future__ import annotations

from src.assembled_core.risk.factor_exposures import (
    FactorExposureConfig,
    compute_factor_exposures,
    summarize_factor_exposures,
)
from src.assembled_core.risk.profit_targets import (  # noqa: F401
    PositionRecord,
    ProfitTargetConfig,
    build_position_records,
    check_profit_targets,
)
from src.assembled_core.risk.regime_analysis import (
    RegimeConfig,
    classify_regimes_from_index,
    compute_regime_transitions,
    summarize_factor_ic_by_regime,
    summarize_metrics_by_regime,
)
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
from src.assembled_core.risk.transaction_costs import (
    compute_cost_adjusted_risk_metrics,
    compute_tca_for_trades,
    estimate_per_trade_cost,
    summarize_tca,
)
from src.assembled_core.risk.garch_vol import (  # noqa: F401
    forecast_vol,
    size_vol_target,
    compute_vol_forecasts,
)

# F-stage1-1 (2026-05-17): garch_vol_forecast is deprecated. Removed from
# package re-export so the DeprecationWarning fires only when callers
# explicitly opt in via direct import. This prevents warning-noise
# pollution in test/CI logs from every risk-package import.
# See KNOWN_ISSUES.md §6.5.2 Phase 2.
from src.assembled_core.risk.margin_call_handler import handle_margin_call  # noqa: F401

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
    # Vol forecasting (GARCH / fallback) — F-stage1-1: garch_vol_forecast
    # symbols removed from __all__ since 2026-05-17 (deprecated; direct import only).
    "forecast_vol",
    "size_vol_target",
    "compute_vol_forecasts",
    # Margin Call Handler (Item 42)
    "handle_margin_call",
]
