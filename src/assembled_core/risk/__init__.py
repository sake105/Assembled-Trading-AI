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

# Wired 2026-04-22: previously orphan risk modules
from src.assembled_core.risk.intraday_monitor import (
    IntradayRiskConfig,
    PositionSnapshot,
)
from src.assembled_core.risk.param_stability import (
    check_drawdown_stability,
    check_turnover_stability,
    check_vol_stability,
    compute_rolling_max_drawdown,
    compute_rolling_vol_estimates,
)
from src.assembled_core.risk.regime_costs import (
    RegimeCostConfig,
    RegimeCostEstimate,
    estimate_regime_costs,
)
from src.assembled_core.risk.antifragility import (  # noqa: F401
    compute_antifragility_score,
    compute_portfolio_antifragility,
)
from src.assembled_core.risk.profit_targets import (  # noqa: F401
    PositionRecord,
    ProfitTargetConfig,
    build_position_records,
    check_profit_targets,
)
from src.assembled_core.risk.systemic_risk import compute_return_network_centrality  # noqa: F401
from src.assembled_core.risk.tail_dependence import (  # noqa: F401
    classify_tail_regime,
    compute_empirical_tail_dependence,
    compute_portfolio_tail_dependence_score,
)
from src.assembled_core.risk.tail_hedge import (  # noqa: F401
    CollarConfig,
    TailHedgeResult,
    compute_collar,
    compute_put_spread,
    dynamic_hedge_ratio,
    estimate_option_premium,
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
    # Intraday monitor
    "IntradayRiskConfig",
    "PositionSnapshot",
    # Parameter stability
    "check_drawdown_stability",
    "check_turnover_stability",
    "check_vol_stability",
    "compute_rolling_max_drawdown",
    "compute_rolling_vol_estimates",
    # Regime costs
    "RegimeCostConfig",
    "RegimeCostEstimate",
    "estimate_regime_costs",
]
