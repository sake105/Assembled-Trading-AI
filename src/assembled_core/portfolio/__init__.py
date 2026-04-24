"""Portfolio management modules.

Public API exposes:
- cost-aware turnover wrapper
- long/short balance enforcement
- market-neutral optimizer
- multi-period optimization
- multi-asset regime allocator
- regime-weighted portfolio templates
- robust optimization
- stress-test constraints
Wired 2026-04-22 from previously orphan modules.
"""

from __future__ import annotations

from src.assembled_core.portfolio.cost_aware_wrapper import (
    apply_cost_aware_from_policy,
    apply_cost_aware_wrapper,
)
from src.assembled_core.portfolio.long_short_balance import (  # noqa: F401
    ExposureMetrics,
    LongShortBalancer,
)
from src.assembled_core.portfolio.market_neutral_optimizer import (  # noqa: F401
    MarketNeutralConfig,
    MarketNeutralResult,
    optimize_market_neutral,
)
from src.assembled_core.portfolio.multi_period import (  # noqa: F401
    MultiPeriodResult,
    compute_trade_speed,
    garleanu_pedersen_target,
    multi_period_optimize,
)
from src.assembled_core.portfolio.multiasset_allocator import (  # noqa: F401
    RegimeAllocation,
    RegimeDetector,
    RegimeDetectorConfig,
    allocate_by_regime,
)
from src.assembled_core.portfolio.robust_optimizer import (  # noqa: F401
    RobustOptResult,
    compute_robust_weights,
)
from src.assembled_core.portfolio.stress_test_constraints import (  # noqa: F401
    StressTestConfig,
    StressTestResult,
    build_scenario_return_matrix,
    evaluate_stress_scenarios,
    get_cvxpy_stress_constraints,
)

__all__ = [
    "apply_cost_aware_wrapper",
    "apply_cost_aware_from_policy",
    "ExposureMetrics",
    "LongShortBalancer",
    "MarketNeutralConfig",
    "MarketNeutralResult",
    "optimize_market_neutral",
    "MultiPeriodResult",
    "compute_trade_speed",
    "garleanu_pedersen_target",
    "multi_period_optimize",
    "RegimeAllocation",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "allocate_by_regime",
    "RobustOptResult",
    "compute_robust_weights",
    "StressTestConfig",
    "StressTestResult",
    "build_scenario_return_matrix",
    "evaluate_stress_scenarios",
    "get_cvxpy_stress_constraints",
]
