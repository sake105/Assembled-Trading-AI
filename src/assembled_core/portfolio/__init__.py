"""Portfolio management modules."""

from __future__ import annotations

from src.assembled_core.portfolio.cost_aware_wrapper import (
    apply_cost_aware_from_policy,
    apply_cost_aware_wrapper,
)
from src.assembled_core.portfolio.market_neutral_optimizer import (  # noqa: F401
    MarketNeutralConfig,
    MarketNeutralResult,
    optimize_market_neutral,
)
from src.assembled_core.portfolio.multiasset_allocator import (  # noqa: F401
    RegimeAllocation,
    RegimeDetector,
    RegimeDetectorConfig,
    allocate_by_regime,
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
    "MarketNeutralConfig",
    "MarketNeutralResult",
    "optimize_market_neutral",
    "RegimeAllocation",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "allocate_by_regime",
    "StressTestConfig",
    "StressTestResult",
    "build_scenario_return_matrix",
    "evaluate_stress_scenarios",
    "get_cvxpy_stress_constraints",
]
