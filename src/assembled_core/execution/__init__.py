"""Order execution and simulation modules.

Public API exposes adaptive execution algos, pre-live gate checks, and
pre-open signal evaluators. Wired 2026-04-22 from orphan modules.
"""

from __future__ import annotations

from src.assembled_core.execution.adaptive_algo import (
    AdaptiveAlgoConfig,
    AggressionLevel,
    MarketCondition,
)
from src.assembled_core.execution.pre_live_gate import (
    GateCheckResult,
    PreLiveGate,
    PreLiveGateResult,
)
from src.assembled_core.execution.pre_open_signals import (
    PreOpenConfig,
    PreOpenSignal,
    compute_overnight_gap_signal,
)
from src.assembled_core.execution.cost_model_calibrator import (  # noqa: F401
    CalibrationResult,
    CostModelPriors,
    calibrate_cost_model,
    write_calibration_report,
)

__all__ = [
    "AdaptiveAlgoConfig",
    "AggressionLevel",
    "MarketCondition",
    "GateCheckResult",
    "PreLiveGate",
    "PreLiveGateResult",
    "PreOpenConfig",
    "PreOpenSignal",
    "compute_overnight_gap_signal",
    "CalibrationResult",
    "CostModelPriors",
    "calibrate_cost_model",
    "write_calibration_report",
]
