"""Strategy modules for Assembled Trading AI Backend.

Exposes:
- EMA trend v0 strategy
- Multi-factor long-short strategy
"""

from __future__ import annotations

from src.assembled_core.strategies.ema_trend_v0 import (
    compute_signals as ema_compute_signals,
    compute_target_positions as ema_compute_target_positions,
)
from src.assembled_core.strategies.multifactor_long_short import (
    MultiFactorStrategyConfig,
    compute_multifactor_long_short_positions,
    generate_multifactor_long_short_signals,
)

__all__ = [
    "MultiFactorStrategyConfig",
    "generate_multifactor_long_short_signals",
    "compute_multifactor_long_short_positions",
    "ema_compute_signals",
    "ema_compute_target_positions",
]
