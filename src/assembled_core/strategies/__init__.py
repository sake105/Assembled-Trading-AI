"""Strategy modules for Assembled Trading AI Backend.

This package contains strategy implementations that can be used with the backtest engine.
"""

from src.assembled_core.strategies.multifactor_long_short import (
    MultiFactorStrategyConfig,
    compute_multifactor_long_short_positions,
    generate_multifactor_long_short_signals,
)

__all__ = [
    "MultiFactorStrategyConfig",
    "generate_multifactor_long_short_signals",
    "compute_multifactor_long_short_positions",
]

