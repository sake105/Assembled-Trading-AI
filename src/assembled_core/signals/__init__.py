"""Signal generation modules.

This package handles:
- Trading signal generation from technical indicators
- Signal rules (trend-following, mean-reversion, etc.)
- Signal filtering and validation
- Signal combination (multi-strategy)
- Multi-factor signal generation
- Signal API for standardized signal representation (A2)

Note: Current EMA crossover signals are in pipeline.signals.compute_ema_signals.
This package will provide a broader signal generation framework.
"""

from src.assembled_core.signals.multifactor_signal import (
    MultiFactorSignalResult,
    build_multifactor_signal,
    select_top_bottom,
)
from src.assembled_core.signals.signal_api import (
    SignalMetadata,
    make_signal_frame,
    normalize_signals,
    validate_signal_frame,
)

__all__ = [
    "MultiFactorSignalResult",
    "build_multifactor_signal",
    "select_top_bottom",
    # Signal API (A2)
    "SignalMetadata",
    "normalize_signals",
    "make_signal_frame",
    "validate_signal_frame",
]
