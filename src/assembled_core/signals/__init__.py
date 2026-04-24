"""Signal generation modules."""

from src.assembled_core.signals.multifactor_signal import (
    MultiFactorSignalResult,
    build_multifactor_signal,
    select_top_bottom,
)
from src.assembled_core.signals import regime as regime  # noqa: F401

__all__ = [
    "MultiFactorSignalResult",
    "build_multifactor_signal",
    "select_top_bottom",
]
