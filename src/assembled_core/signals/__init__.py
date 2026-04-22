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

# Wired 2026-04-22: previously orphan signals.behavioral_finance
from src.assembled_core.signals.behavioral_finance import (
    BehavioralConfig,
    BehavioralSignal,
    compute_anchoring_score,
    compute_disposition_score,
    compute_herding_score,
    compute_overreaction_score,
    generate_behavioral_signals,
)
from src.assembled_core.signals.mean_reversion import (  # noqa: F401
    compute_mean_reversion_signals,
    compute_rsi,
)
from src.assembled_core.signals.ml_integration import (  # noqa: F401
    MLPipelineOutput,
    MLSignalPipeline,
)
from src.assembled_core.signals.plugin_loader import (  # noqa: F401
    discover_signal_plugins,
    load_signal_plugin,
)
from src.assembled_core.signals.risk_aware_combiner import (  # noqa: F401
    CombinerState,
    RiskAwareSignalCombiner,
    SignalPerformance,
)
from src.assembled_core.signals import regime as regime  # noqa: F401

__all__ = [
    "MultiFactorSignalResult",
    "build_multifactor_signal",
    "select_top_bottom",
    "SignalMetadata",
    "normalize_signals",
    "make_signal_frame",
    "validate_signal_frame",
    "BehavioralConfig",
    "BehavioralSignal",
    "compute_disposition_score",
    "compute_anchoring_score",
    "compute_herding_score",
    "compute_overreaction_score",
    "generate_behavioral_signals",
]
