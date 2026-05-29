"""Strategy modules for Assembled Trading AI Backend.

Exposes:
- EMA trend v0 strategy
- Multi-factor long-short strategy
- Dual momentum strategy (Antonacci-variant)
- ETF-Pairs Cointegration Mean-Reversion strategy
"""

from __future__ import annotations

from src.assembled_core.strategies.etf_pairs_meanrev import (
    compute_signals as etf_pairs_compute_signals,
)
from src.assembled_core.strategies.etf_pairs_meanrev import (
    compute_target_positions as etf_pairs_compute_target_positions,
)
from src.assembled_core.strategies.etf_pairs_meanrev import (
    generate_etf_pairs_signals_from_prices,
)
from src.assembled_core.strategies.dual_momentum import (
    compute_signals as dual_momentum_compute_signals,
)
from src.assembled_core.strategies.dual_momentum import (
    compute_target_positions as dual_momentum_compute_target_positions,
)
from src.assembled_core.strategies.dual_momentum import (
    generate_dual_momentum_signals_from_prices,
)
from src.assembled_core.strategies.ema_trend_v0 import (
    compute_signals as ema_compute_signals,
)
from src.assembled_core.strategies.ema_trend_v0 import (
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
    "dual_momentum_compute_signals",
    "dual_momentum_compute_target_positions",
    "generate_dual_momentum_signals_from_prices",
    "etf_pairs_compute_signals",
    "etf_pairs_compute_target_positions",
    "generate_etf_pairs_signals_from_prices",
]
