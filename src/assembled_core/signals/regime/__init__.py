"""Regime-inference subpackage."""

from __future__ import annotations

from src.assembled_core.signals.regime.hmm_posterior import (
    RegimeBlendResult,
    blend_weights_by_regime_posterior,
    smooth_posterior,
)

__all__ = [
    "RegimeBlendResult",
    "blend_weights_by_regime_posterior",
    "smooth_posterior",
]
