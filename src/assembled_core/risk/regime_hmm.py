"""DEPRECATED: use src.assembled_core.ml.regime_hmm instead.

B4 deduplication: this file was a second implementation of RegimeHMM / predict_regime.
Canonical implementation lives in ml/regime_hmm.py. This shim re-exports from there
with a DeprecationWarning so any remaining callers are not silently broken.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "risk.regime_hmm is deprecated — import from ml.regime_hmm instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from src.assembled_core.ml.regime_hmm import (  # noqa: F401
        RegimeHMM,
        HMMLEARN_AVAILABLE,
    )
except ImportError:
    pass


def predict_regime(*args, **kwargs):
    """DEPRECATED: use RegimeHMM(n_states=N).predict_regime(returns)."""
    warnings.warn(
        "risk.regime_hmm.predict_regime is deprecated — use ml.regime_hmm.RegimeHMM instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.assembled_core.ml.regime_hmm import RegimeHMM
    model = RegimeHMM()
    if args:
        return model.predict_regime(args[0])
    return model.predict_regime(kwargs.get("returns", kwargs.get("data")))


__all__ = ["predict_regime", "RegimeHMM"]
