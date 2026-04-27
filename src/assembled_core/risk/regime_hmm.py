"""DEPRECATED: use src.assembled_core.ml.regime_hmm instead.

B4 deduplication: this file was a second implementation of RegimeHMM / predict_regime.
Canonical implementation lives in ml/regime_hmm.py. This shim re-exports from there
with a DeprecationWarning so any remaining callers are not silently broken.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

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


@dataclass
class HMMRegimeConfig:
    """DEPRECATED compat shim. Replaced by ml.regime_hmm.RegimeHMM parameters."""
    n_states: int = 3
    n_iter: int = 100
    covariance_type: str = "full"
    random_state: int = 42


def _try_import_hmm():
    """DEPRECATED internal helper — returns hmmlearn.hmm module or None."""
    try:
        import hmmlearn.hmm
        return hmmlearn.hmm
    except ImportError:
        return None


def fit_regime_hmm(returns: pd.Series, vols: pd.Series | None = None,
                   config: HMMRegimeConfig | None = None):
    """DEPRECATED: use ml.regime_hmm.RegimeHMM instead.

    Returns (model, states_series). When hmmlearn is absent, returns (None, empty Series).
    """
    warnings.warn(
        "risk.regime_hmm.fit_regime_hmm is deprecated — use ml.regime_hmm.RegimeHMM instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _try_import_hmm() is None:
        return None, pd.Series(dtype="object")
    cfg = config or HMMRegimeConfig()
    try:
        from src.assembled_core.ml.regime_hmm import RegimeHMM
        model = RegimeHMM(n_states=cfg.n_states, n_iter=cfg.n_iter,
                          covariance_type=cfg.covariance_type,
                          random_state=cfg.random_state)
        model.fit(returns)
        states = model.predict_regime(returns)
        return model, states
    except Exception:
        return None, pd.Series(dtype="object")


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


__all__ = ["predict_regime", "RegimeHMM", "HMMRegimeConfig", "fit_regime_hmm", "_try_import_hmm"]
