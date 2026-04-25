"""HMM-based Regime Switching via hmmlearn.

From 13_FREE_MODULE.md §13.2 and 11_FREE_MODELLE.md §11.9.
Uses GaussianHMM with log-returns + realized-vol + VIX + term-slope.

State labels assigned post-fit by sorting states on mean-return:
  highest mean-return + lowest vol → 'bull_trend'
  lowest mean-return + highest vol → 'bear_hv'
  middle                           → 'ranging'

Retraining: weekly walk-forward (63-bar window).
Rule: never more than 3-4 states (overfitting risk).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeLabel = Literal["bull_trend", "ranging", "bear_hv", "unknown"]

_LABEL_ORDER: list[str] = ["bull_trend", "ranging", "bear_hv"]


@dataclass
class HMMRegimeConfig:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 200
    random_state: int = 42
    lookback_bars: int = 252  # training window
    retrain_every_bars: int = 63  # weekly walk-forward


def _try_import_hmm():
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not installed — install with: pip install hmmlearn==0.3.3")
        return None


def fit_regime_hmm(
    log_returns: pd.Series,
    realized_vol: pd.Series,
    vix: pd.Series | None = None,
    term_slope: pd.Series | None = None,
    cfg: HMMRegimeConfig | None = None,
) -> tuple[object | None, pd.Series]:
    """Fit GaussianHMM and predict regime states.

    Args:
        log_returns: Log-return series (aligned index)
        realized_vol: 20-day realized vol (same index)
        vix: Optional VIX series (same index)
        term_slope: Optional 2s10s yield curve (same index)
        cfg: HMM configuration

    Returns:
        (fitted_hmm_model, states_series) where states_series has string labels.
        Returns (None, empty Series) if hmmlearn is unavailable.
    """
    GaussianHMM = _try_import_hmm()
    if GaussianHMM is None:
        return None, pd.Series(dtype=str)

    if cfg is None:
        cfg = HMMRegimeConfig()

    # Build feature matrix — only use available series
    parts = [log_returns, realized_vol]
    if vix is not None:
        parts.append(vix)
    if term_slope is not None:
        parts.append(term_slope)

    common_idx = parts[0].index
    for s in parts[1:]:
        common_idx = common_idx.intersection(s.index)

    X = np.column_stack([s.loc[common_idx].values for s in parts])
    X = np.where(np.isfinite(X), X, 0.0)

    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_scaled = (X - mu) / sigma

    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
    )
    model.fit(X_scaled)
    raw_states = model.predict(X_scaled)

    # Assign semantic labels: rank states by mean-return (index 0)
    state_means = {s: X[raw_states == s, 0].mean() if (raw_states == s).any() else 0.0
                   for s in range(cfg.n_states)}
    sorted_states = sorted(state_means, key=state_means.get, reverse=True)
    label_map = {s: _LABEL_ORDER[min(i, len(_LABEL_ORDER) - 1)] for i, s in enumerate(sorted_states)}

    states = pd.Series([label_map[s] for s in raw_states], index=common_idx, name="regime_hmm")
    logger.info("HMM fit complete. State distribution: %s", states.value_counts().to_dict())
    return model, states


def predict_regime(
    model,
    log_returns: pd.Series,
    realized_vol: pd.Series,
    vix: pd.Series | None = None,
    term_slope: pd.Series | None = None,
    cfg: HMMRegimeConfig | None = None,
) -> pd.Series:
    """Predict regime labels using a pre-fitted model."""
    if model is None:
        return pd.Series(["unknown"] * len(log_returns), index=log_returns.index, name="regime_hmm")

    if cfg is None:
        cfg = HMMRegimeConfig()

    parts = [log_returns, realized_vol]
    if vix is not None:
        parts.append(vix)
    if term_slope is not None:
        parts.append(term_slope)

    common_idx = parts[0].index
    for s in parts[1:]:
        common_idx = common_idx.intersection(s.index)

    X = np.column_stack([s.loc[common_idx].values for s in parts])
    X = np.where(np.isfinite(X), X, 0.0)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_scaled = (X - mu) / sigma

    raw_states = model.predict(X_scaled)
    # Use same label ordering as fit
    state_means = {s: X[raw_states == s, 0].mean() if (raw_states == s).any() else 0.0
                   for s in range(cfg.n_states)}
    sorted_states = sorted(state_means, key=state_means.get, reverse=True)
    label_map = {s: _LABEL_ORDER[min(i, len(_LABEL_ORDER) - 1)] for i, s in enumerate(sorted_states)}
    return pd.Series([label_map[s] for s in raw_states], index=common_idx, name="regime_hmm")


__all__ = [
    "HMMRegimeConfig",
    "RegimeLabel",
    "fit_regime_hmm",
    "predict_regime",
]
