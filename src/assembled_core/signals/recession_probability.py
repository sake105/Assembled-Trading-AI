"""Recession Probability via Hamilton Markov Regime Switching.

From 13_FREE_MODULE.md §13.9.
Hamilton 2022: MarkovRegression on T10Y3M + NFCI.
Binary risk-off timing signal.

Library: statsmodels.tsa.regime_switching.MarkovRegression

When recession_prob > 0.5:
  → Scale all Long signals with 0.5 multiplier
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_recession_probability(
    t10y3m: pd.Series,
    nfci: pd.Series,
    k_regimes: int = 2,
) -> pd.Series:
    """Estimate recession probability using Markov regime switching.

    Args:
        t10y3m: 10y-3m yield curve (FRED: T10Y3M) — inversion = recession signal
        nfci: National Financial Conditions Index (FRED: NFCI)
        k_regimes: Number of regimes (default 2: expansion vs recession)

    Returns:
        Series of smoothed recession probabilities [0, 1].
        Returns empty Series if statsmodels unavailable.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        logger.warning("statsmodels not installed — pip install statsmodels")
        return pd.Series(dtype=float)

    # Align to common index, forward-fill NFCI (weekly release)
    common = t10y3m.index.intersection(nfci.index)
    if len(common) < 50:
        logger.debug("Insufficient data for recession probability: %d obs", len(common))
        return pd.Series(dtype=float)

    t10y3m_aligned = t10y3m.loc[common].dropna()
    nfci_aligned = nfci.reindex(t10y3m_aligned.index).ffill().fillna(0)

    try:
        model = MarkovRegression(
            t10y3m_aligned,
            k_regimes=k_regimes,
            trend="c",
            switching_variance=True,
            exog_tvtp=nfci_aligned.values.reshape(-1, 1),
        )
        res = model.fit(disp=False)
        # State 1 = high-yield-curve-inversion regime = recession
        # Identify recession regime by lower mean T10Y3M
        means = [float(res.params[f"const[{i}]"]) for i in range(k_regimes)]
        recession_state = int(np.argmin(means))  # most inverted = recession
        recession_prob = pd.Series(
            res.smoothed_marginal_probabilities[:, recession_state],
            index=t10y3m_aligned.index,
            name="recession_probability",
        )
        logger.info(
            "Recession probability: current=%.2f, mean=%.2f",
            float(recession_prob.iloc[-1]),
            float(recession_prob.mean()),
        )
        return recession_prob
    except Exception as exc:
        logger.warning("Markov regime switching failed: %s", exc)
        return pd.Series(dtype=float)


def recession_signal_multiplier(recession_prob: float, threshold: float = 0.5) -> float:
    """Return Long-signal scaling multiplier based on recession probability.

    Args:
        recession_prob: Current recession probability [0, 1].
        threshold: Probability above which to scale down (default 0.5).

    Returns:
        1.0 if normal, 0.5 if recession likely.
    """
    return 0.5 if recession_prob > threshold else 1.0


def latest_recession_prob_from_fred(fred_client: object) -> float:
    """Fetch T10Y3M and NFCI from FRED and compute latest recession probability.

    Args:
        fred_client: fredapi.Fred instance.

    Returns:
        Latest recession probability [0, 1]. Returns 0.3 (neutral) on failure.
    """
    try:
        t10y3m = fred_client.get_series("T10Y3M")
        nfci = fred_client.get_series("NFCI")
        probs = compute_recession_probability(t10y3m, nfci)
        if probs.empty:
            return 0.3
        return float(probs.iloc[-1])
    except Exception as exc:
        logger.warning("FRED recession probability failed: %s", exc)
        return 0.3


__all__ = [
    "compute_recession_probability",
    "recession_signal_multiplier",
    "latest_recession_prob_from_fred",
]
