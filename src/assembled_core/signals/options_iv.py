"""Options Implied Volatility and Greeks via py_vollib.

From 11_FREE_MODELLE.md §11.14.
IV inversion via Peter Jäckel's "Let's Be Rational" — 2 iterations, no Newton's method.
Vol-Surface features: IV-Rank, Skew, Term-Structure.

Install: pip install py_vollib==1.0.1 py_vollib_vectorized==0.1.1
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OptionType = Literal["c", "p"]


def _try_py_vollib():
    try:
        import py_vollib.black_scholes as bs
        import py_vollib.black_scholes.implied_volatility as bsiv
        return bsiv, bs
    except ImportError:
        logger.warning("py_vollib not installed — pip install py_vollib==1.0.1")
        return None, None


def _try_py_vollib_vectorized():
    try:
        import py_vollib_vectorized as pyvv
        return pyvv
    except ImportError:
        return None


def compute_iv(
    flag: OptionType,
    S: float,
    K: float,
    t: float,
    r: float,
    price: float,
) -> float | None:
    """Compute implied volatility for a single option.

    Args:
        flag: 'c' for call, 'p' for put
        S: Underlying spot price
        K: Strike price
        t: Time to expiration in years
        r: Risk-free rate (annualized, e.g. 0.05)
        price: Market option price

    Returns:
        Implied volatility (annualized), or None if py_vollib unavailable / diverged.
    """
    bsiv, _ = _try_py_vollib()
    if bsiv is None:
        return None

    try:
        iv = bsiv.implied_volatility(price, S, K, t, r, flag)
        return float(iv) if np.isfinite(iv) else None
    except Exception as exc:
        logger.debug("IV computation failed (S=%.2f, K=%.2f, t=%.4f): %s", S, K, t, exc)
        return None


def compute_greeks(
    flag: OptionType,
    S: float,
    K: float,
    t: float,
    r: float,
    sigma: float,
) -> dict[str, float]:
    """Compute BS Greeks for a single option.

    Returns:
        Dict with keys: delta, gamma, theta, vega, rho.
        Returns empty dict if py_vollib unavailable.
    """
    _, bs = _try_py_vollib()
    if bs is None:
        return {}

    try:
        from py_vollib.black_scholes.greeks import analytical
        return {
            "delta": float(analytical.delta(flag, S, K, t, r, sigma)),
            "gamma": float(analytical.gamma(flag, S, K, t, r, sigma)),
            "theta": float(analytical.theta(flag, S, K, t, r, sigma)),
            "vega": float(analytical.vega(flag, S, K, t, r, sigma)),
            "rho": float(analytical.rho(flag, S, K, t, r, sigma)),
        }
    except Exception as exc:
        logger.debug("Greeks computation failed: %s", exc)
        return {}


def iv_rank(
    current_iv: float,
    iv_history: pd.Series,
    lookback_days: int = 252,
) -> float:
    """Compute IV Rank (0–100): where current IV sits in its 52-week range.

    Args:
        current_iv: Current implied volatility.
        iv_history: Historical IV series (at least lookback_days long).
        lookback_days: Rolling window (default 252 = 1 year).

    Returns:
        IV Rank in [0, 100]. 100 = at year high, 0 = at year low.
    """
    window = iv_history.tail(lookback_days).dropna()
    if len(window) < 2:
        return 50.0

    iv_low = float(window.min())
    iv_high = float(window.max())

    if iv_high <= iv_low:
        return 50.0

    return float((current_iv - iv_low) / (iv_high - iv_low) * 100)


def iv_skew(
    S: float,
    K_otm_put: float,
    K_atm: float,
    t: float,
    r: float,
    price_otm_put: float,
    price_atm_call: float,
) -> float | None:
    """Compute simple vol skew: IV(OTM put) - IV(ATM call).

    Positive skew = market paying for downside protection (normal).
    Very high skew = fear / crash insurance demand.

    Returns:
        Skew in vol units, or None if computation fails.
    """
    iv_put = compute_iv("p", S, K_otm_put, t, r, price_otm_put)
    iv_call = compute_iv("c", S, K_atm, t, r, price_atm_call)

    if iv_put is None or iv_call is None:
        return None

    return iv_put - iv_call


def vectorized_iv(
    flags: list[str],
    S_arr: list[float],
    K_arr: list[float],
    t_arr: list[float],
    r_arr: list[float],
    price_arr: list[float],
) -> np.ndarray | None:
    """Batch IV computation using py_vollib_vectorized.

    Falls back to None if vectorized library not available.

    Returns:
        Array of IVs, same length as inputs. NaN where computation failed.
    """
    pyvv = _try_py_vollib_vectorized()
    if pyvv is None:
        return None

    try:
        result = pyvv.implied_volatility(
            np.array(price_arr),
            np.array(S_arr),
            np.array(K_arr),
            np.array(t_arr),
            np.array(r_arr),
            flags,
        )
        return np.asarray(result, dtype=float)
    except Exception as exc:
        logger.debug("Vectorized IV failed: %s", exc)
        return None


__all__ = [
    "compute_iv",
    "compute_greeks",
    "iv_rank",
    "iv_skew",
    "vectorized_iv",
]
