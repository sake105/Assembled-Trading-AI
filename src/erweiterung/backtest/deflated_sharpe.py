"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Problem
-------
Sharpe Ratio kann durch zwei Effekte verzerrt sein:
1. **Selection Bias**: Wer N Strategien testet, findet zwangsläufig mehrere mit
   hohem Sharpe — auch ohne echte Edge ("Backtest-Overfitting").
2. **Non-Normality**: Skew und Kurtosis verzerren die Standardfehler-
   Schätzung der Sharpe Ratio.

Deflated Sharpe Ratio
---------------------
DSR korrigiert beides:
    DSR = Φ((SR - SR_0) × √((T-1) / (1 - γ_3·SR + (γ_4-1)/4·SR²)))

mit
- γ_3 = Skewness der Returns
- γ_4 = Kurtosis (Excess + 3)
- SR_0 = Sharpe-Threshold unter Null-Hypothese (kein Edge), abh. von #trials

Referenzen
----------
- Bailey & Lopez de Prado (2014). The Deflated Sharpe Ratio. *J. Portfolio
  Management* 40(5).
- Bailey & Lopez de Prado (2014). The Probability of Backtest Overfitting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# erfinv approximation; SciPy ist optional
try:
    from scipy.special import erfinv as _erfinv  # type: ignore
except ImportError:

    def _erfinv(x):  # type: ignore
        # Winitzki approx
        a = 0.147
        ln1m = np.log(1 - x * x)
        first = 2 / (math.pi * a) + ln1m / 2
        return np.sign(x) * np.sqrt(np.sqrt(first * first - ln1m / a) - first)


def _norm_ppf(q: float) -> float:
    return float(math.sqrt(2) * _erfinv(2 * q - 1))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def estimated_sharpe_threshold(
    n_trials: int,
    annual_factor: float = 252,
    sr_variance_estimate: float = 1.0,
) -> float:
    """Sharpe-Threshold unter H₀ (kein Edge), gegeben N Trials.

    Approximation aus Bailey/Lopez (2014):
        SR₀(N) ≈ √(V[SR̂]) × ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e)))

    Mit Euler-Mascheroni-γ ≈ 0.5772.
    """
    gamma = 0.5772156649
    if n_trials <= 1:
        return 0.0
    sigma_sr = np.sqrt(sr_variance_estimate)
    inv1 = _norm_ppf(1 - 1 / n_trials)
    inv2 = _norm_ppf(1 - 1 / (n_trials * np.e))
    sr0 = sigma_sr * ((1 - gamma) * inv1 + gamma * inv2)
    # Annualize? In Bailey/Lopez SR is daily; here we keep it raw (caller annualizes).
    return float(sr0)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    annual_factor: float = 252,
) -> dict:
    """Berechne DSR für eine Returns-Series.

    Args:
        returns: Series von periodic returns (z. B. daily).
        n_trials: Anzahl gleichzeitig getesteter Strategien (Selection-Korrektur).
        annual_factor: Annualizing-Faktor (252 für daily, 12 für monatlich).

    Returns:
        Dict mit ``sr``, ``annualized_sr``, ``skew``, ``kurt``, ``sr0``, ``dsr_z``,
        ``dsr_p``.

    Interpretation
    --------------
    - dsr_z > 1.96 (p < 0.05) ⇒ statistisch signifikante Edge nach Korrektur.
    - dsr_z < 1.96 ⇒ Sharpe könnte einfach Backtest-Overfitting sein.
    """
    r = returns.dropna()
    if len(r) < 30:
        return {"error": "too few obs"}
    sr = float(r.mean() / r.std(ddof=0))
    sr_ann = sr * np.sqrt(annual_factor)
    skew = float(r.skew())
    kurt = float(r.kurt() + 3)  # pandas kurt is excess; we want raw

    T = len(r)
    if T <= 1:
        return {"error": "need T > 1"}

    sr0 = estimated_sharpe_threshold(n_trials)
    denom = 1 - skew * sr + ((kurt - 1) / 4) * sr * sr
    if denom <= 0:
        return {"error": "denom not positive (extreme tails)"}
    z = (sr - sr0) * np.sqrt((T - 1) / denom)
    dsr = _norm_cdf(z)

    return {
        "sr": sr,
        "annualized_sr": sr_ann,
        "skew": skew,
        "kurt": kurt,
        "sr0": sr0,
        "dsr_z": z,
        "dsr_p": 1 - dsr,
        "dsr": dsr,
        "n_obs": T,
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark: float = 0.0, annual_factor: float = 252
) -> float:
    """Probabilistic Sharpe Ratio (PSR): P(SR_true > sr_benchmark).

    Bailey/Lopez (2012). PSR(SR_benchmark) ist die Wahrscheinlichkeit,
    dass der wahre SR über sr_benchmark liegt — gegeben skew/kurt-korrigierten
    Standardfehler.
    """
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")
    sr = float(r.mean() / r.std(ddof=0))
    skew = float(r.skew())
    kurt = float(r.kurt() + 3)
    T = len(r)
    denom = 1 - skew * sr + ((kurt - 1) / 4) * sr * sr
    if denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt((T - 1) / denom)
    return float(_norm_cdf(z))


__all__ = [
    "estimated_sharpe_threshold",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]
