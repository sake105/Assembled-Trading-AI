"""Cornish-Fisher modified VaR (Cornish/Fisher 1937).

Theorie
-------
Klassisches Gauss-VaR ignoriert Skew + Kurtosis. Cornish-Fisher liefert
VaR-Approximation für nicht-normale Verteilungen via Taylor-Expansion:

    z_CF = z_α + (z_α² - 1)/6 · S + (z_α³ - 3z_α)/24 · K - (2z_α³ - 5z_α)/36 · S²

mit S = skewness, K = excess-kurtosis.

VaR_CF = μ + σ · z_CF.

CVaR-CF: ähnliche Korrektur für Expected Shortfall.

Reference
---------
- Cornish, E. & Fisher, R. (1937). Moments and cumulants in the specification
  of distributions. *RIS* 5.
- Boudt, K., Peterson, B. & Croux, C. (2008). Estimation and decomposition
  of downside risk for portfolios with non-normal returns. *J. Risk* 11.
"""

from __future__ import annotations

import math

import pandas as pd


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF — using Beasley-Springer-Moro."""
    try:
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf(p))
    except ImportError:
        # Rational approximation (Acklam 2003)
        a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.38357751867269e2,
            -3.066479806614716e1,
            2.506628277459239,
        ]
        b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ]
        c = [
            -7.78489400243029e-3,
            -3.223964580411365e-1,
            -2.400758277161838,
            -2.549732539343734,
            4.374664141464968,
            2.938163982698783,
        ]
        d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996,
            3.754408661907416,
        ]
        p_low = 0.02425
        p_high = 1 - p_low
        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            return (
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                * q
            ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _cornish_fisher_domain_check(skew: float, excess_kurt: float) -> tuple[bool, str]:
    """Check whether Cornish-Fisher is in its valid domain (audit C4-075).

    The CF expansion is a Taylor-series approximation around the standard
    normal. Outside a bounded skew/kurt region the expanded z_cf becomes
    non-monotone in z (Maillard 2018, Jaschke 2002), producing nonsense
    VaR estimates (negative VaR for left tails, monotonicity violations,
    etc.).

    Practical rule of thumb (Boudt-Peterson-Croux 2008 / Maillard 2018):
        |skew| <= 6           — outright bound for monotone correction
        -1 <= excess_kurt <= 9
        And the joint constraint  (excess_kurt + 3) - (skew^2 + 1) >= 0
        which is required for the moment problem to have a solution.

    Returns:
        (in_domain, reason). reason is "ok" when in_domain is True.
    """
    if abs(skew) > 6.0:
        return False, f"|skew|={abs(skew):.3f} > 6.0 (CF expansion non-monotone)"
    if excess_kurt < -1.0:
        return (
            False,
            f"excess_kurt={excess_kurt:.3f} < -1 (sub-Gaussian — moment problem)",
        )
    if excess_kurt > 9.0:
        return False, f"excess_kurt={excess_kurt:.3f} > 9 (CF expansion non-monotone)"
    # Joint moment-problem feasibility: raw kurt - skew^2 - 1 >= 0
    # equivalently: excess_kurt + 2 >= skew^2
    if excess_kurt + 2.0 < skew * skew:
        return (
            False,
            f"infeasible moments: excess_kurt+2={excess_kurt + 2:.3f} < skew^2={skew * skew:.3f}",
        )
    return True, "ok"


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.99) -> dict:
    """Compute Cornish-Fisher VaR + CVaR.

    Args:
        returns: Return-Series.
        alpha: confidence level.

    Returns:
        dict mit ``var_gauss``, ``var_cf``, ``cvar_cf``, ``skew``, ``kurt``,
        ``cf_in_domain`` (bool), ``cf_domain_reason`` (str). When the
        Cornish-Fisher approximation is outside its valid domain (audit
        C4-075), ``cf_in_domain`` is False and the caller MUST fall back
        to a different estimator (historical VaR, EVT, or Monte-Carlo).
        The CF values are STILL returned for inspection but should not
        be used as the binding risk number.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 30:
        return {"error": "too few obs"}
    mu = float(r.mean())
    sigma = float(r.std(ddof=0))
    skew = float(r.skew())
    # pandas .kurt() returns EXCESS kurtosis
    excess_kurt = float(r.kurt())
    if sigma == 0:
        return {"error": "zero vol"}

    in_domain, reason = _cornish_fisher_domain_check(skew, excess_kurt)

    z = _norm_ppf(1 - alpha)  # negative for tail

    # Cornish-Fisher z-correction
    z_cf = (
        z
        + (z**2 - 1) / 6 * skew
        + (z**3 - 3 * z) / 24 * excess_kurt
        - (2 * z**3 - 5 * z) / 36 * skew**2
    )
    var_gauss = -(mu + sigma * z)
    var_cf = -(mu + sigma * z_cf)

    # CVaR-CF (Boudt approximation)
    pdf_z = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    es_factor = -pdf_z / (1 - alpha)
    cvar_gauss = -(mu + sigma * es_factor)
    # Corrected
    es_factor_cf = (
        es_factor
        + (1 / (1 - alpha))
        * (((1 - z * z) / 6) * skew + (z**3 - 3 * z) / 24 * excess_kurt)
        * pdf_z
    )
    cvar_cf = -(mu + sigma * es_factor_cf)

    return {
        "var_gauss": var_gauss,
        "var_cf": var_cf,
        "cvar_gauss": cvar_gauss,
        "cvar_cf": cvar_cf,
        "skew": skew,
        "excess_kurt": excess_kurt,
        "alpha": alpha,
        "n_obs": len(r),
        "cf_in_domain": in_domain,
        "cf_domain_reason": reason,
    }


def rolling_var_comparison(
    returns: pd.Series, window: int = 252, alpha: float = 0.99
) -> pd.DataFrame:
    """Rolling Gauss-VaR vs Cornish-Fisher-VaR."""
    out_rows = []
    r = pd.Series(returns).dropna()
    for end in range(window, len(r) + 1):
        sub = r.iloc[end - window : end]
        res = cornish_fisher_var(sub, alpha=alpha)
        if "error" in res:
            continue
        out_rows.append({"date": r.index[end - 1], **res})
    return pd.DataFrame(out_rows).set_index("date")


__all__ = ["cornish_fisher_var", "rolling_var_comparison"]
