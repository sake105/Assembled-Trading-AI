"""Black-Scholes-Merton Option-Pricing + Greeks + Implied-Vol.

Formel
------
Für European Call: C = S·N(d1) - K·e^(-r·T)·N(d2)
mit  d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
     d2 = d1 - σ·√T

Greeks
------
- Delta = N(d1)        (Call) / -N(-d1) (Put)
- Gamma = φ(d1)/(S·σ·√T)
- Vega  = S·φ(d1)·√T   (per 1.0 vola change)
- Theta = -S·φ(d1)·σ/(2√T) - r·K·e^(-rT)·N(d2)
- Rho   = K·T·e^(-rT)·N(d2)

Implied Volatility
------------------
Newton-Raphson auf σ ↦ BS(σ) - market_price.

Anwendung
---------
- Pricing für Strategy-Vergleich
- Greeks für Risk-Management
- IV-Surface-Konstruktion aus Listed-Options
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


@dataclass
class BSParams:
    spot: float
    strike: float
    time_to_expiry: float  # in years
    risk_free: float = 0.05
    dividend_yield: float = 0.0


def _d1_d2(p: BSParams, sigma: float) -> tuple[float, float]:
    if p.time_to_expiry <= 0 or sigma <= 0:
        return float("nan"), float("nan")
    d1 = (
        math.log(p.spot / p.strike)
        + (p.risk_free - p.dividend_yield + 0.5 * sigma * sigma) * p.time_to_expiry
    ) / (sigma * math.sqrt(p.time_to_expiry))
    d2 = d1 - sigma * math.sqrt(p.time_to_expiry)
    return d1, d2


def bs_price(p: BSParams, sigma: float, is_call: bool = True) -> float:
    """Black-Scholes price."""
    if p.time_to_expiry <= 0:
        # intrinsic
        if is_call:
            return max(0.0, p.spot - p.strike)
        return max(0.0, p.strike - p.spot)
    d1, d2 = _d1_d2(p, sigma)
    disc_r = math.exp(-p.risk_free * p.time_to_expiry)
    disc_q = math.exp(-p.dividend_yield * p.time_to_expiry)
    if is_call:
        return p.spot * disc_q * _normal_cdf(d1) - p.strike * disc_r * _normal_cdf(d2)
    return p.strike * disc_r * _normal_cdf(-d2) - p.spot * disc_q * _normal_cdf(-d1)


def bs_greeks(p: BSParams, sigma: float, is_call: bool = True) -> dict:
    """All Greeks for an option."""
    if p.time_to_expiry <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(p, sigma)
    pdf_d1 = _normal_pdf(d1)
    disc_r = math.exp(-p.risk_free * p.time_to_expiry)
    disc_q = math.exp(-p.dividend_yield * p.time_to_expiry)
    sqrt_T = math.sqrt(p.time_to_expiry)
    if is_call:
        delta = disc_q * _normal_cdf(d1)
        rho = p.strike * p.time_to_expiry * disc_r * _normal_cdf(d2)
        theta = (
            -p.spot * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            - p.risk_free * p.strike * disc_r * _normal_cdf(d2)
            + p.dividend_yield * p.spot * disc_q * _normal_cdf(d1)
        )
    else:
        delta = -disc_q * _normal_cdf(-d1)
        rho = -p.strike * p.time_to_expiry * disc_r * _normal_cdf(-d2)
        theta = (
            -p.spot * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            + p.risk_free * p.strike * disc_r * _normal_cdf(-d2)
            - p.dividend_yield * p.spot * disc_q * _normal_cdf(-d1)
        )
    gamma = disc_q * pdf_d1 / (p.spot * sigma * sqrt_T)
    vega = p.spot * disc_q * pdf_d1 * sqrt_T  # per unit vol
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega) / 100,  # per 1% vol change
        "theta": float(theta) / 365,  # per day
        "rho": float(rho) / 100,  # per 1% rate
    }


def implied_volatility(
    market_price: float,
    p: BSParams,
    is_call: bool = True,
    max_iter: int = 100,
    tol: float = 1e-7,
    initial_guess: float = 0.20,
) -> float:
    """Newton-Raphson implied vol solver.

    Returns NaN if no convergence (e.g. price below intrinsic).
    """
    if market_price <= 0 or p.time_to_expiry <= 0:
        return float("nan")
    # check arbitrage bounds
    disc_r = math.exp(-p.risk_free * p.time_to_expiry)
    disc_q = math.exp(-p.dividend_yield * p.time_to_expiry)
    if is_call:
        lower = max(0.0, p.spot * disc_q - p.strike * disc_r)
        upper = p.spot * disc_q
    else:
        lower = max(0.0, p.strike * disc_r - p.spot * disc_q)
        upper = p.strike * disc_r
    if market_price < lower - tol or market_price > upper + tol:
        return float("nan")

    sigma = initial_guess
    for _ in range(max_iter):
        price = bs_price(p, sigma, is_call)
        greeks = bs_greeks(p, sigma, is_call)
        vega_per_unit = greeks["vega"] * 100  # back to per-unit
        if vega_per_unit < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma = sigma - diff / vega_per_unit
        if sigma <= 0:
            sigma = 0.01
    return float(sigma)


def iv_smile(
    spot: float,
    strikes: np.ndarray,
    market_prices: np.ndarray,
    time_to_expiry: float,
    risk_free: float = 0.05,
    is_call: bool = True,
) -> np.ndarray:
    """Bulk IV computation for an entire smile."""
    iv = np.zeros(len(strikes))
    for i, (k, px) in enumerate(zip(strikes, market_prices)):
        params = BSParams(
            spot=spot, strike=k, time_to_expiry=time_to_expiry, risk_free=risk_free
        )
        iv[i] = implied_volatility(px, params, is_call=is_call)
    return iv


__all__ = [
    "BSParams",
    "bs_price",
    "bs_greeks",
    "implied_volatility",
    "iv_smile",
]
