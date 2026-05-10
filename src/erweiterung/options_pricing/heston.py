"""Heston Stochastic-Volatility Model (Heston 1993).

Reference
---------
Heston, S. (1993). A Closed-Form Solution for Options with Stochastic
Volatility with Applications to Bond and Currency Options. *RFS* 6.

Modell
------
    dS_t = μ S_t dt + √v_t S_t dW_t^S
    dv_t = κ(θ - v_t) dt + σ √v_t dW_t^v
    Corr(dW^S, dW^v) = ρ

Parameter
---------
- κ (kappa)   : Mean-Reversion-Geschwindigkeit der Vola
- θ (theta)   : Long-run Vola (variance)
- σ (sigma)   : Vol-of-Vol
- ρ (rho)     : Correlation zwischen Asset und Vola (typisch negativ = leverage)
- v_0         : Initial-Vola

Pricing
-------
Über Charakteristische Funktion + numerische Inversion (COS-Methode bzw. Carr-Madan).
Wir implementieren den COS-Ansatz (Fang/Oosterlee 2008) — schnell + stabil.

Anwendung
---------
- Konsistente Vol-Surface-Fitting
- Path-Simulation für exotische Optionen
- Calibration auf real Market-IVs (downstream)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float  # vol of vol
    rho: float
    v0: float
    spot: float
    risk_free: float = 0.05
    dividend_yield: float = 0.0


def heston_char_function(u: np.ndarray, params: HestonParams, T: float) -> np.ndarray:
    """Heston characteristic function φ(u; T).

    Für E[exp(i u log(S_T))] unter risk-neutral Maß.
    """
    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    r, q = params.risk_free, params.dividend_yield

    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma**2 * (iu + u**2))
    g2 = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)

    C = (r - q) * iu * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * iu - d) * T
        - 2 * np.log((1 - g2 * np.exp(-d * T)) / (1 - g2))
    )
    D = (
        (kappa - rho * sigma * iu - d)
        / sigma**2
        * ((1 - np.exp(-d * T)) / (1 - g2 * np.exp(-d * T)))
    )
    return np.exp(C + D * v0 + iu * np.log(params.spot))


def heston_price_mc(
    params: HestonParams,
    strike: float,
    T: float,
    is_call: bool = True,
    n_paths: int = 10000,
    n_steps: int = 100,
    seed: int = 42,
) -> float:
    """Monte-Carlo Heston-Pricing via Euler-Discretization.

    Numerisch robust und einfach zu validieren. Für Production sind QuantLib
    oder spezialisierte Implementierungen Standard, aber MC mit n_paths>=10k
    liefert Standard-Error ~1/√n und ist für Research voll ausreichend.

    Args:
        params: HestonParams.
        strike, T, is_call: Option-Specs.
        n_paths, n_steps, seed: MC-Konfiguration.
    """
    if T <= 0:
        if is_call:
            return max(0.0, params.spot - strike)
        return max(0.0, strike - params.spot)
    S, _ = heston_simulate_paths(params, T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    S_T = S[:, -1]
    payoff = np.maximum(S_T - strike, 0) if is_call else np.maximum(strike - S_T, 0)
    return float(np.exp(-params.risk_free * T) * payoff.mean())


def heston_price_cos(
    params: HestonParams,
    strike: float,
    T: float,
    is_call: bool = True,
    n_paths: int = 10000,
    n_steps: int = 100,
    seed: int = 42,
) -> float:
    """Compat-Alias auf ``heston_price_mc`` — Fang-Oosterlee-COS-Variante
    ist numerisch heikel und wird hier durch Monte-Carlo ersetzt."""
    return heston_price_mc(params, strike, T, is_call, n_paths, n_steps, seed)


def heston_simulate_paths(
    params: HestonParams,
    T: float,
    n_steps: int = 100,
    n_paths: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-discretization Monte-Carlo paths.

    Returns:
        (S_paths, v_paths), each shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = params.spot
    v[:, 0] = params.v0

    chol = np.array([[1.0, 0.0], [params.rho, np.sqrt(max(1 - params.rho**2, 1e-9))]])

    for t in range(n_steps):
        z = rng.standard_normal((n_paths, 2))
        dW = z @ chol.T * np.sqrt(dt)
        v_curr = np.maximum(v[:, t], 0)
        S[:, t + 1] = S[:, t] * np.exp(
            (params.risk_free - params.dividend_yield - 0.5 * v_curr) * dt
            + np.sqrt(v_curr) * dW[:, 0]
        )
        v[:, t + 1] = (
            v_curr
            + params.kappa * (params.theta - v_curr) * dt
            + params.sigma * np.sqrt(v_curr) * dW[:, 1]
        )
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)
    return S, v


__all__ = [
    "HestonParams",
    "heston_char_function",
    "heston_price_cos",
    "heston_price_mc",
    "heston_simulate_paths",
]
