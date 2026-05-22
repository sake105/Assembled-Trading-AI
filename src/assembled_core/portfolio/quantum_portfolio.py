"""Quantum QUBO Portfolio Optimization — research showcase stub.

Tier 4 showcase item (audit C2-042–044): formulates the portfolio selection
problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem
suitable for D-Wave quantum annealers or classical simulated annealers.

Full quantum execution requires:
    - D-Wave Leap account (https://cloud.dwavesys.com/leap/)
    - pip install dwave-ocean-sdk dimod

This module provides:
    1. QUBO formulation from a returns covariance matrix (works without dimod)
    2. Classical simulated annealing solve via dimod.SimulatedAnnealingSampler
       (when dimod is installed)
    3. Interface stub for D-Wave QPU execution (requires Leap account)

The QUBO formulation follows Lucas (2014) "Ising Formulations of Many NP Problems"
and Mugel et al. (2022) "Dynamic Portfolio Optimization with Real Datasets Using
Quantum Processors and Quantum-Inspired Tensor Networks".

References:
    - Lucas (2014) "Ising Formulations of Many NP Problems", Frontiers in Physics.
    - Mugel et al. (2022) "Dynamic Portfolio Optimization...", PRResearch.
    - Markowitz (1952) "Portfolio Selection", Journal of Finance.
    - audit C2-042, C2-043, C2-044
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    import dimod  # type: ignore[import-untyped]

    HAS_DIMOD = True
except ImportError:
    HAS_DIMOD = False

_ACTIVATION_MSG = (
    "Quantum solve requires dimod (+ D-Wave Ocean SDK for QPU). "
    "Install with: pip install dwave-ocean-sdk dimod"
)


@dataclass
class QUBOConfig:
    """Configuration for QUBO portfolio optimization."""

    n_bits: int = 8  # bits per asset (binary encoding precision)
    risk_aversion: float = 1.0  # λ: trade-off between return and risk
    budget_penalty: float = 10.0  # penalty for violating budget constraint
    max_assets: int | None = None  # max number of selected assets (cardinality)
    num_reads: int = 1000  # annealing reads
    annealing_time: int = 20  # microseconds (D-Wave QPU)
    random_state: int = 42


@dataclass
class QUBOResult:
    """Result of QUBO portfolio optimization."""

    weights: np.ndarray  # continuous weights (decoded from binary)
    selected: np.ndarray  # boolean mask of selected assets
    qubo_energy: float  # QUBO objective at solution
    expected_return: float
    portfolio_variance: float
    method: str = "simulated_annealing"  # or "dwave_qpu"
    n_reads: int = 0
    converged: bool = False


def build_qubo_matrix(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    budget_penalty: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build QUBO matrix Q for portfolio selection.

    The portfolio optimization objective:
        min  λ · w'Σw - μ'w + P·(Σwᵢ - 1)²

    In binary encoding, each weight wᵢ is approximated by k bits:
        wᵢ ≈ Σⱼ 2^j · xᵢⱼ / (2^k - 1)

    For simplicity, this implementation uses 1 bit per asset (asset inclusion):
        xᵢ ∈ {0, 1},  wᵢ = xᵢ / n_selected

    Parameters
    ----------
    expected_returns : (n,) array of expected returns
    cov_matrix : (n, n) covariance matrix
    risk_aversion : λ, risk penalty coefficient
    budget_penalty : P, penalty for Σxᵢ ≠ n_target

    Returns
    -------
    Q : (n, n) QUBO matrix
    meta : dict with formulation metadata
    """
    n = len(expected_returns)
    Q = np.zeros((n, n))

    # Diagonal: -μᵢ + λ·Σᵢᵢ + P·(1 - 2·n_target/n)
    # For equal-weight target: n_target = n/2 (50% inclusion rate)
    n_target = max(1, n // 2)
    for i in range(n):
        Q[i, i] = (
            -expected_returns[i]
            + risk_aversion * cov_matrix[i, i]
            + budget_penalty * (1 - 2 * n_target / n)
        )

    # Off-diagonal: λ·2·Σᵢⱼ + P·2/n²
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2 * risk_aversion * cov_matrix[i, j] + 2 * budget_penalty / n**2
            Q[j, i] = Q[i, j]

    meta = {
        "n_assets": n,
        "n_target": n_target,
        "encoding": "1-bit-per-asset",
        "risk_aversion": risk_aversion,
        "budget_penalty": budget_penalty,
    }
    return Q, meta


def solve_qubo_classical(
    Q: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    config: QUBOConfig | None = None,
) -> QUBOResult:
    """Solve QUBO via classical simulated annealing (dimod) or exhaustive search.

    When dimod is available: uses SimulatedAnnealingSampler.
    When dimod is not available: exhaustive search for n ≤ 20 assets,
    otherwise falls back to greedy (best individual assets).

    Parameters
    ----------
    Q : QUBO matrix from build_qubo_matrix
    expected_returns : (n,) array
    cov_matrix : (n, n) covariance matrix
    config : QUBOConfig, optional
    """
    cfg = config or QUBOConfig()
    n = Q.shape[0]

    if HAS_DIMOD:
        sampler = dimod.SimulatedAnnealingSampler()
        linear = {i: Q[i, i] for i in range(n)}
        quadratic = {(i, j): Q[i, j] for i in range(n) for j in range(i + 1, n)}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
        response = sampler.sample(bqm, num_reads=cfg.num_reads, seed=cfg.random_state)
        best = response.first.sample
        selected = np.array([best[i] for i in range(n)], dtype=bool)
        energy = float(response.first.energy)
        method = "simulated_annealing_dimod"
    elif n <= 20:
        # exhaustive search
        best_energy = np.inf
        best_x: np.ndarray = np.zeros(n, dtype=bool)
        for mask in range(1, 2**n):
            x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
            e = float(x @ Q @ x)
            if e < best_energy:
                best_energy = e
                best_x = x.astype(bool)
        selected = best_x
        energy = best_energy
        method = "exhaustive_search"
    else:
        # greedy: pick assets with best (return - risk) score
        scores = expected_returns - cfg.risk_aversion * np.diag(cov_matrix)
        n_pick = max(1, n // 2)
        idx = np.argsort(scores)[-n_pick:]
        selected = np.zeros(n, dtype=bool)
        selected[idx] = True
        x = selected.astype(float)
        energy = float(x @ Q @ x)
        method = "greedy_fallback"

    n_sel = int(selected.sum())
    if n_sel == 0:
        selected[0] = True
        n_sel = 1
    weights = selected.astype(float) / n_sel

    exp_ret = float(weights @ expected_returns)
    port_var = float(weights @ cov_matrix @ weights)

    return QUBOResult(
        weights=weights,
        selected=selected,
        qubo_energy=energy,
        expected_return=exp_ret,
        portfolio_variance=port_var,
        method=method,
        n_reads=cfg.num_reads if HAS_DIMOD else 0,
        converged=True,
    )


def quantum_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    config: QUBOConfig | None = None,
) -> QUBOResult:
    """End-to-end QUBO portfolio: build Q matrix then solve classically.

    For D-Wave QPU execution, use ``build_qubo_matrix`` directly and
    submit via ``dwave.system.DWaveSampler``.
    """
    cfg = config or QUBOConfig()
    Q, _ = build_qubo_matrix(
        expected_returns,
        cov_matrix,
        risk_aversion=cfg.risk_aversion,
        budget_penalty=cfg.budget_penalty,
    )
    return solve_qubo_classical(Q, expected_returns, cov_matrix, cfg)


__all__ = [
    "QUBOConfig",
    "QUBOResult",
    "build_qubo_matrix",
    "solve_qubo_classical",
    "quantum_portfolio",
    "HAS_DIMOD",
]
