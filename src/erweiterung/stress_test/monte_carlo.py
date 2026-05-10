"""Monte-Carlo Path-Simulation für Portfolio-Stress-Tests.

Methoden
--------
1. **Bootstrap-Resampling** (Efron): zufällig mit Zurücklegen aus historischen
   Returns ziehen. Erhält Rendite-Verteilung & Tail.
2. **Stationary Bootstrap** (Politis/Romano): erhält Autokorrelation.
3. **Block Bootstrap**: feste Block-Längen.
4. **Parametrische Simulation**: Multivariate-Normal mit historischer μ, Σ.
5. **GARCH-Bootstrap**: GARCH-Modell auf Residuen, dann Resample der Innovations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MCConfig:
    n_paths: int = 1000
    horizon: int = 252  # 1 trading year
    seed: int = 42
    method: str = "bootstrap"  # 'bootstrap' | 'block' | 'stationary' | 'normal'
    block_length: int = 5  # for 'block' & 'stationary'


def _bootstrap(
    returns: np.ndarray, config: MCConfig, rng: np.random.Generator
) -> np.ndarray:
    paths = np.zeros((config.n_paths, config.horizon))
    n = len(returns)
    for i in range(config.n_paths):
        idx = rng.integers(0, n, size=config.horizon)
        paths[i] = returns[idx]
    return paths


def _block_bootstrap(
    returns: np.ndarray, config: MCConfig, rng: np.random.Generator
) -> np.ndarray:
    paths = np.zeros((config.n_paths, config.horizon))
    n = len(returns)
    L = config.block_length
    for i in range(config.n_paths):
        ts = []
        while len(ts) < config.horizon:
            start = rng.integers(0, max(1, n - L + 1))
            ts.extend(returns[start : start + L].tolist())
        paths[i] = ts[: config.horizon]
    return paths


def _stationary_bootstrap(
    returns: np.ndarray, config: MCConfig, rng: np.random.Generator
) -> np.ndarray:
    paths = np.zeros((config.n_paths, config.horizon))
    n = len(returns)
    p = 1.0 / config.block_length
    for i in range(config.n_paths):
        idx = np.empty(config.horizon, dtype=int)
        idx[0] = rng.integers(0, n)
        for t in range(1, config.horizon):
            if rng.random() < p:
                idx[t] = rng.integers(0, n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        paths[i] = returns[idx]
    return paths


def _normal_simulate(
    returns: np.ndarray, config: MCConfig, rng: np.random.Generator
) -> np.ndarray:
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=0))
    return rng.normal(mu, sigma, (config.n_paths, config.horizon))


def simulate_paths(returns: pd.Series, config: MCConfig | None = None) -> np.ndarray:
    """Simulate Monte-Carlo paths.

    Returns:
        Array (n_paths, horizon) of simulated returns.
    """
    cfg = config or MCConfig()
    r = pd.Series(returns).dropna().values
    if len(r) < 30:
        raise ValueError("need >= 30 historical returns")
    rng = np.random.default_rng(cfg.seed)
    if cfg.method == "bootstrap":
        return _bootstrap(r, cfg, rng)
    if cfg.method == "block":
        return _block_bootstrap(r, cfg, rng)
    if cfg.method == "stationary":
        return _stationary_bootstrap(r, cfg, rng)
    if cfg.method == "normal":
        return _normal_simulate(r, cfg, rng)
    raise ValueError(f"unknown method: {cfg.method}")


def path_metrics(paths: np.ndarray, var_alpha: float = 0.95) -> dict:
    """Aggregate metrics über alle simulierten Pfade.

    Returns:
        Dict mit terminal-wealth-Verteilung, max-DD-Verteilung, prob-of-loss.
    """
    n_paths, horizon = paths.shape
    eq_paths = (1 + paths).cumprod(axis=1)
    terminal = eq_paths[:, -1] - 1  # total return

    # max-DD per path
    cummax = np.maximum.accumulate(eq_paths, axis=1)
    dd = eq_paths / cummax - 1
    max_dd = dd.min(axis=1)

    return {
        "terminal_mean": float(terminal.mean()),
        "terminal_median": float(np.median(terminal)),
        "terminal_std": float(terminal.std()),
        "terminal_var_loss": float(np.quantile(terminal, 1 - var_alpha)),
        "terminal_cvar_loss": float(
            terminal[terminal <= np.quantile(terminal, 1 - var_alpha)].mean()
        ),
        "max_dd_mean": float(max_dd.mean()),
        "max_dd_median": float(np.median(max_dd)),
        "max_dd_q05": float(np.quantile(max_dd, 0.05)),
        "max_dd_q25": float(np.quantile(max_dd, 0.25)),
        "prob_of_loss": float((terminal < 0).mean()),
        "n_paths": n_paths,
        "horizon": horizon,
    }


__all__ = ["MCConfig", "simulate_paths", "path_metrics"]
