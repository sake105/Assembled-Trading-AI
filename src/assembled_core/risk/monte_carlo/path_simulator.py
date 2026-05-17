"""Monte-Carlo path simulation for equity-curve distributions.

Two methods:

1. **Parametric i.i.d. normal** (``simulate_paths_iid_normal``): Calibrates
   arithmetic mean μ and std σ from history; draws i.i.d. normal arithmetic
   returns. NOT true Geometric Brownian Motion — GBM uses log-returns with
   ``-0.5 σ²`` drift correction. The previous name ``simulate_paths_gbm`` was
   misleading; renamed per F-risk-4. If you need true GBM (log-normal terminal
   distribution), use a dedicated implementation.
2. **Block bootstrap**: Resample contiguous blocks of historical returns to
   preserve serial correlation (momentum, mean-reversion, vol clustering).

Both return a :class:`PathSimResult` containing a ``(n_paths, n_periods)``
array of simulated equity paths starting at 1.0.

Complements ``qa/scenario_engine.py`` (historical stress-replays) with
forward-looking distributional analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class PathSimResult:
    """Result of a Monte-Carlo path simulation.

    Attributes:
        paths: ``(n_paths, n_periods)`` equity curves, each starting at 1.0
            before the first return.
        method: ``"iid_normal"`` or ``"block_bootstrap"``.
        seed: The RNG seed used (``None`` if unseeded).
    """

    paths: np.ndarray  # (n_paths, n_periods)
    method: Literal["iid_normal", "block_bootstrap"]
    seed: int | None

    def final_value_quantiles(
        self, qs: list[float] | None = None
    ) -> dict[float, float]:
        """Return quantiles of final equity values across all paths.

        Args:
            qs: Quantile levels (default ``[0.05, 0.25, 0.50, 0.75, 0.95]``).

        Returns:
            Dict mapping quantile → final equity value.
        """
        if qs is None:
            qs = [0.05, 0.25, 0.50, 0.75, 0.95]
        final_vals = self.paths[:, -1]
        return {q: float(np.quantile(final_vals, q)) for q in qs}

    def max_drawdown_quantiles(
        self, qs: list[float] | None = None
    ) -> dict[float, float]:
        """Return quantiles of maximum drawdown per path (values <= 0).

        Args:
            qs: Quantile levels (default ``[0.05, 0.25, 0.50, 0.75, 0.95]``).

        Returns:
            Dict mapping quantile → MDD value (all <= 0).
        """
        if qs is None:
            qs = [0.05, 0.25, 0.50, 0.75, 0.95]
        # Prepend starting equity of 1.0 to compute drawdown vs initial level
        ones_col = np.ones((self.paths.shape[0], 1))
        equity_full = np.hstack([ones_col, self.paths])
        running_max = np.maximum.accumulate(equity_full, axis=1)
        drawdowns = equity_full / running_max - 1.0
        mdd_per_path = drawdowns.min(axis=1)
        return {q: float(np.quantile(mdd_per_path, q)) for q in qs}


def _returns_to_paths(returns_matrix: np.ndarray) -> np.ndarray:
    """Convert a ``(n_paths, n_periods)`` returns matrix to equity paths starting at 1.0.

    Each row is the equity level after applying each period's return in sequence.
    The initial level (1.0) is NOT stored; the first column is equity after
    the first period.  This matches typical equity-curve convention.
    """
    return np.cumprod(1.0 + returns_matrix, axis=1)


def simulate_paths_iid_normal(
    historical_returns: pd.Series | np.ndarray,
    n_paths: int = 1000,
    n_periods: int = 252,
    seed: int | None = None,
) -> PathSimResult:
    """I.i.d. normal-return path simulation calibrated to historical returns.

    Calibrates ``mu`` (arithmetic mean) and ``sigma`` (std) from
    ``historical_returns``, then draws i.i.d. normal arithmetic-return shocks
    to build ``n_paths`` equity paths of length ``n_periods``.

    NOTE: This is NOT true Geometric Brownian Motion (despite the previous
    name ``simulate_paths_gbm``). GBM operates on log-returns with a
    ``-0.5 σ²`` Itô drift correction. The arithmetic-return form here is
    mathematically valid as a stand-alone model but should not be confused
    with GBM. Renamed per F-risk-4.

    Args:
        historical_returns: Series or array of historical period returns
            (e.g. daily, weekly — consistent with ``n_periods`` horizon).
        n_paths: Number of simulated paths.
        n_periods: Forecast horizon in periods.
        seed: RNG seed for reproducibility.

    Returns:
        :class:`PathSimResult` with ``(n_paths, n_periods)`` paths starting at 1.0.
    """
    arr = np.asarray(historical_returns, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))

    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=(n_paths, n_periods))
    paths = _returns_to_paths(shocks)

    return PathSimResult(paths=paths, method="iid_normal", seed=seed)


def simulate_paths_block_bootstrap(
    historical_returns: pd.Series | np.ndarray,
    n_paths: int = 1000,
    n_periods: int = 252,
    block_size: int = 5,
    seed: int | None = None,
) -> PathSimResult:
    """Block bootstrap path simulation (preserves serial correlation).

    Samples contiguous blocks of size ``block_size`` from ``historical_returns``
    with replacement, concatenates them, and trims to exactly ``n_periods``.

    ``block_size=1`` degenerates to standard i.i.d. bootstrap.

    Args:
        historical_returns: Series or array of historical period returns.
        n_paths: Number of simulated paths.
        n_periods: Forecast horizon in periods.
        block_size: Block size in periods (default 5 — weekly blocks for
            daily data).  Must be >= 1.
        seed: RNG seed.

    Returns:
        :class:`PathSimResult` with ``(n_paths, n_periods)`` simulated paths.

    Raises:
        ValueError: If ``block_size < 1`` or ``historical_returns`` is shorter
            than ``block_size``.
    """
    arr = np.asarray(historical_returns, dtype=float)
    n_hist = len(arr)

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}.")
    if n_hist < block_size:
        raise ValueError(
            f"historical_returns length ({n_hist}) must be >= block_size ({block_size})."
        )

    # Number of blocks needed to cover n_periods (may overshoot; trim later)
    n_blocks = int(np.ceil(n_periods / block_size))
    # Valid starting indices for a block of size block_size
    max_start = n_hist - block_size  # inclusive
    n_valid_starts = max_start + 1

    rng = np.random.default_rng(seed)

    # Sample all block starts at once: (n_paths, n_blocks)
    block_starts = rng.integers(0, n_valid_starts, size=(n_paths, n_blocks))

    # Build return matrix: (n_paths, n_blocks * block_size)
    # Vectorised: for each block, extract slice using advanced indexing
    # offsets within a block: shape (block_size,)
    offsets = np.arange(block_size)  # (block_size,)

    # block_starts: (n_paths, n_blocks)  ->  add offsets  -> (n_paths, n_blocks, block_size)
    indices = (
        block_starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    )  # broadcast
    # indices: (n_paths, n_blocks, block_size) — all within [0, n_hist)
    returns_raw = arr[indices]  # (n_paths, n_blocks, block_size)

    # Reshape to (n_paths, n_blocks * block_size) then trim to n_periods
    returns_flat = returns_raw.reshape(n_paths, n_blocks * block_size)
    returns_trimmed = returns_flat[:, :n_periods]

    paths = _returns_to_paths(returns_trimmed)

    return PathSimResult(paths=paths, method="block_bootstrap", seed=seed)
