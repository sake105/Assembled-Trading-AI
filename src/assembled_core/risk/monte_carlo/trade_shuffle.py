"""Trade-shuffling bootstrap for backtest robustness.

Given a sequence of trade P&L outcomes, resample with replacement to
estimate the sampling distribution of summary statistics (Sharpe, max
drawdown, total return). Answers: "if I had taken the same trades in a
different order, how confident am I in my Sharpe?"

PIT-safe: trade outcomes are inputs, not predictions. Shuffling is a
post-hoc statistical robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShuffleResult:
    """Result of a trade-shuffle bootstrap run."""

    n_iterations: int
    sharpe_distribution: np.ndarray  # 1D array of N bootstrap Sharpes
    max_drawdown_distribution: np.ndarray  # 1D array of N bootstrap MDDs (negative)
    total_return_distribution: np.ndarray  # 1D array of N bootstrap total returns
    ci_lo: float = 0.05  # default 5th percentile
    ci_hi: float = 0.95  # default 95th percentile

    def confidence_interval(
        self,
        metric: str,
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[float, float]:
        """Return (lower, upper) percentile bounds for a metric.

        Args:
            metric: One of ``"sharpe"``, ``"max_drawdown"``, ``"total_return"``.
            lo: Lower quantile (defaults to ``self.ci_lo``).
            hi: Upper quantile (defaults to ``self.ci_hi``).

        Returns:
            Tuple ``(lower_bound, upper_bound)``.
        """
        lo = lo if lo is not None else self.ci_lo
        hi = hi if hi is not None else self.ci_hi
        arr = getattr(self, f"{metric}_distribution")
        return float(np.quantile(arr, lo)), float(np.quantile(arr, hi))


def _compute_sharpe(
    returns_matrix: np.ndarray, annualization_factor: float
) -> np.ndarray:
    """Compute annualised Sharpe for each row of a 2-D returns matrix.

    Rows where std == 0 (all returns identical) get Sharpe = 0.
    """
    means = returns_matrix.mean(axis=1)
    stds = returns_matrix.std(axis=1, ddof=1)
    sharpes = np.where(stds > 0, means / stds * np.sqrt(annualization_factor), 0.0)
    return sharpes


def _compute_mdd(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute max drawdown (negative float) for each row.

    Uses cumulative product of (1 + r) then running-max / current ratio.
    Prepends 1.0 so drawdown is measured from the starting equity level.
    """
    # equity: (n_iterations, n_trades)
    equity = np.cumprod(1.0 + returns_matrix, axis=1)
    # Prepend 1.0 for the initial equity level so MDD starts from 1.0
    ones_col = np.ones((equity.shape[0], 1))
    equity_full = np.hstack([ones_col, equity])
    running_max_full = np.maximum.accumulate(equity_full, axis=1)
    drawdowns = equity_full / running_max_full - 1.0
    mdd = drawdowns.min(axis=1)
    return mdd


def _compute_total_return(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute total return (prod(1+r) - 1) for each row."""
    return np.prod(1.0 + returns_matrix, axis=1) - 1.0


def shuffle_trades(
    trade_pnl: pd.Series | np.ndarray,
    n_iterations: int = 1000,
    seed: int | None = None,
    annualization_factor: float = 252,
) -> ShuffleResult:
    """Bootstrap-shuffle trade outcomes to estimate Sharpe / MDD / total-return CIs.

    Args:
        trade_pnl: Series or array of per-trade P&L values (in return units —
            e.g. 0.01 = 1 % per trade). NOT cumulative.
        n_iterations: Number of bootstrap iterations (default 1000).
        seed: RNG seed for reproducibility.
        annualization_factor: For Sharpe annualisation. Default 252 (daily trades).
            Use 12 for monthly, 52 for weekly.

    Returns:
        :class:`ShuffleResult` with N-length distributions for each metric.

    Raises:
        ValueError: If ``trade_pnl`` is empty, contains NaN, or contains inf.
    """
    arr = np.asarray(trade_pnl, dtype=float)

    if arr.size == 0:
        raise ValueError("trade_pnl is empty — provide at least one trade outcome.")
    if np.any(np.isnan(arr)):
        raise ValueError("trade_pnl contains NaN values.")
    if np.any(np.isinf(arr)):
        raise ValueError("trade_pnl contains inf values.")

    rng = np.random.default_rng(seed)
    n_trades = len(arr)

    # Sample with replacement: shape (n_iterations, n_trades)
    # Use rng.integers for index sampling — efficient and reproducible.
    indices = rng.integers(0, n_trades, size=(n_iterations, n_trades))
    samples = arr[indices]  # (n_iterations, n_trades)

    sharpes = _compute_sharpe(samples, annualization_factor)
    mdds = _compute_mdd(samples)
    total_returns = _compute_total_return(samples)

    return ShuffleResult(
        n_iterations=n_iterations,
        sharpe_distribution=sharpes,
        max_drawdown_distribution=mdds,
        total_return_distribution=total_returns,
    )
