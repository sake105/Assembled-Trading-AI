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
from typing import cast

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
    # np.where evaluates both branches; suppress the divide-by-zero warning
    # that fires when any row has std==0 (e.g. permutation of identical PnL).
    with np.errstate(divide="ignore", invalid="ignore"):
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
    return cast(np.ndarray, mdd)


def _compute_total_return(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute total return (prod(1+r) - 1) for each row."""
    return cast(np.ndarray, np.prod(1.0 + returns_matrix, axis=1) - 1.0)


def pnl_to_returns(
    pnl: pd.Series | np.ndarray,
    initial_capital: float = 100_000.0,
) -> np.ndarray:
    """Convert per-trade currency PnL to return units (Phase-2 migration helper).

    Approximation used: ``r_i = pnl_i / initial_capital`` — uniform divisor,
    no compounding context per trade. This matches the legacy
    ``qa.monte_carlo_paths.monte_carlo_trade_paths`` equity model
    (``initial + cumsum(pnl)``) within small-return regime where
    ``cumprod(1 + pnl/K) ≈ 1 + cumsum(pnl)/K``.

    Use this at the call site to migrate from legacy currency-PnL inputs
    to the canonical :func:`permute_trades` / :func:`shuffle_trades`
    return-unit API. The ``r <= -1.0`` guard in the canonical functions
    will reject inputs where ``initial_capital`` was set too low — that
    rejection IS the migration safety net.

    Args:
        pnl: Per-trade currency PnL (NOT cumulative).
        initial_capital: Starting capital used to normalise PnL into
            return units. Default 100_000 matches legacy default.

    Returns:
        np.ndarray of per-trade returns suitable for ``permute_trades``
        or ``shuffle_trades``.

    Raises:
        ValueError: ``initial_capital`` must be positive and finite.
    """
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError(
            f"initial_capital must be positive and finite, got {initial_capital}"
        )
    arr = np.asarray(pnl, dtype=float)
    return arr / float(initial_capital)


def shuffle_result_to_quantile_dict(
    result: ShuffleResult,
    n_trades: int,
    initial_capital: float = 100_000.0,
    annual_trading_days: int = 252,
) -> dict:
    """Convert ShuffleResult to the legacy dict schema used by
    ``qa.monte_carlo_paths.monte_carlo_trade_paths`` consumers.

    Phase-2 migration adapter: lets callers swap the legacy function for
    ``permute_trades`` / ``shuffle_trades`` WITHOUT breaking downstream
    JSON-readers (``metrics.json``, API responses, daily reports).

    Schema produced (matches legacy + Phase-2 additions):
        {
            "n_paths": int,
            "n_trades": int,
            "sharpe": {"mean", "std", "p10", "p50", "p90"},
            "mdd":    {"mean", "p10", "p50", "p90", "p99"},
            "cagr":   {"mean", "p10", "p50", "p90"},
            "final_equity": {"mean", "p10", "p50", "p90"},
            "pct_ruined": float,   # NEW: fraction of paths with final equity <= 0
        }

    Args:
        result: ShuffleResult from ``permute_trades`` / ``shuffle_trades``.
        n_trades: Number of trades in the input series. **Mandatory** —
            the ShuffleResult does NOT carry n_trades, so the caller must
            pass it (typically ``len(pnl_series)``). Drives CAGR's
            ``years = n_trades / annual_trading_days`` annualisation.
            F-RISK-MC2-BLOCKER-1 fix: was previously inferred wrongly from
            ``sharpe.shape[0]`` which equals n_iterations, not n_trades.
        initial_capital: For ``final_equity = initial * (1 + total_return)``.
        annual_trading_days: For CAGR annualisation of ``total_return``.

    Returns:
        Dict with the legacy schema (all values are Python floats).

    Raises:
        ValueError: ``n_trades`` must be a positive integer.
    """
    if not isinstance(n_trades, int) or n_trades <= 0:
        raise ValueError(f"n_trades must be a positive int, got {n_trades!r}")
    sharpe = result.sharpe_distribution
    mdd = result.max_drawdown_distribution
    total_ret = result.total_return_distribution
    n_paths = int(result.n_iterations)
    final_equity = initial_capital * (1.0 + total_ret)
    years = n_trades / annual_trading_days
    # F-RISK-MC2-MAJOR-3: count ruined paths BEFORE clipping for CAGR.
    # final_equity <= 0 means the path lost everything (or more). The clip
    # below then makes (1+total_ret) safe for the **(1/years) operation but
    # the ruin information is preserved in pct_ruined.
    #
    # F-SCR-MC2-MINOR-3 note: since permute_trades/shuffle_trades reject any
    # r <= -1.0 input, (1+r) > 0 for every individual trade, so cumprod
    # remains positive. pct_ruined > 0 is therefore FP-underflow-only in
    # realistic inputs (e.g. 50 trades of r=-0.99 may underflow). The field
    # exists as honest visibility-of-ruin under extreme edge cases; in
    # normal backtest data it will report 0.0.
    ruined_mask = (1.0 + total_ret) <= 0.0
    pct_ruined = float(ruined_mask.mean())
    # CAGR per path: (1 + total_return) ** (1/years) - 1
    safe_total = np.clip(1.0 + total_ret, 1e-12, None)
    cagr = safe_total ** (1.0 / max(years, 1e-6)) - 1.0

    def _pct(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p))

    return {
        "n_paths": n_paths,
        "n_trades": int(n_trades),
        "sharpe": {
            "mean": float(np.mean(sharpe)),
            "std": float(np.std(sharpe)),
            "p10": _pct(sharpe, 10),
            "p50": _pct(sharpe, 50),
            "p90": _pct(sharpe, 90),
        },
        "mdd": {
            "mean": float(np.mean(mdd)),
            "p10": _pct(mdd, 10),
            "p50": _pct(mdd, 50),
            "p90": _pct(mdd, 90),
            "p99": _pct(mdd, 99),
        },
        "cagr": {
            "mean": float(np.mean(cagr)),
            "p10": _pct(cagr, 10),
            "p50": _pct(cagr, 50),
            "p90": _pct(cagr, 90),
        },
        "final_equity": {
            "mean": float(np.mean(final_equity)),
            "p10": _pct(final_equity, 10),
            "p50": _pct(final_equity, 50),
            "p90": _pct(final_equity, 90),
        },
        "pct_ruined": pct_ruined,
    }


def permute_trades(
    trade_pnl: pd.Series | np.ndarray,
    n_iterations: int = 1000,
    seed: int | None = None,
    annualization_factor: float = 252,
) -> ShuffleResult:
    """Permute (not bootstrap) trade order to estimate sequencing CIs.

    Semantically distinct from :func:`shuffle_trades`:
        - ``shuffle_trades`` resamples WITH replacement (bootstrap) — answers
          "how confident am I given the *distribution* of my trades".
        - ``permute_trades`` shuffles WITHOUT replacement (permutation) —
          answers "how confident am I given the *exact set* of my trades
          taken in a different order". Each trade appears exactly once per
          iteration.

    Separates 'edge' (mean trade return > 0) from 'lucky sequencing'
    (specific order of wins/losses produced low MDD).

    This is the canonical replacement for the legacy
    ``qa.monte_carlo_paths.monte_carlo_trade_paths`` which is deprecated
    (returned an untyped dict). This function returns the same
    :class:`ShuffleResult` as ``shuffle_trades`` for API consistency.

    Args:
        trade_pnl: Series or array of per-trade returns (e.g. ``0.01 = 1%``).
            NOT cumulative, NOT currency. If migrating from legacy
            ``qa.monte_carlo_paths.monte_carlo_trade_paths`` (which took
            currency PnL): convert via ``pnl_per_trade / entry_notional``.
            Returns ``<= -1.0`` are rejected — they would produce non-positive
            equity in ``cumprod(1+r)``.
        n_iterations: Number of permutation iterations (default 1000).
        seed: RNG seed for reproducibility.
        annualization_factor: For Sharpe annualisation. Default 252.

    Returns:
        :class:`ShuffleResult` with N-length distributions for each metric.

    Raises:
        ValueError: If ``trade_pnl`` is empty, contains NaN, contains inf, or
            contains values ``<= -1.0``.
    """
    arr = np.asarray(trade_pnl, dtype=float)

    if arr.size == 0:
        raise ValueError("trade_pnl is empty — provide at least one trade outcome.")
    if np.any(np.isnan(arr)):
        raise ValueError("trade_pnl contains NaN values.")
    if np.any(np.isinf(arr)):
        raise ValueError("trade_pnl contains inf values.")
    # F-RISK-MC1-MINOR-1: equity uses cumprod(1+r); a return ≤ -1.0 produces
    # zero or negative equity, after which (1+r)-cumprod is sign-incoherent.
    # Reject up front — Risk-Zone primitives must not silently emit bogus MDDs.
    if np.any(arr <= -1.0):
        raise ValueError(
            "trade_pnl contains values <= -1.0 (return units). Equity would "
            "go non-positive and downstream MDD/Sharpe are ill-defined. "
            "Pass per-trade RETURN units (e.g. 0.01 = 1%), NOT currency PnL."
        )

    rng = np.random.default_rng(seed)
    n_trades = len(arr)

    # Permutation WITHOUT replacement: each row is a re-ordering of arr.
    samples = np.empty((n_iterations, n_trades), dtype=float)
    for i in range(n_iterations):
        samples[i] = rng.permutation(arr)

    sharpes = _compute_sharpe(samples, annualization_factor)
    mdds = _compute_mdd(samples)
    total_returns = _compute_total_return(samples)

    return ShuffleResult(
        n_iterations=n_iterations,
        sharpe_distribution=sharpes,
        max_drawdown_distribution=mdds,
        total_return_distribution=total_returns,
    )


def shuffle_trades(
    trade_pnl: pd.Series | np.ndarray,
    n_iterations: int = 1000,
    seed: int | None = None,
    annualization_factor: float = 252,
    block_size: int = 1,
) -> ShuffleResult:
    """Bootstrap-shuffle trade outcomes to estimate Sharpe / MDD / total-return CIs.

    Args:
        trade_pnl: Series or array of per-trade P&L values (in return units —
            e.g. 0.01 = 1 % per trade). NOT cumulative.
        n_iterations: Number of bootstrap iterations (default 1000).
        seed: RNG seed for reproducibility.
        annualization_factor: For Sharpe annualisation. Default 252 (daily trades).
            Use 12 for monthly, 52 for weekly.
        block_size: Block size for moving-block bootstrap (Künsch 1989, Politis
            & Romano 1994). Default 1 = standard i.i.d. bootstrap.
            ``block_size > 1`` preserves local autocorrelation (volatility
            clustering) by drawing contiguous blocks of length ``block_size``
            from the input series and concatenating ``ceil(n / block_size)``
            blocks to length ``n``. Recommended for **daily return series**
            with serial dependence; **per-trade PnL** typically has weak
            autocorrelation and ``block_size=1`` is fine. F-RISK-MC2-MAJOR-1
            enables daily_qa_report re-migration (§6.5.3 Phase 2c).

    Returns:
        :class:`ShuffleResult` with N-length distributions for each metric.

    Raises:
        ValueError: If ``trade_pnl`` is empty, contains NaN, contains inf,
            contains values ``<= -1.0``, or ``block_size`` is non-positive
            or larger than the input length.
    """
    arr = np.asarray(trade_pnl, dtype=float)

    if arr.size == 0:
        raise ValueError("trade_pnl is empty — provide at least one trade outcome.")
    if np.any(np.isnan(arr)):
        raise ValueError("trade_pnl contains NaN values.")
    if np.any(np.isinf(arr)):
        raise ValueError("trade_pnl contains inf values.")
    # F-RISK-MC1-MINOR-1: same r<=-1.0 guard as permute_trades.
    if np.any(arr <= -1.0):
        raise ValueError(
            "trade_pnl contains values <= -1.0 (return units). Equity would "
            "go non-positive and downstream MDD/Sharpe are ill-defined. "
            "Pass per-trade RETURN units (e.g. 0.01 = 1%), NOT currency PnL."
        )
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")

    rng = np.random.default_rng(seed)
    n_trades = len(arr)

    if block_size > n_trades:
        raise ValueError(
            f"block_size={block_size} > len(trade_pnl)={n_trades}; "
            "no valid block can be sampled."
        )

    if block_size == 1:
        # Standard i.i.d. bootstrap — fast vectorised path.
        indices = rng.integers(0, n_trades, size=(n_iterations, n_trades))
        samples = arr[indices]
    else:
        # Moving-block bootstrap: draw blocks of length block_size starting at
        # random indices in [0, n_trades - block_size + 1), concatenate
        # ceil(n_trades / block_size) blocks, then truncate to n_trades.
        n_blocks = -(-n_trades // block_size)  # ceil(n_trades / block_size)
        max_start = n_trades - block_size + 1
        starts = rng.integers(0, max_start, size=(n_iterations, n_blocks))
        # Build offset matrix so that for each (iter, block_idx, offset)
        # we get a per-element index into arr.
        offsets = np.arange(block_size)
        # shape (n_iterations, n_blocks, block_size)
        indices_3d = starts[:, :, None] + offsets[None, None, :]
        # Flatten last two axes → (n_iterations, n_blocks * block_size)
        indices_flat = indices_3d.reshape(n_iterations, -1)[:, :n_trades]
        samples = arr[indices_flat]

    sharpes = _compute_sharpe(samples, annualization_factor)
    mdds = _compute_mdd(samples)
    total_returns = _compute_total_return(samples)

    return ShuffleResult(
        n_iterations=n_iterations,
        sharpe_distribution=sharpes,
        max_drawdown_distribution=mdds,
        total_return_distribution=total_returns,
    )
