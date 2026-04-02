"""Monte Carlo / Bootstrap simulation for backtest confidence analysis.

Provides statistical confidence intervals for strategy performance metrics
via return resampling and geometric Brownian motion forward simulation.

Example:
    from src.assembled_core.qa.monte_carlo import (
        bootstrap_returns,
        forward_simulate_gbm,
        compute_confidence_intervals,
    )

    # Bootstrap existing backtest returns
    ci = bootstrap_returns(daily_returns, n_paths=1000, seed=42)
    print(f"Sharpe 95% CI: [{ci['sharpe']['ci_lower']:.2f}, {ci['sharpe']['ci_upper']:.2f}]")

    # Forward simulate equity paths
    paths = forward_simulate_gbm(daily_returns, n_paths=500, horizon_days=252, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""

    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_simulations: int


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    confidence_intervals: dict[str, ConfidenceInterval]
    sharpe_distribution: np.ndarray
    cagr_distribution: np.ndarray
    max_dd_distribution: np.ndarray
    p_value_vs_zero: float
    n_paths: int
    seed: int | None


def _compute_sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-12:
        return 0.0
    return float(mean_r / std_r * np.sqrt(trading_days))


def _compute_cagr(returns: np.ndarray, trading_days: int = 252) -> float:
    """Compute CAGR from daily returns."""
    if len(returns) < 2:
        return 0.0
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / trading_days
    if n_years < 1e-6:
        return 0.0
    if total_return <= -1:
        return -1.0
    return float((1 + total_return) ** (1 / n_years) - 1)


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from daily returns."""
    if len(returns) < 1:
        return 0.0
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1
    return float(np.min(drawdowns))


def bootstrap_returns(
    daily_returns: np.ndarray | pd.Series,
    n_paths: int = 1000,
    confidence_level: float = 0.95,
    block_size: int | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """Bootstrap resample daily returns to compute confidence intervals.

    Uses block bootstrap (if block_size > 1) to preserve autocorrelation,
    or standard iid bootstrap (block_size=1 or None).

    Args:
        daily_returns: Array of daily returns (not cumulative)
        n_paths: Number of bootstrap samples (default: 1000)
        confidence_level: Confidence level (default: 0.95, i.e. 95%)
        block_size: Block size for block bootstrap (default: None = iid)
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with confidence intervals for Sharpe, CAGR, MaxDD
    """
    if isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.dropna().values

    returns = np.asarray(daily_returns, dtype=np.float64)
    n = len(returns)

    if n < 10:
        raise ValueError(f"Need at least 10 returns for bootstrap, got {n}")

    rng = np.random.default_rng(seed)

    sharpe_dist = np.empty(n_paths)
    cagr_dist = np.empty(n_paths)
    max_dd_dist = np.empty(n_paths)

    if block_size is None or block_size <= 1:
        # Standard iid bootstrap
        for i in range(n_paths):
            idx = rng.integers(0, n, size=n)
            sample = returns[idx]
            sharpe_dist[i] = _compute_sharpe(sample)
            cagr_dist[i] = _compute_cagr(sample)
            max_dd_dist[i] = _compute_max_drawdown(sample)
    else:
        # Block bootstrap
        n_blocks = max(1, n // block_size)
        for i in range(n_paths):
            blocks = [
                returns[start : start + block_size]
                for start in rng.integers(0, n - block_size + 1, size=n_blocks)
            ]
            sample = np.concatenate(blocks)[:n]
            sharpe_dist[i] = _compute_sharpe(sample)
            cagr_dist[i] = _compute_cagr(sample)
            max_dd_dist[i] = _compute_max_drawdown(sample)

    alpha = 1 - confidence_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Point estimates
    sharpe_point = _compute_sharpe(returns)
    cagr_point = _compute_cagr(returns)
    max_dd_point = _compute_max_drawdown(returns)

    # P-value: fraction of bootstrap Sharpes <= 0
    p_value = float(np.mean(sharpe_dist <= 0))

    ci = {
        "sharpe": ConfidenceInterval(
            metric="sharpe",
            point_estimate=sharpe_point,
            ci_lower=float(np.quantile(sharpe_dist, lower_q)),
            ci_upper=float(np.quantile(sharpe_dist, upper_q)),
            confidence_level=confidence_level,
            n_simulations=n_paths,
        ),
        "cagr": ConfidenceInterval(
            metric="cagr",
            point_estimate=cagr_point,
            ci_lower=float(np.quantile(cagr_dist, lower_q)),
            ci_upper=float(np.quantile(cagr_dist, upper_q)),
            confidence_level=confidence_level,
            n_simulations=n_paths,
        ),
        "max_drawdown": ConfidenceInterval(
            metric="max_drawdown",
            point_estimate=max_dd_point,
            ci_lower=float(np.quantile(max_dd_dist, lower_q)),
            ci_upper=float(np.quantile(max_dd_dist, upper_q)),
            confidence_level=confidence_level,
            n_simulations=n_paths,
        ),
    }

    return MonteCarloResult(
        confidence_intervals=ci,
        sharpe_distribution=sharpe_dist,
        cagr_distribution=cagr_dist,
        max_dd_distribution=max_dd_dist,
        p_value_vs_zero=p_value,
        n_paths=n_paths,
        seed=seed,
    )


@dataclass
class ForwardSimulationResult:
    """Result of forward Monte Carlo simulation."""

    paths: np.ndarray  # shape: (n_paths, horizon_days)
    terminal_values: np.ndarray  # shape: (n_paths,)
    prob_loss: float
    prob_dd_exceed: float  # P(MaxDD > threshold)
    dd_threshold: float
    median_terminal: float
    ci_lower_terminal: float
    ci_upper_terminal: float


def forward_simulate_gbm(
    daily_returns: np.ndarray | pd.Series,
    initial_value: float = 100000.0,
    n_paths: int = 500,
    horizon_days: int = 252,
    dd_threshold: float = -0.20,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> ForwardSimulationResult:
    """Forward simulate equity paths using geometric Brownian motion.

    Fits mu and sigma from historical returns, then generates random paths.

    Args:
        daily_returns: Historical daily returns to fit parameters
        initial_value: Starting portfolio value (default: 100,000)
        n_paths: Number of simulation paths (default: 500)
        horizon_days: Simulation horizon in trading days (default: 252 = 1 year)
        dd_threshold: Drawdown threshold for P(ruin) (default: -0.20 = 20%)
        confidence_level: CI level for terminal value (default: 0.95)
        seed: Random seed

    Returns:
        ForwardSimulationResult with paths, terminal stats, and risk metrics
    """
    if isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.dropna().values

    returns = np.asarray(daily_returns, dtype=np.float64)

    if len(returns) < 10:
        raise ValueError(f"Need at least 10 returns, got {len(returns)}")

    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))

    rng = np.random.default_rng(seed)

    # Generate random returns: N(mu, sigma^2)
    random_returns = rng.normal(mu, sigma, size=(n_paths, horizon_days))

    # Build equity paths
    paths = initial_value * np.cumprod(1 + random_returns, axis=1)

    terminal_values = paths[:, -1]

    # Compute drawdowns per path
    dd_exceed_count = 0
    for i in range(n_paths):
        running_max = np.maximum.accumulate(paths[i])
        max_dd = np.min(paths[i] / running_max - 1)
        if max_dd < dd_threshold:
            dd_exceed_count += 1

    alpha = 1 - confidence_level

    return ForwardSimulationResult(
        paths=paths,
        terminal_values=terminal_values,
        prob_loss=float(np.mean(terminal_values < initial_value)),
        prob_dd_exceed=float(dd_exceed_count / n_paths),
        dd_threshold=dd_threshold,
        median_terminal=float(np.median(terminal_values)),
        ci_lower_terminal=float(np.quantile(terminal_values, alpha / 2)),
        ci_upper_terminal=float(np.quantile(terminal_values, 1 - alpha / 2)),
    )


def summarize_monte_carlo(result: MonteCarloResult) -> str:
    """Format Monte Carlo results as readable summary string."""
    lines = ["Monte Carlo Bootstrap Analysis", "=" * 40]

    for name, ci in result.confidence_intervals.items():
        pct = ci.confidence_level * 100
        lines.append(
            f"{name}: {ci.point_estimate:.4f} "
            f"[{pct:.0f}% CI: {ci.ci_lower:.4f} to {ci.ci_upper:.4f}]"
        )

    lines.append(f"P(Sharpe <= 0): {result.p_value_vs_zero:.4f}")
    lines.append(f"Simulations: {result.n_paths}")

    return "\n".join(lines)


def summarize_forward_sim(result: ForwardSimulationResult) -> str:
    """Format forward simulation results as readable summary string."""
    lines = ["Forward Monte Carlo Simulation", "=" * 40]
    lines.append(f"Median terminal value: ${result.median_terminal:,.0f}")
    lines.append(
        f"95% CI: [${result.ci_lower_terminal:,.0f}, ${result.ci_upper_terminal:,.0f}]"
    )
    lines.append(f"P(loss): {result.prob_loss:.2%}")
    lines.append(
        f"P(MaxDD > {result.dd_threshold:.0%}): {result.prob_dd_exceed:.2%}"
    )
    lines.append(f"Paths: {len(result.terminal_values)}")

    return "\n".join(lines)
