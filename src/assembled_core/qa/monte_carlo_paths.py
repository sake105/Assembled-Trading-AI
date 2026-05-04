"""Monte-Carlo trade-order simulation for backtest robustness (Plan 11/10 §2.2).

Permutes the order of historical trades to estimate the distribution of
equity-curve metrics under random trade sequencing. Separates 'edge'
(mean trade return > 0) from 'lucky sequencing' (specific order of
wins/losses produced low MDD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_trade_paths(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    n_paths: int = 5000,
    seed: int = 42,
    pnl_col: str = "pnl",
    annual_trading_days: int = 252,
) -> dict:
    """Permute trade order N times and compute equity-curve metric distributions.

    Args:
        trades: DataFrame with at least a PnL column.
        initial_capital: Starting equity for each simulated path.
        n_paths: Number of Monte-Carlo paths.
        seed: RNG seed for reproducibility.
        pnl_col: Column name containing per-trade PnL in currency units.
        annual_trading_days: Used to annualise per-trade Sharpe approximation.

    Returns:
        Dict with sharpe / mdd / cagr / final_equity distributions.
    """
    if pnl_col not in trades.columns:
        # Fall back to common alternatives
        for alt in ("net_pnl", "gross_pnl", "trade_pnl", "closed_return"):
            if alt in trades.columns:
                pnl_col = alt
                break
        else:
            return {"error": f"no pnl column found in trades (tried {pnl_col})"}

    pnl_array = trades[pnl_col].dropna().values.astype(float)
    n_trades = len(pnl_array)
    if n_trades < 5:
        return {"error": "too_few_trades", "n_trades": n_trades}

    rng = np.random.default_rng(seed)
    years = n_trades / annual_trading_days  # approximate

    sharpe_list: list[float] = []
    mdd_list: list[float] = []
    cagr_list: list[float] = []
    final_eq_list: list[float] = []

    for _ in range(n_paths):
        permuted = rng.permutation(pnl_array)
        equity = initial_capital + np.cumsum(permuted)

        # MDD
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.where(peak == 0, 1.0, peak)
        mdd_list.append(float(drawdown.min()))

        # Sharpe (using trade returns, not daily)
        ret = np.diff(equity) / np.where(equity[:-1] == 0, 1.0, equity[:-1])
        std = ret.std(ddof=1)
        sharpe = (
            float(ret.mean() / std * np.sqrt(annual_trading_days)) if std > 0 else 0.0
        )
        sharpe_list.append(sharpe)

        # CAGR
        final_eq = float(equity[-1])
        cagr = float((final_eq / initial_capital) ** (1.0 / max(years, 1e-6)) - 1.0)
        cagr_list.append(cagr)
        final_eq_list.append(final_eq)

    def _pct(lst: list[float], p: float) -> float:
        return float(np.percentile(lst, p))

    return {
        "n_paths": n_paths,
        "n_trades": n_trades,
        "sharpe": {
            "mean": float(np.mean(sharpe_list)),
            "std": float(np.std(sharpe_list)),
            "p10": _pct(sharpe_list, 10),
            "p50": _pct(sharpe_list, 50),
            "p90": _pct(sharpe_list, 90),
        },
        "mdd": {
            "mean": float(np.mean(mdd_list)),
            "p10": _pct(mdd_list, 10),
            "p50": _pct(mdd_list, 50),
            "p90": _pct(mdd_list, 90),
            "p99": _pct(mdd_list, 99),
        },
        "cagr": {
            "mean": float(np.mean(cagr_list)),
            "p10": _pct(cagr_list, 10),
            "p50": _pct(cagr_list, 50),
            "p90": _pct(cagr_list, 90),
        },
        "final_equity": {
            "mean": float(np.mean(final_eq_list)),
            "p10": _pct(final_eq_list, 10),
            "p50": _pct(final_eq_list, 50),
            "p90": _pct(final_eq_list, 90),
        },
    }
