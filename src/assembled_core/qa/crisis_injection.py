# src/assembled_core/qa/crisis_injection.py
"""Crisis-injection backtest validation (audit C2-017).

Synthetic insertion of three canonical historical-stress regimes into a
returns series so a strategy can be tested against "what if 2008 / 2020
COVID / 2022 inflation grind happened TODAY". The point is not to
replay those exact paths; it is to confirm the strategy:

1. survives — equity does not go to zero,
2. activates the right protective regimes (kill-switch, drawdown derisk,
   vol-targeting), and
3. produces a defensible drawdown profile relative to the unstressed
   counterfactual.

The injected shocks are deliberately stylised:

- ``inject_2008_shock`` — six contiguous weeks of left-skewed, fat-tailed
  returns calibrated to roughly match S&P 500 Q4-2008 daily statistics.
- ``inject_2020_covid_shock`` — three weeks of high-vol but symmetric
  returns followed by three weeks of negative-skew rebound (the actual
  COVID profile was V-shaped, not L-shaped).
- ``inject_2022_inflation_grind`` — a long, slow grind: 6 months of
  modest-negative-drift, modest-vol returns.

Each injector returns the *modified* series so the caller can pipe
through their existing backtest plumbing without renaming columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_2008_shock(
    returns: pd.Series,
    *,
    start_index: int = 0,
    seed: int | None = 2008,
) -> pd.Series:
    """Splice 30 trading days (~6 weeks) of 2008-style returns into ``returns``.

    Starts at ``start_index`` (default: very beginning). Existing returns
    in the injection window are *replaced*, not added — this is the worst
    case for the strategy because there is no benign cushion.
    """
    rng = np.random.default_rng(seed)
    n_shock = 30
    # Q4-2008 S&P 500 stylised facts: mean ~ -0.0025/day, std ~ 0.035,
    # skew ~ -0.4, excess kurt ~ 5. A simple skewed-Student t with df=6
    # captures the fat tail well enough for adversarial validation.
    base = rng.standard_t(df=6.0, size=n_shock) * 0.035 - 0.0025
    # Add a left-skew by squashing the right tail.
    base = np.where(base > 0, base * 0.5, base)
    return _splice(returns, start_index, base)


def inject_2020_covid_shock(
    returns: pd.Series,
    *,
    start_index: int = 0,
    seed: int | None = 2020,
) -> pd.Series:
    """Two phases: 15-day crash + 15-day V-shape rebound."""
    rng = np.random.default_rng(seed)
    crash = rng.standard_t(df=4.0, size=15) * 0.05 - 0.005  # higher vol, fatter tails
    rebound = rng.standard_t(df=8.0, size=15) * 0.025 + 0.004  # positive drift
    injected = np.concatenate([crash, rebound])
    return _splice(returns, start_index, injected)


def inject_2022_inflation_grind(
    returns: pd.Series,
    *,
    start_index: int = 0,
    seed: int | None = 2022,
) -> pd.Series:
    """6 months (~126 trading days) of slow negative drift, modest vol."""
    rng = np.random.default_rng(seed)
    # 2022 H1: mean ~ -0.001/day, std ~ 0.012, near-normal tails. The
    # grind is what kills strategies that rely on Q4-style fast recoveries.
    base = rng.standard_normal(126) * 0.012 - 0.001
    return _splice(returns, start_index, base)


def _splice(returns: pd.Series, start: int, injected: np.ndarray) -> pd.Series:
    """Replace ``injected.size`` values in ``returns`` starting at ``start``."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pd.Series")
    if start < 0 or start + len(injected) > len(returns):
        raise ValueError(
            f"injection window [{start}, {start + len(injected)}) "
            f"exceeds returns length {len(returns)}"
        )
    out = returns.copy().astype(float)
    out.iloc[start : start + len(injected)] = injected
    return out


def run_crisis_battery(
    returns: pd.Series,
    *,
    starts: tuple[int, int, int] | None = None,
) -> dict[str, pd.Series]:
    """Apply all three crisis injections in three independent copies.

    Returns a dict keyed by scenario name with the injected series.
    """
    n = len(returns)
    if starts is None:
        # Place each shock in its own third of the series so they don't
        # overlap when callers want to compare drawdowns side-by-side.
        starts = (
            max(0, n // 6),
            max(0, n // 2),
            max(0, (2 * n) // 3),
        )
    return {
        "2008": inject_2008_shock(returns, start_index=starts[0]),
        "2020": inject_2020_covid_shock(returns, start_index=starts[1]),
        "2022": inject_2022_inflation_grind(returns, start_index=starts[2]),
    }


__all__ = [
    "inject_2008_shock",
    "inject_2020_covid_shock",
    "inject_2022_inflation_grind",
    "run_crisis_battery",
]
