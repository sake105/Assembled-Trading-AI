"""Shared metric helpers for scripts/forensic/.

Extracted 2026-05-18 per F-S2-OOR-1 / F-S2-HOL-1 follow-up: the same
``_annualised_sharpe`` and ``_max_drawdown`` helpers were copied across
``equity_curve_audit.py``, ``out_of_regime_test.py``, and
``hold_out_leakage_test.py`` (Rule of Three triggered at 4 consumers).

This module centralises them with one canonical implementation each.
``survivorship_bias_check.py`` does not use these helpers and is
unaffected.

Public API:
    - annualised_sharpe(returns, periods_per_year=252) -> float
    - max_drawdown(equity) -> float

Both return NaN on insufficient / undefined input rather than raising,
because they are used in audit / reporting paths where soft-fail with
NaN is preferable to halting the whole report on one bad column.
"""

from __future__ import annotations

import numpy as np


def annualised_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio of a returns series.

    Returns NaN if:
        - len(returns) < 2
        - std(returns) == 0 (constant returns)
        - mean(returns) is non-finite

    The annualisation factor is ``sqrt(periods_per_year)``: 252 daily,
    52 weekly, 12 monthly.
    """
    if len(returns) < 2:
        return float("nan")
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std <= 0 or not np.isfinite(mean):
        return float("nan")
    return mean / std * float(np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity series (returns a non-positive float).

    Drawdown is defined as ``(equity[t] - max(equity[0..t])) / max(equity[0..t])``.
    Returns 0.0 if len(equity) < 2 (no drawdown possible).
    """
    if len(equity) < 2:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())
