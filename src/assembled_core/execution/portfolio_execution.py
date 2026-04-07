"""Portfolio Execution Optimizer (Plan 6.10).

Coordinates correlated buys/sells to minimize tracking error during transition.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_execution_sequence(
    orders: pd.DataFrame,
    correlation_matrix: pd.DataFrame | None = None,
    max_parallel: int = 5,
) -> pd.DataFrame:
    """Order execution sequencing to minimize transition tracking error.

    Strategy: execute correlated opposing trades (buy X, sell Y where corr(X,Y) > 0.5)
    simultaneously to reduce market impact.

    Args:
        orders: DataFrame with columns [symbol, qty, direction].
        correlation_matrix: Pairwise correlation of assets.
        max_parallel: Maximum simultaneous orders.

    Returns:
        orders with added 'execution_batch' column (int).
    """
    result = orders.copy()
    n = len(result)

    if n == 0:
        result["execution_batch"] = pd.Series(dtype=int)
        return result

    if correlation_matrix is None or n <= max_parallel:
        result["execution_batch"] = 0
        return result

    # Separate buys and sells
    buys = result[result["qty"] > 0].index.tolist()
    sells = result[result["qty"] < 0].index.tolist()

    batch = np.zeros(n, dtype=int)
    current_batch = 0

    # Pair correlated buys/sells in same batch
    paired_buys = set()
    paired_sells = set()

    for bi in buys:
        sym_b = result.loc[bi, "symbol"]
        if sym_b not in correlation_matrix.index:
            continue
        for si in sells:
            if si in paired_sells:
                continue
            sym_s = result.loc[si, "symbol"]
            if sym_s not in correlation_matrix.columns:
                continue
            corr = correlation_matrix.loc[sym_b, sym_s]
            if abs(corr) > 0.5:
                batch[bi] = current_batch
                batch[si] = current_batch
                paired_buys.add(bi)
                paired_sells.add(si)
                if len(paired_buys) % max_parallel == 0:
                    current_batch += 1
                break

    # Remaining unpaired in subsequent batches
    current_batch += 1
    for i in range(n):
        if i not in paired_buys and i not in paired_sells:
            batch[i] = current_batch

    result["execution_batch"] = batch
    return result


__all__ = ["optimize_execution_sequence"]
