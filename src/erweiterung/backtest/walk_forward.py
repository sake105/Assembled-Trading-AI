"""Walk-Forward Backtest-Engine.

Strikte Walk-Forward-Logik:
- Training-Window: [t - train_size, t)
- OOS-Test:        [t, t + test_size)
- Step:            test_size (rolling) oder train_size + test_size (anchored)

Vorteil
-------
Realistischste Out-of-Sample-Simulation.  Modell wird in jedem Schritt neu
trainiert mit nur historisch verfügbaren Daten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_size: int = 252  # 1 year
    test_size: int = 21  # 1 month
    expanding: bool = False  # If True: train_window grows
    min_train_size: int = 252
    embargo_days: int = 1  # purge between train and test


def walk_forward_run(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    config: Optional[WalkForwardConfig] = None,
) -> pd.DataFrame:
    """Run walk-forward backtest.

    Args:
        df: full panel/dataframe sorted by date.
        strategy_fn: ``f(train_df, test_df) -> Series of test_returns``.
        config: WalkForwardConfig.

    Returns:
        DataFrame [date, return, fold_id] mit OOS-Returns.
    """
    config = config or WalkForwardConfig()
    if df.empty:
        return df.assign(return_=pd.Series(dtype=float))

    n = len(df)
    train_size = config.train_size
    test_size = config.test_size
    embargo = config.embargo_days

    rows = []
    fold = 0
    start = train_size + embargo
    while start + test_size <= n:
        if config.expanding:
            train_df = df.iloc[: start - embargo]
        else:
            train_df = df.iloc[start - embargo - train_size : start - embargo]
        test_df = df.iloc[start : start + test_size]
        try:
            ret = strategy_fn(train_df, test_df)
        except Exception as e:  # noqa: BLE001
            logger.warning("[walk-fwd] fold %d failed: %s", fold, e)
            start += test_size
            fold += 1
            continue
        for d, r in ret.items():
            rows.append({"date": d, "return": float(r), "fold_id": fold})
        start += test_size
        fold += 1

    return pd.DataFrame(rows)


__all__ = ["WalkForwardConfig", "walk_forward_run"]
