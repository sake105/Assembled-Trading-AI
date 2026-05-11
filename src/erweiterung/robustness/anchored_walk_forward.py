"""Anchored vs Rolling Walk-Forward-Analysis.

Anchored (Expanding)
--------------------
Train-window wächst mit jedem Step: ``[start, t]`` → predict ``[t, t+test]``.
Vorteil: nutzt alle bis-dato verfügbare Daten.

Rolling
-------
Fixe Fenstergröße: ``[t-W, t]`` → predict ``[t, t+test]``.
Vorteil: passt sich an Regime-Wechsel an, vergisst alte Daten.

Hybrid
------
Anchored bis ``min_train``, dann rolling. Praktisch oft beste Variante.

Anwendung
---------
Vergleicht ein-Backtest-Strategy unter beiden Schemes. Wenn Sharpe stark
abweicht, deutet das auf Regime-Drift hin (rolling profitabler) oder auf
Stichproben-Effekt (anchored profitabler).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    min_train_size: int = 252
    test_size: int = 21
    mode: str = "anchored"  # 'anchored' | 'rolling' | 'hybrid'
    rolling_window_size: int = 504  # only used for rolling/hybrid
    hybrid_switch_size: int = 1008  # at which train-size hybrid switches to rolling


def walk_forward(
    data_index: pd.DatetimeIndex,
    strategy_fn: Callable[[pd.Index, pd.Index], pd.Series],
    config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """Generic walk-forward driver.

    Args:
        data_index: full datetime-index.
        strategy_fn: callable(train_idx, test_idx) -> pd.Series of test_returns.
        config: WalkForwardConfig.

    Returns:
        DataFrame [date, return, fold_id, mode].
    """
    cfg = config or WalkForwardConfig()
    n = len(data_index)
    if n < cfg.min_train_size + cfg.test_size:
        raise ValueError("not enough data")

    rows: list[dict] = []
    fold = 0
    start = cfg.min_train_size
    while start + cfg.test_size <= n:
        if cfg.mode == "anchored":
            train_idx = data_index[:start]
        elif cfg.mode == "rolling":
            train_start = max(0, start - cfg.rolling_window_size)
            train_idx = data_index[train_start:start]
        elif cfg.mode == "hybrid":
            if start <= cfg.hybrid_switch_size:
                train_idx = data_index[:start]
            else:
                train_start = max(0, start - cfg.rolling_window_size)
                train_idx = data_index[train_start:start]
        else:
            raise ValueError(f"unknown mode: {cfg.mode}")
        test_idx = data_index[start : start + cfg.test_size]
        try:
            ret = strategy_fn(train_idx, test_idx)
        except Exception:  # noqa: BLE001
            start += cfg.test_size
            fold += 1
            continue
        for d, r in ret.items():
            rows.append(
                {"date": d, "return": float(r), "fold_id": fold, "mode": cfg.mode}
            )
        start += cfg.test_size
        fold += 1
    return pd.DataFrame(rows)


def compare_anchored_vs_rolling(
    data_index: pd.DatetimeIndex,
    strategy_fn: Callable[[pd.Index, pd.Index], pd.Series],
    min_train: int = 252,
    test_size: int = 21,
    rolling_window: int = 504,
) -> dict:
    """Run both modes, compare Sharpe, return DataFrame + summary."""
    anchored = walk_forward(
        data_index,
        strategy_fn,
        WalkForwardConfig(
            min_train_size=min_train, test_size=test_size, mode="anchored"
        ),
    )
    rolling = walk_forward(
        data_index,
        strategy_fn,
        WalkForwardConfig(
            min_train_size=min_train,
            test_size=test_size,
            mode="rolling",
            rolling_window_size=rolling_window,
        ),
    )

    def _sharpe(df: pd.DataFrame) -> float:
        if df.empty:
            return float("nan")
        r = df["return"]
        if r.std(ddof=0) == 0:
            return float("nan")
        return float(r.mean() / r.std(ddof=0) * np.sqrt(252))

    return {
        "anchored": anchored,
        "rolling": rolling,
        "sharpe_anchored": _sharpe(anchored),
        "sharpe_rolling": _sharpe(rolling),
        "n_folds_anchored": (
            int(anchored["fold_id"].nunique()) if not anchored.empty else 0
        ),
        "n_folds_rolling": (
            int(rolling["fold_id"].nunique()) if not rolling.empty else 0
        ),
    }


__all__ = ["WalkForwardConfig", "walk_forward", "compare_anchored_vs_rolling"]
