"""Walk-Forward Out-of-Sample-Validation für Regime-Switching.

Theorie
-------
Backtest-Resultate sind nur dann robust, wenn sie aus einer Methode kommen,
die **ohne Future-Information** entscheidet. Walk-Forward-Testing (Lopez de
Prado 2018, §11) erzwingt das durch:

1. Rolle ein **Training-Window** vorwärts.
2. Fitte Hyperparameter (z. B. Threshold) auf Training-Daten.
3. Anwende die gefitteten Parameter **auf das nächste, unsichtbare Test-Window**.
4. Sammle die Test-Window-Returns.
5. Wiederhole bis Ende.

Damit ist sichergestellt, dass keine Future-Information in die Allokations-
Entscheidung einfließt.

Anwendung
---------
Hier konkret: für jedes Test-Window finden wir den **besten Drawdown-Threshold
im Trainings-Window** (gemessen am Calmar-Ratio), und wenden ihn dann auf das
Test-Window an.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    train_days: int = 1260  # ~5 Jahre Training
    test_days: int = 252  # 1 Jahr Test
    step_days: int = 252  # Step size between windows
    min_train_days: int = 504


def walk_forward_threshold_search(
    bench_returns: pd.Series,
    fac_returns: pd.Series,
    threshold_grid: list[float],
    objective_fn: Callable[[pd.Series], float] | None = None,
    config: WalkForwardConfig | None = None,
    from_erweiterung_regime: Callable | None = None,
) -> pd.DataFrame:
    """Walk-Forward-Optimierung des Drawdown-Thresholds.

    Args:
        bench_returns: "calm"-Strategie-Tagesreturns.
        fac_returns: "stress"-Strategie-Tagesreturns.
        threshold_grid: Liste der zu testenden Drawdown-Thresholds.
        objective_fn: Funktion (returns -> float), Default = Calmar-Ratio.
        config: WalkForwardConfig.
        from_erweiterung_regime: Callable, das Threshold + Bench-Returns
            → Regime-Series ("calm"/"stress") berechnet. Default: trailing-60d-DD.

    Returns:
        DataFrame mit Spalten:
        - window_idx
        - train_start, train_end, test_start, test_end
        - best_threshold (aus dem Train-Window)
        - train_calmar, test_calmar
        - train_sharpe, test_sharpe
        Pro Test-Window eine Zeile.
    """
    cfg = config or WalkForwardConfig()
    if objective_fn is None:

        def objective_fn(r: pd.Series) -> float:
            if r.empty or r.std() == 0:
                return -np.inf
            ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
            eq = (1 + r).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            return ann_ret / abs(dd) if dd != 0 else -np.inf

    if from_erweiterung_regime is None:

        def from_erweiterung_regime(thr: float, bench: pd.Series) -> pd.Series:
            eq = (1 + bench.fillna(0)).cumprod()
            rolling_max = eq.rolling(60, min_periods=1).max()
            dd = (1 - eq / rolling_max).abs()
            return pd.Series(np.where(dd > thr, "stress", "calm"), index=bench.index)

    aligned = pd.concat({"bench": bench_returns, "fac": fac_returns}, axis=1).dropna()
    n = len(aligned)
    rows = []
    window_idx = 0
    start = 0
    while start + cfg.train_days + cfg.test_days <= n:
        train_idx = aligned.index[start : start + cfg.train_days]
        test_idx = aligned.index[
            start + cfg.train_days : start + cfg.train_days + cfg.test_days
        ]
        train_bench = aligned.loc[train_idx, "bench"]
        train_fac = aligned.loc[train_idx, "fac"]
        test_bench = aligned.loc[test_idx, "bench"]
        test_fac = aligned.loc[test_idx, "fac"]

        # Find best threshold in train
        best_thr = threshold_grid[0]
        best_obj = -np.inf
        train_objs: dict[float, float] = {}
        for thr in threshold_grid:
            regime = from_erweiterung_regime(thr, train_bench)
            regime_lag = regime.shift(1)
            alloc = pd.Series(
                np.where(regime_lag == "stress", train_fac, train_bench),
                index=train_bench.index,
            ).dropna()
            obj = objective_fn(alloc)
            train_objs[thr] = obj
            if obj > best_obj:
                best_obj = obj
                best_thr = thr

        # Apply best_thr to test (use last train_bench for warmup so DD-trailing is meaningful)
        combined_for_dd = pd.concat(
            [train_bench.iloc[-60:], test_bench]
        )  # for warm DD computation
        regime_full = from_erweiterung_regime(best_thr, combined_for_dd)
        # Trim to test window with t-1 shift
        regime_test = regime_full.shift(1).loc[test_idx]
        test_alloc = pd.Series(
            np.where(regime_test == "stress", test_fac, test_bench),
            index=test_idx,
        ).dropna()

        train_alloc = pd.Series(
            np.where(
                from_erweiterung_regime(best_thr, train_bench).shift(1) == "stress",
                train_fac,
                train_bench,
            ),
            index=train_bench.index,
        ).dropna()

        train_ann = (1 + train_alloc).prod() ** (252 / max(len(train_alloc), 1)) - 1
        train_dd = (
            (1 + train_alloc).cumprod() / (1 + train_alloc).cumprod().cummax() - 1
        ).min()
        train_vol = train_alloc.std() * np.sqrt(252)
        train_sharpe = train_ann / train_vol if train_vol > 0 else 0

        test_ann = (1 + test_alloc).prod() ** (252 / max(len(test_alloc), 1)) - 1
        test_dd = (
            (1 + test_alloc).cumprod() / (1 + test_alloc).cumprod().cummax() - 1
        ).min()
        test_vol = test_alloc.std() * np.sqrt(252)
        test_sharpe = test_ann / test_vol if test_vol > 0 else 0

        rows.append(
            {
                "window_idx": window_idx,
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "test_start": test_idx[0],
                "test_end": test_idx[-1],
                "best_threshold": float(best_thr),
                "train_obj": float(train_objs[best_thr]),
                "train_ann_return": float(train_ann),
                "train_sharpe": float(train_sharpe),
                "train_max_dd": float(train_dd),
                "test_ann_return": float(test_ann),
                "test_sharpe": float(test_sharpe),
                "test_max_dd": float(test_dd),
                "test_n_days": int(len(test_alloc)),
            }
        )
        window_idx += 1
        start += cfg.step_days
    return pd.DataFrame(rows)


def concat_oos_returns(
    bench_returns: pd.Series,
    fac_returns: pd.Series,
    walk_forward_df: pd.DataFrame,
    from_erweiterung_regime: Callable | None = None,
) -> pd.Series:
    """Konkateniere alle Test-Window-Returns zu einer durchgehenden OOS-Series."""
    if from_erweiterung_regime is None:

        def from_erweiterung_regime(thr: float, bench: pd.Series) -> pd.Series:
            eq = (1 + bench.fillna(0)).cumprod()
            rolling_max = eq.rolling(60, min_periods=1).max()
            dd = (1 - eq / rolling_max).abs()
            return pd.Series(np.where(dd > thr, "stress", "calm"), index=bench.index)

    aligned = pd.concat({"bench": bench_returns, "fac": fac_returns}, axis=1).dropna()
    out_chunks = []
    for _, row in walk_forward_df.iterrows():
        thr = float(row["best_threshold"])
        test_start = row["test_start"]
        test_end = row["test_end"]
        # Warmup für trailing-DD: nimm einen größeren Vorlauf
        warmup_start = aligned.index[max(0, aligned.index.get_loc(test_start) - 60)]
        chunk = aligned.loc[warmup_start:test_end]
        regime = from_erweiterung_regime(thr, chunk["bench"])
        regime_lag = regime.shift(1)
        alloc = pd.Series(
            np.where(regime_lag == "stress", chunk["fac"], chunk["bench"]),
            index=chunk.index,
        )
        out_chunks.append(alloc.loc[test_start:test_end])
    if not out_chunks:
        return pd.Series(dtype=float)
    full = pd.concat(out_chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full


__all__ = [
    "WalkForwardConfig",
    "walk_forward_threshold_search",
    "concat_oos_returns",
]
