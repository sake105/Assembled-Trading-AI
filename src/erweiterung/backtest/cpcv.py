"""Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).

DUPLIKAT-HINWEIS
================
Mainline hat ``src/assembled_core/qa/cpcv_validation.py`` (236 LoC), die
``skfolio`` als primäres Backend nutzt mit sklearn-Walk-Forward-Fallback
+ ``meta_labeling_pipeline``. Für Production nutze die mainline-Version.

Diese Erweiterungs-Variante ist **NumPy-only**, ohne externe Lib —
nutzbar wenn skfolio nicht verfügbar oder didaktischer Code gewünscht.

Theorie
-------
Klassisches K-Fold-CV in Time-Series leakt:
- Train- und Test-Samples können zeitlich überlappen.
- Adjacent-Train/Test-Folds teilen ähnliche Zustände.

Lopez de Prado (2018) löst das durch:
1. **Purging**: Aus dem Trainset werden alle Samples entfernt, deren
   Information-Period mit dem Test-Fold überlappt.
2. **Embargo**: Zusätzlicher Time-Buffer nach dem Test-Fold (typisch 1-2 % der
   Daten).
3. **Combinatorial**: Statt k einzelner Test-Folds werden alle C(N, k)
   Kombinationen von k-out-of-N Test-Gruppen durchlaufen → mehrere
   Backtest-Pfade.

Resultat: ``n_paths`` distinct Backtest-Pfade, deren Sharpe-Verteilung über
die kombinatorische Selektion **rigorose Statistik** liefert.

Referenzen
----------
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez de Prado, M. (2018). The 10 Reasons Most Machine Learning Funds Fail.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    n_groups: int = 6  # N
    test_groups_per_split: int = 2  # k
    embargo_pct: float = 0.01  # Embargo-Period nach Test-Fold


def _purge_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    info_period_lookback: int = 0,
    embargo: int = 0,
    label_horizon: int = 0,
) -> np.ndarray:
    """Entferne train-indices, deren Information-Period mit test-Fold überlappt.

    Fix (audit C4-001 / C3-026):
        The information window of training sample i extends BACKWARD,
        not forward. Features at index i typically use the last
        ``info_period_lookback`` bars (rolling vol, MA, RSI etc.), so a
        test sample whose label realises in ``[i - info_period_lookback, i]``
        can leak into the features at i. The previous implementation
        looked forward (``range(i, i + lookback)``) which is the wrong
        direction (Lopez de Prado 2018, Ch. 7.4.1).

    Additionally we now purge the **label-horizon** direction: training
    sample i carries a forward-return label spanning ``[i, i + label_horizon]``
    and a test sample inside that window contaminates the train label.

    Args:
        train_idx: train indices (1-D int array).
        test_idx: test indices (1-D int array).
        info_period_lookback: feature-side lookback in bars. Purge if any
            test index sits in [i - lookback, i].
        label_horizon: forward label horizon in bars. Purge if any test
            index sits in [i, i + horizon].
        embargo: extra time-buffer immediately after the latest test bar.

    Returns:
        Filtered train indices (1-D int array).
    """
    test_set = set(test_idx.tolist())
    keep = []
    test_max = int(test_idx.max()) if len(test_idx) > 0 else -1
    for i in train_idx:
        # Backward purge — feature-leakage protection (audit C4-001 fix).
        if info_period_lookback > 0:
            back_lo = max(0, int(i) - info_period_lookback)
            back_window = range(back_lo, int(i) + 1)
            if any(t in test_set for t in back_window):
                continue
        # Forward purge — label-leakage protection (label horizon).
        if label_horizon > 0:
            fwd_window = range(int(i), int(i) + label_horizon + 1)
            if any(t in test_set for t in fwd_window):
                continue
        # Embargo: drop train samples too close after test_max.
        if embargo > 0 and i > test_max and i - test_max <= embargo:
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def cpcv_splits(
    n_samples: int,
    config: CPCVConfig,
    info_period_lookback: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Erzeuge alle (train, test)-Index-Splits für CPCV.

    Returns:
        Liste von (train_indices, test_indices) — total = C(n_groups, k_test).
    """
    n = n_samples
    group_size = n // config.n_groups
    groups = [
        np.arange(
            g * group_size, (g + 1) * group_size if g < config.n_groups - 1 else n
        )
        for g in range(config.n_groups)
    ]
    embargo = int(np.ceil(n * config.embargo_pct))

    out = []
    for combo in itertools.combinations(
        range(config.n_groups), config.test_groups_per_split
    ):
        test_idx = np.concatenate([groups[g] for g in combo])
        all_idx = np.arange(n)
        train_idx = np.array([i for i in all_idx if i not in set(test_idx.tolist())])
        train_idx = _purge_train_indices(
            train_idx, test_idx, info_period_lookback, embargo
        )
        out.append((train_idx, test_idx))
    return out


def cpcv_backtest_paths(
    n_samples: int,
    config: CPCVConfig,
    fold_strategy_fn: Callable[[np.ndarray, np.ndarray], pd.Series],
) -> list[pd.Series]:
    """Erzeuge ``n_paths`` Backtest-Pfade aus CPCV.

    Args:
        n_samples: Gesamtbeobachtungen.
        config: CPCVConfig.
        fold_strategy_fn: Callable ``(train_idx, test_idx) -> Series of test_returns``.

    Returns:
        Liste von ``pd.Series`` — jeder eine Sequenz von Returns aus jeweils
        passend ausgewählten Test-Folds (n_paths = N!/(k!(N-k)!) / (N!/...)).
    """
    splits = cpcv_splits(n_samples, config)
    # number of paths = C(N-1, k-1)/k? In Lopez de Prado approach, n_paths = C(N, k)*k/N
    paths = []
    for train_idx, test_idx in splits:
        ret = fold_strategy_fn(train_idx, test_idx)
        paths.append(ret)
    return paths


__all__ = ["CPCVConfig", "cpcv_splits", "cpcv_backtest_paths"]
