"""Purged + embargoed cross-validation for time-series financial data.

Implements walk-forward PurgedKFold to prevent label leakage across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """Walk-forward cross-validator with purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    label_horizon : int
        Forward-return horizon in days (used for purging).
    embargo_pct : float
        Fraction of the fold length to embargo after the test set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_pct = embargo_pct

    def split(self, timestamps: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_indices, test_indices) for each fold.

        Uses walk-forward expanding window: fold i trains on all data before
        the i-th test period, with purging near the test boundary.
        """
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        n = len(ts)
        if n < self.n_splits * 2:
            return []

        # Split unique dates into n_splits + 1 equal-ish chunks
        unique_dates = ts.sort_values().unique()
        fold_size = len(unique_dates) // (self.n_splits + 1)
        if fold_size == 0:
            return []

        # Embargo window: label_horizon calendar days
        embargo_days = max(self.label_horizon, 1)

        splits = []
        for k in range(1, self.n_splits + 1):
            test_start_date = unique_dates[k * fold_size]
            if k == self.n_splits:
                test_end_date = unique_dates[-1]
            else:
                test_end_date = unique_dates[min((k + 1) * fold_size - 1, len(unique_dates) - 1)]

            purge_cutoff = test_start_date - pd.Timedelta(days=embargo_days)

            train_mask = ts < purge_cutoff
            test_mask = (ts >= test_start_date) & (ts <= test_end_date)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            splits.append((train_idx, test_idx))

        return splits
