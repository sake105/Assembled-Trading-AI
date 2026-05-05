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

    def split(
        self,
        timestamps: pd.Series,
        train_size: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_indices, test_indices) for each fold.

        Parameters
        ----------
        timestamps : pd.Series
            Timestamp series aligned to the panel index.
        train_size : int | None
            If set, use a rolling window of this many days instead of expanding.

        Raises
        ------
        ValueError
            If timestamps has fewer than n_splits * 2 samples.
        """
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        n = len(ts)
        min_samples = self.n_splits * 4
        if n < min_samples:
            raise ValueError(
                f"Not enough samples for {self.n_splits} splits: "
                f"need at least {min_samples}, got {n}"
            )

        unique_dates = ts.sort_values().unique()
        fold_size = len(unique_dates) // (self.n_splits + 1)
        if fold_size == 0:
            return []

        embargo_days = max(self.label_horizon, 1)

        splits = []
        for k in range(1, self.n_splits + 1):
            test_start_date = unique_dates[k * fold_size]
            if k == self.n_splits:
                test_end_date = unique_dates[-1]
            else:
                test_end_date = unique_dates[
                    min((k + 1) * fold_size - 1, len(unique_dates) - 1)
                ]

            purge_cutoff = test_start_date - pd.Timedelta(days=embargo_days)

            if train_size is not None:
                # Rolling window: only include last train_size days before purge cutoff
                roll_start = purge_cutoff - pd.Timedelta(days=train_size)
                train_mask = (ts >= roll_start) & (ts < purge_cutoff)
            else:
                train_mask = ts < purge_cutoff

            test_mask = (ts >= test_start_date) & (ts <= test_end_date)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            splits.append((train_idx, test_idx))

        return splits


def purged_walk_forward_split(
    timestamps: pd.Series,
    train_window_days: int = 252,
    test_window_days: int = 63,
    label_horizon: int = 5,
    embargo_days: int = 5,
    max_splits: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward split with fixed train/test windows and embargo.

    Parameters
    ----------
    timestamps : pd.Series
        Timestamp series aligned to the panel index.
    train_window_days : int
        Number of calendar days in each training window.
    test_window_days : int
        Number of calendar days in each test window.
    label_horizon : int
        Forward-return horizon in days (purged from train end).
    embargo_days : int
        Additional embargo days after each test window.
    max_splits : int | None
        Cap the number of splits returned.

    Returns
    -------
    List of (train_indices, test_indices) tuples.
    """
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    unique_dates = np.sort(ts.unique())
    if len(unique_dates) == 0:
        return []

    total_days = int((unique_dates[-1] - unique_dates[0]) / np.timedelta64(1, "D"))
    min_required = train_window_days + test_window_days + embargo_days + label_horizon
    if total_days < min_required:
        return []

    splits = []
    test_start = unique_dates[0] + pd.Timedelta(days=train_window_days)

    while True:
        test_end = test_start + pd.Timedelta(days=test_window_days - 1)
        if test_end > unique_dates[-1]:
            break

        purge_cutoff = test_start - pd.Timedelta(days=label_horizon)
        train_start = purge_cutoff - pd.Timedelta(days=train_window_days)

        train_mask = (ts >= train_start) & (ts < purge_cutoff)
        test_mask = (ts >= test_start) & (ts <= test_end)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        if max_splits is not None and len(splits) >= max_splits:
            break

        test_start = test_end + pd.Timedelta(days=embargo_days + 1)

    return splits
