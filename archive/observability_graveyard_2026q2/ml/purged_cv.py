"""Purged and Embargoing Cross-Validation for Financial Time Series.

Implements Lopez de Prado (2018) purged k-fold CV to prevent data leakage
from overlapping label windows. Standard time-series CV has subtle leakage:
a 5-day forward-return label computed at day t uses price information from
t+1..t+5.  If the test split starts at t+3, training samples t, t+1, t+2
are contaminated because their label windows overlap the test period.

This module provides:
- PurgedKFold: purged + embargo time-series splits
- purged_walk_forward_split: expanding/rolling window with purge + embargo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PurgedKFold:
    """Purged K-Fold cross-validator for financial time series.

    For each split, training samples whose label window overlaps the test
    period are *purged* (removed).  An additional *embargo* period after the
    test start is excluded from training to guard against serial correlation.

    Args:
        n_splits: Number of folds (default 5).
        label_horizon: Forward-return label horizon in calendar days.
            Used to determine which training samples have labels that leak
            into the test period.  E.g. ``label_horizon=5`` for 5-day
            forward returns.
        embargo_pct: Fraction of the training set to exclude as embargo
            zone immediately after the test-set start (default 0.01).
    """

    n_splits: int = 5
    label_horizon: int = 5
    embargo_pct: float = 0.01

    def split(
        self,
        timestamps: pd.Series,
        *,
        train_size: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test index arrays.

        Args:
            timestamps: Aligned Series of timestamps (one per sample row).
                Must share the same index as the feature/label DataFrame.
            train_size: If ``None`` (default), expanding window. If an int,
                rolling window of that many calendar days before the test
                start.

        Returns:
            List of ``(train_idx, test_idx)`` where each is a numpy array
            of positional indices into the original DataFrame.
        """
        if len(timestamps) < self.n_splits * 10:
            raise ValueError(
                f"Need >= {self.n_splits * 10} samples for {self.n_splits} splits, "
                f"got {len(timestamps)}"
            )

        unique_ts = pd.Series(timestamps.unique()).sort_values().reset_index(drop=True)
        n_ts = len(unique_ts)

        # Label horizon as Timedelta for comparison
        label_td = pd.Timedelta(days=self.label_horizon)

        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for fold in range(self.n_splits):
            # --- test window ---
            test_start_idx = n_ts * (fold + 1) // (self.n_splits + 1)
            test_end_idx = n_ts * (fold + 2) // (self.n_splits + 1)
            if test_end_idx <= test_start_idx:
                continue

            test_ts = unique_ts.iloc[test_start_idx:test_end_idx]
            test_start_date = test_ts.min()
            _test_end_date = test_ts.max()

            # --- raw train window ---
            if train_size is None:
                # Expanding: all dates strictly before test start
                train_ts = unique_ts.iloc[:test_start_idx]
            else:
                train_start_date = test_start_date - pd.Timedelta(days=train_size)
                train_mask = (unique_ts >= train_start_date) & (unique_ts < test_start_date)
                train_ts = unique_ts[train_mask]

            if train_ts.empty or test_ts.empty:
                continue

            # --- purge: remove train samples whose label window overlaps test ---
            # A training sample at time t has a label that uses data up to t + label_horizon.
            # If t + label_horizon >= test_start_date, the sample is contaminated.
            purge_cutoff = test_start_date - label_td
            train_ts_purged = train_ts[train_ts <= purge_cutoff]

            # --- embargo: remove additional samples after purge cutoff ---
            n_embargo = max(1, int(len(train_ts_purged) * self.embargo_pct))
            if n_embargo < len(train_ts_purged):
                train_ts_final = train_ts_purged.iloc[:-n_embargo]
            else:
                train_ts_final = train_ts_purged  # don't embargo everything

            if train_ts_final.empty:
                logger.debug(
                    "Fold %d: all training samples purged/embargoed, skipping", fold
                )
                continue

            # --- convert to row indices ---
            train_row_mask = timestamps.isin(train_ts_final)
            test_row_mask = timestamps.isin(test_ts)

            train_idx = np.where(train_row_mask)[0]
            test_idx = np.where(test_row_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            splits.append((train_idx, test_idx))

        if not splits:
            logger.warning(
                "PurgedKFold produced 0 valid splits (n_splits=%d, label_horizon=%d, "
                "n_timestamps=%d). Falling back to single expanding split.",
                self.n_splits, self.label_horizon, n_ts,
            )
            # Fallback: 80/20 expanding split with purge only
            cutoff_idx = int(n_ts * 0.8)
            purge_cutoff = unique_ts.iloc[cutoff_idx] - label_td
            train_ts_fb = unique_ts[unique_ts <= purge_cutoff]
            test_ts_fb = unique_ts.iloc[cutoff_idx:]
            if not train_ts_fb.empty and not test_ts_fb.empty:
                train_idx = np.where(timestamps.isin(train_ts_fb))[0]
                test_idx = np.where(timestamps.isin(test_ts_fb))[0]
                if len(train_idx) > 0 and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))

        return splits


def purged_walk_forward_split(
    timestamps: pd.Series,
    *,
    train_window_days: int = 252,
    test_window_days: int = 63,
    step_days: int | None = None,
    label_horizon: int = 5,
    embargo_days: int = 5,
    mode: str = "expanding",
    max_splits: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward splits with purge and embargo.

    More fine-grained than ``PurgedKFold``: gives explicit calendar-day
    control over window sizes, step sizes and embargo durations.

    Args:
        timestamps: Aligned Series of timestamps.
        train_window_days: Training window (calendar days). Ignored in
            ``"expanding"`` mode (where training grows from earliest data).
        test_window_days: Test window (calendar days).
        step_days: How many calendar days to advance between consecutive
            test windows.  Defaults to ``test_window_days`` (non-overlapping).
        label_horizon: Label horizon (calendar days) for purging.
        embargo_days: Calendar-day embargo after purge cutoff.
        mode: ``"expanding"`` or ``"rolling"``.
        max_splits: Upper bound on number of splits.

    Returns:
        List of ``(train_idx, test_idx)`` numpy arrays.
    """
    if step_days is None:
        step_days = test_window_days

    unique_ts = pd.Series(timestamps.unique()).sort_values().reset_index(drop=True)
    if unique_ts.empty:
        return []

    ts_min, ts_max = unique_ts.min(), unique_ts.max()
    label_td = pd.Timedelta(days=label_horizon)
    embargo_td = pd.Timedelta(days=embargo_days)

    # First possible test start: after enough training data
    if mode == "expanding":
        first_test_start = ts_min + pd.Timedelta(days=train_window_days)
    else:
        first_test_start = ts_min + pd.Timedelta(days=train_window_days)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    test_start = first_test_start

    while test_start <= ts_max and len(splits) < max_splits:
        test_end = test_start + pd.Timedelta(days=test_window_days)

        # Test timestamps
        test_mask_ts = (unique_ts >= test_start) & (unique_ts < test_end)
        test_ts = unique_ts[test_mask_ts]

        if test_ts.empty:
            test_start += pd.Timedelta(days=step_days)
            continue

        # Training timestamps
        if mode == "expanding":
            raw_train_mask = unique_ts < test_start
        else:
            train_start = test_start - pd.Timedelta(days=train_window_days)
            raw_train_mask = (unique_ts >= train_start) & (unique_ts < test_start)

        raw_train_ts = unique_ts[raw_train_mask]

        # Purge + embargo
        purge_cutoff = test_start - label_td - embargo_td
        train_ts = raw_train_ts[raw_train_ts <= purge_cutoff]

        if train_ts.empty or test_ts.empty:
            test_start += pd.Timedelta(days=step_days)
            continue

        # Convert to row indices
        train_idx = np.where(timestamps.isin(train_ts))[0]
        test_idx = np.where(timestamps.isin(test_ts))[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        test_start += pd.Timedelta(days=step_days)

    return splits


__all__ = [
    "PurgedKFold",
    "purged_walk_forward_split",
]
