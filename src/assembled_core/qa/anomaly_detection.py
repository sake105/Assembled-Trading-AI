"""Trade/signal anomaly detection — fat-finger, manipulation, strategy drift.

Uses PyOD ensemble if installed; falls back to numpy-based IQR/Z-score detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    scores: pd.Series
    flags: pd.Series  # 1 = anomaly
    method: str
    n_anomalies: int
    contamination: float
    details: dict[str, Any] = field(default_factory=dict)


class TradeAnomalyDetector:
    """Detect anomalies in trades / signals / market data.

    Ensemble of PyOD detectors (IForest + COPOD + ECOD) when installed;
    falls back to IQR + Z-score when PyOD is not available.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalies (0.0–0.5). Default 0.01 = 1%.
    """

    def __init__(self, contamination: float = 0.01) -> None:
        self.contamination = contamination
        self._detectors: dict[str, Any] = {}
        self._fitted = False
        self._method = "unfit"
        self._baseline_stats: dict[str, tuple[float, float, float, float, float]] = {}

    def fit(self, baseline_df: pd.DataFrame) -> "TradeAnomalyDetector":
        """Fit on historical normal data."""
        X = self._prepare(baseline_df)
        if self._try_fit_pyod(X):
            self._method = "pyod_ensemble"
        else:
            self._fit_iqr_zscore(X, baseline_df.columns.tolist())
            self._method = "iqr_zscore_fallback"
        self._fitted = True
        logger.info("[Anomaly] Fitted %s on %d rows", self._method, len(baseline_df))
        return self

    def score(self, current_df: pd.DataFrame) -> AnomalyResult:
        """Score new data; flag if majority of detectors say anomaly."""
        if not self._fitted:
            raise RuntimeError("Call fit() before score()")
        X = self._prepare(current_df)

        if self._method == "pyod_ensemble":
            return self._score_pyod(X, current_df.index)
        return self._score_iqr_zscore(X, current_df)

    # ------------------------------------------------------------------
    # PyOD backend
    # ------------------------------------------------------------------

    def _try_fit_pyod(self, X: np.ndarray) -> bool:
        try:
            from pyod.models.copod import COPOD  # noqa: PLC0415
            from pyod.models.ecod import ECOD  # noqa: PLC0415
            from pyod.models.iforest import IForest  # noqa: PLC0415

            self._detectors = {
                "iforest": IForest(contamination=self.contamination),
                "copod": COPOD(contamination=self.contamination),
                "ecod": ECOD(contamination=self.contamination),
            }
            for det in self._detectors.values():
                det.fit(X)
            return True
        except ImportError:
            return False

    def _score_pyod(self, X: np.ndarray, index: Any) -> AnomalyResult:
        scores_dict: dict[str, np.ndarray] = {}
        flags_dict: dict[str, np.ndarray] = {}
        for name, det in self._detectors.items():
            scores_dict[name] = det.decision_function(X)
            flags_dict[name] = det.predict(X)

        flags_arr = np.column_stack(list(flags_dict.values()))
        consensus = (flags_arr.sum(axis=1) >= 2).astype(int)
        avg_score = np.column_stack(list(scores_dict.values())).mean(axis=1)

        flags = pd.Series(consensus, index=index, name="anomaly_flag")
        scores = pd.Series(avg_score, index=index, name="anomaly_score")
        return AnomalyResult(
            scores=scores,
            flags=flags,
            method="pyod_ensemble",
            n_anomalies=int(consensus.sum()),
            contamination=self.contamination,
            details={k: v.tolist() for k, v in scores_dict.items()},
        )

    # ------------------------------------------------------------------
    # Pure numpy IQR + Z-score fallback
    # ------------------------------------------------------------------

    def _fit_iqr_zscore(self, X: np.ndarray, cols: list[str]) -> None:
        self._baseline_stats = {}
        for i, col in enumerate(cols):
            col_data = X[:, i]
            q1, q3 = (
                float(np.percentile(col_data, 25)),
                float(np.percentile(col_data, 75)),
            )
            iqr = q3 - q1
            mean = float(col_data.mean())
            std = float(col_data.std(ddof=1)) or 1.0
            self._baseline_stats[col] = (q1, q3, iqr, mean, std)
        self._cols = cols

    def _score_iqr_zscore(self, X: np.ndarray, df: pd.DataFrame) -> AnomalyResult:
        n = X.shape[0]
        iqr_flags = np.zeros(n)
        z_flags = np.zeros(n)
        z_scores_all = []

        for i, col in enumerate(self._cols):
            if col not in self._baseline_stats:
                continue
            q1, q3, iqr, mean, std = self._baseline_stats[col]
            col_data = X[:, i]
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            iqr_flags += (col_data < lo) | (col_data > hi)
            z = np.abs((col_data - mean) / std)
            z_flags += z > 4.0
            z_scores_all.append(z)

        consensus = ((iqr_flags >= 1) | (z_flags >= 1)).astype(int)
        avg_z = (
            np.column_stack(z_scores_all).mean(axis=1) if z_scores_all else np.zeros(n)
        )
        flags = pd.Series(consensus, index=df.index, name="anomaly_flag")
        scores = pd.Series(avg_z, index=df.index, name="anomaly_score")
        return AnomalyResult(
            scores=scores,
            flags=flags,
            method="iqr_zscore_fallback",
            n_anomalies=int(consensus.sum()),
            contamination=self.contamination,
        )

    @staticmethod
    def _prepare(df: pd.DataFrame) -> np.ndarray:
        numeric = df.select_dtypes(include=[np.number])
        # pandas is Any under the mypy overrides; cast is a no-op.
        return cast(np.ndarray, numeric.fillna(numeric.mean()).values)


def detect_fat_finger(
    trade_sizes: pd.Series,
    multiplier: float = 10.0,
    min_samples: int = 20,
) -> pd.Series:
    """Flag trades where size is > multiplier × median of recent history.

    Returns boolean Series: True = suspected fat-finger.
    """
    if len(trade_sizes) < min_samples:
        return pd.Series(False, index=trade_sizes.index, name="fat_finger")
    median = trade_sizes.rolling(
        min(50, len(trade_sizes)), min_periods=min_samples
    ).median()
    return (trade_sizes > median * multiplier).rename("fat_finger")
