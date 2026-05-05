from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QuantilePrediction:
    """Per-symbol quantile prediction result."""

    symbol: str
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float

    @property
    def confidence(self) -> float:
        """Width of central IQR as confidence proxy (always > 0)."""
        return max(float(self.q75 - self.q25), 1e-8)

    @property
    def asymmetry(self) -> float:
        """Upside/downside ratio (always > 0)."""
        upside = self.q95 - self.q50
        downside = self.q50 - self.q05
        return max(upside, 1e-8) / max(downside, 1e-8)

    @property
    def expected_direction(self) -> str:
        """Sign of median prediction (threshold = 0.001 to avoid noise bias)."""
        if self.q50 > 0.001:
            return "positive"
        if self.q50 < -0.001:
            return "negative"
        return "neutral"


def _fallback_quantiles(
    y_train: np.ndarray,
    X_predict: np.ndarray,
    quantiles_tuple: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Return empirical quantiles from y_train broadcast across all prediction rows."""
    n_pred = len(X_predict)
    result: dict[float, np.ndarray] = {}
    for q in quantiles_tuple:
        val = float(np.nanpercentile(y_train, q * 100)) if len(y_train) > 0 else 0.0
        result[q] = np.full(n_pred, val)
    return result


def predict_quantiles(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> list[QuantilePrediction]:
    """Return empirical quantile predictions per symbol from historical target_col values."""
    if df is None or df.empty or target_col not in df.columns:
        logger.debug("[SKIP] predict_quantiles: empty df or missing target_col=%s", target_col)
        return []

    if "symbol" not in df.columns:
        logger.debug("[SKIP] predict_quantiles: no 'symbol' column")
        return []

    results: list[QuantilePrediction] = []
    for sym, grp in df.groupby("symbol", sort=False):
        series = grp[target_col].dropna()
        if len(series) < 50:
            continue
        q05, q25, q50, q75, q95 = (
            float(np.nanpercentile(series, p)) for p in (5, 25, 50, 75, 95)
        )
        results.append(QuantilePrediction(
            symbol=str(sym),
            q05=q05, q25=q25, q50=q50, q75=q75, q95=q95,
        ))

    logger.debug("[OK] predict_quantiles: %d symbols processed", len(results))
    return results
