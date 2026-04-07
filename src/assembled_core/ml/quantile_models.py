"""Quantile Regression for prediction intervals (Plan 2.4).

Point forecasts suggest false precision.  Quantile regression produces
the full predictive distribution: ``(q05, q25, q50, q75, q95)``.

Position sizing integration:
- ``confidence = 1 / (q95 - q05)`` — narrow CI → higher conviction
- Asymmetry signal: if q50 > 0 but q05 << 0 → positive expectation
  but high downside → smaller position despite positive signal

Uses LightGBM with ``objective='quantile'`` when available, falls back
to numpy percentile-based estimates.

Requires: ``lightgbm`` Python package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


@dataclass
class QuantilePrediction:
    """Quantile prediction for a single symbol."""

    symbol: str
    q05: float
    q25: float
    q50: float  # median forecast
    q75: float
    q95: float

    @property
    def confidence(self) -> float:
        """Prediction interval width → inverse confidence."""
        width = self.q95 - self.q05
        return 1.0 / max(width, 1e-6)

    @property
    def asymmetry(self) -> float:
        """Downside risk relative to upside.  >1 = more downside risk."""
        upside = self.q95 - self.q50
        downside = self.q50 - self.q05
        if upside < 1e-10:
            return float("inf")
        return downside / upside

    @property
    def expected_direction(self) -> str:
        """Positive/negative/neutral based on median."""
        if self.q50 > 0.001:
            return "positive"
        elif self.q50 < -0.001:
            return "negative"
        return "neutral"


def fit_quantile_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict[float, np.ndarray]:
    """Fit LightGBM quantile regression models and predict.

    Trains one model per quantile level.

    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training targets (n_samples,).
        X_predict: Features to predict on.
        quantiles: Quantile levels to model.
        n_estimators: Number of boosting rounds.
        max_depth: Max tree depth.
        learning_rate: Learning rate.
        seed: Random seed.

    Returns:
        Dict mapping quantile → prediction array.
    """
    if not LGB_AVAILABLE:
        logger.debug("[Quantile] lightgbm not available — using fallback")
        return _fallback_quantiles(y_train, X_predict, quantiles)

    predictions: dict[float, np.ndarray] = {}

    for q in quantiles:
        try:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=seed,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_predict)
            predictions[q] = preds
        except Exception as exc:
            logger.warning("[Quantile] Failed for q=%.2f: %s", q, exc)
            predictions[q] = np.full(len(X_predict), np.quantile(y_train, q))

    return predictions


def _fallback_quantiles(
    y_train: np.ndarray,
    X_predict: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Fallback: historical quantiles when LightGBM unavailable."""
    n = len(X_predict)
    return {q: np.full(n, float(np.quantile(y_train, q))) for q in quantiles}


def predict_quantiles(
    features_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    *,
    train_fraction: float = 0.8,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> list[QuantilePrediction]:
    """Predict quantiles for the test portion of the data.

    Simple train/test split for cross-sectional quantile prediction.

    Args:
        features_df: Panel DataFrame with features and target.
        target_col: Target column (e.g., forward returns).
        feature_cols: Feature column names.
        train_fraction: Fraction of data for training.
        quantiles: Quantile levels.

    Returns:
        List of QuantilePrediction (one per test-set row).
    """
    clean = features_df[feature_cols + [target_col]].dropna()
    if len(clean) < 50:
        return []

    n_train = int(len(clean) * train_fraction)
    train = clean.iloc[:n_train]
    test = clean.iloc[n_train:]

    if len(test) == 0:
        return []

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values

    preds = fit_quantile_lgbm(X_train, y_train, X_test, quantiles=quantiles)

    results = []
    symbols = features_df.get("symbol", pd.Series(range(len(test))))
    test_symbols = symbols.iloc[n_train:n_train + len(test)].values if "symbol" in features_df.columns else [f"row_{i}" for i in range(len(test))]

    for i in range(len(test)):
        sym = str(test_symbols[i]) if i < len(test_symbols) else f"row_{i}"
        results.append(QuantilePrediction(
            symbol=sym,
            q05=round(float(preds.get(0.05, np.zeros(1))[i]), 6),
            q25=round(float(preds.get(0.25, np.zeros(1))[i]), 6),
            q50=round(float(preds.get(0.50, np.zeros(1))[i]), 6),
            q75=round(float(preds.get(0.75, np.zeros(1))[i]), 6),
            q95=round(float(preds.get(0.95, np.zeros(1))[i]), 6),
        ))

    return results


__all__ = [
    "LGB_AVAILABLE",
    "QuantilePrediction",
    "fit_quantile_lgbm",
    "predict_quantiles",
]
