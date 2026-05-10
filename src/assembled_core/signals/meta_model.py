"""Meta-Model for predicting setup success probability.

This module provides functionality to train, save, load, and evaluate meta-models
that predict the success probability (confidence_score ∈ [0,1]) of trading setups.

Example:
    >>> from src.assembled_core.signals.meta_model import train_meta_model, save_meta_model, load_meta_model
    >>> import pandas as pd
    >>>
    >>> # Load ML dataset
    >>> df = pd.read_parquet("output/ml_datasets/trend_baseline_1d.parquet")
    >>>
    >>> # Train model
    >>> model = train_meta_model(df, model_type="gradient_boosting")
    >>>
    >>> # Save model
    >>> save_meta_model(model, "models/meta/trend_baseline_meta.joblib")
    >>>
    >>> # Load model
    >>> loaded_model = load_meta_model("models/meta/trend_baseline_meta.joblib")
    >>>
    >>> # Predict confidence scores
    >>> X = df[model.feature_names]
    >>> confidence_scores = loaded_model.predict_proba(X)
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import joblib (optional, for model saving/loading)
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Install with: pip install joblib")

# Try to import sklearn, fallback to None if not available
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class MetaModel:
    """Wrapper class for meta-model with feature names and label information.

    Attributes:
        model: The trained model (e.g., sklearn classifier)
        feature_names: List of feature column names used for training
        label_name: Name of the label column (default: "label")
    """

    model: Any
    feature_names: list[str]
    label_name: str = "label"

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Predict confidence scores (probability of success) for each row.

        Args:
            X: DataFrame with feature columns matching feature_names

        Returns:
            Series with confidence_score ∈ [0, 1] for each row

        Raises:
            ValueError: If required feature columns are missing
        """
        # Validate feature columns
        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")

        # Extract features in correct order
        X_features = X[self.feature_names].copy()

        # Handle NaN values (fill with 0 for now, could be improved)
        if X_features.isnull().any().any():
            logger.warning("NaN values found in features, filling with 0")
            X_features = X_features.fillna(0)

        # Predict probabilities
        # For binary classification, predict_proba returns shape (n_samples, 2)
        # We want the probability of class 1 (success)
        proba = self.model.predict_proba(X_features)

        # Extract probability of positive class (class 1)
        if proba.shape[1] == 2:
            confidence_scores = pd.Series(
                proba[:, 1], index=X.index, name="confidence_score"
            )
        else:
            # Multi-class case (shouldn't happen with binary classification, but handle gracefully)
            confidence_scores = pd.Series(
                proba[:, -1], index=X.index, name="confidence_score"
            )

        # Ensure values are in [0, 1]
        confidence_scores = confidence_scores.clip(0.0, 1.0)

        return confidence_scores

    def predict_proba_with_meta(
        self,
        X: pd.DataFrame,
        meta_labeler: object | None = None,
    ) -> pd.Series:
        """Konfidenz-Scores mit optionalem Meta-Labeling.

        Wenn meta_labeler=None → identisch zu predict_proba (keine Änderung).
        Wenn meta_labeler gesetzt → output = primary_score × meta_confidence.
        Bestehende Aufrufer von predict_proba() bleiben unverändert.
        """
        primary_scores = self.predict_proba(X)

        if meta_labeler is None:
            return primary_scores

        try:
            meta_conf = meta_labeler.predict_confidence(X)  # type: ignore[union-attr]
            scaled = primary_scores * meta_conf
            logger.debug(
                "[MetaModel] Meta-Scaling: mean_primary=%.3f → mean_scaled=%.3f",
                float(primary_scores.mean()),
                float(scaled.mean()),
            )
            return scaled
        except Exception as exc:
            logger.warning(
                "[MetaModel] Meta-Labeling fehlgeschlagen: %s — Primary-Scores unverändert",
                exc,
            )
            return primary_scores

    def predict_with_intervals(
        self,
        X: pd.DataFrame,
        X_calib: pd.DataFrame | None = None,
        y_calib: pd.Series | None = None,
        alpha: float = 0.1,
    ) -> dict:
        """Predictions mit Conformal-Intervallen.

        Args:
            X: Inference-Features
            X_calib: Kalibrierungs-Features (zeitlich vor X)
            y_calib: Kalibrierungs-Labels
            alpha: Miscoverage-Level (0.1 = 90%-Intervall)

        Returns:
            dict mit 'predictions', 'lower', 'upper', 'confidence', 'half_width'.
            Ohne X_calib: nur 'predictions' + 'confidence'=1.0.
        """
        primary = self.predict_proba(X)

        if X_calib is None or y_calib is None:
            return {
                "predictions": primary,
                "lower": primary.copy(),
                "upper": primary.copy(),
                "confidence": pd.Series(
                    np.ones(len(primary)), index=primary.index, name="confidence"
                ),
                "half_width": 0.0,
            }

        try:
            from src.assembled_core.ml.conformal import ConformalResult

            # Direkt residuals auf Calib-Set rechnen — MetaModel ist bereits
            # trainiert, daher kein Wrapper/Re-Fit nötig.
            calib_preds = self.predict_proba(X_calib).values
            residuals = np.abs(y_calib.values - calib_preds)
            n = len(residuals)
            q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            q = float(np.quantile(residuals, q_level))

            preds_vals = primary.values
            result = ConformalResult(
                point_predictions=pd.Series(
                    preds_vals, index=X.index, name="prediction"
                ),
                lower_bounds=pd.Series(preds_vals - q, index=X.index, name="lower"),
                upper_bounds=pd.Series(preds_vals + q, index=X.index, name="upper"),
                half_width=q,
                alpha=alpha,
            )
            return {
                "predictions": result.point_predictions,
                "lower": result.lower_bounds,
                "upper": result.upper_bounds,
                "confidence": result.confidence(),
                "half_width": result.half_width,
            }
        except Exception as exc:
            logger.warning(
                "[MetaModel] Conformal failed: %s — returning point predictions", exc
            )
            return {
                "predictions": primary,
                "lower": primary.copy(),
                "upper": primary.copy(),
                "confidence": pd.Series(np.ones(len(primary)), index=primary.index),
                "half_width": 0.0,
            }

    def explain_prediction_lime(
        self,
        X_row: pd.Series | dict,
        training_data: pd.DataFrame | None = None,
        num_features: int = 10,
    ) -> dict:
        """LIME Local-Explanation für eine einzelne Prediction (Round 7K).

        Args:
            X_row: Feature-Werte der zu erklärenden Instance
            training_data: Baseline-Distribution (für LIME-Init).
                           Falls None → Permutation-Fallback.
            num_features: Top-N Features im Output

        Returns:
            dict mit top_features + prediction_value + source.
        """
        try:
            from src.assembled_core.ml.lime_explainer import LIMEExplainerWrapper
        except ImportError:
            return {"error": "lime_explainer module fehlt"}
        try:
            wrapper = LIMEExplainerWrapper(
                model=self.model,
                feature_names=list(self.feature_names),
                training_data=training_data,
                mode=(
                    "classification"
                    if hasattr(self.model, "predict_proba")
                    else "regression"
                ),
            )
            expl = wrapper.explain(X_row, num_features=num_features)
            return {
                "top_features": expl.top_features(num_features),
                "predicted_value": expl.predicted_value,
                "source": expl.source,
            }
        except Exception as exc:
            logger.warning("[MetaModel] LIME failed: %s", exc)
            return {"error": str(exc)}


def train_meta_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    label_col: str = "label",
    model_type: str = "gradient_boosting",
    random_state: int = 42,
    test_size: float = 0.2,
) -> MetaModel:
    """Train a meta-model on the provided ML dataset.

    Args:
        df: DataFrame with ML dataset (must contain label column and feature columns)
        feature_cols: Optional list of feature column names. If None, auto-detects
            feature columns (excludes timestamp, symbol, label, and metadata columns).
        label_col: Name of the label column (default: "label")
        model_type: Type of model to train:
            - "gradient_boosting": GradientBoostingClassifier (default)
            - "random_forest": RandomForestClassifier
        random_state: Random seed for reproducibility (default: 42)
        test_size: Fraction of data to use for testing (default: 0.2, not used for training but for validation)

    Returns:
        MetaModel instance with trained model and feature names

    Raises:
        ValueError: If required columns are missing or model_type is unsupported
        ImportError: If scikit-learn is not available
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for meta-model training. "
            "Install with: pip install scikit-learn"
        )

    # Validate label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Auto-detect feature columns if not provided
    if feature_cols is None:
        # Exclude standard metadata columns
        exclude_cols = {
            "timestamp",
            "symbol",
            "date",
            label_col,
            "entry_price",
            "exit_price",
            "realized_return",
            "horizon_days",
            "open_time",
            "close_time",
            "open_price",
            "close_price",
            "pnl_pct",
        }
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not feature_cols:
        raise ValueError(
            "No feature columns found. Provide feature_cols explicitly or ensure DataFrame has numeric columns."
        )

    logger.info(f"Training meta-model with {len(feature_cols)} features")
    logger.info(
        f"Feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}"
    )

    # Sort chronologically so train/val split is temporally ordered.
    sort_col = next((c for c in ("timestamp", "date") if c in df.columns), None)
    if sort_col is not None:
        df = df.sort_values(sort_col, na_position="last").reset_index(drop=True)
        non_nat = df[sort_col].dropna()
        if not non_nat.is_monotonic_increasing:
            raise RuntimeError(
                f"DataFrame not sorted by {sort_col} after sort_values()"
            )

    # Extract features and labels
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Handle NaN values
    if X.isnull().any().any():
        logger.warning("NaN values found in features, filling with 0")
        X = X.fillna(0)

    if y.isnull().any():
        logger.warning("NaN values found in labels, dropping rows")
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]

    if X.empty:
        raise ValueError("No valid training data after removing NaN values")

    # Validate label distribution
    label_counts = y.value_counts()
    logger.info(f"Label distribution: {label_counts.to_dict()}")

    if len(label_counts) < 2:
        raise ValueError(
            f"Need at least 2 classes for binary classification. Found: {label_counts.to_dict()}"
        )

    # Chronological validation split — evaluate OOS before training on full data
    X_train = X
    y_train = y
    if 0.0 < test_size < 1.0 and len(X) >= 10:
        split_idx = int(len(X) * (1.0 - test_size))
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
        X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
        if len(X_tr) >= 2 and len(y_tr.unique()) >= 2:
            try:
                val_model_type = model_type  # same type for quick val run
                if val_model_type == "gradient_boosting":
                    from sklearn.ensemble import GradientBoostingClassifier as _GBC

                    _vm = _GBC(n_estimators=50, max_depth=3, random_state=random_state)
                else:
                    from sklearn.ensemble import RandomForestClassifier as _RFC

                    _vm = _RFC(
                        n_estimators=50,
                        max_depth=5,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                _vm.fit(X_tr, y_tr)
                val_acc = float((_vm.predict(X_val) == y_val).mean())
                logger.info(
                    "[MetaModel] OOS validation: n_train=%d n_val=%d accuracy=%.4f",
                    len(X_tr),
                    len(X_val),
                    val_acc,
                )
            except Exception as exc:
                logger.warning("[MetaModel] Validation split failed: %s", exc)

    # Initialize model
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
            verbose=0,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported: 'gradient_boosting', 'random_forest'"
        )

    # Train model
    logger.info(f"Training {model_type} model on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Create MetaModel wrapper
    meta_model = MetaModel(
        model=model,
        feature_names=list(feature_cols),
        label_name=label_col,
    )

    return meta_model


def save_meta_model(meta_model: MetaModel, path: str | pathlib.Path) -> None:
    """Save a MetaModel to disk.

    Args:
        meta_model: MetaModel instance to save
        path: Path to save the model (will create parent directories if needed)

    Raises:
        IOError: If file cannot be written
        ImportError: If joblib is not available
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required for saving meta-models. "
            "Install with: pip install joblib"
        )

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save using joblib (standard for sklearn models)
    joblib.dump(meta_model, path)

    logger.info(f"Saved meta-model to {path}")


def load_meta_model(path: str | pathlib.Path) -> MetaModel:
    """Load a MetaModel from disk.

    Args:
        path: Path to the saved model file

    Returns:
        Loaded MetaModel instance

    Raises:
        FileNotFoundError: If model file does not exist
        IOError: If file cannot be read
        ImportError: If joblib is not available
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required for loading meta-models. "
            "Install with: pip install joblib"
        )

    path = pathlib.Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        try:
            from src.assembled_core.ml.model_registry import verify_model_hash

            if not verify_model_hash(path):
                logger.warning(
                    "[meta_model] Hash mismatch for %s — loading anyway (non-strict)",
                    path.name,
                )
        except Exception:
            pass
        meta_model = joblib.load(path)
    except (EOFError, Exception) as exc:
        raise RuntimeError(
            f"[meta_model] Failed to load model from {path}: {exc}"
        ) from exc

    logger.info(f"Loaded meta-model from {path}")

    return meta_model
