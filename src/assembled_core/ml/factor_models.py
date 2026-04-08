"""Machine Learning Models for Factor-Based Return Prediction.

This module provides ML models for predicting forward returns from factor panels.
It implements Phase E1 (ML Validation & Model Comparison) from the Advanced Analytics roadmap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import sklearn - raise clear error if not available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy classes for type hints
    LinearRegression = None  # type: ignore
    Ridge = None  # type: ignore
    Lasso = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore

# Optional gradient boosting imports
try:
    from xgboost import XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor  # type: ignore

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor  # type: ignore

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None  # type: ignore

_TREE_MODEL_TYPES = {"random_forest", "xgboost", "lightgbm", "catboost"}
_ALL_MODEL_TYPES = {"linear", "ridge", "lasso", "random_forest", "xgboost", "lightgbm", "catboost"}


@dataclass
class MLModelConfig:
    """Configuration for an ML model.

    Attributes:
        name: Model name (e.g., "ridge_20d", "random_forest_20d")
        model_type: Type of model ("linear", "ridge", "lasso", "random_forest")
        params: Model-specific hyperparameters (dict, default: empty dict)
    """

    name: str
    model_type: Literal["linear", "ridge", "lasso", "random_forest", "xgboost", "lightgbm", "catboost"]
    params: dict | None = None
    early_stopping_rounds: int | None = None

    def __post_init__(self) -> None:
        """Validate model type and ensure params is a dict."""
        if self.params is None:
            self.params = {}

        if self.model_type not in _ALL_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Must be one of: {sorted(_ALL_MODEL_TYPES)}"
            )


@dataclass
class MLExperimentConfig:
    """Configuration for an ML experiment.

    Attributes:
        label_col: Name of label column (e.g., "fwd_return_20d")
        feature_cols: List of feature column names (None = auto-detect factor_* columns)
        test_start: Optional start date for test set (None = use all data)
        test_end: Optional end date for test set (None = use all data)
        n_splits: Number of time-series CV splits (default: 5)
        train_size: Training window size in days (None = expanding window, int = rolling window)
        standardize: Whether to standardize features (default: True, only for linear models)
        min_train_samples: Minimum number of samples required for training (default: 100)
    """

    label_col: str
    feature_cols: list[str] | None = None
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None
    n_splits: int = 5
    train_size: int | None = None  # None = expanding, int = rolling
    standardize: bool = True
    min_train_samples: int = 100

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.min_train_samples < 10:
            raise ValueError("min_train_samples must be >= 10")


def detect_feature_cols(
    factor_panel_df: pd.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> list[str]:
    """
    Auto-detect feature columns in factor panel.

    Features are identified as columns that:
    - Start with "factor_" OR
    - Are common Alt-Data prefixes (returns_*, trend_*, rv_*, earnings_*, insider_*, news_*, macro_*)
    - Are NOT the label column, timestamp, or symbol

    Args:
        factor_panel_df: Factor panel DataFrame
        label_col: Name of label column (to exclude)
        timestamp_col: Name of timestamp column (to exclude, default: "timestamp")
        symbol_col: Name of symbol column (to exclude, default: "symbol")

    Returns:
        List of feature column names

    Example:
        >>> df = pd.DataFrame({
        ...     "timestamp": [...],
        ...     "symbol": [...],
        ...     "fwd_return_20d": [...],
        ...     "factor_returns_12m": [...],
        ...     "returns_12m": [...],
        ...     "close": [...],
        ... })
        >>> features = detect_feature_cols(df, label_col="fwd_return_20d")
        >>> # Returns: ["factor_returns_12m", "returns_12m"]
    """
    exclude_cols = {label_col, timestamp_col, symbol_col}

    # Common Alt-Data/Factor prefixes
    feature_prefixes = [
        "factor_",
        "returns_",
        "trend_",
        "rv_",
        "vov_",
        "earnings_",
        "insider_",
        "news_",
        "macro_",
        "momentum_",
        "reversal_",
    ]

    feature_cols = []

    for col in factor_panel_df.columns:
        if col in exclude_cols:
            continue

        # Check if column starts with any feature prefix
        if any(col.startswith(prefix) for prefix in feature_prefixes):
            feature_cols.append(col)

    logger.info(
        f"Auto-detected {len(feature_cols)} feature columns: {feature_cols[:5]}"
        + (f" ... ({len(feature_cols) - 5} more)" if len(feature_cols) > 5 else "")
    )

    return feature_cols


def prepare_ml_dataset(
    factor_panel_df: pd.DataFrame,
    experiment: MLExperimentConfig,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare ML dataset from factor panel.

    Filters time range, selects features & label column, handles missing values.

    Args:
        factor_panel_df: Factor panel DataFrame (panel format: timestamp, symbol, factor_*, fwd_return_*)
        experiment: MLExperimentConfig with label_col, feature_cols, test_start, test_end
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")

    Returns:
        Tuple of (X, y):
        - X: DataFrame with features (rows = samples, columns = features, index = original index)
        - y: Series with labels (same index as X)

    Raises:
        ValueError: If label column or required features are missing
        ValueError: If insufficient data after filtering
    """
    if factor_panel_df.empty:
        raise ValueError("factor_panel_df is empty")

    # Ensure timestamp is datetime
    if timestamp_col in factor_panel_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(factor_panel_df[timestamp_col]):
            factor_panel_df = factor_panel_df.copy()
            factor_panel_df[timestamp_col] = pd.to_datetime(
                factor_panel_df[timestamp_col], utc=True
            )

    # Check label column exists
    if experiment.label_col not in factor_panel_df.columns:
        raise ValueError(
            f"Label column '{experiment.label_col}' not found in factor_panel_df. "
            f"Available columns: {list(factor_panel_df.columns)}"
        )

    # Auto-detect features if not specified
    if experiment.feature_cols is None:
        feature_cols = detect_feature_cols(
            factor_panel_df, experiment.label_col, timestamp_col, symbol_col
        )
    else:
        feature_cols = experiment.feature_cols.copy()

    if not feature_cols:
        raise ValueError(
            "No feature columns found. Either specify feature_cols explicitly "
            "or ensure factor panel has columns starting with 'factor_' or common prefixes."
        )

    # Check that all specified feature columns exist
    missing_features = [
        col for col in feature_cols if col not in factor_panel_df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Missing feature columns: {', '.join(missing_features)}. "
            f"Available columns: {list(factor_panel_df.columns)}"
        )

    # Filter by time range (if specified)
    df_filtered = factor_panel_df.copy()

    if timestamp_col in df_filtered.columns:
        if experiment.test_start is not None:
            df_filtered = df_filtered[
                df_filtered[timestamp_col] >= experiment.test_start
            ]
        if experiment.test_end is not None:
            df_filtered = df_filtered[df_filtered[timestamp_col] <= experiment.test_end]

        # Sort by timestamp for time-series CV
        df_filtered = df_filtered.sort_values(timestamp_col).reset_index(drop=True)

    # Extract X and y
    # Drop rows with NaN in label (we cannot use these for training/testing)
    df_clean = df_filtered.dropna(subset=[experiment.label_col])

    # For features: we can optionally drop rows with too many missing features
    # For now: drop rows where ALL features are NaN (too few features)
    feature_mask = df_clean[feature_cols].notna().sum(axis=1) > 0
    df_clean = df_clean[feature_mask]

    if len(df_clean) == 0:
        raise ValueError(
            "No valid samples after filtering. Check for missing labels or features."
        )

    # Extract X and y
    X = df_clean[feature_cols].copy()
    y = df_clean[experiment.label_col].copy()

    # Note: X and y have the same index (from df_clean)
    # We could use MultiIndex (symbol, timestamp), but for simplicity, keep sequential index

    logger.info(
        f"Prepared dataset: {len(X)} samples, {len(feature_cols)} features, "
        f"label range: [{y.min():.4f}, {y.max():.4f}]"
    )

    return X, y


def _create_model(model_cfg: MLModelConfig):
    """Create sklearn model from config.

    Args:
        model_cfg: MLModelConfig

    Returns:
        sklearn model instance

    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If model_type is unsupported
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )

    model_type = model_cfg.model_type
    params = model_cfg.params or {}

    if model_type == "linear":
        return LinearRegression(**params)
    elif model_type == "ridge":
        return Ridge(**params)
    elif model_type == "lasso":
        return Lasso(**params)
    elif model_type == "random_forest":
        return RandomForestRegressor(**params)
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        xgb_params = {"tree_method": "hist", "n_jobs": -1, "random_state": 42}
        xgb_params.update(params)
        return XGBRegressor(**xgb_params)
    elif model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        lgbm_params = {"n_estimators": 500, "learning_rate": 0.05, "random_state": 42, "n_jobs": -1, "verbose": -1}
        lgbm_params.update(params)
        return LGBMRegressor(**lgbm_params)
    elif model_type == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost is not installed. Run: pip install catboost")
        cb_params = {"silent": True, "random_state": 42}
        cb_params.update(params)
        return CatBoostRegressor(**cb_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _split_time_series(
    timestamps: pd.Series,
    n_splits: int,
    train_size: int | None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Split time series into train/test splits.

    Args:
        timestamps: Series of timestamps (must be sorted)
        n_splits: Number of splits
        train_size: Training window size in days (None = expanding window, int = rolling window)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    if len(timestamps) < n_splits * 10:  # Minimum 10 samples per split
        raise ValueError(
            f"Insufficient data for {n_splits} splits. Need at least {n_splits * 10} samples, "
            f"got {len(timestamps)}"
        )

    # Get unique timestamps (sorted)
    unique_timestamps = timestamps.unique()
    unique_timestamps = pd.Series(unique_timestamps).sort_values()

    # Calculate test size (approximately equal splits)
    total_days = (unique_timestamps.max() - unique_timestamps.min()).days
    total_days // (n_splits + 1)  # Reserve some data for first train set

    splits = []

    for i in range(n_splits):
        # Calculate test window
        test_start_idx = len(unique_timestamps) * (i + 1) // (n_splits + 1)
        test_end_idx = len(unique_timestamps) * (i + 2) // (n_splits + 1)

        if test_end_idx <= test_start_idx:
            continue  # Skip if test window is too small

        test_timestamps = unique_timestamps.iloc[test_start_idx:test_end_idx]

        # Calculate train window
        if train_size is None:
            # Expanding window: use all data before test window
            train_timestamps = unique_timestamps.iloc[:test_start_idx]
        else:
            # Rolling window: use fixed-size window before test
            train_start_date = test_timestamps.min() - pd.Timedelta(days=train_size)
            train_mask = (unique_timestamps >= train_start_date) & (
                unique_timestamps < test_timestamps.min()
            )
            train_timestamps = unique_timestamps[train_mask]

        if len(train_timestamps) == 0 or len(test_timestamps) == 0:
            continue  # Skip if either window is empty

        # Convert to indices in original dataframe
        train_mask = timestamps.isin(train_timestamps)
        test_mask = timestamps.isin(test_timestamps)

        train_indices = timestamps[train_mask].index
        test_indices = timestamps[test_mask].index

        splits.append((train_indices, test_indices))

    return splits


def run_time_series_cv(
    factor_panel_df: pd.DataFrame,
    experiment: MLExperimentConfig,
    model_cfg: MLModelConfig,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> dict:
    """
    Run time-series cross-validation.

    Splits data into train/test windows (expanding or rolling) and trains/predicts for each split.

    Args:
        factor_panel_df: Factor panel DataFrame
        experiment: MLExperimentConfig
        model_cfg: MLModelConfig
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")

    Returns:
        Dictionary with:
        - predictions_df: DataFrame with columns: timestamp (if available), symbol (if available),
          y_true, y_pred, split_index
        - metrics_df: DataFrame with columns: split_index, mse, mae, r2, n_train_samples, n_test_samples
        - global_metrics: Dictionary with aggregated metrics across all splits
        - per_split_metrics: List of dictionaries, one per split

    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If insufficient data for CV splits
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )

    # Prepare dataset
    X, y = prepare_ml_dataset(factor_panel_df, experiment, timestamp_col, symbol_col)

    # Get timestamps for splitting (use index alignment)
    if timestamp_col in factor_panel_df.columns:
        # Reindex timestamps to match X/y index
        timestamps = factor_panel_df.loc[X.index, timestamp_col]
    else:
        # Fallback: use sequential indices
        timestamps = pd.Series(range(len(X)), index=X.index)
        timestamps = pd.to_datetime(timestamps, unit="D", origin="2000-01-01")

    # Create splits
    splits = _split_time_series(timestamps, experiment.n_splits, experiment.train_size)

    if len(splits) == 0:
        raise ValueError(
            "No valid splits created. Check data range and split configuration."
        )

    logger.info(f"Created {len(splits)} time-series splits")

    # Collect predictions and metrics
    all_predictions = []
    all_metrics = []
    last_model = None  # Store last trained model for feature importance analysis

    for split_idx, (train_indices, test_indices) in enumerate(splits):
        # Extract train/test sets
        X_train = X.loc[train_indices]
        y_train = y.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y.loc[test_indices]

        # Check minimum train samples
        if len(X_train) < experiment.min_train_samples:
            logger.warning(
                f"Split {split_idx}: Train set has only {len(X_train)} samples "
                f"(minimum: {experiment.min_train_samples}). Skipping."
            )
            continue

        # Drop rows with NaN features in train/test (imputation could be added later)
        train_valid = X_train.notna().all(axis=1)
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]

        test_valid = X_test.notna().all(axis=1)
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(
                f"Split {split_idx}: Empty train or test set after NaN removal. Skipping."
            )
            continue

        # Standardize features (only for linear models, not for tree-based)
        is_tree_model = model_cfg.model_type in _TREE_MODEL_TYPES
        scaler = None
        if experiment.standardize and not is_tree_model:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Handle remaining NaN (fill with 0 for linear models, median for tree models)
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            if is_tree_model:
                # Fill with median for tree models
                median_train = np.nanmedian(X_train_scaled, axis=0)
                X_train_scaled = np.nan_to_num(X_train_scaled, nan=median_train)
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=median_train)
            else:
                # Fill with 0 for linear models
                X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

        # Create and train model
        model = _create_model(model_cfg)
        model.fit(X_train_scaled, y_train.values)

        # Store last model for feature importance analysis
        last_model = model

        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Compute metrics on test set
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mse_test = float(mean_squared_error(y_test, y_pred_test))
        mae_test = float(mean_absolute_error(y_test, y_pred_test))
        r2_test = float(r2_score(y_test, y_pred_test))

        # Also compute train metrics for overfitting detection
        r2_train = float(r2_score(y_train, y_pred_train))

        # Store predictions
        pred_df = pd.DataFrame(
            {
                "y_true": y_test.values,
                "y_pred": y_pred_test,
                "split_index": split_idx,
            },
            index=y_test.index,
        )

        # Add timestamp and symbol if available
        if timestamp_col in factor_panel_df.columns:
            pred_df[timestamp_col] = factor_panel_df.loc[
                y_test.index, timestamp_col
            ].values
        if symbol_col in factor_panel_df.columns:
            pred_df[symbol_col] = factor_panel_df.loc[y_test.index, symbol_col].values

        all_predictions.append(pred_df)

        # Store metrics
        split_metrics = {
            "split_index": split_idx,
            "mse": mse_test,
            "mae": mae_test,
            "r2": r2_test,
            "r2_train": r2_train,
            "train_test_gap_r2": r2_train - r2_test,  # Overfitting indicator
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
        all_metrics.append(split_metrics)

    if len(all_predictions) == 0:
        raise ValueError(
            "No valid splits after processing. Check data quality and configuration."
        )

    # Combine predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Compute global metrics (average across splits)
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        global_metrics = {
            "mse": float(metrics_df["mse"].mean()),
            "mae": float(metrics_df["mae"].mean()),
            "r2": float(metrics_df["r2"].mean()),
            "r2_train": float(metrics_df["r2_train"].mean()),
            "train_test_gap_r2": float(metrics_df["train_test_gap_r2"].mean()),
            "n_splits": len(all_metrics),
            "avg_train_samples": float(metrics_df["n_train_samples"].mean()),
            "avg_test_samples": float(metrics_df["n_test_samples"].mean()),
        }
    else:
        metrics_df = pd.DataFrame()
        global_metrics = {}

    logger.info(
        f"CV completed: {len(all_metrics)} splits, "
        f"global R² = {global_metrics.get('r2', 0):.4f}, "
        f"train/test gap = {global_metrics.get('train_test_gap_r2', 0):.4f}"
    )

    return {
        "predictions_df": predictions_df,
        "metrics_df": metrics_df,
        "global_metrics": global_metrics,
        "per_split_metrics": all_metrics,
        "last_model": last_model,  # Last trained model (for feature importance)
        "feature_names": X.columns.tolist(),  # Feature names for feature importance
    }


def evaluate_ml_predictions(
    predictions_df: pd.DataFrame,
    horizon_days: int,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict:
    """
    Compute evaluation metrics from ML predictions.

    Metrics include:
    - Classical ML: MSE, MAE, R²
    - Factor-specific: IC, Rank-IC, IC-IR
    - Directional Accuracy
    - Long/Short Portfolio: Sharpe, Annualized Return

    Args:
        predictions_df: DataFrame with columns: y_true, y_pred (and optionally timestamp, symbol)
        horizon_days: Forward return horizon (for annualization)
        timestamp_col: Name of timestamp column (default: "timestamp")
        symbol_col: Name of symbol column (default: "symbol")
        y_true_col: Name of true labels column (default: "y_true")
        y_pred_col: Name of predicted labels column (default: "y_pred")

    Returns:
        Dictionary with metrics:
        - mse, mae, r2: float
        - ic_mean, ic_ir: float | None (if timestamp available for cross-sectional IC)
        - rank_ic_mean, rank_ic_ir: float | None
        - directional_accuracy: float
        - ls_return_mean, ls_return_std, ls_sharpe: float | None (if timestamp+symbol available)
        - n_samples: int
        - n_timestamps: int | None
    """
    if predictions_df.empty:
        return {
            "mse": None,
            "mae": None,
            "r2": None,
            "ic_mean": None,
            "ic_ir": None,
            "rank_ic_mean": None,
            "rank_ic_ir": None,
            "directional_accuracy": None,
            "ls_return_mean": None,
            "ls_return_std": None,
            "ls_sharpe": None,
            "n_samples": 0,
            "n_timestamps": None,
        }

    # Drop rows with NaN in y_true or y_pred
    pred_clean = predictions_df[[y_true_col, y_pred_col]].dropna()

    if len(pred_clean) == 0:
        raise ValueError("No valid predictions after dropping NaN values")

    y_true = pred_clean[y_true_col].values
    y_pred = pred_clean[y_pred_col].values

    # Classical ML metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Directional Accuracy
    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "n_samples": len(pred_clean),
    }

    # IC and Rank-IC (cross-sectional, per timestamp)
    if timestamp_col in predictions_df.columns:
        # Compute IC per timestamp
        ic_values = []
        rank_ic_values = []

        for timestamp in predictions_df[timestamp_col].unique():
            mask = predictions_df[timestamp_col] == timestamp
            ts_pred = predictions_df.loc[mask, [y_true_col, y_pred_col]].dropna()

            if len(ts_pred) < 2:
                continue  # Need at least 2 symbols for correlation

            y_true_ts = ts_pred[y_true_col].values
            y_pred_ts = ts_pred[y_pred_col].values

            # Pearson IC
            ic = float(np.corrcoef(y_pred_ts, y_true_ts)[0, 1])
            if not np.isnan(ic):
                ic_values.append(ic)

            # Spearman Rank-IC (using pandas rank correlation as fallback if scipy not available)
            try:
                from scipy.stats import spearmanr

                rank_ic, _ = spearmanr(y_pred_ts, y_true_ts)
                if not np.isnan(rank_ic):
                    rank_ic_values.append(float(rank_ic))
            except (ValueError, ImportError):
                # Fallback: use pandas rank correlation
                try:
                    ts_df = pd.DataFrame({"pred": y_pred_ts, "true": y_true_ts})
                    rank_ic = float(
                        ts_df["pred"].corr(ts_df["true"], method="spearman")
                    )
                    if not np.isnan(rank_ic):
                        rank_ic_values.append(rank_ic)
                except (ValueError, np.linalg.LinAlgError, TypeError):
                    pass  # Skip if correlation computation fails

        if ic_values:
            ic_mean = float(np.mean(ic_values))
            ic_std = float(np.std(ic_values))
            ic_ir = ic_mean / ic_std if ic_std > 0 else None
            metrics["ic_mean"] = ic_mean
            metrics["ic_ir"] = ic_ir
            metrics["n_timestamps"] = len(ic_values)
        else:
            metrics["ic_mean"] = None
            metrics["ic_ir"] = None
            metrics["n_timestamps"] = None

        if rank_ic_values:
            rank_ic_mean = float(np.mean(rank_ic_values))
            rank_ic_std = float(np.std(rank_ic_values))
            rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else None
            metrics["rank_ic_mean"] = rank_ic_mean
            metrics["rank_ic_ir"] = rank_ic_ir
        else:
            metrics["rank_ic_mean"] = None
            metrics["rank_ic_ir"] = None
    else:
        metrics["ic_mean"] = None
        metrics["ic_ir"] = None
        metrics["rank_ic_mean"] = None
        metrics["rank_ic_ir"] = None
        metrics["n_timestamps"] = None

    # Long/Short Portfolio metrics (if timestamp and symbol available)
    if timestamp_col in predictions_df.columns and symbol_col in predictions_df.columns:
        ls_returns = []

        for timestamp in predictions_df[timestamp_col].unique():
            mask = predictions_df[timestamp_col] == timestamp
            ts_pred = predictions_df.loc[
                mask, [symbol_col, y_true_col, y_pred_col]
            ].dropna()

            if len(ts_pred) < 10:  # Need at least 10 symbols for quintiles
                continue

            # Sort by y_pred and create quintiles
            ts_pred_sorted = ts_pred.sort_values(y_pred_col)
            n = len(ts_pred_sorted)
            ts_pred_sorted.iloc[: n // 5]  # Bottom 20%
            ts_pred_sorted.iloc[-n // 5 :]  # Top 20%

            # Average returns in bottom and top quintiles
            bottom_return = float(ts_pred_sorted[y_true_col].iloc[: n // 5].mean())
            top_return = float(ts_pred_sorted[y_true_col].iloc[-n // 5 :].mean())

            # Long/Short return (Top - Bottom)
            ls_return = top_return - bottom_return
            ls_returns.append(ls_return)

        if ls_returns:
            ls_return_mean = float(np.mean(ls_returns))
            ls_return_std = float(np.std(ls_returns))
            # Annualized Sharpe (assuming daily returns, scale by sqrt(252/horizon_days))
            periods_per_year = 252.0 / horizon_days
            ls_sharpe = (
                (ls_return_mean / ls_return_std * np.sqrt(periods_per_year))
                if ls_return_std > 0
                else None
            )
            metrics["ls_return_mean"] = ls_return_mean
            metrics["ls_return_std"] = ls_return_std
            metrics["ls_sharpe"] = float(ls_sharpe) if ls_sharpe is not None else None
        else:
            metrics["ls_return_mean"] = None
            metrics["ls_return_std"] = None
            metrics["ls_sharpe"] = None
    else:
        metrics["ls_return_mean"] = None
        metrics["ls_return_std"] = None
        metrics["ls_sharpe"] = None

    return metrics


# ---------------------------------------------------------------------------
# D4: Stacking ensemble entry point
# ---------------------------------------------------------------------------

def train_stacked_ensemble(
    panel_df: pd.DataFrame,
    experiment: "MLExperimentConfig",
    base_model_types: list[str] | None = None,
    meta_model_type: str = "ridge",
) -> dict:
    """Train a stacked ensemble (D4): multiple base learners + meta-learner.

    Trains several base models via time-series CV, then trains a Ridge
    meta-learner on their out-of-fold predictions.

    Args:
        panel_df: Factor panel DataFrame (timestamp, symbol, features, label).
        experiment: MLExperimentConfig controlling CV and label settings.
        base_model_types: List of base model types to use
                          (default: ["xgboost", "random_forest", "ridge"]).
        meta_model_type: Meta-learner model type (default: "ridge").

    Returns:
        Dict with keys:
            - "base_results": {model_type: run_time_series_cv result}
            - "stacked_predictions": DataFrame with ensemble predictions
            - "meta_model": Trained meta-learner model object
            - "global_metrics": Ensemble metrics (IC, R², Sharpe)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for stacking ensemble")

    if base_model_types is None:
        base_model_types = ["xgboost", "random_forest", "ridge"]

    base_results = {}
    oof_predictions: list[pd.DataFrame] = []

    for mtype in base_model_types:
        cfg = MLModelConfig(name=f"base_{mtype}", model_type=mtype)
        try:
            res = run_time_series_cv(panel_df, experiment, cfg)
            base_results[mtype] = res
            preds = res.get("predictions_df")
            if preds is not None and not preds.empty:
                preds = preds[["timestamp", "symbol", "y_true", "y_pred"]].copy()
                preds = preds.rename(columns={"y_pred": f"pred_{mtype}"})
                oof_predictions.append(preds)
            logger.info("[StackingEnsemble] Base model %s IC=%.4f", mtype,
                        res.get("global_metrics", {}).get("ic_mean", float("nan")))
        except Exception as e:
            logger.warning("[StackingEnsemble] Base model %s failed: %s", mtype, e)

    if not oof_predictions:
        logger.warning("[StackingEnsemble] No base models produced predictions")
        return {"base_results": base_results, "stacked_predictions": pd.DataFrame()}

    # Merge OOF predictions on (timestamp, symbol, y_true)
    merged = oof_predictions[0]
    for df in oof_predictions[1:]:
        drop_cols = ["y_true"] if "y_true" in df.columns else []
        merged = merged.merge(
            df.drop(columns=drop_cols),
            on=["timestamp", "symbol"],
            how="inner",
        )

    # Train meta-learner on OOF predictions
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    if not pred_cols or "y_true" not in merged.columns:
        return {"base_results": base_results, "stacked_predictions": merged}

    X_meta = merged[pred_cols].fillna(0.0).values
    y_meta = merged["y_true"].fillna(0.0).values

    if meta_model_type == "ridge":
        from sklearn.linear_model import Ridge as _Ridge
        meta = _Ridge(alpha=1.0)
    else:
        from sklearn.linear_model import LinearRegression as _LR
        meta = _LR()

    meta.fit(X_meta, y_meta)
    merged["y_pred_ensemble"] = meta.predict(X_meta)

    # Compute ensemble IC
    try:
        from scipy.stats import spearmanr  # type: ignore
        ic, _ = spearmanr(merged["y_true"], merged["y_pred_ensemble"])
    except Exception:
        ic = float("nan")

    logger.info("[StackingEnsemble] Ensemble trained: IC=%.4f using %d base models", ic, len(pred_cols))

    return {
        "base_results": base_results,
        "stacked_predictions": merged,
        "meta_model": meta,
        "global_metrics": {"ic_mean": ic, "n_base_models": len(pred_cols)},
    }


# ---------------------------------------------------------------------------
# D11: Optuna hyperopt entry point (proxy / convenience wrapper)
# ---------------------------------------------------------------------------

def train_with_hyperopt(
    panel_df: pd.DataFrame,
    experiment: "MLExperimentConfig",
    model_type: str = "xgboost",
    n_trials: int = 50,
    metric: str = "ic",
    timeout_seconds: int = 3600,
) -> dict:
    """D11: Train a model using Optuna hyperparameter optimisation.

    Thin wrapper around hyperopt.tune_model_optuna that returns a dict
    with the best model config and CV results.

    Args:
        panel_df: Factor panel DataFrame.
        experiment: MLExperimentConfig.
        model_type: Model type to tune (default: "xgboost").
        n_trials: Number of Optuna trials (default: 50).
        metric: Metric to optimise — "ic", "r2", "sharpe" (default: "ic").
        timeout_seconds: Max time per study (default: 3600).

    Returns:
        Dict with "best_config" (MLModelConfig) and "cv_results".
    """
    from src.assembled_core.ml.hyperopt import tune_model_optuna

    best_cfg = tune_model_optuna(
        panel_df=panel_df,
        experiment=experiment,
        model_type=model_type,
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        metric=metric,
    )
    # Run final CV with best config
    cv_results = run_time_series_cv(panel_df, experiment, best_cfg)
    logger.info(
        "[Hyperopt] Best %s config IC=%.4f",
        model_type, cv_results.get("global_metrics", {}).get("ic_mean", float("nan")),
    )
    return {"best_config": best_cfg, "cv_results": cv_results}


# ---------------------------------------------------------------------------
# Feature-Specific NaN Imputation (Plan 2.8)
# ---------------------------------------------------------------------------

FEATURE_NEUTRAL_LOOKUP: dict[str, object] = {
    "returns": 0.0,
    "rsi_14": 50.0,
    "macd": 0.0,
    "bollinger_pctb": 0.5,
    "volume": "rolling_median",
    "vix": "rolling_median",
    "atr": "rolling_median",
    "momentum": 0.0,
    "mf_score": 0.0,
}


def impute_features_smart(
    df: "pd.DataFrame",
    feature_cols: list[str] | None = None,
    add_nan_indicators: bool = True,
    lookup: dict[str, object] | None = None,
) -> "pd.DataFrame":
    """Feature-specific NaN imputation with optional NaN indicators.

    Uses domain-appropriate neutral values instead of blind mean/zero fill.
    Adds ``is_nan_{feature}`` binary columns when ``add_nan_indicators=True``
    so tree models can learn whether data absence is informative.

    Args:
        df: Feature DataFrame.
        feature_cols: Columns to impute. If None, uses all numeric columns.
        add_nan_indicators: Add binary is_nan columns (default True).
        lookup: Override neutral value lookup.

    Returns:
        DataFrame with imputed values and optional indicator columns.
    """
    import pandas as pd

    df = df.copy()
    lut = lookup or FEATURE_NEUTRAL_LOOKUP
    cols = feature_cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in cols:
        if col not in df.columns:
            continue
        has_nan = df[col].isna()
        if not has_nan.any():
            continue

        if add_nan_indicators:
            df[f"is_nan_{col}"] = has_nan.astype(int)

        neutral = lut.get(col)
        if neutral == "rolling_median":
            fill_val = df[col].rolling(20, min_periods=1).median()
            df[col] = df[col].fillna(fill_val).fillna(df[col].median()).fillna(0.0)
        elif neutral is not None:
            df[col] = df[col].fillna(float(neutral))
        else:
            df[col] = df[col].fillna(df[col].median()).fillna(0.0)

    return df
