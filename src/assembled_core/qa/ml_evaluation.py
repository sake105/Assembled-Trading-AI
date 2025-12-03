"""ML Model Evaluation Routines.

This module provides evaluation functions for meta-models, including:
- ROC/AUC calculation
- Brier Score
- Log Loss
- Calibration Curve plotting

Example:
    >>> from src.assembled_core.qa.ml_evaluation import evaluate_meta_model, plot_calibration_curve
    >>> import pandas as pd
    >>> 
    >>> # Load predictions
    >>> y_true = pd.Series([0, 1, 1, 0, 1])
    >>> y_prob = pd.Series([0.1, 0.9, 0.8, 0.2, 0.7])
    >>> 
    >>> # Evaluate
    >>> metrics = evaluate_meta_model(y_true, y_prob)
    >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    >>> print(f"Brier Score: {metrics['brier_score']:.4f}")
    >>> 
    >>> # Plot calibration curve
    >>> plot_calibration_curve(y_true, y_prob, "output/reports/meta/calibration.png")
"""
from __future__ import annotations

import logging
import pathlib
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import sklearn metrics
try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


def evaluate_meta_model(
    y_true: pd.Series,
    y_prob: pd.Series,
) -> Mapping[str, float]:
    """Evaluate meta-model predictions.
    
    Calculates key metrics for binary classification:
    - ROC-AUC: Area under the ROC curve (higher is better, max 1.0)
    - Brier Score: Mean squared error of probabilities (lower is better, min 0.0)
    - Log Loss: Logarithmic loss (lower is better, min 0.0)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (confidence scores ∈ [0, 1])
    
    Returns:
        Dictionary with metrics:
        - roc_auc: float (or NaN if insufficient data)
        - brier_score: float
        - log_loss: float (or NaN if insufficient data)
    
    Raises:
        ValueError: If inputs have different lengths or invalid values
        ImportError: If scikit-learn is not available
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for evaluation. "
            "Install with: pip install scikit-learn"
        )
    
    # Validate inputs
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true and y_prob must have same length. Got {len(y_true)} and {len(y_prob)}")
    
    # Convert to numpy arrays for sklearn
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    
    # Validate values
    if not np.all((y_true_arr == 0) | (y_true_arr == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    
    if not np.all((y_prob_arr >= 0) & (y_prob_arr <= 1)):
        raise ValueError("y_prob must contain values in [0, 1]")
    
    # Calculate metrics
    metrics: dict[str, float] = {}
    
    # ROC-AUC (requires at least one sample of each class)
    try:
        if len(np.unique(y_true_arr)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        else:
            logger.warning("Insufficient classes for ROC-AUC calculation")
            metrics["roc_auc"] = float("nan")
    except Exception as e:
        logger.warning(f"Failed to calculate ROC-AUC: {e}")
        metrics["roc_auc"] = float("nan")
    
    # Brier Score (always calculable)
    metrics["brier_score"] = float(brier_score_loss(y_true_arr, y_prob_arr))
    
    # Log Loss (requires at least one sample of each class)
    try:
        if len(np.unique(y_true_arr)) >= 2:
            metrics["log_loss"] = float(log_loss(y_true_arr, y_prob_arr))
        else:
            logger.warning("Insufficient classes for Log Loss calculation")
            metrics["log_loss"] = float("nan")
    except Exception as e:
        logger.warning(f"Failed to calculate Log Loss: {e}")
        metrics["log_loss"] = float("nan")
    
    return metrics


def plot_calibration_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    output_path: str | pathlib.Path,
    n_bins: int = 10,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """Plot calibration curve for meta-model predictions.
    
    A calibration curve shows how well-calibrated the predicted probabilities are.
    Perfect calibration means that if we predict 0.7, the actual success rate is 70%.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities (confidence scores ∈ [0, 1])
        output_path: Path to save the plot (PNG format)
        n_bins: Number of bins for calibration curve (default: 10)
        figsize: Figure size (width, height) in inches (default: (8, 6))
    
    Raises:
        ValueError: If inputs have different lengths or invalid values
        ImportError: If scikit-learn or matplotlib is not available
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for calibration curve. "
            "Install with: pip install scikit-learn"
        )
    
    # Import matplotlib only when needed (lazy import)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for calibration curve plotting. "
            "Install with: pip install matplotlib"
        )
    
    # Validate inputs
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true and y_prob must have same length. Got {len(y_true)} and {len(y_prob)}")
    
    # Convert to numpy arrays
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    
    # Validate values
    if not np.all((y_true_arr == 0) | (y_true_arr == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    
    if not np.all((y_prob_arr >= 0) & (y_prob_arr <= 1)):
        raise ValueError("y_prob must contain values in [0, 1]")
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_arr, y_prob_arr, n_bins=n_bins, strategy="uniform"
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Meta-Model", linewidth=2, markersize=8)
    
    # Plot perfect calibration line (diagonal)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1)
    
    # Formatting
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Save plot
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved calibration curve to {output_path}")

