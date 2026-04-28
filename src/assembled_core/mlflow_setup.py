"""MLflow experiment-tracking setup.

From 39_HYPERPARAMETER_GOVERNANCE.md §3.4.

Provides a one-call setup that reads MLFLOW_TRACKING_URI from the environment
and sets the default experiment name for this project.

Usage::

    from assembled_core.mlflow_setup import setup_mlflow

    setup_mlflow()
    with mlflow.start_run(run_name="news_clf_v3"):
        mlflow.log_params({"n_estimators": 200, "max_depth": 5})
        ...
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT_NAME = "assembled_trading_ai"


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> bool:
    """Configure MLflow tracking URI and experiment.

    Args:
        tracking_uri: Override for the tracking server URI.
            Falls back to ``MLFLOW_TRACKING_URI`` env var, then localhost.
        experiment_name: MLflow experiment name (default ``assembled_trading_ai``).

    Returns:
        ``True`` if MLflow was configured successfully, ``False`` if mlflow is
        not installed (non-fatal — the module degrades gracefully).
    """
    try:
        import mlflow  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("mlflow not installed — experiment tracking disabled.")
        return False

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.debug("MLflow configured: uri=%s experiment=%s", uri, experiment_name)
    return True
