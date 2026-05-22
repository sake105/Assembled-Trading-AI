"""MLflow experiment tracking integration — scaffold.

Tier 3 infra item (audit C2-046): wires MLflow tracking into backtest and
paper-pilot runs so experiments are reproducible and comparable.

Activation:
    1. Install: pip install mlflow
    2. Start tracking server:
         mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db
    3. Set env var: MLFLOW_TRACKING_URI=http://localhost:5000
    4. Set env var: MLFLOW_EXPERIMENT_NAME=assembled_trading (default)

When mlflow is not installed or MLFLOW_TRACKING_URI is not set, all calls
are no-ops — the trading pipeline is never blocked by tracking infra.

References:
    - audit C2-046
    - MLflow documentation: https://mlflow.org/docs/latest/index.html
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore[import-untyped]

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

_DEFAULT_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "assembled_trading")


def _is_active() -> bool:
    """True when mlflow is installed AND MLFLOW_TRACKING_URI is configured."""
    return HAS_MLFLOW and bool(os.environ.get("MLFLOW_TRACKING_URI"))


@dataclass
class RunMetrics:
    """Metrics to log for a single backtest or pilot run."""

    sharpe: float | None = None
    cagr: float | None = None
    max_drawdown: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    n_trades: int | None = None
    win_rate: float | None = None
    avg_holding_days: float | None = None
    turnover: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class RunParams:
    """Hyperparameters / config to log."""

    strategy: str = "unknown"
    universe: str = "unknown"
    start_date: str = ""
    end_date: str = ""
    rebalance_freq: str = "daily"
    commission_bps: float = 0.0
    risk_aversion: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "strategy": self.strategy,
            "universe": self.universe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_freq": self.rebalance_freq,
            "commission_bps": self.commission_bps,
            "risk_aversion": self.risk_aversion,
        }
        d.update(self.extra)
        return d


@contextmanager
def tracking_run(
    run_name: str,
    params: RunParams | None = None,
    experiment_name: str = _DEFAULT_EXPERIMENT,
) -> Generator[Any, None, None]:
    """Context manager: wraps a backtest or pilot run with MLflow tracking.

    Usage::

        with tracking_run("backtest_mfv2_2025", params=RunParams(strategy="mfv2")) as run:
            result = run_backtest(...)
            log_metrics(RunMetrics(sharpe=result.sharpe, cagr=result.cagr), run_id=run.info.run_id)

    When mlflow is inactive, yields None and the body runs normally.
    """
    if not _is_active():
        logger.debug("[mlflow_tracking] MLflow not active — tracking skipped")
        yield None
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if params is not None:
            mlflow.log_params(params.to_dict())
        logger.info(
            "[mlflow_tracking] started run %s (id=%s)", run_name, run.info.run_id
        )
        yield run


def log_metrics(
    metrics: RunMetrics, run_id: str | None = None, step: int | None = None
) -> None:
    """Log RunMetrics to the active (or specified) MLflow run.

    No-op when mlflow is inactive.
    """
    if not _is_active():
        return
    kw: dict[str, Any] = {}
    if run_id:
        kw["run_id"] = run_id
    if step is not None:
        kw["step"] = step
    mlflow.log_metrics(metrics.to_dict(), **kw)


def log_equity_curve(
    equity: np.ndarray,
    dates: list[str] | None = None,
    artifact_name: str = "equity_curve.npy",
) -> None:
    """Save equity curve array as an MLflow artifact.

    No-op when mlflow is inactive.
    """
    if not _is_active():
        return
    import tempfile, pathlib  # noqa: E401

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / artifact_name
        np.save(p, equity)
        mlflow.log_artifact(str(p))


def log_model_params(model_name: str, params: dict[str, Any]) -> None:
    """Log a named model's hyperparameters.

    No-op when mlflow is inactive.
    """
    if not _is_active():
        return
    prefixed = {f"{model_name}.{k}": v for k, v in params.items()}
    mlflow.log_params(prefixed)


def get_best_run(
    experiment_name: str = _DEFAULT_EXPERIMENT,
    metric: str = "sharpe",
    ascending: bool = False,
) -> dict[str, Any] | None:
    """Retrieve the best run from an experiment by a metric.

    Returns None when mlflow is inactive or no runs exist.
    """
    if not _is_active():
        return None
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    if runs.empty:
        return None
    return runs.iloc[0].to_dict()


__all__ = [
    "HAS_MLFLOW",
    "RunMetrics",
    "RunParams",
    "tracking_run",
    "log_metrics",
    "log_equity_curve",
    "log_model_params",
    "get_best_run",
]
