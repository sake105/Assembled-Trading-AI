"""MLflow integration for backtest certificate logging.

Soft-optional: requires mlflow>=2.0.0. Falls back to a no-op logger if not installed.

Usage:
    from assembled_core.certify.mlflow_integration import log_certificate_to_mlflow
    log_certificate_to_mlflow(cert, experiment_name="backtest_v4")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def log_certificate_to_mlflow(
    certificate: Any,
    experiment_name: str = "assembled_trading_backtests",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log a ReproducibilityCertificate to an MLflow run.

    Parameters
    ----------
    certificate:
        A ``ReproducibilityCertificate`` dataclass instance (from certify.schema).
    experiment_name:
        MLflow experiment to log into (created if it doesn't exist).
    run_name:
        Optional run name; defaults to ``certificate.certificate_id``.
    tracking_uri:
        MLflow tracking server URI. ``None`` uses local ``mlruns/`` directory.
    tags:
        Extra key-value tags logged to the run.

    Returns
    -------
    str | None
        The MLflow run_id, or ``None`` if mlflow is not installed.
    """
    if not _mlflow_available():
        logger.warning(
            "[MLflow] mlflow not installed — certificate not logged. pip install mlflow>=2.0.0"
        )
        return None

    import mlflow  # noqa: PLC0415

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    run_name = run_name or getattr(certificate, "certificate_id", "backtest_run")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # --- Params: environment fingerprint ---
        env = getattr(certificate, "environment", None)
        if env is not None:
            env_dict = env.__dict__ if hasattr(env, "__dict__") else dict(env)
            for k, v in env_dict.items():
                try:
                    mlflow.log_param(f"env.{k}", str(v)[:250])
                except Exception:
                    pass

        # --- Params: certificate metadata ---
        for attr in ("certificate_id", "created_at", "strategy_version"):
            val = getattr(certificate, attr, None)
            if val is not None:
                mlflow.log_param(attr, str(val)[:250])

        # --- Metrics: output fingerprint ---
        out = getattr(certificate, "output", None)
        if out is not None:
            _log_output_metrics(out)

        # --- Tags ---
        if tags:
            mlflow.set_tags(tags)

        git_info = getattr(certificate, "git_info", None)
        if git_info:
            git_dict = (
                git_info.__dict__ if hasattr(git_info, "__dict__") else dict(git_info)
            )
            mlflow.set_tags({f"git.{k}": str(v)[:250] for k, v in git_dict.items()})

        # --- Artifact: full certificate JSON ---
        cert_json = _certificate_to_json(certificate)
        if cert_json:
            import os
            import tempfile  # noqa: E401

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="certificate_"
            ) as f:
                f.write(cert_json)
                tmp_path = f.name
            try:
                mlflow.log_artifact(tmp_path, artifact_path="certificate")
            finally:
                os.unlink(tmp_path)

        logger.info(
            "[MLflow] Certificate logged — experiment=%s run_id=%s",
            experiment_name,
            run_id,
        )
        return run_id


def _log_output_metrics(output: Any) -> None:
    """Extract numeric metrics from OutputFingerprint and log to mlflow."""
    import mlflow  # noqa: PLC0415

    metric_attrs = (
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "n_trades",
        "win_rate",
        "avg_return_per_trade",
        "annualized_volatility",
    )
    out_dict = (
        output.__dict__
        if hasattr(output, "__dict__")
        else dict(output)
        if hasattr(output, "__iter__")
        else {}
    )
    if not out_dict:
        out_dict = {attr: getattr(output, attr, None) for attr in metric_attrs}

    for k, v in out_dict.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass


def _certificate_to_json(certificate: Any) -> str | None:
    """Serialize a certificate to JSON string (dataclass → dict → json)."""
    import dataclasses  # noqa: PLC0415

    try:
        if dataclasses.is_dataclass(certificate):
            d = dataclasses.asdict(certificate)
        else:
            d = (
                certificate.__dict__
                if hasattr(certificate, "__dict__")
                else str(certificate)
            )
        return json.dumps(d, default=str, indent=2)
    except Exception as exc:
        logger.debug("Certificate serialization failed: %s", exc)
        return None


def log_backtest_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    git_commit: str | None = None,
    experiment_name: str = "assembled_trading_backtests",
    run_name: str | None = None,
    artifact_paths: list[str] | None = None,
) -> str | None:
    """Lightweight mlflow logging for a backtest run without full certificate.

    Parameters
    ----------
    params:
        Strategy / config parameters (strings or scalars).
    metrics:
        Numeric performance metrics (Sharpe, CAGR, etc.).
    git_commit:
        Current git commit SHA for reproducibility tracking.
    artifact_paths:
        Local file paths to log as artifacts (e.g. equity curve CSV).

    Returns
    -------
    str | None
        MLflow run_id, or ``None`` if mlflow unavailable.
    """
    if not _mlflow_available():
        logger.warning("[MLflow] mlflow not installed — run not logged.")
        return None

    import mlflow  # noqa: PLC0415

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        for k, v in params.items():
            mlflow.log_param(k, str(v)[:250])

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        if git_commit:
            mlflow.set_tag("git.commit", git_commit[:40])

        for path in artifact_paths or []:
            if Path(path).exists():
                mlflow.log_artifact(path)

        logger.info("[MLflow] Backtest run logged — run_id=%s", run_id)
        return run_id
