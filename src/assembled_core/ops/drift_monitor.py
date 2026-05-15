"""Feature drift monitoring via Evidently + NannyML.

From 11_FREE_MODELLE.md §11.17.

Evidently: PSI/KS/JS/Wasserstein feature drift, HTML reports.
NannyML: performance estimation WITHOUT ground truth (CBPE, DLE).

Prometheus alert thresholds (from plan):
  PSI > 0.25 for 2d → 25% size reduction
  PSI > 0.35 for 1d → signal pause

Install: pip install evidently nannyml

Usage:
  monitor = DriftMonitor(reference_df)
  report = monitor.check_drift(current_df)
  if report.max_psi > 0.25:
      apply_size_multiplier(0.75)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Result of a drift check."""

    date: date
    feature_psi: dict[str, float] = field(default_factory=dict)
    max_psi: float = 0.0
    drifted_features: list[str] = field(default_factory=list)
    action: str = "none"  # 'none' | 'reduce_size' | 'pause'
    html_report_path: str | None = None


def _try_evidently():
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        return Report, DataDriftPreset
    except ImportError:
        logger.warning("evidently not installed — pip install evidently")
        return None, None


def _try_nannyml():
    try:
        import nannyml

        return nannyml
    except ImportError:
        logger.warning("nannyml not installed — pip install nannyml")
        return None


class DriftMonitor:
    """Daily feature drift monitor.

    Args:
        reference: Reference DataFrame (training period features).
        output_dir: Directory for HTML reports (optional).
        psi_warn_threshold: PSI above this → reduce size by 25%.
        psi_pause_threshold: PSI above this → pause signal.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        output_dir: str | Path | None = None,
        psi_warn_threshold: float = 0.25,
        psi_pause_threshold: float = 0.35,
    ):
        self.reference = reference
        self.output_dir = Path(output_dir) if output_dir else None
        self.psi_warn = psi_warn_threshold
        self.psi_pause = psi_pause_threshold

    def check_drift(
        self,
        current: pd.DataFrame,
        report_date: date | None = None,
        save_html: bool = False,
    ) -> DriftReport:
        """Run drift check between reference and current data.

        Args:
            current: Current period feature DataFrame.
            report_date: Date for the report (default today).
            save_html: Whether to save Evidently HTML report.

        Returns:
            DriftReport with PSI values and recommended action.
        """
        if report_date is None:
            # date.today() sweep: use UTC to avoid cross-platform local-tz drift.
            from datetime import datetime as _dt, timezone as _tz

            report_date = _dt.now(tz=_tz.utc).date()

        report = DriftReport(date=report_date)

        Report, DataDriftPreset = _try_evidently()
        if Report is None:
            logger.debug("Evidently unavailable — skipping drift check")
            return report

        try:
            ev_report = Report(metrics=[DataDriftPreset()])
            ev_report.run(reference_data=self.reference, current_data=current)

            result = ev_report.as_dict()
            metrics = result.get("metrics", [])

            # Extract per-feature PSI from Evidently output
            feature_psi: dict[str, float] = {}
            for metric in metrics:
                if metric.get("metric") == "DataDriftTable":
                    drift_results = metric.get("result", {}).get("drift_by_columns", {})
                    for col, col_data in drift_results.items():
                        # Evidently uses various drift scores; use stattest_threshold as proxy
                        stat_val = col_data.get("drift_score", 0.0)
                        feature_psi[col] = float(stat_val or 0.0)
                    break

            report.feature_psi = feature_psi
            if feature_psi:
                report.max_psi = max(feature_psi.values())
                report.drifted_features = [
                    k for k, v in feature_psi.items() if v > self.psi_warn
                ]

            # Determine action
            if report.max_psi > self.psi_pause:
                report.action = "pause"
            elif report.max_psi > self.psi_warn:
                report.action = "reduce_size"
            else:
                report.action = "none"

            logger.info(
                "Drift check %s: max_psi=%.3f, action=%s, drifted=%d features",
                report_date,
                report.max_psi,
                report.action,
                len(report.drifted_features),
            )

            # Save HTML report
            if save_html and self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / f"drift_report_{report_date}.html"
                ev_report.save_html(str(path))
                report.html_report_path = str(path)

        except Exception as exc:
            logger.warning("Evidently drift check failed: %s", exc)

        return report

    def size_multiplier(self, report: DriftReport) -> float:
        """Return position-size multiplier based on drift action.

        Returns:
          1.0 if no drift, 0.75 if reduce_size, 0.0 if pause.
        """
        return {"none": 1.0, "reduce_size": 0.75, "pause": 0.0}.get(report.action, 1.0)


def estimate_performance_without_labels(
    reference_with_targets: pd.DataFrame,
    current_without_targets: pd.DataFrame,
    target_col: str = "y",
    prediction_col: str = "y_pred",
) -> dict[str, float]:
    """NannyML CBPE: estimate model performance without ground truth.

    Useful for monitoring live model quality before labels arrive.

    Args:
        reference_with_targets: Training data with actual labels.
        current_without_targets: Current data, no labels yet.
        target_col: Name of the target column.
        prediction_col: Name of the prediction column.

    Returns:
        Dict with estimated performance metrics. Empty if NannyML unavailable.
    """
    nannyml = _try_nannyml()
    if nannyml is None:
        return {}

    if target_col not in reference_with_targets.columns:
        logger.debug("NannyML: target column %s not found in reference", target_col)
        return {}

    try:
        estimator = nannyml.CBPE(
            y_pred_proba=prediction_col,
            y_pred=prediction_col,
            y_true=target_col,
            timestamp_column_name=(
                "timestamp" if "timestamp" in reference_with_targets.columns else None
            ),
            metrics=["roc_auc", "f1"],
            chunk_size=50,
        )
        estimator.fit(reference_with_targets)
        results = estimator.estimate(current_without_targets)
        return {
            "estimated_roc_auc": float(
                results.filter(period="analysis").to_df()["estimated_roc_auc"].mean()
            ),
        }
    except Exception as exc:
        logger.debug("NannyML CBPE failed: %s", exc)
        return {}


__all__ = [
    "DriftReport",
    "DriftMonitor",
    "estimate_performance_without_labels",
]
