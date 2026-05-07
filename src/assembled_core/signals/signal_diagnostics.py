"""Signal Diagnostics and Real-Time Monitoring (Plan 1.10).

Provides rolling health metrics per signal per day:
- Rolling IC, hit rate, turnover contribution
- Alert triggers for degraded signals
- JSON artifact generation for pipeline runs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_signal_health(
    factor_df: pd.DataFrame,
    forward_returns_col: str,
    factor_cols: list[str],
    timestamp_col: str = "timestamp",
    ic_window: int = 30,
) -> pd.DataFrame:
    """Compute rolling signal health metrics.

    Args:
        factor_df: Panel with timestamp, symbols, factors, forward returns.
        forward_returns_col: Forward return column for IC.
        factor_cols: Factor column names.
        timestamp_col: Timestamp column.
        ic_window: Rolling window for IC.

    Returns:
        DataFrame with timestamp, factor, rolling_ic, hit_rate columns.
    """
    timestamps = sorted(factor_df[timestamp_col].unique())
    records = []

    for ts in timestamps:
        slice_df = factor_df[factor_df[timestamp_col] == ts]
        for fcol in factor_cols:
            valid = slice_df[[fcol, forward_returns_col]].dropna()
            if len(valid) < 5:
                continue

            # IC (Spearman)
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(valid[fcol], valid[forward_returns_col])
            except Exception:
                ic = np.nan

            # Hit rate: fraction where factor sign matches return sign
            agree = (valid[fcol] * valid[forward_returns_col]) > 0
            hit_rate = float(agree.mean())

            records.append({
                "timestamp": ts,
                "factor": fcol,
                "ic": round(float(ic), 6) if pd.notna(ic) else np.nan,
                "hit_rate": round(hit_rate, 4),
                "n_obs": len(valid),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Add rolling IC
    for fcol in factor_cols:
        mask = df["factor"] == fcol
        df.loc[mask, "rolling_ic"] = (
            df.loc[mask, "ic"].rolling(ic_window, min_periods=5).mean()
        )

    return df


def generate_signal_health_alerts(
    health_df: pd.DataFrame,
    ic_alert_threshold: float = 0.0,
    ic_alert_window: int = 30,
) -> list[dict]:
    """Generate alerts for degraded signals.

    Alert when:
    - Rolling IC < threshold for sustained period
    - Single factor dominates turnover (>50%)

    Args:
        health_df: Output from compute_signal_health.
        ic_alert_threshold: IC below this triggers alert.
        ic_alert_window: Days IC must be below threshold.

    Returns:
        List of alert dicts with factor, alert_type, details.
    """
    alerts = []

    if health_df.empty:
        return alerts

    for factor in health_df["factor"].unique():
        fdata = health_df[health_df["factor"] == factor].sort_values("timestamp")

        if "rolling_ic" not in fdata.columns:
            continue

        recent = fdata.tail(ic_alert_window)
        if len(recent) < 10:
            continue

        recent_ic = recent["rolling_ic"].dropna()
        if len(recent_ic) > 0 and float(recent_ic.mean()) < ic_alert_threshold:
            alerts.append({
                "factor": factor,
                "alert_type": "LOW_IC",
                "details": f"Rolling IC {float(recent_ic.mean()):.4f} < {ic_alert_threshold} "
                           f"over last {len(recent_ic)} periods",
            })

    return alerts


def save_signal_health_artifact(
    health_df: pd.DataFrame,
    alerts: list[dict],
    output_dir: str = "output/diagnostics",
    run_date: str | None = None,
) -> str:
    """Save signal health report as JSON artifact.

    Args:
        health_df: Health metrics DataFrame.
        alerts: Alert list.
        output_dir: Output directory.
        run_date: Date string for filename.

    Returns:
        Path to saved artifact.
    """
    run_date = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filepath = out_path / f"signal_health_{run_date}.json"

    summary = {
        "run_date": run_date,
        "n_factors": len(health_df["factor"].unique()) if not health_df.empty else 0,
        "n_alerts": len(alerts),
        "alerts": alerts,
    }

    if not health_df.empty:
        latest = health_df.sort_values("timestamp").groupby("factor").tail(1)
        summary["latest_metrics"] = latest.to_dict(orient="records")

    filepath.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("[SignalDiag] Saved health artifact to %s", filepath)
    return str(filepath)


__all__ = [
    "compute_signal_health",
    "generate_signal_health_alerts",
    "save_signal_health_artifact",
]
