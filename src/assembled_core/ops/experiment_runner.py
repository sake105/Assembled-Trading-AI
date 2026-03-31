"""OPS-7: A/B paper experiment runner — run paper range with policy overrides and write summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml  # type: ignore[import]

log = logging.getLogger(__name__)


__all__ = ["deep_merge_policy", "run_experiment"]


def deep_merge_policy(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep-merge overrides into base. Only override keys are changed; nested dicts merged recursively."""
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_policy(out[k], v)
        else:
            out[k] = v
    return out


def run_experiment(
    name: str,
    start_date: str,
    end_date: str,
    mode: str,
    output_root: str | Path,
    policy_overrides: Dict[str, Any],
    app_overrides: Dict[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Run paper range for [start_date, end_date] with policy and app overrides; write summary and snapshots.

    - experiment_root = output_root / "_experiments" / name
    - runs at experiment_root / "runs"
    - summary at experiment_root / "summary.json"
    - policy_snapshot at experiment_root / "policy_snapshot.yaml"
    - app_snapshot at experiment_root / "app_snapshot.yaml"
    Returns experiment_root.
    """
    from src.assembled_core.config.config import get_base_dir
    from src.assembled_core.config import policy_loader as pl
    from src.assembled_core.config.policy_loader import load_policy

    output_root = Path(output_root)
    repo_root = Path(root) if root is not None else get_base_dir()
    experiment_root = output_root / "_experiments" / name
    runs_root = experiment_root / "runs"
    experiment_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    base_policy = load_policy(repo_root / "configs" / "policy.yaml")
    merged_policy = deep_merge_policy(base_policy, policy_overrides)
    policy_snapshot_path = experiment_root / "policy_snapshot.yaml"
    with policy_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            merged_policy,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    app_cfg_path = repo_root / "configs" / "app.yaml"
    app_cfg = {}
    if app_cfg_path.exists():
        try:
            with app_cfg_path.open("r", encoding="utf-8") as f:
                app_cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    app_cfg = deep_merge_policy(app_cfg, app_overrides or {})
    app_snapshot_path = experiment_root / "app_snapshot.yaml"
    with app_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            app_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    original_load_policy = pl.load_policy
    pl.load_policy = lambda _path=None: merged_policy

    try:
        from src.assembled_core.data.prices_ingest import load_eod_prices
        from src.assembled_core.ops.paper_runner import run_paper_daily_one
        from src.assembled_core.ops.paper_summary import build_paper_summary

        prices = load_eod_prices(freq="1d")
    except Exception as e:
        log.error("Failed to load EOD prices: %s", e)
        pl.load_policy = original_load_policy
        raise
    if prices.empty or "timestamp" not in prices.columns:
        pl.load_policy = original_load_policy
        raise ValueError("No price data")

    try:
        ts = pd.to_datetime(prices["timestamp"], utc=True)
        dates_sorted = sorted(ts.dt.normalize().dt.date.unique())
        start_ts = pd.to_datetime(start_date, utc=True).normalize()
        end_ts = pd.to_datetime(end_date, utc=True).normalize()
        start_d = start_ts.date()
        end_d = end_ts.date()
        trading_dates = [d for d in dates_sorted if start_d <= d <= end_d]
        date_strs = [d.isoformat() for d in trading_dates]

        # Isolate paper ledger per experiment so baseline and treatment do not share state (A/B validity).
        ledger_dir = experiment_root / "_paper_ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        experiment_ledger_path = ledger_dir / "ledger_state.json"
        if "paper_runner" not in app_cfg or not isinstance(
            app_cfg["paper_runner"], dict
        ):
            app_cfg["paper_runner"] = {}
        app_cfg["paper_runner"] = dict(app_cfg["paper_runner"])
        app_cfg["paper_runner"]["ledger_path"] = str(experiment_ledger_path.resolve())
        # Experiment-isolated intel output dirs (fetch_state, dedupe_store, artifacts per experiment)
        pr = app_cfg["paper_runner"]
        if "intel" not in pr or not isinstance(pr["intel"], dict):
            pr["intel"] = {}
        pr["intel"] = dict(pr["intel"])
        pr["intel"]["news"] = dict(pr["intel"].get("news") or {})
        pr["intel"]["news"]["output_dir"] = str(
            (experiment_root / "intel" / "news").resolve()
        )
        pr["intel"]["disclosures"] = dict(pr["intel"].get("disclosures") or {})
        pr["intel"]["disclosures"]["output_dir"] = str(
            (experiment_root / "intel" / "disclosures").resolve()
        )

        import os

        _prev_run_id = os.environ.get("ASSEMBLED_RUN_ID")
        try:
            os.environ["ASSEMBLED_RUN_ID"] = f"exp_{name}"
            for i, d in enumerate(trading_dates):
                date_str = date_strs[i]
                day_ts = pd.Timestamp(d, tz="UTC")
                out_dir = runs_root / date_str
                out_dir.mkdir(parents=True, exist_ok=True)
                log.info(
                    "Experiment %s: run paper daily for %s (%d/%d)",
                    name,
                    date_str,
                    i + 1,
                    len(trading_dates),
                )
                exit_code, _ = run_paper_daily_one(
                    day_ts, out_dir, mode, app_cfg, prices, root=repo_root, day_index=i
                )
                if exit_code != 0:
                    log.warning(
                        "run_paper_daily failed for %s (exit_code=%d)",
                        date_str,
                        exit_code,
                    )
        finally:
            if _prev_run_id is not None:
                os.environ["ASSEMBLED_RUN_ID"] = _prev_run_id
            elif "ASSEMBLED_RUN_ID" in os.environ:
                os.environ.pop("ASSEMBLED_RUN_ID")

        summary = build_paper_summary(runs_root, date_strs)
        summary_path = experiment_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        log.info("Summary written to %s", summary_path)
        return experiment_root
    finally:
        pl.load_policy = original_load_policy
