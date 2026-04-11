"""Stress scenario replay runner (Sprint 3 / Plan C6).

Loads historical stress scenarios from configs/stress_scenarios.yaml, applies
each one to a synthetic or provided equity curve, and writes per-scenario
risk-delta reports under output/stress_reports/.

This is a lightweight runner, not a full backtest harness: it shocks an
equity curve and compares baseline vs shocked risk metrics. For a full
price-level replay against a strategy, wire this into run_backtest_strategy.

Example:
    python scripts/run_stress_replay.py
    python scripts/run_stress_replay.py --config configs/stress_scenarios.yaml \
        --output-dir output/stress_reports
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.scenario_engine import (  # noqa: E402
    load_scenarios_from_yaml,
    run_scenario_on_equity,
)


def build_synthetic_equity(
    start: str = "2007-01-01",
    end: str = "2025-01-01",
    start_equity: float = 1_000_000.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.18,
    seed: int = 42,
) -> pd.Series:
    """Build a synthetic daily equity curve spanning all scenario windows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B", tz="UTC")
    dt = 1.0 / 252.0
    rets = rng.normal(annual_drift * dt, annual_vol * np.sqrt(dt), size=len(dates))
    equity = start_equity * np.cumprod(1.0 + rets)
    return pd.Series(equity, index=dates, name="equity")


def _to_jsonable(obj):  # pragma: no cover - trivial
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    return str(obj)


def run_replay(
    config_path: str,
    output_dir: str,
    equity: pd.Series | None = None,
) -> list[dict]:
    """Run stress replay for every scenario in ``config_path``.

    Returns a list of summary dicts (one per scenario). Also writes per-scenario
    JSON files plus a combined summary.json under ``output_dir``.
    """
    scenarios = load_scenarios_from_yaml(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if equity is None:
        equity = build_synthetic_equity()

    summaries: list[dict] = []
    for scenario in scenarios:
        try:
            result = run_scenario_on_equity(equity, scenario, freq="1d")
            summary = {
                "name": scenario.name,
                "shock_type": scenario.shock_type,
                "shock_magnitude": scenario.shock_magnitude,
                "shock_start": (
                    scenario.shock_start.isoformat() if scenario.shock_start else None
                ),
                "shock_end": (
                    scenario.shock_end.isoformat() if scenario.shock_end else None
                ),
                "baseline_metrics": _to_jsonable(result.get("baseline_metrics", {})),
                "shocked_metrics": _to_jsonable(result.get("shocked_metrics", {})),
                "delta_metrics": _to_jsonable(result.get("delta_metrics", {})),
                "status": "ok",
            }
        except Exception as exc:  # noqa: BLE001 - replay must not crash the whole run
            summary = {
                "name": scenario.name,
                "shock_type": scenario.shock_type,
                "status": "error",
                "error": str(exc),
            }

        per_file = out_dir / f"{scenario.name}.json"
        per_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config_path,
        "scenario_count": len(summaries),
        "scenarios": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stress scenario replay")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "stress_scenarios.yaml"),
        help="Path to stress_scenarios.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "stress_reports"),
        help="Directory for per-scenario JSON reports",
    )
    args = parser.parse_args(argv)

    summaries = run_replay(args.config, args.output_dir)
    ok = sum(1 for s in summaries if s.get("status") == "ok")
    err = len(summaries) - ok
    print(f"[stress_replay] wrote {len(summaries)} reports to {args.output_dir} "
          f"(ok={ok}, errors={err})")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
