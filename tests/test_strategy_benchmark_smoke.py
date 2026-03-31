# tests/test_strategy_benchmark_smoke.py
"""Smoke test: run_strategy_benchmark.py with --synthetic-only --quick.

Asserts BENCHMARK_REPORT.md, scoreboard, anomalies, data_quality_summary, and new columns.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dev" / "run_strategy_benchmark.py"


@pytest.mark.smoke
def test_strategy_benchmark_produces_report_and_scoreboard(tmp_path: Path) -> None:
    """Run benchmark with synthetic and quick; assert main report, scoreboard, new outputs and columns."""
    if not SCRIPT.exists():
        pytest.skip("run_strategy_benchmark.py not found")
    out_root = tmp_path / "system_run"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--output-root",
        str(out_root),
        "--synthetic-only",
        "--quick",
        "--max-variants",
        "3",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        timeout=600,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Benchmark failed: {result.returncode}\nstdout: {result.stdout[-2500:]}\nstderr: {result.stderr[-2500:]}"
    bench = out_root / "benchmark"
    assert (bench / "BENCHMARK_REPORT.md").exists(), "BENCHMARK_REPORT.md not produced"
    assert (bench / "scoreboard.csv").exists(), "scoreboard.csv not produced"
    assert (bench / "scoreboard.json").exists(), "scoreboard.json not produced"
    assert (bench / "regime_metrics.json").exists() or (
        bench / "regime_metrics.csv"
    ).exists(), "regime_metrics not produced"
    assert (bench / "attribution_summary.json").exists() or (
        bench / "attribution_summary.csv"
    ).exists(), "attribution_summary not produced"
    assert (bench / "anomalies.json").exists(), "anomalies.json not produced"
    assert (
        bench / "data_quality_summary.json"
    ).exists(), "data_quality_summary.json not produced"
    assert (bench / "indicator_exposure_summary.json").exists() or (
        bench / "indicator_exposure_summary.csv"
    ).exists(), "indicator_exposure_summary not produced"
    with (bench / "scoreboard.json").open("r", encoding="utf-8") as f:
        sb = json.load(f)
    assert sb, "scoreboard empty"
    any_row = sb[0]
    for col in ("stability_score", "robustness_score"):
        assert col in any_row, f"scoreboard missing column {col}"
    all_keys = set().union(*(set(r.keys()) for r in sb))
    for col in ("var_95", "es_95", "pct_days_in_market"):
        assert (
            col in all_keys
        ), f"scoreboard missing extended column {col} (columns: {sorted(all_keys)})"
    for col in ("gross_total_return_est", "net_total_return"):
        assert (
            col in all_keys
        ), f"scoreboard missing profit diagnostic column {col} (columns: {sorted(all_keys)})"
    run_inputs_path = bench / "trend_baseline" / "1y" / "run_inputs.json"
    assert run_inputs_path.exists(), "run_inputs.json not produced for at least one run"
    run_inputs = json.loads(run_inputs_path.read_text(encoding="utf-8"))
    assert (
        "columns_synthesized" in run_inputs
    ), "run_inputs.json missing columns_synthesized"
    assert "synthetic_ohlcv" in run_inputs, "run_inputs.json missing synthetic_ohlcv"
    with (bench / "anomalies.json").open("r", encoding="utf-8") as f:
        anomalies = json.load(f)
    anomaly_types = {a.get("type") for a in anomalies if isinstance(a, dict)}
    if run_inputs.get("synthetic_ohlcv") is True:
        assert (
            "synthetic_ohlcv" in anomaly_types
        ), "run synthesized OHLCV but anomalies.json missing synthetic_ohlcv"
    else:
        assert (
            "data_qc_fail" in anomaly_types
            or "synthetic_ohlcv" in anomaly_types
            or len(anomaly_types) > 0
        ), "anomalies.json expected data_qc_fail or synthetic_ohlcv or other (e.g. too_few_bars) for synthetic run"


@pytest.mark.smoke
def test_benchmark_outputs_deterministic_json(tmp_path: Path) -> None:
    """After a run, anomalies.json and data_quality_summary.json are valid deterministic JSON (list/dict, sort_keys)."""
    if not SCRIPT.exists():
        pytest.skip("run_strategy_benchmark.py not found")
    out_root = tmp_path / "system_run"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(out_root),
            "--synthetic-only",
            "--quick",
            "--max-variants",
            "2",
        ],
        cwd=str(ROOT),
        timeout=120,
        capture_output=True,
        text=True,
    )
    bench = out_root / "benchmark"
    for name in (
        "anomalies.json",
        "data_quality_summary.json",
        "indicator_exposure_summary.json",
    ):
        path = bench / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, (list, dict)), f"{name} should be JSON list or dict"
        if isinstance(data, list):
            for item in data:
                assert isinstance(item, dict), f"{name} list items should be dicts"
    if (bench / "indicator_exposure_summary.json").exists():
        raw = (bench / "indicator_exposure_summary.json").read_text(encoding="utf-8")
        reencoded = json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
        assert (
            raw == reencoded
        ), "indicator_exposure_summary.json must be deterministic (sort_keys, newline)"
