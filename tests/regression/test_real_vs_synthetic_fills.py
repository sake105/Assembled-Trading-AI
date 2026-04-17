"""E5 — Real-vs-synthetic fill calibration regression pins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.phase_realism]

from scripts.compare_real_vs_synthetic_fills import (  # noqa: E402
    build_calibration_report,
)


def _write_fills(tmp_path: Path, rows: list[dict]) -> Path:
    root = tmp_path / "executions"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "fills.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return root


def test_empty_dir_returns_zero_deltas(tmp_path: Path) -> None:
    report = build_calibration_report(tmp_path / "missing")
    assert report["fills_considered"] == 0
    assert report["deltas_computed"] == 0
    assert report["passes"] is False


def test_tight_fills_pass_gate(tmp_path: Path) -> None:
    fills = _write_fills(
        tmp_path,
        [
            {
                "order_id": f"o{i}",
                "symbol": "AAPL",
                "side": "buy",
                "arrival_price": 100.0,
                "fill_price": 100.0 + 0.001 * i,  # 0.1 bps per unit
                "synthetic_fill_price": 100.0 + 0.0005 * i,
            }
            for i in range(20)
        ],
    )
    report = build_calibration_report(fills, p95_threshold_bps=2.0)
    assert report["deltas_computed"] == 20
    assert report["p95_abs_bps"] <= 2.0
    assert report["passes"] is True


def test_wide_fills_fail_gate(tmp_path: Path) -> None:
    fills = _write_fills(
        tmp_path,
        [
            {
                "order_id": f"o{i}",
                "symbol": "XYZ",
                "arrival_price": 10.0,
                "fill_price": 10.0 + 0.05,  # 500 bps (huge)
                "synthetic_fill_price": 10.0,
            }
            for i in range(10)
        ],
    )
    report = build_calibration_report(fills, p95_threshold_bps=2.0)
    assert report["p95_abs_bps"] > 2.0
    assert report["passes"] is False


def test_missing_fields_are_skipped(tmp_path: Path) -> None:
    fills = _write_fills(
        tmp_path,
        [
            {"order_id": "ok", "arrival_price": 10.0, "fill_price": 10.01, "synthetic_fill_price": 10.005},
            {"order_id": "no_synth", "arrival_price": 10.0, "fill_price": 10.01},
            {"order_id": "no_arrival", "fill_price": 10.01, "synthetic_fill_price": 10.005},
        ],
    )
    report = build_calibration_report(fills)
    assert report["fills_considered"] == 3
    assert report["deltas_computed"] == 1
