"""Phase 9.5 tests for the offline cost-model calibrator.

Covers:

* No TCA files → priors returned unchanged, no recommendation drift.
* Synthetic TCA aggregates with known means → shrinkage blend matches the
  closed-form ``shrinkage*prior + (1-shrinkage)*realised``.
* Malformed JSON files are skipped rather than crashing.
* The recommendation report file is written and round-trips the keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.assembled_core.execution.cost_model_calibrator import (
    CostModelPriors,
    calibrate_cost_model,
    write_calibration_report,
)


def _write_tca(tca_dir: Path, name: str, spread: float, impact: float, fill_rate: float) -> None:
    tca_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "r",
        "date": name,
        "n_orders": 10,
        "n_fills": int(10 * fill_rate),
        "fill_rate": fill_rate,
        "cost_bps_avg": {
            "spread": spread, "impact": impact,
            "adversarial": 0.0, "sor": 0.0, "total": spread + impact,
        },
    }
    (tca_dir / f"tca_r_{name}.json").write_text(json.dumps(payload))


def test_calibrate_no_files_returns_priors(tmp_path: Path) -> None:
    res = calibrate_cost_model(tmp_path / "paper_tca", min_runs=1)
    assert res.n_runs == 0
    assert res.half_spread_bps == CostModelPriors().half_spread_bps
    assert res.impact_bps_per_pct_adv == CostModelPriors().impact_bps_per_pct_adv
    assert res.participation_cap == CostModelPriors().participation_cap


def test_calibrate_shrinkage_blend(tmp_path: Path) -> None:
    d = tmp_path / "paper_tca"
    # Two runs — realised means will be spread=3.0, impact=6.0, fill_rate=0.8.
    _write_tca(d, "2025-01-01", spread=2.0, impact=4.0, fill_rate=0.7)
    _write_tca(d, "2025-01-02", spread=4.0, impact=8.0, fill_rate=0.9)

    res = calibrate_cost_model(d, shrinkage=0.30)
    # Closed-form: 0.30*prior + 0.70*realised
    # half_spread: 0.3*5 + 0.7*3 = 1.5 + 2.1 = 3.6
    assert abs(res.half_spread_bps - 3.6) < 1e-9
    # impact: 0.3*10 + 0.7*6 = 3 + 4.2 = 7.2
    assert abs(res.impact_bps_per_pct_adv - 7.2) < 1e-9
    # cap realised = 0.05 * 0.8 = 0.04; blended = 0.3*0.05 + 0.7*0.04 = 0.015 + 0.028 = 0.043
    assert abs(res.participation_cap - 0.043) < 1e-9
    assert res.n_runs == 2


def test_calibrate_skips_malformed(tmp_path: Path) -> None:
    d = tmp_path / "paper_tca"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tca_r_broken.json").write_text("{not json")
    _write_tca(d, "2025-01-01", spread=2.0, impact=4.0, fill_rate=1.0)

    res = calibrate_cost_model(d, shrinkage=0.30)
    assert res.n_runs == 2  # file count counted; malformed skipped during mean
    # Only the valid file contributes to the mean (spread=2.0, impact=4.0).
    assert abs(res.half_spread_bps - (0.3 * 5.0 + 0.7 * 2.0)) < 1e-9


def test_write_calibration_report_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "paper_tca"
    _write_tca(d, "2025-01-01", spread=3.0, impact=6.0, fill_rate=0.8)
    res = calibrate_cost_model(d)

    out = write_calibration_report(res, tmp_path / "configs" / "fill_model_calibrated.yaml")
    assert out.exists()
    text = out.read_text()
    assert "half_spread_bps" in text
    assert "deploy" in text  # opt-in flag present
