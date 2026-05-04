"""Tests for exposure-report artifact (Sprint 1 / W5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.kpi_artifacts import (
    build_exposure_report,
    write_exposure_report,
)


def _mk_targets(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "target_qty": q} for s, q in rows])


def _mk_prices(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "close": p} for s, p in rows])


def test_empty_targets_returns_zero_payload() -> None:
    payload = build_exposure_report(
        pd.DataFrame(), _mk_prices([("AAA", 100.0)]), equity=1000.0
    )
    assert payload["summary"]["n_positions"] == 0
    assert payload["summary"]["gross_exposure"] == 0.0
    assert payload["positions"] == []


def test_basic_long_portfolio() -> None:
    targets = _mk_targets([("AAA", 10.0), ("BBB", 20.0)])
    prices = _mk_prices([("AAA", 100.0), ("BBB", 50.0)])
    payload = build_exposure_report(targets, prices, equity=10_000.0)

    # AAA: 10*100=1000, BBB: 20*50=1000, gross=2000, net=2000
    assert payload["summary"]["gross_exposure"] == 2000.0
    assert payload["summary"]["net_exposure"] == 2000.0
    assert payload["summary"]["gross_exposure_pct"] == 20.0
    assert payload["summary"]["n_positions"] == 2
    assert len(payload["positions"]) == 2


def test_long_short_portfolio_net_vs_gross() -> None:
    targets = _mk_targets([("AAA", 10.0), ("BBB", -10.0)])
    prices = _mk_prices([("AAA", 100.0), ("BBB", 100.0)])
    payload = build_exposure_report(targets, prices, equity=10_000.0)
    assert payload["summary"]["gross_exposure"] == 2000.0
    assert payload["summary"]["net_exposure"] == 0.0


def test_hhi_is_maximal_for_single_position() -> None:
    targets = _mk_targets([("AAA", 10.0)])
    prices = _mk_prices([("AAA", 100.0)])
    payload = build_exposure_report(targets, prices, equity=10_000.0)
    assert abs(payload["summary"]["hhi"] - 1.0) < 1e-9


def test_hhi_is_1_over_n_for_equal_weights() -> None:
    targets = _mk_targets([("AAA", 10.0), ("BBB", 10.0), ("CCC", 10.0), ("DDD", 10.0)])
    prices = _mk_prices(
        [("AAA", 100.0), ("BBB", 100.0), ("CCC", 100.0), ("DDD", 100.0)]
    )
    payload = build_exposure_report(targets, prices, equity=10_000.0)
    assert abs(payload["summary"]["hhi"] - 0.25) < 1e-9


def test_top_concentration_is_sorted_descending() -> None:
    targets = _mk_targets([("AAA", 1.0), ("BBB", 5.0), ("CCC", 3.0)])
    prices = _mk_prices([("AAA", 100.0), ("BBB", 100.0), ("CCC", 100.0)])
    payload = build_exposure_report(targets, prices, equity=10_000.0, top_n=2)
    top = payload["top_concentration"]
    assert len(top) == 2
    assert top[0]["symbol"] == "BBB"
    assert top[1]["symbol"] == "CCC"


def test_missing_equity_returns_empty_payload() -> None:
    targets = _mk_targets([("AAA", 10.0)])
    prices = _mk_prices([("AAA", 100.0)])
    assert (
        build_exposure_report(targets, prices, equity=0.0)["summary"]["n_positions"]
        == 0
    )


def test_missing_prices_handled_gracefully() -> None:
    targets = _mk_targets([("AAA", 10.0), ("BBB", 10.0)])
    prices = _mk_prices([("AAA", 100.0)])  # BBB missing → price=0 via "zero"
    payload = build_exposure_report(targets, prices, equity=10_000.0)
    # BBB contributes nothing, AAA counts
    assert payload["summary"]["gross_exposure"] == 1000.0
    assert payload["summary"]["n_positions"] == 2


def test_write_exposure_report_creates_file(tmp_path: Path) -> None:
    targets = _mk_targets([("AAA", 10.0)])
    prices = _mk_prices([("AAA", 100.0)])
    out = write_exposure_report(tmp_path, targets, prices, equity=10_000.0)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "run.exposure_report.v1"
    assert payload["summary"]["gross_exposure"] == 1000.0


def test_payload_schema_keys_present() -> None:
    targets = _mk_targets([("AAA", 10.0)])
    prices = _mk_prices([("AAA", 100.0)])
    payload = build_exposure_report(targets, prices, equity=10_000.0)
    for key in (
        "schema_version",
        "generated_utc",
        "equity",
        "summary",
        "positions",
        "top_concentration",
    ):
        assert key in payload
    for key in (
        "gross_exposure",
        "net_exposure",
        "gross_exposure_pct",
        "net_exposure_pct",
        "n_positions",
        "hhi",
    ):
        assert key in payload["summary"]
