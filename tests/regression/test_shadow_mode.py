"""Part D — Shadow-mode snapshot writer regression pins.

Ensures the shared helper that every D-module (correlation_guard,
zombie_killer, crash_prediction, inverse_etf, signal_decay) uses for its
shadow artifact is deterministic and atomic.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.assembled_core.ops.shadow_mode import (
    read_shadow_snapshot,
    write_shadow_snapshot,
)

pytestmark = pytest.mark.phase_depth


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    path = write_shadow_snapshot(
        module="correlation_guard",
        payload={"max_pairwise_corr": 0.85, "reduced_pairs": [["AAPL", "MSFT"]]},
        snapshot_date=date(2026, 4, 17),
        shadow_root=tmp_path,
    )
    assert path.exists()
    envelope = read_shadow_snapshot(path)
    assert envelope["module"] == "correlation_guard"
    assert envelope["snapshot_date"] == "2026-04-17"
    assert envelope["payload"]["max_pairwise_corr"] == 0.85


def test_snapshot_is_single_line(tmp_path: Path) -> None:
    path = write_shadow_snapshot(
        module="signal_decay",
        payload={"multipliers": {"trend_ema_spread": 0.0}},
        snapshot_date=date(2026, 4, 17),
        shadow_root=tmp_path,
    )
    raw = path.read_text(encoding="utf-8")
    assert raw.count("\n") <= 1, (
        "Shadow snapshots must be single-line to keep per-cycle writes cheap"
    )


def test_run_id_suffix_disambiguates(tmp_path: Path) -> None:
    a = write_shadow_snapshot(
        module="zombie_killer",
        payload={"closed": ["AAPL"]},
        snapshot_date=date(2026, 4, 17),
        run_id="pre",
        shadow_root=tmp_path,
    )
    b = write_shadow_snapshot(
        module="zombie_killer",
        payload={"closed": ["AAPL", "MSFT"]},
        snapshot_date=date(2026, 4, 17),
        run_id="post",
        shadow_root=tmp_path,
    )
    assert a != b
    assert a.exists() and b.exists()


def test_invalid_module_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_shadow_snapshot(module="", payload={}, shadow_root=tmp_path)
    with pytest.raises(ValueError):
        write_shadow_snapshot(module="../evil", payload={}, shadow_root=tmp_path)


def test_envelope_schema_enforced(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.json"
    bad.write_text(json.dumps({"module": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        read_shadow_snapshot(bad)
