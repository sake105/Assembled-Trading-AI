"""Tests for the one-time adj_close backfill.

This script already modified production data irreversibly once
(98,279 rows of output/aggregates/daily.parquet). Its safety invariant — abort
rather than write if any populated adj_close differs from close — was untested
at that point, which is the wrong order. These tests close that gap.

The invariant is the whole argument for the script's existence: mirroring close
into adj_close is only correct BECAUSE the two are already identical wherever
both are present. If that ever stops holding, adj_close carries independent
information and mirroring would destroy it. The script must then refuse.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "ops" / "backfill_adj_close.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_adj_close_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _row(ts: str, sym: str, close: float, adj: float | None) -> dict:
    return {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "symbol": sym,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": float("nan") if adj is None else adj,
        "volume": 1_000.0,
    }


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """Load the script with its status path redirected into tmp_path.

    Deliberately NOT autouse: ``_load_module`` builds a fresh module object on
    every call, so a fixture that patched its own instance would leave the one
    the test uses pointing at the real output/ops/. Each test takes this
    fixture's module, never its own.
    """
    m = _load_module()
    monkeypatch.setattr(m, "STATUS_PATH", tmp_path / "ops" / "status.json")
    return m


# --- the safety invariant ------------------------------------------------


def test_aborts_when_populated_adj_close_differs_from_close(mod, tmp_path):
    """THE guard: a real adj_close != close means the column carries information."""
    status = mod.STATUS_PATH  # already redirected into tmp_path by the fixture

    cache = _cache(
        tmp_path,
        [
            _row("2024-01-02", "AAA", 100.0, 90.0),  # genuinely different
            _row("2024-01-03", "AAA", 101.0, None),
        ],
    )
    before = pd.read_parquet(cache)

    rc = mod.backfill(cache, apply=True, backup=False)

    assert rc == -1, "must refuse rather than destroy independent information"
    pd.testing.assert_frame_equal(before, pd.read_parquet(cache))
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["rc"] == -1
    assert "differs from close" in payload["error"]
    assert payload["n_mismatch"] == 1


def test_aborts_when_close_has_nan(mod, tmp_path):
    """Nothing to mirror FROM is also a refusal, not a silent partial fill."""

    rows = [_row("2024-01-02", "AAA", 100.0, None)]
    rows[0]["close"] = float("nan")
    cache = _cache(tmp_path, rows)

    assert mod.backfill(cache, apply=True, backup=False) == -1


def test_float_epsilon_difference_does_not_trip_the_guard(mod, tmp_path):
    """Real panels carry epsilon-level noise; that must not block the repair.

    Measured on the real cache: 15 of 279,013 rows have close marginally
    outside [low, high], all at float epsilon (max 2.8e-14 absolute). The guard has to tolerate that while
    still catching a genuine difference.
    """

    cache = _cache(
        tmp_path,
        [
            _row("2024-01-02", "AAA", 100.0, 100.0 + 1e-13),
            _row("2024-01-03", "AAA", 101.0, None),
        ],
    )
    assert mod.backfill(cache, apply=True, backup=False) == 1


# --- normal operation ----------------------------------------------------


def test_dry_run_reports_but_writes_nothing(mod, tmp_path):
    cache = _cache(
        tmp_path,
        [
            _row("2024-01-02", "AAA", 100.0, 100.0),
            _row("2024-01-03", "AAA", 101.0, None),
        ],
    )
    before = pd.read_parquet(cache)

    rc = mod.backfill(cache, apply=False, backup=False)

    assert rc == 1, "reports what WOULD be repaired"
    pd.testing.assert_frame_equal(before, pd.read_parquet(cache))


def test_apply_mirrors_close_and_leaves_no_nan(mod, tmp_path):
    cache = _cache(
        tmp_path,
        [
            _row("2024-01-02", "AAA", 100.0, 100.0),
            _row("2024-01-03", "AAA", 101.0, None),
            _row("2024-01-02", "BBB", 50.0, None),
        ],
    )

    rc = mod.backfill(cache, apply=True, backup=False)

    out = pd.read_parquet(cache)
    assert rc == 2
    assert out["adj_close"].isna().sum() == 0
    assert (out["adj_close"] == out["close"]).all()


def test_backup_is_written_next_to_the_cache(mod, tmp_path):
    cache = _cache(tmp_path, [_row("2024-01-02", "AAA", 100.0, None)])
    mod.backfill(cache, apply=True, backup=True)

    assert cache.with_suffix(".parquet.bak").exists()


def test_idempotent_second_run_is_a_noop(mod, tmp_path):
    cache = _cache(tmp_path, [_row("2024-01-02", "AAA", 100.0, None)])
    mod.backfill(cache, apply=True, backup=False)
    after_first = pd.read_parquet(cache)

    assert mod.backfill(cache, apply=True, backup=False) == 0
    pd.testing.assert_frame_equal(after_first, pd.read_parquet(cache))


def test_missing_cache_returns_minus_one(mod, tmp_path):
    assert mod.backfill(tmp_path / "nope.parquet", apply=True) == -1


def test_missing_adj_close_column_returns_minus_one(mod, tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02", tz="UTC")],
            "symbol": ["AAA"],
            "close": [100.0],
        }
    )
    p = tmp_path / "daily.parquet"
    df.to_parquet(p, index=False)

    assert mod.backfill(p, apply=True) == -1
