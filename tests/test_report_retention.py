"""Tests for ops.report_retention.purge_old_dated_reports."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def _touch(p: Path, mtime: float) -> None:
    p.write_text("{}", encoding="utf-8")
    os.utime(p, (mtime, mtime))


def test_purge_noop_when_below_limit(tmp_path):
    from src.assembled_core.ops.report_retention import purge_old_dated_reports

    for i in range(3):
        _touch(tmp_path / f"tca_report_2026010{i}.json", time.time() - i)

    assert (
        purge_old_dated_reports(tmp_path, "tca_report_", ".json", keep_last_n=10) == 0
    )
    assert len(list(tmp_path.glob("tca_report_*.json"))) == 3


def test_purge_keeps_most_recent_n(tmp_path):
    from src.assembled_core.ops.report_retention import purge_old_dated_reports

    now = time.time()
    for i in range(10):
        _touch(tmp_path / f"signal_decay_2026{i:04d}.json", now - i * 60)

    deleted = purge_old_dated_reports(tmp_path, "signal_decay_", ".json", keep_last_n=3)
    assert deleted == 7
    remaining = sorted(tmp_path.glob("signal_decay_*.json"))
    assert len(remaining) == 3
    # Newest (highest mtime) survive — that's i=0,1,2
    remaining_names = {p.name for p in remaining}
    assert remaining_names == {
        "signal_decay_20260000.json",
        "signal_decay_20260001.json",
        "signal_decay_20260002.json",
    }


def test_purge_ignores_other_prefixes(tmp_path):
    from src.assembled_core.ops.report_retention import purge_old_dated_reports

    now = time.time()
    for i in range(5):
        _touch(tmp_path / f"tca_report_a{i}.json", now - i)
        _touch(tmp_path / f"other_report_a{i}.json", now - i)

    purge_old_dated_reports(tmp_path, "tca_report_", ".json", keep_last_n=1)
    assert len(list(tmp_path.glob("tca_report_*.json"))) == 1
    assert len(list(tmp_path.glob("other_report_*.json"))) == 5


def test_purge_nonexistent_dir_safe(tmp_path):
    from src.assembled_core.ops.report_retention import purge_old_dated_reports

    missing = tmp_path / "does_not_exist"
    assert purge_old_dated_reports(missing, "x_", ".json", keep_last_n=3) == 0


def test_purge_negative_keep_is_noop(tmp_path):
    from src.assembled_core.ops.report_retention import purge_old_dated_reports

    _touch(tmp_path / "x_1.json", time.time())
    assert purge_old_dated_reports(tmp_path, "x_", ".json", keep_last_n=-1) == 0
    assert (tmp_path / "x_1.json").exists()


def test_purge_respects_suffix(tmp_path):
    from src.assembled_core.ops.report_retention import purge_old_dated_reports

    now = time.time()
    for i in range(5):
        _touch(tmp_path / f"x_{i}.json", now - i)
        _touch(tmp_path / f"x_{i}.csv", now - i)

    purge_old_dated_reports(tmp_path, "x_", ".json", keep_last_n=1)
    assert len(list(tmp_path.glob("x_*.json"))) == 1
    assert len(list(tmp_path.glob("x_*.csv"))) == 5
