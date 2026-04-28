"""Tests for RejectionCollector."""
from __future__ import annotations

import pandas as pd

from assembled_core.ops.rejection_collector import RejectionCollector


def _fills_df(statuses, reject_reasons=None):
    n = len(statuses)
    df = pd.DataFrame({
        "fill_price": [100.0] * n,
        "mid_price": [100.0] * n,
        "status": statuses,
    })
    if reject_reasons is not None:
        df["reject_reason"] = reject_reasons
    return df


def test_record_single():
    rc = RejectionCollector()
    rc.record("UNKNOWN_ADV")
    assert rc.snapshot() == {"UNKNOWN_ADV": 1}


def test_record_accumulates():
    rc = RejectionCollector()
    rc.record("UNKNOWN_ADV")
    rc.record("UNKNOWN_ADV")
    rc.record("MIN_FILL_QTY")
    counts = rc.snapshot()
    assert counts["UNKNOWN_ADV"] == 2
    assert counts["MIN_FILL_QTY"] == 1


def test_record_fills_basic():
    fills = _fills_df(
        ["filled", "rejected", "rejected"],
        [None, "UNKNOWN_ADV", "MIN_FILL_QTY"],
    )
    rc = RejectionCollector()
    rc.record_fills(fills)
    counts = rc.snapshot()
    assert counts.get("UNKNOWN_ADV") == 1
    assert counts.get("MIN_FILL_QTY") == 1
    assert sum(counts.values()) == 2


def test_record_fills_no_reject_reason_column():
    fills = _fills_df(["filled", "rejected"])  # no reject_reason column
    rc = RejectionCollector()
    rc.record_fills(fills)
    counts = rc.snapshot()
    assert counts.get("UNKNOWN") == 1


def test_record_fills_all_filled():
    fills = _fills_df(["filled", "filled"])
    rc = RejectionCollector()
    rc.record_fills(fills)
    assert rc.total() == 0


def test_record_fills_empty():
    rc = RejectionCollector()
    rc.record_fills(pd.DataFrame())
    assert rc.total() == 0


def test_record_blocked_reasons():
    rc = RejectionCollector()
    rc.record_blocked_reasons(["FAT_FINGER", "FAT_FINGER", "VAR_LIMIT"])
    counts = rc.snapshot()
    assert counts["FAT_FINGER"] == 2
    assert counts["VAR_LIMIT"] == 1


def test_snapshot_reset():
    rc = RejectionCollector()
    rc.record("X")
    rc.record("X")
    result = rc.snapshot(reset=True)
    assert result == {"X": 2}
    assert rc.total() == 0


def test_total_and_len():
    rc = RejectionCollector()
    assert rc.total() == 0
    assert len(rc) == 0
    rc.record("A")
    rc.record("B")
    assert rc.total() == 2
    assert len(rc) == 2


def test_thread_safety():
    import threading
    rc = RejectionCollector()
    threads = [threading.Thread(target=rc.record, args=("REASON",)) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rc.total() == 50
