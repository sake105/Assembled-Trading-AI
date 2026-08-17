"""Tests for SlippageCollector."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.ops.slippage_collector import SlippageCollector


def _fills_df(fill_prices, mid_prices, statuses=None):
    n = len(fill_prices)
    if statuses is None:
        statuses = ["filled"] * n
    return pd.DataFrame(
        {
            "fill_price": fill_prices,
            "mid_price": mid_prices,
            "status": statuses,
        }
    )


def test_record_single():
    sc = SlippageCollector()
    sc.record(5.0)
    assert sc.snapshot() == [5.0]


def test_record_fills_basic():
    fills = _fills_df([100.5, 99.5], [100.0, 100.0])
    sc = SlippageCollector()
    sc.record_fills(fills)
    obs = sc.snapshot()
    assert len(obs) == 2
    assert abs(obs[0] - 50.0) < 0.01  # (100.5 - 100) / 100 * 10000 = 50
    assert abs(obs[1] - (-50.0)) < 0.01  # (99.5 - 100) / 100 * 10000 = -50


def test_record_fills_skips_rejected():
    fills = _fills_df([100.5, 100.5], [100.0, 100.0], ["filled", "rejected"])
    sc = SlippageCollector()
    sc.record_fills(fills)
    assert len(sc) == 1


def test_record_fills_partial_counted():
    fills = _fills_df([100.5], [100.0], ["partial"])
    sc = SlippageCollector()
    sc.record_fills(fills)
    assert len(sc) == 1


def test_record_fills_zero_mid_skipped():
    fills = _fills_df([100.5, 100.5], [0.0, 100.0], ["filled", "filled"])
    sc = SlippageCollector()
    sc.record_fills(fills)
    assert len(sc) == 1  # zero-mid row dropped


def test_record_fills_empty_df():
    sc = SlippageCollector()
    sc.record_fills(pd.DataFrame())
    assert len(sc) == 0


def test_record_fills_missing_columns():
    df = pd.DataFrame({"fill_price": [100.0], "status": ["filled"]})  # no mid_price
    sc = SlippageCollector()
    sc.record_fills(df)
    assert len(sc) == 0


def test_snapshot_reset():
    sc = SlippageCollector()
    sc.record(1.0)
    sc.record(2.0)
    result = sc.snapshot(reset=True)
    assert result == [1.0, 2.0]
    assert len(sc) == 0


def test_snapshot_no_reset():
    sc = SlippageCollector()
    sc.record(1.0)
    sc.snapshot(reset=False)
    assert len(sc) == 1


def test_len():
    sc = SlippageCollector()
    assert len(sc) == 0
    sc.record(1.0)
    sc.record(2.0)
    assert len(sc) == 2


def test_thread_safety():
    import threading

    sc = SlippageCollector()
    threads = [threading.Thread(target=sc.record, args=(float(i),)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sc) == 100
