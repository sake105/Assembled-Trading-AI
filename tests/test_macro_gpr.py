"""Tests for the Caldara-Iacoviello GPR panel merge helper (§9.9 wiring).

Covers:
- File missing → panel returned unchanged (graceful degradation)
- File present with valid schema → gpr_index column added
- File present with missing schema columns → panel returned unchanged
- merge_asof PIT semantics (monthly value broadcast forward to daily rows)
- Idempotency: re-running on a panel that already has gpr_index is a no-op
- Row-order preservation: panel rows stay in their original (symbol, timestamp)
  order after the merge
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.data.macro.gpr import (
    load_gpr_series,
    merge_gpr_index_into_panel,
)


def _make_daily_panel(symbols: list[str], start: str, periods: int) -> pd.DataFrame:
    """Tiny per-symbol panel fixture."""
    dates = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({"timestamp": d, "symbol": sym, "close": 100.0})
    return pd.DataFrame(rows)


def _make_gpr_parquet(tmp_path, values: list[tuple[str, float]]) -> str:
    """Write a tiny monthly GPR parquet, return its path."""
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(ts, tz="UTC") for ts, _ in values],
            "gpr_index": [v for _, v in values],
        }
    )
    out = tmp_path / "macro_gpr.parquet"
    df.to_parquet(out, index=False)
    return str(out)


def test_merge_missing_file_returns_panel_unchanged(tmp_path):
    panel = _make_daily_panel(["AAPL"], "2024-01-01", 5)
    out = merge_gpr_index_into_panel(panel, tmp_path / "does_not_exist.parquet")
    assert "gpr_index" not in out.columns
    # Same shape & values
    assert out.shape == panel.shape
    pd.testing.assert_frame_equal(out, panel)


def test_merge_adds_gpr_index_column_with_raw_backward_fill_lag0(tmp_path):
    """Raw asof behavior (lag=0): each row sees the latest month-start value.

    This documents the helper's behavior WITHOUT the production
    release-lag shift. Production callers must pass the default
    release_lag_days=32 for PIT-safety.
    """
    gpr_path = _make_gpr_parquet(
        tmp_path,
        [("2024-01-01", 100.0), ("2024-02-01", 150.0)],
    )
    panel = _make_daily_panel(["AAPL"], "2024-01-15", 60)
    out = merge_gpr_index_into_panel(panel, gpr_path, release_lag_days=0)

    assert "gpr_index" in out.columns
    out_jan = out[out["timestamp"] < pd.Timestamp("2024-02-01", tz="UTC")]
    out_feb = out[out["timestamp"] >= pd.Timestamp("2024-02-01", tz="UTC")]
    assert (out_jan["gpr_index"] == 100.0).all(), out_jan
    assert (out_feb["gpr_index"] == 150.0).all(), out_feb


def test_merge_default_release_lag_prevents_future_leak(tmp_path):
    """F-GPR-1 regression guard: default release_lag_days=32 prevents
    the Feb-2024 value (stamped 2024-02-01) from leaking into a backtest
    bar dated 2024-02-01 (would only be public around 2024-03-04).
    """
    gpr_path = _make_gpr_parquet(
        tmp_path,
        [("2024-01-01", 100.0), ("2024-02-01", 150.0)],
    )
    # Panel covers Jan 15 → Mar 31 → spans the publication delay
    panel = _make_daily_panel(["AAPL"], "2024-01-15", 80)
    out = merge_gpr_index_into_panel(panel, gpr_path)  # default lag=32

    # Jan-01 GPR (stamped 2024-01-01) becomes publishable at 2024-02-02
    # → rows dated < 2024-02-02 see NaN (no GPR yet in the lagged series)
    # → rows dated >= 2024-02-02 see 100.0 until Feb-01 GPR publishes (2024-03-04)
    # → rows dated >= 2024-03-04 see 150.0
    bar_2024_02_01 = out[out["timestamp"] == pd.Timestamp("2024-02-01", tz="UTC")]
    assert bar_2024_02_01["gpr_index"].isna().all(), (
        "2024-02-01 must NOT see any GPR (default lag=32 means Jan-01 "
        "value becomes publishable 2024-02-02, Feb-01 value 2024-03-04)"
    )

    bar_2024_02_15 = out[out["timestamp"] == pd.Timestamp("2024-02-15", tz="UTC")]
    assert (bar_2024_02_15["gpr_index"] == 100.0).all(), (
        "2024-02-15 must see Jan-01 GPR=100 (publishable 2024-02-02)"
    )

    bar_2024_03_15 = out[out["timestamp"] == pd.Timestamp("2024-03-15", tz="UTC")]
    assert (bar_2024_03_15["gpr_index"] == 150.0).all(), (
        "2024-03-15 must see Feb-01 GPR=150 (publishable 2024-03-04)"
    )


def test_merge_nat_in_panel_timestamp_returns_unchanged_with_warning(tmp_path, caplog):
    """F-tr-1 guard: merge_asof raises ValueError on NaT left keys.
    Helper degrades gracefully with a warning log.
    """
    gpr_path = _make_gpr_parquet(tmp_path, [("2024-01-01", 100.0)])
    panel = _make_daily_panel(["AAPL"], "2024-01-15", 5)
    panel.loc[0, "timestamp"] = pd.NaT

    with caplog.at_level("WARNING", logger="src.assembled_core.data.macro.gpr"):
        out = merge_gpr_index_into_panel(panel, gpr_path, release_lag_days=0)

    assert "gpr_index" not in out.columns
    assert any("NaT" in rec.message for rec in caplog.records)


def test_merge_missing_required_columns_returns_panel_unchanged(tmp_path):
    # Parquet exists but lacks gpr_index column
    bad = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "wrong_col": [1.0],
        }
    )
    bad_path = tmp_path / "macro_gpr.parquet"
    bad.to_parquet(bad_path, index=False)

    panel = _make_daily_panel(["AAPL"], "2024-01-15", 5)
    out = merge_gpr_index_into_panel(panel, bad_path)
    assert "gpr_index" not in out.columns
    pd.testing.assert_frame_equal(out, panel)


def test_merge_idempotent_when_column_already_present(tmp_path):
    gpr_path = _make_gpr_parquet(tmp_path, [("2024-01-01", 100.0)])
    panel = _make_daily_panel(["AAPL"], "2024-01-15", 5)
    panel["gpr_index"] = 99.0  # pre-existing value should not be clobbered

    out = merge_gpr_index_into_panel(panel, gpr_path)
    assert (out["gpr_index"] == 99.0).all()


def test_merge_preserves_row_order_multi_symbol(tmp_path):
    gpr_path = _make_gpr_parquet(
        tmp_path,
        [("2024-01-01", 100.0), ("2024-02-01", 150.0)],
    )
    # Build panel ordered (symbol, timestamp) but interleave so we can verify
    panel = _make_daily_panel(["MSFT", "AAPL"], "2024-01-15", 30)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Use release_lag_days=0 so we can assert raw-asof values directly.
    out = merge_gpr_index_into_panel(panel, gpr_path, release_lag_days=0)

    # Original (symbol, timestamp) ordering preserved
    expected_order = panel[["symbol", "timestamp"]].reset_index(drop=True)
    actual_order = out[["symbol", "timestamp"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_order, expected_order)

    # Each row carries the correct backward-asof GPR value
    out_jan = out[out["timestamp"] < pd.Timestamp("2024-02-01", tz="UTC")]
    out_feb = out[out["timestamp"] >= pd.Timestamp("2024-02-01", tz="UTC")]
    assert (out_jan["gpr_index"] == 100.0).all()
    assert (out_feb["gpr_index"] == 150.0).all()


def test_load_gpr_series_drops_duplicates_and_sorts(tmp_path):
    # Out-of-order with a duplicate timestamp (keep last value)
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-02-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),  # duplicate, value below wins
            ],
            "gpr_index": [150.0, 100.0, 110.0],
        }
    )
    p = tmp_path / "macro_gpr.parquet"
    df.to_parquet(p, index=False)

    out = load_gpr_series(p)
    assert out is not None
    # Sorted by timestamp
    assert (out["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()
    # Duplicate kept the LAST value (110.0 for 2024-01-01)
    jan = out[out["timestamp"] == pd.Timestamp("2024-01-01", tz="UTC")]
    assert (jan["gpr_index"] == 110.0).all()


def test_merge_empty_panel_returns_unchanged(tmp_path):
    gpr_path = _make_gpr_parquet(tmp_path, [("2024-01-01", 100.0)])
    panel = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    out = merge_gpr_index_into_panel(panel, gpr_path)
    assert out.empty
    assert "gpr_index" not in out.columns


def test_merge_panel_without_timestamp_returns_unchanged(tmp_path):
    gpr_path = _make_gpr_parquet(tmp_path, [("2024-01-01", 100.0)])
    panel = pd.DataFrame({"symbol": ["AAPL"], "close": [100.0]})
    out = merge_gpr_index_into_panel(panel, gpr_path)
    assert "gpr_index" not in out.columns
