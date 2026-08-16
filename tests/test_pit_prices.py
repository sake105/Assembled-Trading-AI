"""Tests for the point-in-time price adapter.

Two things are being protected here:

1. the adapter emits the operational schema, so it can be swapped in without
   touching prices_ingest.py
2. the SYNTHETIC OHLC is impossible to consume by accident

(2) matters more than it looks. The PIT panel has close only; the adapter fills
open/high/low from close so the frame satisfies the loader contract. Any
range-dependent feature (ATR, candlesticks, spread models) then computes a
confident zero. Trading a measured survivorship bias for an unmeasurable
feature bias would be a bad trade, so the marker and the guard are load-bearing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data.pit_prices import (
    DAILY_SCHEMA,
    SYNTHETIC_OHLC_ATTR,
    SyntheticOHLCError,
    assert_no_synthetic_ohlc,
    is_synthetic_ohlc,
    load_pit_prices,
    pit_members,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _panel(tmp_path):
    """A minimal stand-in for prices_verdict.parquet."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2008-01-02", "2008-01-03", "2008-01-02", "2008-01-03"], utc=True
            ),
            "symbol": ["AAPL", "AAPL", "BSC", "BSC"],
            "close": [10.0, 11.0, 80.0, 2.0],
            "volume": [100, 200, 300, 400],
        }
    )
    path = tmp_path / "prices_verdict.parquet"
    df.to_parquet(path, index=False)
    return path


def test_emits_operational_schema(tmp_path):
    out = load_pit_prices(panel_path=_panel(tmp_path), warn_synthetic=False)
    assert list(out.columns) == DAILY_SCHEMA
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert out["volume"].dtype == "float64"


def test_ohlc_is_synthesised_from_close(tmp_path):
    out = load_pit_prices(panel_path=_panel(tmp_path), warn_synthetic=False)
    for col in ("open", "high", "low", "adj_close"):
        assert (out[col] == out["close"]).all(), f"{col} must mirror close"


def test_frame_is_marked_synthetic(tmp_path):
    out = load_pit_prices(panel_path=_panel(tmp_path), warn_synthetic=False)
    assert out.attrs.get(SYNTHETIC_OHLC_ATTR) is True
    assert is_synthetic_ohlc(out) is True


def test_guard_refuses_range_dependent_consumers(tmp_path):
    """A consumer that reads high/low must fail loudly, not compute ATR = 0."""
    out = load_pit_prices(panel_path=_panel(tmp_path), warn_synthetic=False)
    with pytest.raises(SyntheticOHLCError, match="SYNTHETIC OHLC"):
        assert_no_synthetic_ohlc(out, "ta_factors_core.atr")


def test_guard_passes_a_real_panel():
    real = pd.DataFrame({"close": [1.0], "high": [2.0], "low": [0.5]})
    assert is_synthetic_ohlc(real) is False
    assert_no_synthetic_ohlc(real, "ta_factors_core.atr")  # must not raise


def test_symbol_and_date_filters(tmp_path):
    path = _panel(tmp_path)
    only = load_pit_prices(symbols=["BSC"], panel_path=path, warn_synthetic=False)
    assert set(only["symbol"]) == {"BSC"}

    window = load_pit_prices(start="2008-01-03", panel_path=path, warn_synthetic=False)
    assert (window["timestamp"] >= pd.Timestamp("2008-01-03", tz="UTC")).all()


def test_empty_result_still_carries_schema_and_marker(tmp_path):
    out = load_pit_prices(
        symbols=["DOES_NOT_EXIST"], panel_path=_panel(tmp_path), warn_synthetic=False
    )
    assert out.empty
    assert list(out.columns) == DAILY_SCHEMA
    assert is_synthetic_ohlc(out) is True


def test_missing_panel_names_the_reason(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATENZUGANG_STATUS"):
        load_pit_prices(panel_path=tmp_path / "nope.parquet")


# --- membership ---------------------------------------------------------


def _constituents(tmp_path):
    path = tmp_path / "constituents.csv"
    path.write_text(
        'date,tickers\n2008-01-02,"AAPL,BSC,MSFT"\n2010-01-04,"AAPL,MSFT"\n',
        encoding="utf-8",
    )
    return path


def test_members_come_from_the_snapshot_at_or_before_as_of(tmp_path):
    path = _constituents(tmp_path)
    assert pit_members("2008-06-30", constituents_path=path) == ["AAPL", "BSC", "MSFT"]
    assert pit_members("2011-01-01", constituents_path=path) == ["AAPL", "MSFT"]


def test_members_are_sorted(tmp_path):
    """Sorted output, not frozenset iteration order — that was E-051."""
    members = pit_members("2008-06-30", constituents_path=_constituents(tmp_path))
    assert members == sorted(members)


def test_as_of_before_first_snapshot_raises_instead_of_returning_empty(tmp_path):
    """An empty list would read as 'no members', which is a different claim."""
    with pytest.raises(ValueError, match="predates the first membership snapshot"):
        pit_members("1990-01-01", constituents_path=_constituents(tmp_path))


# --- marker robustness (Stage-1 finding B-2a) ---------------------------
#
# df.attrs survives assign / boolean masks / reset_index, but NOT merge,
# concat with mixed attrs, pivot or groupby().apply() — and those sit between
# the loader and any feature or sizing path. A marker that evaporates exactly
# where it matters would make the guard decorative (E-142), so detection also
# inspects the data.


def _synthetic_frame():
    close = pd.Series([10.0, 11.0, 12.0])
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06"], utc=True
            ),
            "symbol": ["AAA"] * 3,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": [1.0, 2.0, 3.0],
        }
    )


def test_detection_survives_merge_which_drops_attrs(tmp_path):
    out = load_pit_prices(panel_path=_panel(tmp_path), warn_synthetic=False)
    other = pd.DataFrame({"symbol": ["AAPL", "BSC"], "sector": ["Tech", "Fin"]})
    merged = out.merge(other, on="symbol", how="left")

    assert merged.attrs.get(SYNTHETIC_OHLC_ATTR) is None, (
        "precondition: merge is expected to drop attrs"
    )
    assert is_synthetic_ohlc(merged) is True, (
        "detection must fall back to the data when the attrs marker is gone"
    )
    with pytest.raises(SyntheticOHLCError):
        assert_no_synthetic_ohlc(merged, "ta_factors_core.atr")


def test_detection_survives_concat_with_mixed_attrs():
    a = _synthetic_frame()
    a.attrs[SYNTHETIC_OHLC_ATTR] = True
    b = _synthetic_frame()  # no attrs at all
    combined = pd.concat([a, b], ignore_index=True)

    assert is_synthetic_ohlc(combined) is True


def test_real_ohlc_is_not_flagged_by_the_data_fallback():
    """The fallback must not produce false positives on genuine bars."""
    real = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.5, 11.4],
            "low": [9.8, 10.7],
            "close": [10.2, 11.1],
        }
    )
    assert is_synthetic_ohlc(real) is False
    assert_no_synthetic_ohlc(real, "ta_factors_core.atr")  # must not raise


def test_frame_without_ohlc_columns_is_not_flagged():
    thin = pd.DataFrame(
        {"timestamp": [1, 2], "symbol": ["A", "A"], "close": [1.0, 2.0]}
    )
    assert is_synthetic_ohlc(thin) is False
