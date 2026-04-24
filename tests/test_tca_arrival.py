"""Tests for TCA Implementation Shortfall vs Arrival Price (Sprint 2 / C11).

Covers the sidecar module ``qa/tca_arrival.py``. Does NOT exercise the
existing ``qa/tca.py`` aggregator.
"""

from __future__ import annotations

import math

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest; pytest.importorskip('src.assembled_core.qa.tca_arrival')
from src.assembled_core.qa.tca_arrival import (  # noqa: E402
    compute_implementation_shortfall,
    summarize_implementation_shortfall,
)


TS = pd.Timestamp("2026-04-11 14:30:00", tz="UTC")
TS2 = pd.Timestamp("2026-04-11 14:31:00", tz="UTC")


def _arrivals(rows):
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "arrival_price"])


def _fills(rows):
    return pd.DataFrame(
        rows, columns=["timestamp", "symbol", "side", "qty", "fill_price"]
    )


# ---------------------------------------------------------------------------
# 1. BUY at arrival price -> 0 bps
# ---------------------------------------------------------------------------
def test_buy_at_arrival_is_zero_bps():
    fills = _fills([(TS, "AAA", "BUY", 100, 100.0)])
    arr = _arrivals([(TS, "AAA", 100.0)])
    out = compute_implementation_shortfall(fills, arr)
    assert len(out) == 1
    assert out.loc[0, "is_bps"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. BUY 10 bps above arrival -> +10 bps (unfavorable)
# ---------------------------------------------------------------------------
def test_buy_above_arrival_positive_bps():
    fill_px = 100.0 * (1 + 10 / 10_000)  # +10 bps
    fills = _fills([(TS, "AAA", "BUY", 100, fill_px)])
    arr = _arrivals([(TS, "AAA", 100.0)])
    out = compute_implementation_shortfall(fills, arr)
    assert out.loc[0, "is_bps"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. SELL 10 bps below arrival -> +10 bps (unfavorable)
# ---------------------------------------------------------------------------
def test_sell_below_arrival_positive_bps():
    fill_px = 100.0 * (1 - 10 / 10_000)  # -10 bps from arrival
    fills = _fills([(TS, "AAA", "SELL", 100, fill_px)])
    arr = _arrivals([(TS, "AAA", 100.0)])
    out = compute_implementation_shortfall(fills, arr)
    # (fill - arrival)/arrival*10000*sign(-1) = (-10)*(-1) = +10
    assert out.loc[0, "is_bps"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. BUY below arrival -> negative bps (favorable)
# ---------------------------------------------------------------------------
def test_buy_below_arrival_is_favorable_negative_bps():
    fill_px = 100.0 * (1 - 5 / 10_000)  # -5 bps
    fills = _fills([(TS, "AAA", "BUY", 100, fill_px)])
    arr = _arrivals([(TS, "AAA", 100.0)])
    out = compute_implementation_shortfall(fills, arr)
    val = out.loc[0, "is_bps"]
    assert val < 0
    assert val == pytest.approx(-5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Unmatched fill -> NaN, counted in summary
# ---------------------------------------------------------------------------
def test_unmatched_fill_nan_and_counted():
    fills = _fills(
        [
            (TS, "AAA", "BUY", 100, 100.5),
            (TS, "BBB", "BUY", 50, 200.0),  # no arrival for BBB
        ]
    )
    arr = _arrivals([(TS, "AAA", 100.0)])
    out = compute_implementation_shortfall(fills, arr)
    assert len(out) == 2
    # row order preserved
    bbb_row = out[out["symbol"] == "BBB"].iloc[0]
    assert math.isnan(bbb_row["is_bps"])
    summary = summarize_implementation_shortfall(out)
    assert summary["n_fills"] == 2
    assert summary["matched"] == 1
    assert len(summary["unmatched"]) == 1
    assert summary["unmatched"][0][1] == "BBB"


# ---------------------------------------------------------------------------
# 6. Zero arrival price -> NaN, no crash
# ---------------------------------------------------------------------------
def test_zero_arrival_price_is_nan_not_crash():
    fills = _fills([(TS, "AAA", "BUY", 100, 100.0)])
    arr = _arrivals([(TS, "AAA", 0.0)])
    out = compute_implementation_shortfall(fills, arr)
    assert math.isnan(out.loc[0, "is_bps"])
    summary = summarize_implementation_shortfall(out)
    assert summary["matched"] == 0
    assert len(summary["unmatched"]) == 1


# ---------------------------------------------------------------------------
# 7. Mixed BUY/SELL batch, per-symbol breakdown
# ---------------------------------------------------------------------------
def test_mixed_batch_summary_and_per_symbol():
    fills = _fills(
        [
            (TS, "AAA", "BUY", 100, 100.1),    # +10 bps
            (TS, "AAA", "SELL", 100, 99.95),   # +5 bps (below arrival on sell)
            (TS2, "BBB", "BUY", 50, 200.4),    # +20 bps
        ]
    )
    arr = _arrivals(
        [
            (TS, "AAA", 100.0),
            (TS2, "BBB", 200.0),
        ]
    )
    out = compute_implementation_shortfall(fills, arr)
    assert len(out) == 3
    assert out["is_bps"].notna().all()

    summary = summarize_implementation_shortfall(out)
    assert summary["n_fills"] == 3
    assert summary["matched"] == 3
    assert summary["n_buy"] == 2
    assert summary["n_sell"] == 1
    # mean = (10 + 5 + 20) / 3
    assert summary["mean_bps"] == pytest.approx(35.0 / 3.0, abs=1e-6)
    assert "AAA" in summary["per_symbol"]
    assert "BBB" in summary["per_symbol"]
    assert summary["per_symbol"]["AAA"]["n_fills"] == 2
    assert summary["per_symbol"]["BBB"]["n_fills"] == 1
    assert summary["per_symbol"]["BBB"]["mean_bps"] == pytest.approx(20.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Empty fills -> empty output + zero-count summary, no crash
# ---------------------------------------------------------------------------
def test_empty_fills_safe():
    fills = _fills([])
    arr = _arrivals([(TS, "AAA", 100.0)])
    out = compute_implementation_shortfall(fills, arr)
    assert out.empty
    assert "is_bps" in out.columns
    summary = summarize_implementation_shortfall(out)
    assert summary["n_fills"] == 0
    assert summary["matched"] == 0
    assert summary["unmatched"] == []
    assert math.isnan(summary["mean_bps"])
    assert math.isnan(summary["median_bps"])
    assert math.isnan(summary["p95_bps"])
    assert summary["per_symbol"] == {}


# ---------------------------------------------------------------------------
# 9. Case-insensitive side ("buy" == "BUY")
# ---------------------------------------------------------------------------
def test_case_insensitive_side():
    fill_px = 100.0 * (1 + 10 / 10_000)
    fills_upper = _fills([(TS, "AAA", "BUY", 100, fill_px)])
    fills_lower = _fills([(TS, "AAA", "buy", 100, fill_px)])
    arr = _arrivals([(TS, "AAA", 100.0)])
    out_u = compute_implementation_shortfall(fills_upper, arr)
    out_l = compute_implementation_shortfall(fills_lower, arr)
    assert out_u.loc[0, "is_bps"] == pytest.approx(out_l.loc[0, "is_bps"], abs=1e-12)
    assert out_l.loc[0, "is_bps"] == pytest.approx(10.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 10. Idempotency: same inputs -> identical outputs
# ---------------------------------------------------------------------------
def test_idempotent_twice_same_output():
    fills = _fills(
        [
            (TS, "AAA", "BUY", 100, 100.1),
            (TS, "BBB", "SELL", 50, 199.95),
            (TS2, "CCC", "BUY", 25, 50.0),  # unmatched
        ]
    )
    arr = _arrivals(
        [
            (TS, "AAA", 100.0),
            (TS, "BBB", 200.0),
        ]
    )
    out1 = compute_implementation_shortfall(fills, arr)
    out2 = compute_implementation_shortfall(fills, arr)
    pd.testing.assert_frame_equal(out1, out2)

    s1 = summarize_implementation_shortfall(out1)
    s2 = summarize_implementation_shortfall(out2)
    # dict-level comparison; NaNs compared via repr is fragile, so check fields
    assert s1["n_fills"] == s2["n_fills"]
    assert s1["matched"] == s2["matched"]
    assert s1["unmatched"] == s2["unmatched"]
    assert s1["per_symbol"] == s2["per_symbol"]
    # numeric equality (allowing NaN equality)
    for key in ("mean_bps", "median_bps", "p95_bps"):
        a, b = s1[key], s2[key]
        assert (math.isnan(a) and math.isnan(b)) or a == b
