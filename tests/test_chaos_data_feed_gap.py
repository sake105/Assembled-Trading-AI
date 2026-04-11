"""Chaos test: data feed stall / gap detection (Plan C21).

``detect_stale_features`` is the backstop against a silent data feed
outage: if a feature value is constant for N trading days, that is
almost certainly a dead feed rather than a feature that genuinely
stopped moving. This test injects three realistic chaos scenarios
and verifies that the detector fires on the right cases and stays
silent on the wrong ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.freshness_monitor import detect_stale_features  # noqa: E402


def _base_frame(n_days: int = 10) -> pd.DataFrame:
    """Two symbols, two features, fresh every day."""
    rows = []
    base = pd.Timestamp("2026-04-01", tz="UTC")
    for i in range(n_days):
        ts = base + pd.Timedelta(days=i)
        rows.append(
            {"timestamp": ts, "symbol": "AAPL", "feat_a": 100.0 + i, "feat_b": 10.0 + i * 0.5}
        )
        rows.append(
            {"timestamp": ts, "symbol": "MSFT", "feat_a": 200.0 + i, "feat_b": 20.0 + i * 0.3}
        )
    return pd.DataFrame(rows)


def test_healthy_feed_has_no_stale_alerts() -> None:
    df = _base_frame(n_days=10)
    alerts = detect_stale_features(
        df, feature_cols=["feat_a", "feat_b"], stale_days=5
    )
    assert alerts == []


def test_single_symbol_feed_stall_is_detected() -> None:
    """AAPL feat_a freezes for the last 6 days. Must be detected."""
    df = _base_frame(n_days=10)
    # Freeze AAPL feat_a for the last 6 rows (sorted order may matter —
    # we patch by mask on symbol + by last timestamps).
    aapl_mask = df["symbol"] == "AAPL"
    aapl_df = df[aapl_mask].sort_values("timestamp")
    freeze_value = aapl_df.iloc[-6]["feat_a"]
    frozen_idx = aapl_df.iloc[-6:].index
    df.loc[frozen_idx, "feat_a"] = freeze_value

    alerts = detect_stale_features(
        df, feature_cols=["feat_a", "feat_b"], stale_days=5
    )
    aapl_feat_a_alerts = [
        a for a in alerts if a["symbol"] == "AAPL" and a["feature"] == "feat_a"
    ]
    assert aapl_feat_a_alerts, f"expected stale alert on AAPL.feat_a, got {alerts}"


def test_unaffected_symbols_are_not_alerted() -> None:
    """When AAPL feat_a stalls, MSFT.feat_a and both feat_b values
    must not be flagged. This rules out false-positive broadcast."""
    df = _base_frame(n_days=10)
    aapl_df = df[df["symbol"] == "AAPL"].sort_values("timestamp")
    freeze_value = aapl_df.iloc[-6]["feat_a"]
    df.loc[aapl_df.iloc[-6:].index, "feat_a"] = freeze_value

    alerts = detect_stale_features(
        df, feature_cols=["feat_a", "feat_b"], stale_days=5
    )
    for a in alerts:
        # Only AAPL.feat_a should be flagged, nothing else.
        assert (a["symbol"], a["feature"]) == ("AAPL", "feat_a"), (
            f"unexpected alert: {a}"
        )


def test_short_gap_below_threshold_is_not_alerted() -> None:
    """A 3-day freeze with stale_days=5 must not trip — otherwise
    weekend / holiday gaps would produce false positives."""
    df = _base_frame(n_days=10)
    aapl_df = df[df["symbol"] == "AAPL"].sort_values("timestamp")
    freeze_value = aapl_df.iloc[-3]["feat_a"]
    df.loc[aapl_df.iloc[-3:].index, "feat_a"] = freeze_value

    alerts = detect_stale_features(
        df, feature_cols=["feat_a", "feat_b"], stale_days=5
    )
    aapl_feat_a_alerts = [
        a for a in alerts if a["symbol"] == "AAPL" and a["feature"] == "feat_a"
    ]
    assert aapl_feat_a_alerts == []


def test_all_features_all_symbols_stalled() -> None:
    """Worst case: the whole feed is dead for 7 days. Must produce
    alerts for every (symbol, feature) pair, not just one."""
    rows = []
    base = pd.Timestamp("2026-04-01", tz="UTC")
    for i in range(10):
        ts = base + pd.Timedelta(days=i)
        # Every row has the same value — the feed has been dead
        # from the start.
        rows.append({"timestamp": ts, "symbol": "AAPL", "feat_a": 100.0, "feat_b": 10.0})
        rows.append({"timestamp": ts, "symbol": "MSFT", "feat_a": 200.0, "feat_b": 20.0})
    df = pd.DataFrame(rows)

    alerts = detect_stale_features(
        df, feature_cols=["feat_a", "feat_b"], stale_days=5
    )
    flagged = {(a["symbol"], a["feature"]) for a in alerts}
    assert ("AAPL", "feat_a") in flagged
    assert ("AAPL", "feat_b") in flagged
    assert ("MSFT", "feat_a") in flagged
    assert ("MSFT", "feat_b") in flagged
