"""Tests for wave-4 module wiring into trading_cycle.py.

Covers:
  Step 1.8  — data_versioning (compute_data_hash, create_lineage_record)
  Step 1.9  — quality_checks (check_panel_quality)
  Step 2.3  — freshness_monitor (detect_stale_features)
  Step 6.7  — fat_finger_guard (apply_fat_finger_guard_from_policy)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data.data_versioning import compute_data_hash, create_lineage_record
from src.assembled_core.data.quality_checks import check_panel_quality
from src.assembled_core.data.freshness_monitor import detect_stale_features
from src.assembled_core.execution.fat_finger_guard import (
    apply_fat_finger_guard,
    apply_fat_finger_guard_from_policy,
)


# ---------------------------------------------------------------------------
# data_versioning (Step 1.8)
# ---------------------------------------------------------------------------

def _make_prices(n_symbols: int = 3, n_rows: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_symbols):
        sym = f"S{i}"
        for j in range(n_rows):
            rows.append({
                "symbol": sym,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=j),
                "close": 100.0 + j + i * 5,
            })
    return pd.DataFrame(rows)


def test_data_hash_deterministic():
    df = _make_prices()
    h1 = compute_data_hash(df, columns=["symbol", "timestamp", "close"])
    h2 = compute_data_hash(df, columns=["symbol", "timestamp", "close"])
    assert h1 == h2


def test_data_hash_differs_after_change():
    df = _make_prices()
    h1 = compute_data_hash(df, columns=["symbol", "timestamp", "close"])
    df2 = df.copy()
    df2.loc[0, "close"] = 999.0
    h2 = compute_data_hash(df2, columns=["symbol", "timestamp", "close"])
    assert h1 != h2


def test_create_lineage_record_fields():
    rec = create_lineage_record(data_hash="abc123", source="test", n_rows=100, n_symbols=5)
    assert rec["data_hash"] == "abc123"
    assert rec["n_rows"] == 100
    assert rec["n_symbols"] == 5
    assert "created_at" in rec


def test_data_hash_empty_df():
    empty = pd.DataFrame(columns=["symbol", "timestamp", "close"])
    h = compute_data_hash(empty)
    assert isinstance(h, str) and len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# quality_checks (Step 1.9)
# ---------------------------------------------------------------------------

def test_check_panel_quality_clean_data():
    df = _make_prices()
    results = check_panel_quality(df)
    assert len(results) == 3
    assert all(r.passed for r in results)


def test_check_panel_quality_detects_null_prices():
    df = _make_prices()
    # Set >10% of S0 rows to null so passed=False (threshold is >10%)
    s0_idx = df[df["symbol"] == "S0"].index[:3]  # 3/10 = 30% > threshold
    df.loc[s0_idx, "close"] = None
    results = check_panel_quality(df)
    failed = [r for r in results if not r.passed]
    assert any(r.symbol == "S0" for r in failed)


def test_check_panel_quality_empty_df():
    df = pd.DataFrame(columns=["symbol", "timestamp", "close"])
    results = check_panel_quality(df)
    assert results == []


# ---------------------------------------------------------------------------
# freshness_monitor / detect_stale_features (Step 2.3)
# ---------------------------------------------------------------------------

def test_detect_stale_features_finds_constant_feature():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10),
        "symbol": ["A"] * 10,
        "feat_const": [1.0] * 10,
        "feat_vary": list(range(10)),
    })
    stale = detect_stale_features(df, ["feat_const", "feat_vary"], stale_days=5)
    assert any(s["feature"] == "feat_const" for s in stale)
    assert not any(s["feature"] == "feat_vary" for s in stale)


def test_detect_stale_features_no_stale_below_threshold():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "symbol": ["A"] * 3,
        "feat": [1.0, 1.0, 1.0],
    })
    stale = detect_stale_features(df, ["feat"], stale_days=5)
    assert stale == []


def test_detect_stale_features_multi_symbol():
    rows = []
    for sym in ["A", "B"]:
        for i in range(6):
            rows.append({"timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "symbol": sym, "feat": 1.0})
    df = pd.DataFrame(rows)
    stale = detect_stale_features(df, ["feat"], stale_days=5)
    stale_syms = {s["symbol"] for s in stale}
    assert "A" in stale_syms and "B" in stale_syms


# ---------------------------------------------------------------------------
# fat_finger_guard (Step 6.7)
# ---------------------------------------------------------------------------

def _make_orders(symbols, qtys, prices=None) -> pd.DataFrame:
    prices = prices or [100.0] * len(symbols)
    return pd.DataFrame({
        "symbol": symbols,
        "side": ["buy"] * len(symbols),
        "qty": qtys,
        "price": prices,
        "timestamp": [0] * len(symbols),
    })


def test_fat_finger_blocks_oversized_notional():
    orders = _make_orders(["A", "B"], [100, 10000], [50.0, 100.0])
    filtered, reasons = apply_fat_finger_guard(orders, max_notional_usd=100_000.0)
    assert len(filtered) == 1
    assert filtered.iloc[0]["symbol"] == "A"
    assert len(reasons) == 1


def test_fat_finger_passes_all_when_no_limits():
    orders = _make_orders(["A", "B"], [100, 200])
    filtered, reasons = apply_fat_finger_guard(orders)
    assert len(filtered) == 2
    assert reasons == []


def test_fat_finger_from_policy_disabled():
    orders = _make_orders(["A"], [99999])
    policy = {"fat_finger_guard": {"enabled": False, "max_notional_usd": 1.0}}
    filtered, reasons = apply_fat_finger_guard_from_policy(orders, policy)
    assert len(filtered) == 1  # not filtered when disabled


def test_fat_finger_from_policy_enabled():
    orders = _make_orders(["A"], [10000], [100.0])  # notional = $1M
    policy = {"fat_finger_guard": {"enabled": True, "max_notional_usd": 500_000.0}}
    filtered, reasons = apply_fat_finger_guard_from_policy(orders, policy)
    assert len(filtered) == 0
    assert len(reasons) == 1


def test_fat_finger_empty_orders_passthrough():
    orders = pd.DataFrame(columns=["symbol", "qty", "price", "side", "timestamp"])
    filtered, reasons = apply_fat_finger_guard(orders, max_notional_usd=1.0)
    assert filtered.empty
    assert reasons == []
