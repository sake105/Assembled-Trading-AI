"""Tests for group-exposure cap helper (Sprint 2 / W3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.trading_cycle import _apply_group_exposure_caps


def _meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "sector": "Technology", "region": "US", "currency": "USD"},
            {"symbol": "BBB", "sector": "Technology", "region": "US", "currency": "USD"},
            {"symbol": "CCC", "sector": "Technology", "region": "US", "currency": "USD"},
            {"symbol": "DDD", "sector": "Energy", "region": "US", "currency": "USD"},
        ]
    )


def _orders(qtys: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-03-01", tz="UTC"),
                "symbol": sym,
                "side": "BUY",
                "qty": q,
                "price": 100.0,
            }
            for sym, q in qtys.items()
        ]
    )


def test_no_cap_exceedance_no_scale() -> None:
    orders = _orders({"AAA": 10, "BBB": 10, "DDD": 10})
    cfg = {"max_sector_gross": 0.9}  # 2/3 tech < 0.9
    out, meta = _apply_group_exposure_caps(orders, _meta(), cfg)
    assert (out["qty"] == orders["qty"]).all()
    assert meta["scaled_groups"] == []


def test_sector_overcap_scales_down() -> None:
    orders = _orders({"AAA": 50, "BBB": 50, "DDD": 10})  # 100/110 tech ~= 0.909
    cfg = {"max_sector_gross": 0.5}
    out, meta = _apply_group_exposure_caps(orders, _meta(), cfg)
    # AAA and BBB should be scaled by 0.5/0.909 ~= 0.55
    assert out[out["symbol"] == "AAA"].iloc[0]["qty"] < 50
    assert out[out["symbol"] == "BBB"].iloc[0]["qty"] < 50
    # DDD (Energy) untouched
    assert out[out["symbol"] == "DDD"].iloc[0]["qty"] == 10
    assert any(g["dim"] == "sector" for g in meta["scaled_groups"])


def test_disabled_when_no_active_dims() -> None:
    orders = _orders({"AAA": 50, "BBB": 50})
    out, meta = _apply_group_exposure_caps(orders, _meta(), {})
    assert (out["qty"] == orders["qty"]).all()
    assert meta["scaled_groups"] == []


def test_empty_orders() -> None:
    empty = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    out, meta = _apply_group_exposure_caps(empty, _meta(), {"max_sector_gross": 0.5})
    assert out.empty
    assert meta["n_orders"] == 0


def test_missing_security_meta_is_noop() -> None:
    orders = _orders({"AAA": 50})
    out, meta = _apply_group_exposure_caps(orders, None, {"max_sector_gross": 0.5})
    assert out is orders
    assert meta["scaled_groups"] == []


def test_multi_dim_takes_most_restrictive() -> None:
    orders = _orders({"AAA": 100, "DDD": 10})  # AAA=Tech/US; DDD=Energy/US
    # Sector cap 0.5 → AAA ~0.909/0.5 scale=0.55
    # Region US cap 0.95 (all US = 1.0 → scale=0.95); AAA should get min scale
    cfg = {"max_sector_gross": 0.5, "max_region_gross": 0.95}
    out, meta = _apply_group_exposure_caps(orders, _meta(), cfg)
    # AAA: most restrictive scale should apply
    aaa_qty = float(out[out["symbol"] == "AAA"].iloc[0]["qty"])
    assert aaa_qty < 100 * 0.95  # more restrictive than region cap
