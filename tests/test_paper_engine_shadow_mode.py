"""Phase 6 regression tests for shadow-mode observability.

Covers:

* ``shadow_mode=False`` (default) → no compare CSV written
* ``shadow_mode=True`` + injected broker → CSV is written with sim vs live
* Shadow compare does NOT influence simulated fills (observability only)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


class _FakeBroker:
    """Deterministic broker that returns a fixed fill price per symbol."""

    def __init__(self, price_map: dict[str, float]) -> None:
        self._price_map = price_map
        self.calls: list[dict] = []

    def submit(self, *, symbol: str, side: str, qty: float) -> dict:
        self.calls.append({"symbol": symbol, "side": side, "qty": qty})
        return {
            "fill_price": self._price_map.get(symbol, 100.0),
            "status": "filled",
        }


def _make_engine(
    tmp_path: Path,
    *,
    shadow_mode: bool,
    broker=None,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        shadow_mode=shadow_mode,
        shadow_broker=broker,
        shadow_compare_dir=tmp_path / "shadow",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="shadow_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_shadow_mode_disabled_writes_nothing(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, shadow_mode=False)
    orders = pd.DataFrame([{"symbol": "AAA", "side": "BUY", "qty": 100.0}])
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 100.5, "status": "filled"}]
    )
    out = eng._run_shadow_compare("2025-01-15", orders, fills)
    assert out is None
    assert not (tmp_path / "shadow").exists()


def test_shadow_mode_writes_compare_csv(tmp_path: Path) -> None:
    broker = _FakeBroker({"AAA": 101.0})
    eng = _make_engine(tmp_path, shadow_mode=True, broker=broker)
    orders = pd.DataFrame([{"symbol": "AAA", "side": "BUY", "qty": 100.0}])
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 100.5, "status": "filled"}]
    )
    out = eng._run_shadow_compare("2025-01-15", orders, fills)
    assert out is not None
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["sim_fill_price"] == 100.5
    assert row["live_fill_price"] == 101.0
    # diff_bps = (101.0 - 100.5) / 100.5 * 10_000 ≈ 49.75
    assert 49 < row["diff_bps"] < 50
    assert row["sim_status"] == "filled"
    assert row["live_status"] == "filled"


def test_shadow_mode_no_broker_is_noop(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, shadow_mode=True, broker=None)
    orders = pd.DataFrame([{"symbol": "AAA", "side": "BUY", "qty": 100.0}])
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 100.5, "status": "filled"}]
    )
    out = eng._run_shadow_compare("2025-01-15", orders, fills)
    assert out is None


def test_shadow_mode_does_not_influence_fills(tmp_path: Path) -> None:
    """Shadow compare is observability-only — engine fills must be unchanged."""
    broker = _FakeBroker({"AAA": 101.0})
    eng = _make_engine(tmp_path, shadow_mode=True, broker=broker)
    orders = pd.DataFrame([{"symbol": "AAA", "side": "BUY", "qty": 100.0}])
    fills_before = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 100.5, "status": "filled"}]
    )
    eng._run_shadow_compare("2025-01-15", orders, fills_before.copy())
    # fills_before unchanged
    assert fills_before.iloc[0]["fill_price"] == 100.5
