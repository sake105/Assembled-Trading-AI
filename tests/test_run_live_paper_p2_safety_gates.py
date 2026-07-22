"""Regression guards for the 2026-07-21 GESAMTBEWERTUNG P2 fixes in
``scripts/run_live_paper.py`` and ``scripts/ops/refresh_daily_cache_from_panel.py``.

Pins four behaviours:
  K2a — market-hours gate: broker cycles must not submit into a closed or
        nearly-closed market (root cause of the 2026-07-14 after-hours fills).
  K3  — stale-order cleanup must never cancel recent orders: per-order cancel
        when available; blanket cancel only when no recent orders exist.
  W7a — pending order intents from a prior crash BLOCK preflight (return
        False), they are no longer warn-and-continue.
  W7b — a failed broker sync with position mismatches trips the halt even
        when cash_diff is below the cash thresholds (live-verified gap:
        2026-07-20 halt JSON showed mismatches_count=0 with 5 missing symbols).
  K6  — panel→daily.parquet refresh drops same-day (forming) bars (E-053).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]
RLP_PATH = REPO_ROOT / "scripts" / "run_live_paper.py"
REFRESH_PATH = REPO_ROOT / "scripts" / "ops" / "refresh_daily_cache_from_panel.py"


def _load(path: Path, name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rlp():
    return _load(RLP_PATH, "rlp_p2_mod")


# ---------------------------------------------------------------------------
# K2a — market-hours gate
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, is_open: bool, next_close=None):
        self.is_open = is_open
        self.next_close = next_close


class _FakeApi:
    def __init__(self, clock):
        self._clock = clock

    def get_clock(self):
        if isinstance(self._clock, Exception):
            raise self._clock
        return self._clock


def _adapter_with_clock(clock):
    return SimpleNamespace(_api=_FakeApi(clock))


def test_k2a_closed_market_blocks(rlp):
    ok, reason = rlp._market_open_for_submission(
        _adapter_with_clock(_FakeClock(is_open=False))
    )
    assert ok is False
    assert "closed" in reason


def test_k2a_open_market_far_from_close_passes(rlp):
    nc = datetime.now(timezone.utc) + timedelta(hours=3)
    ok, reason = rlp._market_open_for_submission(
        _adapter_with_clock(_FakeClock(is_open=True, next_close=nc))
    )
    assert ok is True
    assert "open" in reason


def test_k2a_near_close_blocks(rlp):
    nc = datetime.now(timezone.utc) + timedelta(minutes=4)
    ok, reason = rlp._market_open_for_submission(
        _adapter_with_clock(_FakeClock(is_open=True, next_close=nc))
    )
    assert ok is False
    assert "close" in reason


def test_k2a_clock_failure_falls_back_deterministically(rlp):
    ok, reason = rlp._market_open_for_submission(
        _adapter_with_clock(RuntimeError("simulated clock outage"))
    )
    assert isinstance(ok, bool)
    assert reason.startswith("fallback")


def test_k2a_adapter_without_api_uses_fallback(rlp):
    ok, reason = rlp._market_open_for_submission(SimpleNamespace())
    assert isinstance(ok, bool)
    assert reason.startswith("fallback")


# ---------------------------------------------------------------------------
# K3 — stale-order cleanup must not cancel recent orders
# ---------------------------------------------------------------------------


class _Order(SimpleNamespace):
    pass


def _mk_order(order_id: str, age_seconds: float) -> _Order:
    submitted = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return _Order(order_id=order_id, submitted_at=submitted.isoformat())


class _CleanupAdapter:
    """Adapter stub reaching only the stale-cleanup part of preflight."""

    def __init__(self, orders, with_cancel_order: bool):
        self._orders = orders
        self.cancelled_ids: list[str] = []
        self.cancel_all_called = False
        if with_cancel_order:
            self.cancel_order = self._cancel_order  # feature-detected

    def get_account(self):
        return {"equity": 999_999.0}  # far above any drawdown stop

    def get_open_orders(self):
        return self._orders

    def _cancel_order(self, order_id: str):
        self.cancelled_ids.append(order_id)

    def cancel_all_orders(self) -> int:
        self.cancel_all_called = True
        return len(self._orders)


def _neutralised_preflight(rlp, monkeypatch, tmp_path, adapter):
    monkeypatch.setattr(rlp, "HALT_FLAG_PATH", tmp_path / "halt.json")
    import src.assembled_core.execution.kill_switch as ks

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    import src.assembled_core.execution.intent_store as intent_store

    monkeypatch.setattr(intent_store, "find_pending_order_intents", lambda: [])
    app_cfg = {"paper_runner": {"start_capital": 100.0, "dd_stop_pct": 0.99}}
    return rlp._preflight_checks(adapter, app_cfg)


def test_k3_per_order_cancel_spares_recent(rlp, monkeypatch, tmp_path):
    stale = _mk_order("stale-1", age_seconds=900)
    recent = _mk_order("recent-1", age_seconds=30)
    adapter = _CleanupAdapter([stale, recent], with_cancel_order=True)
    assert _neutralised_preflight(rlp, monkeypatch, tmp_path, adapter) is True
    assert adapter.cancelled_ids == ["stale-1"]
    assert adapter.cancel_all_called is False


def test_k3_no_per_order_cancel_and_recent_present_blocks_cycle(
    rlp, monkeypatch, tmp_path
):
    # Stage-1 review M1: un-cancellable stale orders + recent orders in the
    # blast radius -> BLOCK the cycle (trading on top of live stale orders
    # risks double exposure). Previously this warned and traded anyway.
    stale = _mk_order("stale-1", age_seconds=900)
    recent = _mk_order("recent-1", age_seconds=30)
    adapter = _CleanupAdapter([stale, recent], with_cancel_order=False)
    assert _neutralised_preflight(rlp, monkeypatch, tmp_path, adapter) is False
    assert adapter.cancel_all_called is False  # recent order in blast radius


def test_k3_blanket_cancel_only_when_all_stale(rlp, monkeypatch, tmp_path):
    stale1 = _mk_order("stale-1", age_seconds=900)
    stale2 = _mk_order("stale-2", age_seconds=1200)
    adapter = _CleanupAdapter([stale1, stale2], with_cancel_order=False)
    assert _neutralised_preflight(rlp, monkeypatch, tmp_path, adapter) is True
    assert adapter.cancel_all_called is True


# ---------------------------------------------------------------------------
# W7a — pending intents block preflight
# ---------------------------------------------------------------------------


def test_w7a_pending_intents_block_preflight(rlp, monkeypatch, tmp_path):
    adapter = _CleanupAdapter([], with_cancel_order=True)
    monkeypatch.setattr(rlp, "HALT_FLAG_PATH", tmp_path / "halt.json")
    import src.assembled_core.execution.kill_switch as ks

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    import src.assembled_core.execution.intent_store as intent_store

    monkeypatch.setattr(
        intent_store,
        "find_pending_order_intents",
        lambda: [{"intent_id": "crash-residue-1"}],
    )
    app_cfg = {"paper_runner": {"start_capital": 100.0, "dd_stop_pct": 0.99}}
    assert rlp._preflight_checks(adapter, app_cfg) is False


# ---------------------------------------------------------------------------
# W7b — position mismatches trip the halt below the cash threshold
# ---------------------------------------------------------------------------


def _policy():
    return {
        "halt_on_mismatch": True,
        "cash_threshold_usd": 100.0,
        "cash_threshold_bps": 10.0,
    }


def test_w7a_broken_intent_checker_fails_closed(rlp, monkeypatch, tmp_path):
    # Stage-1 review M2: a corrupt intent store (likely exactly after a
    # crash) must not fail open into trading.
    adapter = _CleanupAdapter([], with_cancel_order=True)
    monkeypatch.setattr(rlp, "HALT_FLAG_PATH", tmp_path / "halt.json")
    import src.assembled_core.execution.kill_switch as ks

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    import src.assembled_core.execution.intent_store as intent_store

    def _boom():
        raise RuntimeError("simulated corrupt intent store")

    monkeypatch.setattr(intent_store, "find_pending_order_intents", _boom)
    app_cfg = {"paper_runner": {"start_capital": 100.0, "dd_stop_pct": 0.99}}
    assert rlp._preflight_checks(adapter, app_cfg) is False


def test_w7b_missing_in_ledger_trips_the_halt(rlp):
    # Stage-1 review B1: the 2026-07-20 incident class — broker positions
    # entirely unknown to the ledger land in missing_in_ledger, NOT in
    # mismatches. They must trip even with a tiny cash_diff.
    sync = SimpleNamespace(
        cash_diff=1.83,
        broker_equity=86_878.0,
        mismatches=[],
        missing_in_ledger=["AAL", "BIIB", "MRNA", "TDG", "V"],
        missing_in_broker=[],
    )
    tripped, reason = rlp._sync_trips_halt(sync, _policy())
    assert tripped is True
    assert "missing_in_ledger=5" in reason


def test_w7b_missing_in_broker_trips_the_halt(rlp):
    sync = SimpleNamespace(
        cash_diff=0.0,
        broker_equity=86_878.0,
        mismatches=[],
        missing_in_ledger=[],
        missing_in_broker=["GLD"],
    )
    tripped, reason = rlp._sync_trips_halt(sync, _policy())
    assert tripped is True
    assert "missing_in_broker=1" in reason


def test_w7b_position_mismatch_trips_despite_small_cash_diff(rlp):
    sync = SimpleNamespace(
        cash_diff=1.83,
        broker_equity=86_878.0,
        mismatches=[{"symbol": "AAL", "ledger_qty": 0, "broker_qty": 213}],
    )
    tripped, reason = rlp._sync_trips_halt(sync, _policy())
    assert tripped is True
    assert "position_mismatches=1" in reason


def test_w7b_cash_breach_still_trips(rlp):
    sync = SimpleNamespace(cash_diff=17_237.33, broker_equity=86_661.0, mismatches=[])
    tripped, reason = rlp._sync_trips_halt(sync, _policy())
    assert tripped is True


def test_w7b_clean_sync_does_not_trip(rlp):
    sync = SimpleNamespace(cash_diff=1.83, broker_equity=86_878.0, mismatches=[])
    tripped, _reason = rlp._sync_trips_halt(sync, _policy())
    assert tripped is False


# ---------------------------------------------------------------------------
# K6 — E-053 PIT cutoff in the panel→daily.parquet refresh
# ---------------------------------------------------------------------------


def test_k6_same_day_panel_rows_are_dropped(tmp_path):
    mod = _load(REFRESH_PATH, "refresh_panel_p2_mod")
    today = pd.Timestamp.now("UTC").normalize()
    cache = pd.DataFrame(
        {
            "timestamp": [today - pd.Timedelta(days=3)],
            "symbol": ["AAPL"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "adj_close": [1.0],
            "volume": [100],
        }
    )
    panel = pd.DataFrame(
        {
            "timestamp": [
                today - pd.Timedelta(days=1),  # completed session -> append
                today,  # forming bar -> must be dropped (E-053)
            ],
            "symbol": ["AAPL", "AAPL"],
            "open": [2.0, 3.0],
            "high": [2.0, 3.0],
            "low": [2.0, 3.0],
            "close": [2.0, 3.0],
            "volume": [200, 300],
        }
    )
    cache_path = tmp_path / "daily.parquet"
    panel_path = tmp_path / "panel.parquet"
    cache.to_parquet(cache_path, index=False)
    panel.to_parquet(panel_path, index=False)
    # Redirect the status sidecar away from the repo output dir.
    mod.STATUS_PATH = tmp_path / "status.json"

    appended = mod.refresh(cache_path, panel_path, dry_run=False)
    assert appended == 1  # only the completed session

    merged = pd.read_parquet(cache_path)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    assert merged["timestamp"].max() == today - pd.Timedelta(days=1)
    assert (merged["timestamp"] < today).all()
