from datetime import datetime, timezone, timedelta
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "ops_watchdog",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ops_watchdog.py",
)
ow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ow)

CFG = {
    "warn_after_trading_days": 1,
    "liquidate_after_warning_hours": 4,
    "heartbeat_stale_hours": 26,
    "zero_order_days": 2,
    "dd_breach_pct": -8.0,
    "flatten_mode": "shadow",
}
NOW = datetime(2026, 6, 22, 18, 0, tzinfo=timezone.utc)


def _snap(**kw):
    base = {
        "halt": None,
        "sched_hb": None,
        "state_hb": None,
        "manifest": None,
        "equity": None,
        "peak": None,
    }
    base.update(kw)
    return base


def test_no_halt_no_actions():
    acts = ow.evaluate(state={}, snap=_snap(), cfg=CFG, now=NOW)
    assert acts == []


def test_fresh_halt_fires_once():
    halt = {"ts_utc": (NOW - timedelta(hours=2)).isoformat(), "reason": "soft-timeout"}
    acts = ow.evaluate(state={}, snap=_snap(halt=halt, equity=88000), cfg=CFG, now=NOW)
    assert ("fire", "halt_flag_set") in [(a[0], a[1]) for a in acts]


def test_halt_already_seen_no_refire():
    halt = {"ts_utc": (NOW - timedelta(hours=2)).isoformat(), "reason": "x"}
    state = {"last_seen_halt_ts": halt["ts_utc"]}
    acts = ow.evaluate(state=state, snap=_snap(halt=halt), cfg=CFG, now=NOW)
    assert "halt_flag_set" not in [a[1] for a in acts if a[0] == "fire"]


def test_unacked_past_warn_fires_warning():
    halt = {"ts_utc": (NOW - timedelta(days=2)).isoformat(), "reason": "x"}
    state = {"last_seen_halt_ts": halt["ts_utc"]}
    acts = ow.evaluate(state=state, snap=_snap(halt=halt), cfg=CFG, now=NOW)
    assert "liquidation_warning" in [a[1] for a in acts if a[0] == "fire"]


def test_unacked_past_window_liquidates():
    halt = {"ts_utc": (NOW - timedelta(days=2)).isoformat(), "reason": "x"}
    state = {
        "last_seen_halt_ts": halt["ts_utc"],
        "warning_sent_at": (NOW - timedelta(hours=5)).isoformat(),
    }
    acts = ow.evaluate(state=state, snap=_snap(halt=halt), cfg=CFG, now=NOW)
    assert any(a[0] == "liquidate" for a in acts)


def test_liquidation_only_once():
    halt = {"ts_utc": (NOW - timedelta(days=2)).isoformat(), "reason": "x"}
    state = {
        "last_seen_halt_ts": halt["ts_utc"],
        "warning_sent_at": (NOW - timedelta(hours=5)).isoformat(),
        "liquidation_done": True,
    }
    acts = ow.evaluate(state=state, snap=_snap(halt=halt), cfg=CFG, now=NOW)
    assert not any(a[0] == "liquidate" for a in acts)


def test_ack_clears_escalation():
    state = {
        "last_seen_halt_ts": "2026-06-20T00:00:00+00:00",
        "warning_sent_at": "2026-06-21T00:00:00+00:00",
    }
    acts = ow.evaluate(state=state, snap=_snap(halt=None), cfg=CFG, now=NOW)
    assert "halt_cleared" in [a[1] for a in acts if a[0] == "fire"]
    assert not any(a[0] == "liquidate" for a in acts)
