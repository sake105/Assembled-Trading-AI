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


def test_stale_heartbeat_fires():
    hb = {"timestamp_utc": (NOW - timedelta(hours=30)).isoformat()}
    acts = ow.evaluate(state={}, snap=_snap(sched_hb=hb), cfg=CFG, now=NOW)
    assert "heartbeat_stale" in [a[1] for a in acts if a[0] == "fire"]


def test_fresh_heartbeat_silent():
    hb = {"timestamp_utc": (NOW - timedelta(hours=2)).isoformat()}
    acts = ow.evaluate(state={}, snap=_snap(sched_hb=hb), cfg=CFG, now=NOW)
    assert "heartbeat_stale" not in [a[1] for a in acts if a[0] == "fire"]


def test_zero_orders_streak_fires():
    days = [{"rc": 1, "n_orders_detected": 0}, {"rc": 1, "n_orders_detected": 0}]
    acts = ow.evaluate(state={}, snap=_snap(manifest={"days": days}), cfg=CFG, now=NOW)
    assert "zero_orders_unexpected" in [a[1] for a in acts if a[0] == "fire"]


def test_one_zero_order_day_silent():
    days = [{"rc": 0, "n_orders_detected": 3}, {"rc": 1, "n_orders_detected": 0}]
    acts = ow.evaluate(state={}, snap=_snap(manifest={"days": days}), cfg=CFG, now=NOW)
    assert "zero_orders_unexpected" not in [a[1] for a in acts if a[0] == "fire"]


def test_drawdown_breach_fires():
    acts = ow.evaluate(
        state={}, snap=_snap(equity=88000, peak=100000), cfg=CFG, now=NOW
    )
    assert "drawdown_breach" in [a[1] for a in acts if a[0] == "fire"]


def test_drawdown_ok_silent():
    acts = ow.evaluate(
        state={}, snap=_snap(equity=97000, peak=100000), cfg=CFG, now=NOW
    )
    assert "drawdown_breach" not in [a[1] for a in acts if a[0] == "fire"]


class _FakeAM:
    def __init__(self):
        self.fired = []

    def fire(self, rule, ctx=None):
        self.fired.append((rule, ctx))
        return True


def test_apply_actions_fires_and_shadow_liquidation(monkeypatch):
    am = _FakeAM()
    called = {"flatten": 0}
    monkeypatch.setattr(
        ow,
        "_do_liquidation",
        lambda reason, ctx, policy: called.__setitem__(
            "flatten", called["flatten"] + 1
        ),
    )
    acts = [
        ("fire", "halt_flag_set", {"reason": "x", "equity": 1}),
        ("liquidate", "halt_unacked_grace_exceeded", {"mode": "shadow"}),
    ]
    state = {}
    ow.apply_actions(acts, am=am, state=state, policy={}, now=NOW)
    assert ("halt_flag_set", {"reason": "x", "equity": 1}) in am.fired
    assert called["flatten"] == 1
    assert state.get("liquidation_done") is True
