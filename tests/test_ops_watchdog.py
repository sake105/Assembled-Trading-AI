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
    """UMGESTELLT 2026-08-18 auf den KANONISCHEN Heartbeat.

    Der Test nutzte ``sched_hb`` (output/ops/scheduler_heartbeat.json). Diese
    Datei wurde mit OPS-02 im Juni 2026 durch output/state/heartbeat.json
    ersetzt und seither von NIEMANDEM mehr geschrieben — der Watchdog
    alarmierte dafuer taeglich CRITICAL "Alter=1661h" (Alert-Fatigue, E-189).
    Der Alarm haengt jetzt am lebenden Producer; die Staleness-Semantik selbst
    ist unveraendert und hier weiter gepinnt."""
    hb = {"timestamp": (NOW - timedelta(hours=30)).isoformat()}
    acts = ow.evaluate(state={}, snap=_snap(state_hb=hb), cfg=CFG, now=NOW)
    assert "heartbeat_stale" in [a[1] for a in acts if a[0] == "fire"]


def test_fresh_heartbeat_silent():
    hb = {"timestamp": (NOW - timedelta(hours=2)).isoformat()}
    acts = ow.evaluate(state={}, snap=_snap(state_hb=hb), cfg=CFG, now=NOW)
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


def test_do_liquidation_shadow_does_not_call_primitive(monkeypatch):
    import sys
    import types

    called = {"n": 0}
    fake = types.ModuleType("src.assembled_core.ops.dead_man_switch")
    fake.auto_flatten_on_stale = lambda policy, reason="": called.__setitem__(
        "n", called["n"] + 1
    )
    monkeypatch.setitem(sys.modules, "src.assembled_core.ops.dead_man_switch", fake)
    ow._do_liquidation("grace", {"mode": "shadow"}, {})
    assert (
        called["n"] == 0
    )  # shadow short-circuits BEFORE importing/calling the primitive


def test_do_liquidation_market_calls_primitive(monkeypatch):
    import sys
    import types

    called = {"reason": None}
    fake = types.ModuleType("src.assembled_core.ops.dead_man_switch")
    fake.auto_flatten_on_stale = lambda policy, reason="": called.__setitem__(
        "reason", reason
    )
    monkeypatch.setitem(sys.modules, "src.assembled_core.ops.dead_man_switch", fake)
    ow._do_liquidation(
        "grace", {"mode": "market"}, {"dead_man_switch": {"flatten_mode": "shadow"}}
    )
    assert called["reason"] == "grace"


def test_corrupt_state_fail_closed():
    st = ow._resolve_state(ow._CORRUPT)
    assert st.get("liquidation_done") is True


def test_naive_ts_halt_does_not_crash():
    halt = {"ts_utc": "2026-06-20T00:00:00", "reason": "x"}  # naive, no offset
    acts = ow.evaluate(
        state={"last_seen_halt_ts": halt["ts_utc"]},
        snap=_snap(halt=halt),
        cfg=CFG,
        now=NOW,
    )
    assert isinstance(acts, list)  # must not raise


# --- pull_log-Konsument (Audit-Plan 2.5b, 2026-08-16) -------------------------


def test_pull_log_high_error_ratio_fires():
    plog = {
        "requested": 10,
        "error": 6,
        "ok": 4,
        "log_name": "pull_log_yfinance_x.json",
    }
    acts = ow.evaluate(state={}, snap=_snap(pull_log=plog), cfg=CFG, now=NOW)
    rules = [a[1] for a in acts if a[0] == "fire"]
    assert "pull_log_errors" in rules
    ctx = next(a[2] for a in acts if a[0] == "fire" and a[1] == "pull_log_errors")
    assert ctx["n_error"] == 6 and ctx["requested"] == 10


def test_pull_log_low_error_ratio_silent():
    plog = {"requested": 10, "error": 1, "ok": 9, "log_name": "x.json"}
    acts = ow.evaluate(state={}, snap=_snap(pull_log=plog), cfg=CFG, now=NOW)
    assert all(a[1] != "pull_log_errors" for a in acts if a[0] == "fire")


def test_pull_log_missing_or_empty_silent():
    # kein Protokoll -> kein Alarm aus Buchhaltung
    acts = ow.evaluate(state={}, snap=_snap(pull_log=None), cfg=CFG, now=NOW)
    assert all(a[1] != "pull_log_errors" for a in acts if a[0] == "fire")
    # requested=0 (leerer Lauf) -> kein Nulldivisions-Alarm
    acts = ow.evaluate(
        state={}, snap=_snap(pull_log={"requested": 0, "error": 0}), cfg=CFG, now=NOW
    )
    assert all(a[1] != "pull_log_errors" for a in acts if a[0] == "fire")


def test_pull_log_skipped_counts_as_bad():
    """F-senior-1: Rate-Limit-Abbruch — skipped zaehlt in die Quote, sonst
    unterdrueckt der gekuerzte Nenner den schweren Ausfall."""
    plog = {"requested": 220, "error": 1, "skipped": 217, "log_name": "x.json"}
    acts = ow.evaluate(state={}, snap=_snap(pull_log=plog), cfg=CFG, now=NOW)
    assert any(a[1] == "pull_log_errors" for a in acts if a[0] == "fire")


def test_pull_log_min_requested_guards_small_runs():
    """F-senior-2: 1-von-2-Kleinlauf darf nicht alarmieren."""
    plog = {"requested": 2, "error": 1, "skipped": 0, "log_name": "x.json"}
    acts = ow.evaluate(state={}, snap=_snap(pull_log=plog), cfg=CFG, now=NOW)
    assert all(a[1] != "pull_log_errors" for a in acts if a[0] == "fire")


def test_load_snapshot_aggregates_and_filters(tmp_path, monkeypatch):
    """F-senior-7: load_snapshot-Bindung — Aggregation ueber frische Logs,
    Fremdquellen und alte Logs fallen raus."""
    import json as _json
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    monkeypatch.setattr(ow, "PULL_LOG_DIR", tmp_path)
    now = _dt.now(tz=_tz.utc)

    def _write(name, source, finished, requested, error, skipped=0):
        (tmp_path / name).write_text(
            _json.dumps(
                {
                    "source": source,
                    "finished_at": finished.isoformat(timespec="seconds"),
                    "summary": {
                        "requested": requested,
                        "error": error,
                        "skipped": skipped,
                    },
                }
            ),
            encoding="utf-8",
        )

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    _write(f"pull_log_yfinance_{stamp}.json", "yfinance", now, 200, 100)
    _write(
        "pull_log_yfinance_20260101T000000Z.json",
        "yfinance",
        now - _td(days=30),
        50,
        50,
    )  # zu alt
    _write(
        f"pull_log_yfinance_intraday_{stamp}.json", "yfinance_intraday", now, 999, 999
    )  # Fremdquelle
    snap = ow.load_snapshot()
    pl = snap["pull_log"]
    assert pl is not None
    assert pl["requested"] == 200 and pl["error"] == 100  # nur das frische echte
    assert pl["n_logs"] == 1


# --- sector_status-Konsument (E-176, 2026-08-17) ------------------------------


def _sector_status(**kw):
    base = {
        "ts_utc": "2026-08-17T05:00:00+00:00",
        "rc": 11,
        "ok": True,
        "dropped_symbols": [],
        "error": None,
    }
    base.update(kw)
    return base


def test_sector_status_seam_abort_fires():
    """ok=false (Naht-Abbruch, rc=-2) muss alarmieren."""
    sstat = _sector_status(rc=-2, ok=False, error="seam_guard: MERGE ABORTED")
    acts = ow.evaluate(state={}, snap=_snap(sector_status=sstat), cfg=CFG, now=NOW)
    hits = [a for a in acts if a[0] == "fire" and a[1] == "sector_refresh_degraded"]
    assert len(hits) == 1
    assert "seam_guard" in hits[0][2]["error"]


def test_sector_status_dropped_symbols_fires_despite_ok_true():
    """Der Drop-Pfad endet mit Exit 0 und ok=true — genau deshalb muss der
    Watchdog auf dropped_symbols anspringen, nicht auf ok."""
    sstat = _sector_status(
        dropped_symbols=["XLK"], error="overlap_ratio_not_constant: ['XLK']"
    )
    acts = ow.evaluate(state={}, snap=_snap(sector_status=sstat), cfg=CFG, now=NOW)
    hits = [a for a in acts if a[0] == "fire" and a[1] == "sector_refresh_degraded"]
    assert len(hits) == 1
    assert hits[0][2]["dropped"] == "XLK"


def test_sector_status_healthy_silent():
    acts = ow.evaluate(
        state={}, snap=_snap(sector_status=_sector_status()), cfg=CFG, now=NOW
    )
    assert all(a[1] != "sector_refresh_degraded" for a in acts if a[0] == "fire")
    # fehlendes Status-JSON -> still (kein Fehlalarm vor dem ersten Lauf)
    acts = ow.evaluate(state={}, snap=_snap(sector_status=None), cfg=CFG, now=NOW)
    assert all(a[1] != "sector_refresh_degraded" for a in acts if a[0] == "fire")


def test_sector_status_dedupe_via_state_ts():
    """Gleicher ts_utc darf nur einmal gemeldet werden (15-min-Ticks vs.
    Tages-Producer); ein NEUER degradierter Status feuert wieder."""
    sstat = _sector_status(rc=-2, ok=False, error="seam_guard: x")
    state = {"last_alerted_sector_status_ts": sstat["ts_utc"]}
    acts = ow.evaluate(state=state, snap=_snap(sector_status=sstat), cfg=CFG, now=NOW)
    assert all(a[1] != "sector_refresh_degraded" for a in acts if a[0] == "fire")
    fresh = _sector_status(
        ts_utc="2026-08-18T05:00:00+00:00", rc=-2, ok=False, error="seam_guard: y"
    )
    acts = ow.evaluate(state=state, snap=_snap(sector_status=fresh), cfg=CFG, now=NOW)
    assert any(a[1] == "sector_refresh_degraded" for a in acts if a[0] == "fire")


def test_sector_status_apply_actions_records_ts():
    """apply_actions muss den gemeldeten ts_utc in den State schreiben
    (Dedupe-Gedaechtnis), ohne echten AlertManager."""

    class _AM:
        def __init__(self):
            self.fired = []

        def fire(self, rule, ctx):
            self.fired.append((rule, ctx))
            return True

    sstat = _sector_status(rc=-2, ok=False, error="seam_guard: x")
    acts = ow.evaluate(state={}, snap=_snap(sector_status=sstat), cfg=CFG, now=NOW)
    am = _AM()
    state = ow.apply_actions(acts, am, {}, policy={}, now=NOW)
    assert state["last_alerted_sector_status_ts"] == sstat["ts_utc"]
    assert any(r == "sector_refresh_degraded" for r, _ in am.fired)


def test_sector_status_cooldown_suppression_keeps_dedupe_open():
    """F-senior-1 (E-181): gibt fire() False zurueck (Cooldown/unbekannte
    Regel), darf der ts NICHT als gemeldet gelten — sonst macht die
    Rate-Limitierung die Degradation dauerhaft stumm."""

    class _SuppressingAM:
        def fire(self, rule, ctx):
            return False  # z.B. aktives Cooldown-Fenster

    sstat = _sector_status(rc=-2, ok=False, error="seam_guard: x")
    acts = ow.evaluate(state={}, snap=_snap(sector_status=sstat), cfg=CFG, now=NOW)
    state = ow.apply_actions(acts, _SuppressingAM(), {}, policy={}, now=NOW)
    assert "last_alerted_sector_status_ts" not in state
    # naechster Tick: derselbe Status muss erneut zur Meldung anstehen
    acts2 = ow.evaluate(state=state, snap=_snap(sector_status=sstat), cfg=CFG, now=NOW)
    assert any(a[1] == "sector_refresh_degraded" for a in acts2 if a[0] == "fire")


# --- Alert-Fatigue-Fixes 2026-08-18 (E-189) ---------------------------------


def test_no_heartbeat_alert_for_retired_scheduler_file():
    """Der Relikt-Heartbeat (output/ops/scheduler_heartbeat.json, mit OPS-02
    im Juni 2026 durch output/state/heartbeat.json ERSETZT) darf keinen
    Alarm mehr ausloesen — er meldete taeglich CRITICAL 'Alter=1661h' fuer
    einen Producer, den es nicht mehr gibt."""
    ancient = {"timestamp_utc": (NOW - timedelta(days=69)).isoformat()}
    fresh = {"timestamp": (NOW - timedelta(hours=1)).isoformat()}
    acts = ow.evaluate(
        state={}, snap=_snap(sched_hb=ancient, state_hb=fresh), cfg=CFG, now=NOW
    )
    assert all(a[1] != "heartbeat_stale" for a in acts if a[0] == "fire")


def test_canonical_heartbeat_still_alerts_when_stale():
    """Gegenprobe: der KANONISCHE Heartbeat muss weiterhin alarmieren."""
    stale = {"timestamp": (NOW - timedelta(hours=48)).isoformat()}
    acts = ow.evaluate(state={}, snap=_snap(state_hb=stale), cfg=CFG, now=NOW)
    fired = [a for a in acts if a[0] == "fire" and a[1] == "heartbeat_stale"]
    assert len(fired) == 1
    assert fired[0][2]["source"] == "state"


def test_fallback_receipt_is_anchored_to_the_log_run_not_to_now():
    """E-189-Nachbesserung: die Quittung muss zum LAUF passen, nicht zu
    `now`. Erste Fassung prueft "juenger als 6h" gegen jetzt — nach einem
    Nachtlauf waren Log UND Quittung gleich alt, der Alarm kehrte zurueck,
    obwohl der Fallback die Daten geliefert hatte."""
    old_run = NOW - timedelta(hours=11)
    plog = {
        "requested": 1138,
        "error": 1056,
        "skipped": 0,
        "log_name": "x.json",
        "finished_at": old_run.isoformat(),
    }
    fb = {
        "ts_utc": (old_run + timedelta(minutes=2)).isoformat(),
        "data_latest": "2026-08-17",
    }
    acts = ow.evaluate(
        state={}, snap=_snap(pull_log=plog, pull_fallback=fb), cfg=CFG, now=NOW
    )
    assert all(a[1] != "pull_log_errors" for a in acts if a[0] == "fire")

    # Gegenprobe: Quittung stammt aus einem FRUEHEREN Lauf (Stunden vor dem
    # Log) -> sie deckt diesen Ausfall nicht.
    stale_fb = {
        "ts_utc": (old_run - timedelta(hours=5)).isoformat(),
        "data_latest": "2026-08-10",
    }
    acts2 = ow.evaluate(
        state={}, snap=_snap(pull_log=plog, pull_fallback=stale_fb), cfg=CFG, now=NOW
    )
    assert any(a[1] == "pull_log_errors" for a in acts2 if a[0] == "fire")


def test_pull_log_alert_suppressed_when_fallback_delivered():
    """Eine hohe yfinance-Fehlerquote ist KEIN Datenausfall, wenn der
    Alpaca-Fallback im selben Lauf die Daten geliefert hat (der Alarm
    feuerte sonst taeglich, obwohl das Panel vollstaendig war)."""
    plog = {"requested": 1117, "error": 1037, "skipped": 0, "log_name": "x.json"}
    fb = {
        "ts_utc": (NOW - timedelta(minutes=20)).isoformat(),
        "fallback_source": "alpaca",
        "fallback_rows": 15030,
    }
    acts = ow.evaluate(
        state={}, snap=_snap(pull_log=plog, pull_fallback=fb), cfg=CFG, now=NOW
    )
    assert all(a[1] != "pull_log_errors" for a in acts if a[0] == "fire")


def test_pull_log_alert_fires_when_fallback_is_stale_or_empty():
    """Gegenprobe in beide Richtungen: ein ALTER Fallback (anderer Lauf) und
    ein LEERER Fallback duerfen den echten Ausfall nicht verdecken."""
    plog = {"requested": 1117, "error": 1037, "skipped": 0, "log_name": "x.json"}
    old_fb = {
        "ts_utc": (NOW - timedelta(days=3)).isoformat(),
        "fallback_rows": 15030,
    }
    acts = ow.evaluate(
        state={}, snap=_snap(pull_log=plog, pull_fallback=old_fb), cfg=CFG, now=NOW
    )
    assert any(a[1] == "pull_log_errors" for a in acts if a[0] == "fire")

    empty_fb = {"ts_utc": (NOW - timedelta(minutes=5)).isoformat(), "fallback_rows": 0}
    acts2 = ow.evaluate(
        state={}, snap=_snap(pull_log=plog, pull_fallback=empty_fb), cfg=CFG, now=NOW
    )
    assert any(a[1] == "pull_log_errors" for a in acts2 if a[0] == "fire")
