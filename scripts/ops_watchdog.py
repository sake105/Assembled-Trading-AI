"""Paper-pilot ops watchdog — single idempotent pass (Task Scheduler ~every 15-30 min).
evaluate() is PURE (no I/O) and returns a list of Action tuples; apply_actions() performs
side effects. Actions: ("fire", rule_name, ctx) | ("liquidate", reason, ctx)."""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import logging

logger = logging.getLogger("ops_watchdog")

HALT_FLAG = Path("output/ops/halt_ack_required.json")
SCHED_HB = Path("output/ops/scheduler_heartbeat.json")
STATE_HB = Path("output/state/heartbeat.json")
PILOT_MANIFEST = Path("output/pilot/pilot_manifest.json")
# Repo-verankert wie der Producer (pull_log.py schreibt _REPO_ROOT/output/ops,
# E-146) — CWD-relativ faende der Check ausserhalb des Repo-Roots nie ein Log
# und waere still inert (m2, Stage-1-Review 2026-08-16).
PULL_LOG_DIR = _REPO / "output" / "ops"
# E-176-Konsument (2026-08-17): erster Leser des sector_etf-Status-JSONs —
# repo-verankert wie PULL_LOG_DIR (E-146), Producer schreibt ROOT/output/ops.
SECTOR_STATUS = _REPO / "output" / "ops" / "refresh_sector_etf_status.json"
WATCHDOG_STATE = Path("output/ops/watchdog_state.json")
ALERT_CFG = Path("configs/alerting.yaml")
POLICY = Path("configs/policy.yaml")


def _parse_ts(s):
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def evaluate(state, snap, cfg, now):
    """Pure decision function. Returns list of Action tuples."""
    actions = []
    halt = snap.get("halt")

    if halt:
        halt_ts = _parse_ts(halt.get("ts_utc"))
        if state.get("last_seen_halt_ts") != halt.get("ts_utc"):
            actions.append(
                (
                    "fire",
                    "halt_flag_set",
                    {"reason": halt.get("reason", "?"), "equity": snap.get("equity")},
                )
            )
        if halt_ts is not None:
            age_h = (now - halt_ts).total_seconds() / 3600.0
            warn_after_h = cfg["warn_after_trading_days"] * 24
            window_h = cfg["liquidate_after_warning_hours"]
            warned_at = _parse_ts(state.get("warning_sent_at"))
            if age_h >= warn_after_h and warned_at is None:
                actions.append(
                    (
                        "fire",
                        "liquidation_warning",
                        {"window_hours": window_h, "age_h": round(age_h, 1)},
                    )
                )
            elif warned_at is not None and not state.get("liquidation_done"):
                since_warn_h = (now - warned_at).total_seconds() / 3600.0
                if since_warn_h >= window_h:
                    actions.append(
                        (
                            "liquidate",
                            "halt_unacked_grace_exceeded",
                            {"mode": cfg["flatten_mode"], "age_h": round(age_h, 1)},
                        )
                    )
    else:
        if state.get("last_seen_halt_ts") or state.get("warning_sent_at"):
            actions.append(
                (
                    "fire",
                    "halt_cleared",
                    {"actor": "operator", "reason": "flag_cleared"},
                )
            )

    # --- heartbeat staleness (alert only; DMS daemon owns the flatten) ---
    for source, key in (("scheduler", "sched_hb"), ("state", "state_hb")):
        hb = snap.get(key)
        if hb:
            hb_ts = _parse_ts(hb.get("timestamp_utc") or hb.get("timestamp"))
            if hb_ts is not None:
                age_h = (now - hb_ts).total_seconds() / 3600.0
                if age_h >= cfg["heartbeat_stale_hours"]:
                    actions.append(
                        (
                            "fire",
                            "heartbeat_stale",
                            {
                                "source": source,
                                "age_h": round(age_h, 1),
                                "threshold_h": cfg["heartbeat_stale_hours"],
                            },
                        )
                    )

    # --- run quality: N consecutive trailing runs with 0 orders ---
    manifest = snap.get("manifest")
    if manifest and manifest.get("days"):
        tail = manifest["days"][-cfg["zero_order_days"] :]
        if len(tail) >= cfg["zero_order_days"] and all(
            (d.get("n_orders_detected", 0) == 0) for d in tail
        ):
            actions.append(
                (
                    "fire",
                    "zero_orders_unexpected",
                    {"streak": cfg["zero_order_days"], "rc": tail[-1].get("rc")},
                )
            )

    # --- pull-log error ratio (E-112-Konsument, Audit-Plan 2.5b) ---
    # yfinance ist der einzige lebende Preispfad; sein Protokoll hatte bis
    # 2026-08-16 keinen einzigen Leser (Lautstaerke ohne Konsument, E-142).
    plog = snap.get("pull_log")
    if plog:
        requested = plog.get("requested") or 0
        # error + skipped: ein Rate-Limit-Abbruch protokolliert die nie
        # angefragten Symbole als skipped — ohne sie unterdrueckte der
        # gekuerzte Nenner genau den schweren Ausfall (F-senior-1/E-158).
        n_bad = (plog.get("error") or 0) + (plog.get("skipped") or 0)
        threshold = cfg.get("pull_log_error_ratio", 0.5)
        min_req = cfg.get("pull_log_min_requested", 5)
        if requested >= min_req and (n_bad / requested) >= threshold:
            actions.append(
                (
                    "fire",
                    "pull_log_errors",
                    {
                        "log": plog.get("log_name", "?"),
                        "n_error": n_bad,
                        "requested": requested,
                        "ratio": round(n_bad / requested, 2),
                        "threshold": threshold,
                    },
                )
            )

    # --- sector-refresh degradation (E-176-Konsument, 2026-08-17) ---
    # Der Drop-Pfad des Refreshers endet mit Exit 0 und ein Naht-Abbruch
    # (ok=false) erreichte nur die Tageslogdatei — das Status-JSON hatte
    # KEINEN Leser. Dedupe ueber ts_utc: jeder Status wird genau einmal
    # gemeldet (der Refresher schreibt taeglich, der Watchdog tickt 15-min).
    sstat = snap.get("sector_status")
    if sstat:
        degraded = (
            (not sstat.get("ok", True))
            or bool(sstat.get("dropped_symbols"))
            or bool(sstat.get("error"))
        )
        if degraded and state.get("last_alerted_sector_status_ts") != sstat.get(
            "ts_utc"
        ):
            actions.append(
                (
                    "fire",
                    "sector_refresh_degraded",
                    {
                        "ts_utc": sstat.get("ts_utc"),
                        "rc": sstat.get("rc"),
                        "dropped": ",".join(sstat.get("dropped_symbols") or []) or "-",
                        "error": str(sstat.get("error") or "")[:160],
                    },
                )
            )

    # --- drawdown breach vs peak ---
    equity, peak = snap.get("equity"), snap.get("peak")
    if equity is not None and peak and peak > 0:
        dd_pct = (equity / peak - 1.0) * 100.0
        if dd_pct <= cfg["dd_breach_pct"]:
            actions.append(
                (
                    "fire",
                    "drawdown_breach",
                    {
                        "dd_pct": round(dd_pct, 1),
                        "limit_pct": cfg["dd_breach_pct"],
                        "equity": equity,
                    },
                )
            )

    return actions


_CORRUPT = object()


def _load_json(path, *, on_error=None):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("[watchdog] failed to read/parse %s: %s", path, exc)
        return on_error


def _load_yaml(path):
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("[watchdog] failed to read/parse %s: %s", path, exc)
        return {}


def _resolve_state(loaded):
    """Fail-closed: a corrupt state file (lost escalation memory) must NOT let us
    re-liquidate. Treat corruption as 'already liquidated' for this tick + log."""
    if loaded is _CORRUPT:
        logger.error(
            "[watchdog] watchdog_state.json corrupt — fail-closed: suppressing liquidation this tick"
        )
        return {"liquidation_done": True}
    return loaded or {}


# NOTE (follow-up, review IMPORTANT #2): equity is scraped from the manifest output_snippet
# ("equity=" = broker-connect-time NAV at START of run, possibly truncated). peak=max over
# parsed days can under-report drawdown on parse-misses. drawdown_breach is alert-ONLY (never
# liquidation), so this is non-safety-critical; replace with a structured equity field later.
def load_snapshot():
    manifest = _load_json(PILOT_MANIFEST)
    equity = peak = None
    if manifest and manifest.get("days"):
        eqs = []
        for d in manifest["days"]:
            snip = d.get("output_snippet", "")
            i = snip.find("equity=")
            if i != -1:
                try:
                    eqs.append(float(snip[i + 7 :].split()[0].rstrip("\n")))
                except Exception:
                    pass
        if eqs:
            equity, peak = eqs[-1], max(eqs)
    # yfinance-Pull-Protokolle (E-112-Konsument), AGGREGIERT ueber alle
    # frischen Logs (F-senior-2, Stage 2 2026-08-16): fetch_prices_yfinance hat
    # vier Aufrufer mit voellig verschiedenen Symbolmengen (Pilot ~200,
    # Sektor-ETF ~11, prewarm 1-3) — nur das juengste Log zu lesen liesse
    # einen sauberen Kleinlauf einen kaputten Grosslauf maskieren. Aggregation
    # macht den Nenner ehrlich; ein Mindestnenner schuetzt vor
    # 1-von-2-Fehlalarmen.
    # M3 (Stage 1): Glob mit Ziffern-Praefix, damit 'pull_log_yfinance_
    # intraday_*' (Fremdquelle, sortiert lexikographisch IMMER zuoberst) gar
    # nicht erst gelesen wird; payload['source'] bleibt der autoritative
    # Filter (F-senior-12: Namensfilter != Quellenfilter).
    # BLINDFLECK, bewusst (F-senior-6): die Quote zaehlt error+skipped, NICHT
    # 'empty' — 220x empty (kein einziger Kurs) alarmiert hier nicht, weil
    # empty auch legitim ist (Feiertag, leeres Fenster). Der Total-Ausfall
    # dieser Form wird vom heartbeat-/zero-orders-Check getragen.
    pull_log = None
    _agg = {"requested": 0, "error": 0, "skipped": 0, "n_logs": 0, "log_name": ""}
    # Beide Seiten auf YYYY-MM-DDTHH:MM:SS geschnitten: der Producer schreibt
    # UTC-isoformat(timespec="seconds"); unparsbares finished_at ueberspringt
    # NUR dieses Log statt den Check stumm zu beenden (F-senior-5, E-142).
    _fresh_cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=3)).isoformat()[:19]
    for log_path in sorted(
        PULL_LOG_DIR.glob("pull_log_yfinance_2*.json"), reverse=True
    ):
        payload = _load_json(log_path)
        if not payload or payload.get("source") != "yfinance":
            continue
        finished = str(payload.get("finished_at") or "")[:19]
        if not finished:
            continue  # halbgeschriebenes/altes Schema: Log skippen, nicht Check
        if finished < _fresh_cutoff:
            break  # Namen tragen Zeitstempel: ab hier nur noch aeltere
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        _agg["requested"] += int(summary.get("requested") or 0)
        _agg["error"] += int(summary.get("error") or 0)
        _agg["skipped"] += int(summary.get("skipped") or 0)
        _agg["n_logs"] += 1
        if not _agg["log_name"]:
            _agg["log_name"] = log_path.name  # juengstes als Referenz
    if _agg["n_logs"]:
        pull_log = _agg

    return {
        "halt": _load_json(HALT_FLAG),
        "sched_hb": _load_json(SCHED_HB),
        "state_hb": _load_json(STATE_HB),
        "manifest": manifest,
        "equity": equity,
        "peak": peak,
        "pull_log": pull_log,
        "sector_status": _load_json(SECTOR_STATUS),
    }


def _do_liquidation(reason, ctx, policy):
    """Phase 1: the watchdog's OWN flatten_mode (carried in ctx['mode'], sourced from
    alerting.yaml alerts.watchdog.flatten_mode) is AUTHORITATIVE. Only 'market' may reach
    the kill-switch primitive; anything else logs a shadow event and returns — so a
    missing/corrupt policy.yaml can NEVER silently escalate to a real liquidation."""
    mode = str((ctx or {}).get("mode", "shadow")).lower()
    if mode != "market":
        logger.warning(
            "[watchdog] SHADOW liquidation (mode=%s) — logged, NOT executed. reason=%s",
            mode,
            reason,
        )
        return
    from src.assembled_core.ops.dead_man_switch import auto_flatten_on_stale

    auto_flatten_on_stale(policy, reason=reason)


def apply_actions(acts, am, state, policy, now):
    for a in acts:
        kind = a[0]
        if kind == "fire":
            _, rule, ctx = a
            delivered = am.fire(rule, ctx)
            if rule == "liquidation_warning":
                state["warning_sent_at"] = now.isoformat()
            elif rule == "sector_refresh_degraded" and delivered:
                # F-senior-1 (Stage 2, 2026-08-17): fire() gibt False bei
                # Cooldown/unbekannter Regel — Dedupe-Gedaechtnis nur
                # fortschreiben, was WIRKLICH zugestellt wurde (E-181),
                # sonst macht die Rate-Limitierung die Degradation stumm.
                state["last_alerted_sector_status_ts"] = ctx.get("ts_utc")
        elif kind == "liquidate":
            _, reason, ctx = a
            _do_liquidation(reason, ctx, policy)
            state["liquidation_done"] = True
            am.fire("liquidation_executed", {"mode": ctx.get("mode"), "detail": reason})
    return state


def main(
    argv=None,
):  # pragma: no cover (thin I/O wiring; logic covered by evaluate/apply tests)
    argparse.ArgumentParser(description="paper-pilot ops watchdog").parse_args(argv)
    # Load .env so alert credentials (TELEGRAM_*) are present when run headless under
    # Task Scheduler — without this, AlertManager silently logs "credentials not set".
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO / ".env")
    except Exception:
        pass
    from src.assembled_core.ops.alerting import AlertManager

    cfg_all = _load_yaml(ALERT_CFG).get("alerts", {})
    cfg = cfg_all.get("watchdog", {})
    policy = _load_yaml(POLICY)
    state = _resolve_state(_load_json(WATCHDOG_STATE, on_error=_CORRUPT))
    snap = load_snapshot()
    now = datetime.now(timezone.utc)
    acts = evaluate(state, snap, cfg, now)
    am = AlertManager(ALERT_CFG)
    apply_actions(acts, am, state, policy, now)
    halt = snap.get("halt")
    state["last_seen_halt_ts"] = (halt or {}).get("ts_utc")
    if not halt:
        state.pop("warning_sent_at", None)
        state.pop("liquidation_done", None)
    try:
        WATCHDOG_STATE.parent.mkdir(parents=True, exist_ok=True)
        WATCHDOG_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("[watchdog] failed to persist state %s: %s", WATCHDOG_STATE, exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
