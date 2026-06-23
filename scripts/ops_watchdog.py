"""Paper-pilot ops watchdog — single idempotent pass (Task Scheduler ~every 15-30 min).
evaluate() is PURE (no I/O) and returns a list of Action tuples; apply_actions() performs
side effects. Actions: ("fire", rule_name, ctx) | ("liquidate", reason, ctx)."""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

HALT_FLAG = Path("output/ops/halt_ack_required.json")
SCHED_HB = Path("output/ops/scheduler_heartbeat.json")
STATE_HB = Path("output/state/heartbeat.json")
PILOT_MANIFEST = Path("output/pilot/pilot_manifest.json")
WATCHDOG_STATE = Path("output/ops/watchdog_state.json")
ALERT_CFG = Path("configs/alerting.yaml")
POLICY = Path("configs/policy.yaml")


def _parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


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


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_yaml(path):
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


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
    return {
        "halt": _load_json(HALT_FLAG),
        "sched_hb": _load_json(SCHED_HB),
        "state_hb": _load_json(STATE_HB),
        "manifest": manifest,
        "equity": equity,
        "peak": peak,
    }


def _do_liquidation(reason, ctx, policy):
    """Phase 1: shadow only — delegate to the existing kill-switch primitive (does NOT sell).
    Phase 2 will replace this body with broker.close_all_positions() under approval."""
    from src.assembled_core.ops.dead_man_switch import auto_flatten_on_stale

    auto_flatten_on_stale(policy, reason=reason)


def apply_actions(acts, am, state, policy, now):
    for a in acts:
        kind = a[0]
        if kind == "fire":
            _, rule, ctx = a
            am.fire(rule, ctx)
            if rule == "liquidation_warning":
                state["warning_sent_at"] = now.isoformat()
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
    from src.assembled_core.ops.alerting import AlertManager

    cfg_all = _load_yaml(ALERT_CFG).get("alerts", {})
    cfg = cfg_all.get("watchdog", {})
    policy = _load_yaml(POLICY)
    state = _load_json(WATCHDOG_STATE) or {}
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
    WATCHDOG_STATE.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
