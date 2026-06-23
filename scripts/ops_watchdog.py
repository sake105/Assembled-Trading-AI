"""Paper-pilot ops watchdog — single idempotent pass (Task Scheduler ~every 15-30 min).
evaluate() is PURE (no I/O) and returns a list of Action tuples; apply_actions() performs
side effects. Actions: ("fire", rule_name, ctx) | ("liquidate", reason, ctx)."""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
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


def main(argv=None):  # pragma: no cover (I/O wiring added in a later task)
    argparse.ArgumentParser(description="paper-pilot ops watchdog").parse_args(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
