# Paper-Pilot Alerting + Halt-Handling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the silent-halt monitoring gap: when the paper pilot sets a halt flag (or heartbeat goes stale / runs produce 0 orders / drawdown breaches), deliver a real Telegram alert; after a *warned* grace window without operator ack, escalate to liquidation.

**Architecture:** Standalone idempotent watchdog (`scripts/ops_watchdog.py`) run by Task Scheduler, orchestrating the existing `AlertManager` (real delivery) and — Phase 2 only — a new protected `close_all_positions` broker primitive. Two-stage halt escalation: warn → intervention window → liquidate. Shadow-first.

**Tech Stack:** Python 3.13, PyYAML, existing `ops/alerting.py` AlertManager, `ops/dead_man_switch.py`, pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-06-22-pilot-alerting-halt-handling-design.md`

---

## File Structure

| File | Responsibility | Phase | Protected? |
|---|---|---|---|
| `scripts/ops_watchdog.py` | watchdog: pure `evaluate()` (decision) + `apply_actions()` (I/O) + `main()` | 1 | no |
| `tests/test_ops_watchdog.py` | unit tests for `evaluate()` across all fixtures | 1 | no |
| `configs/alerting.yaml` | telegram channel + rules + thresholds | 1 | no |
| `scripts/run_live_paper.py` | +`fire("halt_flag_set")` at the halt-write site | 1 | no (review-chain) |
| `scripts/ack_halt.py` | +`fire("halt_cleared")` after clear | 1 | no (review-chain) |
| `scripts/register_ops_tasks.ps1` | document/register Task Scheduler entries (watchdog + DMS) | 1 | no |
| `src/assembled_core/execution/broker_adapter.py` | +`close_all_positions()` (Phase 2) | 2 | **YES — gated** |

**Decision/IO split:** `evaluate(state, snapshot, cfg, now) -> list[Action]` is pure (no I/O, no broker, no telegram) → fully unit-testable. `apply_actions(actions, ...)` performs the side effects (fire alerts, trigger liquidation, persist state). Tests target `evaluate()` only.

---

## PHASE 1 — Alerting + Watchdog + Warning (non-protected)

### Task 1: Alerting config with telegram + rules + thresholds

**Files:**
- Create/extend: `configs/alerting.yaml`

- [ ] **Step 1: Inspect whether the file already exists and its shape**

Run: `python -c "import os;p='configs/alerting.yaml';print(open(p).read() if os.path.exists(p) else 'ABSENT')"`
If present, MERGE the keys below into the existing `alerts:` block (do not clobber existing rules). If absent, create with exactly this content.

- [ ] **Step 2: Write the config**

```yaml
# configs/alerting.yaml — paper-pilot ops alerting (read by ops/alerting.py AlertManager)
alerts:
  # Channels per severity. log_only is always safe; telegram needs env creds.
  channels:
    critical:
      - type: log_only
      - type: telegram
        bot_token_env: TELEGRAM_BOT_TOKEN
        chat_id_env: TELEGRAM_CHAT_ID
    warning:
      - type: log_only
      - type: telegram
        bot_token_env: TELEGRAM_BOT_TOKEN
        chat_id_env: TELEGRAM_CHAT_ID
  rules:
    - name: halt_flag_set
      severity: critical
      cooldown_minutes: 60
      message: "PILOT HALT set: {reason} (equity={equity}). Ack via scripts/ack_halt.py."
    - name: halt_cleared
      severity: warning
      cooldown_minutes: 0
      message: "PILOT halt cleared by {actor}: {reason}."
    - name: liquidation_warning
      severity: critical
      cooldown_minutes: 120
      message: "AUTO-LIQUIDATION in {window_hours}h unless you ack the halt now (unacked {age_h}h). scripts/ack_halt.py"
    - name: liquidation_executed
      severity: critical
      cooldown_minutes: 0
      message: "AUTO-LIQUIDATION triggered (mode={mode}): {detail}"
    - name: heartbeat_stale
      severity: critical
      cooldown_minutes: 180
      message: "Pilot heartbeat STALE: {source} age={age_h}h (threshold {threshold_h}h)."
    - name: zero_orders_unexpected
      severity: warning
      cooldown_minutes: 720
      message: "Pilot produced 0 orders for {streak} weekday runs (last rc={rc})."
    - name: drawdown_breach
      severity: critical
      cooldown_minutes: 360
      message: "Pilot drawdown {dd_pct}% breached limit {limit_pct}% (equity={equity})."
  # Watchdog thresholds (read by scripts/ops_watchdog.py)
  watchdog:
    warn_after_trading_days: 1
    liquidate_after_warning_hours: 4
    heartbeat_stale_hours: 26
    zero_order_days: 2
    dd_breach_pct: -8.0
    flatten_mode: shadow   # shadow until Phase-1 telegram path verified, then -> market in Phase 2
```

- [ ] **Step 3: Validate it parses**

Run: `python -c "import yaml;c=yaml.safe_load(open('configs/alerting.yaml'));assert c['alerts']['watchdog']['flatten_mode']=='shadow';print('OK',list(c['alerts']))"`
Expected: `OK ['channels', 'rules', 'watchdog']`

- [ ] **Step 4: Commit**

```bash
git add configs/alerting.yaml
git commit -m "feat(ops): alerting.yaml telegram channel + watchdog rules/thresholds"
```

---

### Task 2: Watchdog `evaluate()` — halt escalation (TDD)

**Files:**
- Create: `tests/test_ops_watchdog.py`
- Create: `scripts/ops_watchdog.py`

- [ ] **Step 1: Write the failing test (halt escalation)**

```python
# tests/test_ops_watchdog.py
from datetime import datetime, timezone, timedelta
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "ops_watchdog", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ops_watchdog.py")
ow = importlib.util.module_from_spec(spec); spec.loader.exec_module(ow)

CFG = {"warn_after_trading_days": 1, "liquidate_after_warning_hours": 4,
       "heartbeat_stale_hours": 26, "zero_order_days": 2, "dd_breach_pct": -8.0,
       "flatten_mode": "shadow"}
NOW = datetime(2026, 6, 22, 18, 0, tzinfo=timezone.utc)

def _snap(**kw):
    base = {"halt": None, "sched_hb": None, "state_hb": None, "manifest": None, "equity": None, "peak": None}
    base.update(kw); return base

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
    state = {"last_seen_halt_ts": halt["ts_utc"],
             "warning_sent_at": (NOW - timedelta(hours=5)).isoformat()}
    acts = ow.evaluate(state=state, snap=_snap(halt=halt), cfg=CFG, now=NOW)
    assert any(a[0] == "liquidate" for a in acts)

def test_liquidation_only_once():
    halt = {"ts_utc": (NOW - timedelta(days=2)).isoformat(), "reason": "x"}
    state = {"last_seen_halt_ts": halt["ts_utc"],
             "warning_sent_at": (NOW - timedelta(hours=5)).isoformat(),
             "liquidation_done": True}
    acts = ow.evaluate(state=state, snap=_snap(halt=halt), cfg=CFG, now=NOW)
    assert not any(a[0] == "liquidate" for a in acts)

def test_ack_clears_escalation():
    # halt gone but state remembers a prior warning -> emit halt_cleared, no liquidation
    state = {"last_seen_halt_ts": "2026-06-20T00:00:00+00:00", "warning_sent_at": "2026-06-21T00:00:00+00:00"}
    acts = ow.evaluate(state=state, snap=_snap(halt=None), cfg=CFG, now=NOW)
    assert "halt_cleared" in [a[1] for a in acts if a[0] == "fire"]
    assert not any(a[0] == "liquidate" for a in acts)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ops_watchdog.py -q`
Expected: FAIL — `scripts/ops_watchdog.py` has no `evaluate` (ModuleNotFound on exec / AttributeError).

- [ ] **Step 3: Implement the watchdog skeleton + halt escalation in `evaluate()`**

```python
# scripts/ops_watchdog.py
"""Paper-pilot ops watchdog — single idempotent pass (Task Scheduler ~every 15-30 min).
evaluate() is PURE (no I/O) and returns a list of Action tuples; apply_actions() performs
side effects. Actions: ("fire", rule_name, ctx) | ("liquidate", reason, ctx)."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone, timedelta
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

    # --- halt escalation (two-stage with intervention window) ---
    if halt:
        halt_ts = _parse_ts(halt.get("ts_utc"))
        # stage 0: fresh halt -> fire once
        if state.get("last_seen_halt_ts") != halt.get("ts_utc"):
            actions.append(("fire", "halt_flag_set",
                            {"reason": halt.get("reason", "?"), "equity": snap.get("equity")}))
        if halt_ts is not None:
            age_h = (now - halt_ts).total_seconds() / 3600.0
            warn_after_h = cfg["warn_after_trading_days"] * 24
            window_h = cfg["liquidate_after_warning_hours"]
            warned_at = _parse_ts(state.get("warning_sent_at"))
            # stage 1: past warn threshold and not yet warned -> warn
            if age_h >= warn_after_h and warned_at is None:
                actions.append(("fire", "liquidation_warning",
                                {"window_hours": window_h, "age_h": round(age_h, 1)}))
            # stage 2: warned, window elapsed, not yet liquidated -> liquidate
            elif warned_at is not None and not state.get("liquidation_done"):
                since_warn_h = (now - warned_at).total_seconds() / 3600.0
                if since_warn_h >= window_h:
                    actions.append(("liquidate", "halt_unacked_grace_exceeded",
                                    {"mode": cfg["flatten_mode"], "age_h": round(age_h, 1)}))
    else:
        # halt gone: if we had escalated, announce all-clear
        if state.get("last_seen_halt_ts") or state.get("warning_sent_at"):
            actions.append(("fire", "halt_cleared", {"actor": "operator", "reason": "flag_cleared"}))

    return actions


def main(argv=None):  # pragma: no cover (I/O wiring; covered by smoke)
    argparse.ArgumentParser(description="paper-pilot ops watchdog").parse_args(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run to verify halt tests pass**

Run: `pytest tests/test_ops_watchdog.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/ops_watchdog.py tests/test_ops_watchdog.py
git commit -m "feat(ops): watchdog evaluate() two-stage halt escalation (pure, TDD)"
```

---

### Task 3: Watchdog `evaluate()` — heartbeat / run-quality / drawdown (TDD)

**Files:**
- Modify: `tests/test_ops_watchdog.py`
- Modify: `scripts/ops_watchdog.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_ops_watchdog.py
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
    acts = ow.evaluate(state={}, snap=_snap(equity=88000, peak=100000), cfg=CFG, now=NOW)
    assert "drawdown_breach" in [a[1] for a in acts if a[0] == "fire"]

def test_drawdown_ok_silent():
    acts = ow.evaluate(state={}, snap=_snap(equity=97000, peak=100000), cfg=CFG, now=NOW)
    assert "drawdown_breach" not in [a[1] for a in acts if a[0] == "fire"]
```

- [ ] **Step 2: Run to verify new tests fail**

Run: `pytest tests/test_ops_watchdog.py -q`
Expected: FAIL on the 6 new tests (no heartbeat/run/dd logic yet).

- [ ] **Step 3: Add the three checks to `evaluate()` (before `return actions`)**

```python
    # --- heartbeat staleness (alert only; DMS daemon owns the flatten) ---
    for source, key in (("scheduler", "sched_hb"), ("state", "state_hb")):
        hb = snap.get(key)
        if hb:
            hb_ts = _parse_ts(hb.get("timestamp_utc") or hb.get("timestamp"))
            if hb_ts is not None:
                age_h = (now - hb_ts).total_seconds() / 3600.0
                if age_h >= cfg["heartbeat_stale_hours"]:
                    actions.append(("fire", "heartbeat_stale",
                                    {"source": source, "age_h": round(age_h, 1),
                                     "threshold_h": cfg["heartbeat_stale_hours"]}))

    # --- run quality: N consecutive trailing runs with 0 orders ---
    manifest = snap.get("manifest")
    if manifest and manifest.get("days"):
        tail = manifest["days"][-cfg["zero_order_days"]:]
        if len(tail) >= cfg["zero_order_days"] and all(
            (d.get("n_orders_detected", 0) == 0) for d in tail
        ):
            actions.append(("fire", "zero_orders_unexpected",
                            {"streak": cfg["zero_order_days"], "rc": tail[-1].get("rc")}))

    # --- drawdown breach vs peak ---
    equity, peak = snap.get("equity"), snap.get("peak")
    if equity is not None and peak and peak > 0:
        dd_pct = (equity / peak - 1.0) * 100.0
        if dd_pct <= cfg["dd_breach_pct"]:
            actions.append(("fire", "drawdown_breach",
                            {"dd_pct": round(dd_pct, 1), "limit_pct": cfg["dd_breach_pct"],
                             "equity": equity}))
```

- [ ] **Step 4: Run all watchdog tests**

Run: `pytest tests/test_ops_watchdog.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/ops_watchdog.py tests/test_ops_watchdog.py
git commit -m "feat(ops): watchdog heartbeat/run-quality/drawdown checks (TDD)"
```

---

### Task 4: Watchdog I/O wiring — `load_snapshot()`, `apply_actions()`, `main()`

**Files:**
- Modify: `scripts/ops_watchdog.py`
- Modify: `tests/test_ops_watchdog.py`

- [ ] **Step 1: Add a test that apply_actions routes fires to AlertManager and liquidation respects shadow**

```python
# append to tests/test_ops_watchdog.py
class _FakeAM:
    def __init__(self): self.fired = []
    def fire(self, rule, ctx=None): self.fired.append((rule, ctx)); return True

def test_apply_actions_fires_and_shadow_liquidation(monkeypatch):
    am = _FakeAM()
    called = {"flatten": 0}
    monkeypatch.setattr(ow, "_do_liquidation", lambda reason, ctx, policy: called.__setitem__("flatten", called["flatten"] + 1))
    acts = [("fire", "halt_flag_set", {"reason": "x", "equity": 1}),
            ("liquidate", "halt_unacked_grace_exceeded", {"mode": "shadow"})]
    state = {}
    ow.apply_actions(acts, am=am, state=state, policy={}, now=NOW)
    assert ("halt_flag_set", {"reason": "x", "equity": 1}) in am.fired
    assert called["flatten"] == 1
    assert state.get("liquidation_done") is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_ops_watchdog.py::test_apply_actions_fires_and_shadow_liquidation -q`
Expected: FAIL (`apply_actions` / `_do_liquidation` not defined).

- [ ] **Step 3: Implement loaders, apply_actions, _do_liquidation, and wire main()**

```python
# add to scripts/ops_watchdog.py (above main)
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
                try: eqs.append(float(snip[i + 7:].split()[0].rstrip("\\n")))
                except Exception: pass
        if eqs:
            equity, peak = eqs[-1], max(eqs)
    return {"halt": _load_json(HALT_FLAG), "sched_hb": _load_json(SCHED_HB),
            "state_hb": _load_json(STATE_HB), "manifest": manifest,
            "equity": equity, "peak": peak}

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
    # refresh halt-seen marker from current snapshot handled in main()
    return state
```

Replace the placeholder `main()` with:

```python
def main(argv=None):  # pragma: no cover (thin I/O wiring; logic covered by evaluate/apply tests)
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
    # update halt-seen marker / clear escalation when halt is gone
    halt = snap.get("halt")
    state["last_seen_halt_ts"] = (halt or {}).get("ts_utc")
    if not halt:
        state.pop("warning_sent_at", None); state.pop("liquidation_done", None)
    WATCHDOG_STATE.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return 0
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_ops_watchdog.py -q`
Expected: PASS (14 tests).

- [ ] **Step 5: Smoke-run the watchdog against live state (no halt expected → no fire)**

Run: `python scripts/ops_watchdog.py && type output\ops\watchdog_state.json`
Expected: exit 0; `watchdog_state.json` written; no exception.

- [ ] **Step 6: Commit**

```bash
git add scripts/ops_watchdog.py tests/test_ops_watchdog.py
git commit -m "feat(ops): watchdog I/O wiring + shadow liquidation delegate"
```

---

### Task 5: Fire alerts at the halt source + on ack

**Files:**
- Modify: `scripts/run_live_paper.py` (halt-flag write site, near `_arm_soft_timeout` / where `halt_ack_required.json` is written)
- Modify: `scripts/ack_halt.py:81` (after successful clear, before `return 0`)

- [ ] **Step 1: Locate the halt-write site**

Run: `python -c "import re;s=open('scripts/run_live_paper.py',encoding='utf-8').read();[print(i+1,l) for i,l in enumerate(s.splitlines()) if 'halt_ack_required' in l or 'def _arm_soft_timeout' in l]"`
Read ~15 lines around the write (where the JSON `{ts_utc, reason, source}` is dumped).

- [ ] **Step 2: Add a best-effort fire right after the flag is written** (never let alerting break the halt path):

```python
            try:
                from src.assembled_core.ops.alerting import AlertManager
                AlertManager().fire("halt_flag_set", {"reason": reason, "equity": "n/a"})
            except Exception as _alert_exc:  # alerting must never break the halt write
                logger.error("[run_live_paper] halt alert failed: %s", _alert_exc)
```

(Place it immediately after the existing `halt_ack_required.json` write; reuse the in-scope `reason` variable. If `reason` isn't in scope there, pass the same string written to the flag.)

- [ ] **Step 3: Add the all-clear fire in `ack_halt.py`** right before `return 0` (after the success log at line ~83):

```python
    try:
        from src.assembled_core.ops.alerting import AlertManager
        AlertManager().fire("halt_cleared", {"actor": args.actor, "reason": reason})
    except Exception as exc:
        logger.error("[ACK_HALT] all-clear alert failed: %s", exc)
```

- [ ] **Step 4: Verify both files still import + ack_halt no-op path works**

Run: `python -c "import ast;ast.parse(open('scripts/run_live_paper.py',encoding='utf-8').read());ast.parse(open('scripts/ack_halt.py',encoding='utf-8').read());print('parse OK')"`
Run: `python scripts/ack_halt.py --reason="noop_check_2026-06-22" ; echo rc=$LASTEXITCODE`
Expected: parse OK; ack_halt logs "no halt flag present … nothing to clear" (since none currently set) and exits 0 — the fire path is skipped when nothing to clear. (If a halt flag IS present, this would clear it — only run when you intend to.)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_live_paper.py scripts/ack_halt.py
git commit -m "feat(ops): fire halt_flag_set on halt write + halt_cleared on ack"
```

---

### Task 6: Task Scheduler registration (watchdog + DMS daemon)

**Files:**
- Create: `scripts/register_ops_tasks.ps1`

- [ ] **Step 1: Write the registration script (documents + creates both tasks)**

```powershell
# scripts/register_ops_tasks.ps1 — register the ops watchdog (every 20 min) + DMS daemon (on logon).
# Run ONCE in an elevated PowerShell. Idempotent: deletes+recreates by name.
$ErrorActionPreference = "Stop"
$repo = "F:\Python_Projekt\Aktiengerüst"
$py   = "python"

# 1) Watchdog — every 20 minutes, indefinitely
$wdName = "AssembledTradingAI-OpsWatchdog"
$wdAct  = New-ScheduledTaskAction -Execute $py -Argument "scripts\ops_watchdog.py" -WorkingDirectory $repo
$wdTrig = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 20)
schtasks /delete /tn $wdName /f 2>$null
Register-ScheduledTask -TaskName $wdName -Action $wdAct -Trigger $wdTrig -Description "Paper-pilot ops watchdog: halt/heartbeat/run-quality/drawdown alerts"

# 2) DMS daemon — at logon, long-running heartbeat-stale flatten guard (was never registered)
$dmsName = "AssembledTradingAI-DMSDaemon"
$dmsAct  = New-ScheduledTaskAction -Execute $py -Argument "scripts\dms_daemon.py" -WorkingDirectory $repo
$dmsTrig = New-ScheduledTaskTrigger -AtLogOn
schtasks /delete /tn $dmsName /f 2>$null
Register-ScheduledTask -TaskName $dmsName -Action $dmsAct -Trigger $dmsTrig -Description "Dead-Man's Switch: heartbeat-stale auto-flatten guard"

Write-Host "Registered: $wdName (every 20m), $dmsName (at logon)."
```

- [ ] **Step 2: Verify the script parses (no execution — operator runs it elevated)**

Run: `pwsh -NoProfile -Command "$null = [ScriptBlock]::Create((Get-Content -Raw scripts/register_ops_tasks.ps1)); Write-Host 'parse OK'"`
Expected: `parse OK`. (Operator runs the script itself once, elevated, to actually register.)

- [ ] **Step 3: Commit**

```bash
git add scripts/register_ops_tasks.ps1
git commit -m "feat(ops): scheduled-task registration for watchdog + DMS daemon"
```

---

### Task 7: Phase-1 verification + go-live of the telegram path

- [ ] **Step 1: Set telegram creds in `.env`** (operator step; never commit): `TELEGRAM_BOT_TOKEN=…`, `TELEGRAM_CHAT_ID=…`.

- [ ] **Step 2: Live alert smoke** via the existing drill, then a watchdog tick:

Run: `python scripts/drills/drill_halt_flag.py` (sets a halt flag) then `python scripts/ops_watchdog.py`
Expected: a `halt_flag_set` Telegram message arrives; `watchdog_state.json` records `last_seen_halt_ts`.

- [ ] **Step 3: Clear + confirm all-clear**

Run: `python scripts/ack_halt.py --reason="drill_verified_2026-06-22_telegram_path"`
Expected: `halt_cleared` Telegram message arrives.

- [ ] **Step 4: Run the full relevant suite**

Run: `pytest tests/test_ops_watchdog.py tests/test_dead_man_switch.py tests/test_heartbeat.py -q`
Expected: all PASS. Report pass/fail counts + date.

- [ ] **Step 5: Commit any fixups + tag Phase 1 done in the spec.**

---

## PHASE 2 — Real auto-liquidation (PROTECTED — DO NOT START WITHOUT EXPLICIT OPERATOR GO)

> **GATE:** Start only after (a) Phase 1 telegram path is verified live, and (b) the operator
> explicitly approves touching `execution/`. Edits to `execution/broker_adapter.py` are
> deny-guarded → require the scoped deny-lift workflow + full review chain
> (`risk-execution-reviewer` → `senior-code-reviewer` → `task-completion-auditor`).

### Task 8 (GATED): `close_all_positions()` broker primitive + flip watchdog to real

**Files:**
- Modify (PROTECTED): `src/assembled_core/execution/broker_adapter.py` — add method
- Modify: `scripts/ops_watchdog.py` — `_do_liquidation` calls broker when `flatten_mode == "market"`
- Test: `tests/test_broker_close_all_positions.py`

- [ ] **Step 1: TDD against a fake broker** — test that `close_all_positions(shadow=True)` submits nothing and returns a per-symbol "would-close" report; `shadow=False` submits one closing SELL per long position and cancels open orders first.

- [ ] **Step 2: Implement the method** (signature):

```python
def close_all_positions(self, *, shadow: bool = True) -> dict[str, str]:
    """Cancel open orders, then submit a closing SELL (market, day) for each long
    position. Long-only system → SELL only. shadow=True logs intended closes without
    submitting. Returns {symbol: 'closed'|'would_close'|'failed:<err>'}."""
```

- [ ] **Step 3: Wire `_do_liquidation`** to call `broker.close_all_positions(shadow=(mode!='market'))` when a real adapter is available; keep the kill-switch call as a belt-and-suspenders block-orders step.

- [ ] **Step 4: Review chain** (mandatory for protected path) + operator sign-off, THEN commit.

- [ ] **Step 5: Flip `configs/alerting.yaml` watchdog.flatten_mode `shadow` → `market`** as the final deliberate go-live step, after a shadow-mode dry-run confirms the intended closes are correct.

---

## Self-Review (against spec)

- **Spec §3 Phase 1 (K1–K4):** Task 1 (alerting.yaml/K1), Tasks 2–4 (watchdog/K3), Task 5 (fire at source/K2), Task 6 (scheduler/K4). ✓
- **Spec §3 Phase 2 (K5):** Task 8, gated. ✓
- **Spec §4 two-stage warning + intervention window:** Task 2 (warn→window→liquidate, ack aborts). ✓
- **Spec §4 shadow-first:** `flatten_mode: shadow` default (Task 1), flipped only in Task 8 Step 5. ✓
- **Spec §5 testing (no real telegram/broker in tests):** `_FakeAM`, monkeypatched `_do_liquidation`, fake broker (Task 8). ✓
- **Spec §6 protected isolation:** only Task 8 touches `execution/`, behind a GATE. ✓
- **Placeholder scan:** every code step has concrete code; commands have expected output. ✓
- **Naming consistency:** `evaluate`, `apply_actions`, `_do_liquidation`, `load_snapshot`, action tuples `("fire"|"liquidate", …)` consistent across Tasks 2/3/4/8. ✓
