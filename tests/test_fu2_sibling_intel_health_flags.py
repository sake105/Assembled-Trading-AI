"""FU-2 sibling fix — _load_intel clears the three sibling intel-health flags.

Companion to ``test_fu2_pipeline_risk.py`` FIX 2 (daily_circuit_breaker non-trip
reset). The SAME structural asymmetry exists for three sibling flags set inside
``trading_cycle_v2._load_intel``:

* ``intel_disclosures_triggers`` — set "DEGRADED" on the disclosures load-failure
  path (no ``generated_utc`` / except), NEVER reset to healthy on success.
* ``intel_crisis_alpha`` — set "DEGRADED" on the crisis-load except path only.
* ``intel_market_stress`` — set "DEGRADED" on the market-stress PIT-filter except
  path only.

``intel_health_flags`` is a ``field(default_factory=dict)`` on ``TradingContext``
and the canonical backtest driver builds each bar's ctx with ``dataclasses.replace``
WITHOUT passing it — ``replace`` shallow-copies, so the SAME dict is SHARED BY
REFERENCE across all bars. A "DEGRADED" on any bar would therefore LATCH for the
rest of the run.

The pre-fix latch is a WHOLE-RUN persistence bug: once "DEGRADED" was set on any
bar, it survived for the rest of the run on the shared-by-reference flags dict.
For ``intel_disclosures_triggers`` this had two consumers:

* ``apply_disclosures_confirm`` (``risk/disclosures_confirm.py:39``) runs LATER
  inside the SAME ``_load_intel`` call (after the clear + producer) — so the clear
  de-latches it SAME-CYCLE.
* ``compute_next_state`` (``risk/state_machine.py:302`` forces
  ``disclosures_confirmed=False`` while the flag is "DEGRADED") is the STATE-MACHINE
  consumer. In production it runs in ``ingest_data`` (~L170/L188) BEFORE
  ``_load_intel`` (~L195), so by the EXISTING cycle design it reads the most-recent
  COMPLETED bar's disclosures health (intel is one bar old by availability design).
  The clear removes the whole-run latch there too — the state machine now reads a
  value that is at most ONE bar old, never latched permanently. The clear does NOT
  make the state machine see THIS bar's disclosures health; that same-bar
  re-ordering is a SEPARATE scoped follow-up on the risk-state path.

The fix clears all three flags at the TOP of ``_load_intel`` so each bar reflects
only its own load outcome (pop, not assign — convention is healthy == key absent).

These tests DISCRIMINATE: they fail against the pre-fix sticky behaviour
(no clear-at-top), because a bar-1 DEGRADED would survive into a later bar on a
reused-flags ctx.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline import trading_cycle_v2 as tc_v2  # noqa: E402
from src.assembled_core.pipeline.trading_cycle_shared import (  # noqa: E402
    TradingContext,
)
from src.assembled_core.risk.state_machine import (  # noqa: E402
    RiskStateRecord,
    compute_next_state,
)

pytestmark = pytest.mark.fast

_LOG = logging.getLogger("test_fu2_sibling_intel_health_flags")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _ctx() -> TradingContext:
    dates = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    prices = pd.DataFrame(
        {"timestamp": dates, "symbol": ["SPY", "SPY"], "close": [100.0, 99.5]}
    )
    return TradingContext(
        prices=prices,
        as_of=prices["timestamp"].max(),
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
    )


def _disc_policy(path: str) -> dict:
    """Policy that ONLY enables the disclosures-triggers loader (CB/market-stress
    off), so _load_intel touches just the disclosures producer + the flag clear."""
    return {"intel": {"disclosures_triggers": {"enabled": True, "path": path}}}


def _write_healthy_triggers(p: Path, *, max_severity: int = 2) -> None:
    """Write a valid disclosures.triggers.v1 artifact with generated_utc set and a
    sev>=1 item, so the loader returns a healthy snapshot (no DEGRADED) AND
    max_severity is high enough to confirm (disclosures_min_severity default 1).

    generated_utc is kept <= the _ctx() as_of (2025-01-02) so the snapshot passes
    the now-wired _load_intel PIT gate (as_of=ctx.as_of) and actually loads — these
    tests exercise the sibling-flag CLEAR mechanism on a HEALTHY load, not the PIT
    drop path. A future-dated generated_utc would (correctly) be PIT-dropped and
    leave the flag DEGRADED, which is a different code path covered separately in
    test_trading_cycle_load_intel_pit_feeds.py."""
    p.write_text(
        json.dumps(
            {
                "schema_version": "disclosures.triggers.v1",
                "generated_utc": "2025-01-01T00:00:00Z",
                "items": [
                    {"symbol": "SPY", "severity": int(max_severity), "kind": "test"}
                ],
            }
        ),
        encoding="utf-8",
    )


# ===========================================================================
# 1) intel_disclosures_triggers — REAL latch: cleared on the healthy bar
# ===========================================================================


def test_disclosures_flag_degraded_then_cleared_on_healthy_bar(tmp_path: Path) -> None:
    """Reused-flags ctx: bar 1 (missing triggers file) sets the flag "DEGRADED";
    bar 2 (healthy triggers file) must clear it. Pre-fix the stale "DEGRADED"
    would survive (the success path never resets it)."""
    ctx = _ctx()

    # Bar 1: path to a non-existent file -> loader returns empty snapshot with
    # generated_utc == "" -> producer sets "DEGRADED".
    missing = tmp_path / "does_not_exist.json"
    tc_v2._load_intel(ctx, _disc_policy(str(missing)), tmp_path, _LOG)
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") == "DEGRADED"

    # Bar 2: REUSE the same ctx (do NOT clear flags) with a healthy triggers file.
    healthy = tmp_path / "triggers_latest.json"
    _write_healthy_triggers(healthy)
    tc_v2._load_intel(ctx, _disc_policy(str(healthy)), tmp_path, _LOG)
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") is None, (
        "healthy bar must clear the stale intel_disclosures_triggers DEGRADED flag"
    )
    # And the healthy snapshot actually loaded (sanity that bar 2 ran the producer).
    assert ctx.disclosures_triggers is not None


def test_disclosures_failure_on_this_bar_still_sets_degraded(tmp_path: Path) -> None:
    """Degraded handling is PRESERVED: a failed load on THIS bar still sets
    "DEGRADED" after the top-of-function clear (the clear only removes cross-bar
    leakage, it does not swallow a genuine current-bar failure)."""
    ctx = _ctx()
    missing = tmp_path / "nope.json"
    tc_v2._load_intel(ctx, _disc_policy(str(missing)), tmp_path, _LOG)
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") == "DEGRADED"


# ===========================================================================
# 2) state-machine consumer of intel_disclosures_triggers
# ===========================================================================

# Drive the gate directly: require_disclosures_confirm forces require_confirm_now,
# and _effective_geo keys on intel_geo_score / intel_news_triggers (NOT on
# intel_disclosures_triggers), so score/conf are clean and the ONLY thing gating
# WATCH -> ACTIVE is the intel_disclosures_triggers flag + the snapshot severity.
_SM_POLICY = {
    "risk_state_machine": {
        "enabled": True,
        "hysteresis": {
            "activate_score": 2,
            "confidence_floor": 0.60,
            "require_disclosures_confirm": True,
            "disclosures_min_severity": 1,
        },
    }
}


def _watch_prev() -> RiskStateRecord:
    return RiskStateRecord(
        state="WATCH",
        since_utc="2025-01-01T00:00:00Z",
        last_transition_utc="2025-01-01T00:00:00Z",
        reason="seed",
        geo_score=3,
        geo_confidence=0.9,
    )


def test_clear_mechanism_resets_disclosures_flag_in_isolation(tmp_path: Path) -> None:
    """Tests the CLEAR MECHANISM in isolation — NOT production cycle ordering.

    This drives _load_intel FIRST, then compute_next_state on the SAME bar, which
    is the REVERSE of production ingest_data ordering (compute_next_state at
    ~L170/L188 runs BEFORE _load_intel at ~L195). It therefore does NOT prove a
    same-cycle state-machine de-latch — in production the state machine reads the
    PRIOR completed bar's disclosures health (see the production-ordering test
    below). What it DOES verify, in isolation, is the mechanical contract that
    compute_next_state consumes intel_disclosures_triggers correctly: when the
    flag is "DEGRADED" the disclosures-confirm gate blocks WATCH->ACTIVE, and once
    the flag is absent (cleared) + a confirming snapshot is present, the gate opens.
    The honest cross-bar latch behaviour through real ordering is asserted by
    test_state_machine_prior_bar_intel_no_whole_run_latch.
    """
    ctx = _ctx()
    # Provide a clean geo signal (>= activate_score, above confidence_floor).
    ctx.news_geo = {"geo_score": 3, "geo_confidence": 0.9}

    # Degrade: missing file -> producer sets "DEGRADED".
    tc_v2._load_intel(ctx, _disc_policy(str(tmp_path / "missing.json")), tmp_path, _LOG)
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") == "DEGRADED"

    # WITH the DEGRADED flag set, the gate blocks (mechanical contract).
    blocked = compute_next_state(ctx, _SM_POLICY, "2025-01-02T00:00:00Z", _watch_prev())
    assert blocked.state == "WATCH"
    assert blocked.reason == "disclosures_confirm"

    # Healthy load on the same ctx (clears the flag) + confirming severity.
    healthy = tmp_path / "triggers_latest.json"
    _write_healthy_triggers(healthy, max_severity=2)
    ctx.news_geo = {"geo_score": 3, "geo_confidence": 0.9}
    tc_v2._load_intel(ctx, _disc_policy(str(healthy)), tmp_path, _LOG)
    assert ctx.intel_health_flags.get("intel_disclosures_triggers") is None

    # With the flag absent + confirming snapshot, the gate opens.
    escalated = compute_next_state(
        ctx, _SM_POLICY, "2025-01-03T00:00:00Z", _watch_prev()
    )
    assert escalated.state == "ACTIVE", (
        "clear mechanism (isolation): once the flag is absent + a confirming "
        "snapshot is present, the disclosures-confirm gate must open"
    )
    assert escalated.reason == "activate_score"


def test_state_machine_prior_bar_intel_no_whole_run_latch(tmp_path: Path) -> None:
    """HONEST production-ordering regression: the state machine reads PRIOR-bar
    disclosures health, and the clear removes the WHOLE-RUN latch.

    This simulates the EXACT production ordering per bar — compute_next_state
    (reads the shared intel_health_flags) runs BEFORE _load_intel (clears +
    re-derives) — which is the reverse of the isolation test above. We thread the
    SAME flags dict by reference across bars (as dataclasses.replace does in the
    canonical backtest driver), so a stale DEGRADED would leak forward unless the
    clear removes it.

    Per-bar production order (mirrors ingest_data L170/L188 -> L195):
        compute_next_state(ctx, ...)      # state machine reads PRIOR bar's flags
        _load_intel(ctx, ...)             # THEN clear + re-derive THIS bar's flags

    Timeline (shared/reference flags dict across all three bars):
      * Bar N   — load FAILS -> _load_intel sets DEGRADED.
      * Bar N+1 — compute_next_state reads the shared dict: it STILL sees the
                  prior (bar-N) DEGRADED -> gate blocks (this is the honest
                  prior-bar-availability behaviour, NOT a bug). THEN _load_intel
                  succeeds (healthy file) -> the clear+re-derive removes DEGRADED.
      * Bar N+2 — compute_next_state reads the shared dict: DEGRADED is now GONE
                  -> the gate is NOT forced shut by a stale flag; with a confirming
                  snapshot it escalates to ACTIVE.

    DISCRIMINATION: pre-fix (no clear-at-top) the bar-N DEGRADED would PERSIST on
    the shared dict to bar N+2's compute_next_state, keeping it WATCH forever
    (reason 'disclosures_confirm'). Post-fix bar N+1's _load_intel clears it, so
    bar N+2 escalates. This test FAILS against the sticky pre-fix behaviour.

    It does NOT assert a same-cycle state-machine de-latch: bar N+1's state machine
    still sees the bar-N DEGRADED (prior-bar intel by design). The escalation only
    appears at bar N+2, one bar after the healthy load — exactly the one-bar
    availability lag the production design carries.
    """
    ctx = _ctx()
    flags = (
        ctx.intel_health_flags
    )  # the shared-by-reference dict (replace shallow-copy)

    # ---- Bar N: load FAILS -> _load_intel sets DEGRADED on the shared dict.
    tc_v2._load_intel(ctx, _disc_policy(str(tmp_path / "missing.json")), tmp_path, _LOG)
    assert flags.get("intel_disclosures_triggers") == "DEGRADED"

    # ---- Bar N+1: PRODUCTION ORDER — state machine reads PRIOR (bar-N) flags FIRST.
    ctx.news_geo = {"geo_score": 3, "geo_confidence": 0.9}
    sm_bar_n1 = compute_next_state(
        ctx, _SM_POLICY, "2025-01-02T00:00:00Z", _watch_prev()
    )
    # Honest prior-bar behaviour: the state machine still sees bar-N DEGRADED here.
    assert sm_bar_n1.state == "WATCH"
    assert sm_bar_n1.reason == "disclosures_confirm"
    # THEN _load_intel runs with a healthy file -> clears + re-derives THIS bar.
    healthy = tmp_path / "triggers_latest.json"
    _write_healthy_triggers(healthy, max_severity=2)
    tc_v2._load_intel(ctx, _disc_policy(str(healthy)), tmp_path, _LOG)
    # Post-fix: the shared dict no longer carries the stale DEGRADED.
    assert flags.get("intel_disclosures_triggers") is None, (
        "post-fix: bar N+1's healthy _load_intel must clear the bar-N DEGRADED on "
        "the shared-by-reference flags dict (pre-fix it would persist -> whole-run latch)"
    )

    # ---- Bar N+2: PRODUCTION ORDER — state machine reads the now-cleared flags.
    ctx.news_geo = {"geo_score": 3, "geo_confidence": 0.9}
    sm_bar_n2 = compute_next_state(
        ctx, _SM_POLICY, "2025-01-03T00:00:00Z", _watch_prev()
    )
    assert sm_bar_n2.state == "ACTIVE", (
        "whole-run latch is GONE: the bar-N transient DEGRADED does NOT persist to "
        "bar N+2's state machine, so with a confirming snapshot WATCH escalates. "
        "Pre-fix the stale DEGRADED would survive on the shared dict and keep it WATCH."
    )
    assert sm_bar_n2.reason == "activate_score"


# ===========================================================================
# 3) inert siblings — crisis_alpha / market_stress are cleared too
# ===========================================================================


def test_inert_siblings_cleared_on_healthy_load(tmp_path: Path) -> None:
    """intel_crisis_alpha and intel_market_stress have no "DEGRADED" consumer
    today, but the clear must still remove a stale value so a FUTURE
    "DEGRADED"-sensitive consumer is protected. Seed both stale, then run a
    healthy _load_intel (no crisis/market-stress producer enabled) and assert
    both are gone."""
    ctx = _ctx()
    ctx.intel_health_flags["intel_crisis_alpha"] = "DEGRADED"
    ctx.intel_health_flags["intel_market_stress"] = "DEGRADED"

    # Healthy disclosures-only load (crisis_alpha / market_stress producers off →
    # they do not re-set DEGRADED this bar).
    healthy = tmp_path / "triggers_latest.json"
    _write_healthy_triggers(healthy)
    tc_v2._load_intel(ctx, _disc_policy(str(healthy)), tmp_path, _LOG)

    assert ctx.intel_health_flags.get("intel_crisis_alpha") is None
    assert ctx.intel_health_flags.get("intel_market_stress") is None


def test_clear_is_noop_when_flags_absent(tmp_path: Path) -> None:
    """Fresh-ctx / live shape: the three siblings were never set, the loaders
    succeed, so pop(...) is a no-op and no sibling flag appears — mirrors the
    live byte-identical argument (clear removes nothing that was not there)."""
    ctx = _ctx()
    healthy = tmp_path / "triggers_latest.json"
    _write_healthy_triggers(healthy)
    tc_v2._load_intel(ctx, _disc_policy(str(healthy)), tmp_path, _LOG)

    for k in (
        "intel_disclosures_triggers",
        "intel_crisis_alpha",
        "intel_market_stress",
    ):
        assert ctx.intel_health_flags.get(k) is None
