# tests/pipeline/test_tc_sizing_target_qty_notional_drift.py
"""GUARDRAIL — _tc_sizing overlays mutate ONLY target_qty, never re-read/maintain
target_notional.

Pins FACT A (the load-bearing half): the live pipeline overlays in
``src/assembled_core/pipeline/_tc_sizing.py`` recompute / scale ``target_qty``
in place (always derived from ``target_weight * capital`` or ``*= scale``) and
NEVER re-read or re-sync ``target_notional``. ``target_notional`` is an
emit-time honest-name marker only (write-only; see position_sizing.py header
~lines 13-26). Therefore, after an overlay runs over an emitted frame:

  * ``target_qty`` is mutated (changed), and
  * ``target_notional`` is left at its STALE pre-overlay value (NOT re-synced),
    so ``target_qty`` and ``target_notional`` intentionally DIVERGE.

This divergence is BY DESIGN and harmless precisely because no src/ code reads
``target_notional`` as a DataFrame column post-overlay.

Representative overlay pinned: ``_sp_apply_crowding_cap`` (HHI concentration
cap), src/assembled_core/pipeline/_tc_sizing.py lines 1318-1343. Its mutation
site (lines 1339-1340) does:

    target_positions.loc[mask, "target_qty"] = _max_w * ctx.capital

— it touches ``target_qty`` and ``ctx.capital`` only; ``target_notional`` is
never referenced. This is the same pattern used by every other overlay
``target_qty`` mutation in the module (e.g. correlation_guard 1108-1111/1120-1121,
crash_cap 1169-1171, quantile_asymmetry 1310-1312, news_alpha 2004-2050).

``_sp_apply_crowding_cap`` is chosen because it is a clean, dependency-light
unit seam: it imports only ``compute_hhi`` (resolves), takes a plain
``target_positions`` frame + a ``ctx`` exposing ``.capital``, and does NOT pull
the archived ``shadow_recorder`` (whose absence turns several other overlays
into no-ops via their bare ``except``), so the behavioural assertion is real.

WHAT THIS TEST PINS / DISCRIMINATION:
A FUTURE edit that makes an overlay START reading or maintaining
``target_notional`` post-overlay (e.g. re-syncing target_notional = target_qty,
or sourcing the cap from target_notional) would resurrect the drift hazard —
and would flip either "target_notional left stale" or "drift created" below,
failing this test loudly. The negative control
(``test_overlay_noop_when_not_crowded``) ensures the cap actually has to fire
for the mutation assertion to hold, so the test cannot pass vacuously.

The module is READ-ONLY here — this test pins existing behaviour; no src/ file
is edited.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Repo root: tests/pipeline/<file> -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline._tc_sizing import (  # noqa: E402
    _sp_apply_crowding_cap,
)
from src.assembled_core.risk.crowding_detector import compute_hhi  # noqa: E402

pytestmark = pytest.mark.fast

CAPITAL = 100_000.0
_LOG = logging.getLogger("test_tc_sizing_drift")


def _emitted_frame(dominant_weight: float = 0.60) -> pd.DataFrame:
    """An emit-shaped frame: target_qty == target_notional == weight*CAPITAL.

    With dominant_weight=0.60 and four 0.10 satellites: HHI = 0.40 > 0.25 and
    n=5 >= 5, so _sp_apply_crowding_cap fires and caps the 0.60 weight to 0.10.
    """
    satellites = (1.0 - dominant_weight) / 4.0
    weights = [dominant_weight, satellites, satellites, satellites, satellites]
    notional = [w * CAPITAL for w in weights]
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "target_weight": weights,
            "target_notional": list(notional),  # emit-time marker (will go stale)
            "target_qty": list(notional),  # value-identical alias at emit
        }
    )


def test_crowding_overlay_setup_is_valid():
    """Precondition: the constructed frame actually trips the crowding cap
    (HHI > 0.25, n >= 5, a weight > 0.10) — otherwise the overlay would be a
    no-op and the mutation assertion below would be vacuous."""
    frame = _emitted_frame()
    weights_map = dict(zip(frame["symbol"], frame["target_weight"]))
    assert compute_hhi(weights_map) > 0.25
    assert len(frame) >= 5
    assert (frame["target_weight"] > 0.10).any()
    # Emit invariant holds before the overlay runs.
    assert (frame["target_qty"] == frame["target_notional"]).all()


def test_overlay_mutates_target_qty_only_leaving_notional_stale():
    """Core guardrail. After _sp_apply_crowding_cap over an emitted frame:
    target_qty is mutated; target_notional is left at the stale pre-overlay
    value; the two columns therefore DIVERGE (by design)."""
    before = _emitted_frame()
    ctx = SimpleNamespace(capital=CAPITAL)

    after = _sp_apply_crowding_cap(before.copy(), ctx, _LOG)

    # 1) target_qty WAS mutated by the overlay (the dominant position was capped).
    assert not before["target_qty"].equals(after["target_qty"]), (
        "overlay did not mutate target_qty (cap did not fire?)"
    )
    capped = after.loc[after["symbol"] == "A", "target_qty"].iloc[0]
    # Mutation derives from target_weight * ctx.capital (= 0.10 * CAPITAL), NOT
    # from any read of target_notional.
    assert capped == pytest.approx(0.10 * CAPITAL), (
        "capped target_qty must equal _max_w * ctx.capital "
        f"(got {capped}, expected {0.10 * CAPITAL})"
    )

    # 2) target_notional was NOT touched — it holds the stale pre-overlay value.
    assert after["target_notional"].equals(before["target_notional"]), (
        "target_notional was re-synced/maintained post-overlay (drift hazard resurrected)"
    )

    # 3) The two columns now intentionally DIVERGE (this is FACT A — harmless,
    #    because nothing in src/ reads target_notional as a column post-overlay).
    assert (after["target_qty"] != after["target_notional"]).any(), (
        "expected post-overlay drift between target_qty and target_notional "
        "(if they are still equal, an overlay is silently maintaining "
        "target_notional — the latent hazard this test guards against)"
    )


def test_overlay_noop_when_not_crowded_preserves_emit_parity():
    """Negative control: when the portfolio is NOT crowded (HHI below threshold,
    no weight above the cap), the overlay is a no-op — target_qty unchanged and
    emit parity preserved. Confirms the divergence in the main test is caused by
    the cap firing, not by the overlay always desyncing."""
    # Five equal 0.20 weights: HHI = 0.20 < 0.25 -> cap does not fire.
    weights = [0.20, 0.20, 0.20, 0.20, 0.20]
    notional = [w * CAPITAL for w in weights]
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "target_weight": weights,
            "target_notional": list(notional),
            "target_qty": list(notional),
        }
    )
    assert compute_hhi(dict(zip(frame["symbol"], frame["target_weight"]))) < 0.25

    ctx = SimpleNamespace(capital=CAPITAL)
    after = _sp_apply_crowding_cap(frame.copy(), ctx, _LOG)

    assert after["target_qty"].equals(frame["target_qty"]), (
        "no-op case mutated target_qty"
    )
    assert (after["target_qty"] == after["target_notional"]).all(), (
        "no-op case unexpectedly desynced the columns"
    )
