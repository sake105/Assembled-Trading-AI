"""Unit tests for the verdict-gating logic of the full-history sector-rotation
robustness harness (scripts/_oos_wf_sector_rotation_fullhist.py).

These lock the normally-DEAD PROSPECT branch: in every real run so far ALL books
are REJECTED, so the PROSPECT-summary text never executes. If a future data/feed
change ever produced a deflated, significant edge over SPY, the report must flip
to the PROSPECT wording — these tests pin that contract with synthetic edges so a
regression in the gate (e.g. dropping the IR-t or DSR condition) is caught.

Pure-function tests only; no walk-forward run, no network, no I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _oos_wf_sector_rotation_fullhist as fh  # type: ignore  # noqa: E402

MODES = fh.oos.MODES  # ("sector_ls", "sector_lo", "eq_sector")


def _edge(ann_sharpe: float, dsr_pass: bool, ir_t: float) -> dict:
    return {"ann_sharpe": ann_sharpe, "dsr_pass": dsr_pass, "ir_t": ir_t}


def _edges(**overrides: dict) -> dict:
    """All three books default to a clearly-failing edge; override per book."""
    base = {m: _edge(0.1, False, 0.0) for m in MODES}
    base.update(overrides)
    return base


SPY = {"ann_sharpe": 0.5}


def test_prospects_passing_edge_returns_book() -> None:
    # sector_lo clears all three conditions: sharpe>spy, dsr_pass, ir_t>1.96.
    edges = _edges(**{"sector_lo": _edge(0.9, True, 2.5)})
    assert fh._prospects(edges, SPY) == ["sector_lo"]


def test_prospects_all_failing_returns_empty() -> None:
    assert fh._prospects(_edges(), SPY) == []


def test_prospects_requires_dsr_pass() -> None:
    # High sharpe + significant IR-t but DSR fails -> not a prospect.
    edges = _edges(**{"sector_ls": _edge(0.9, False, 5.0)})
    assert fh._prospects(edges, SPY) == []


def test_prospects_requires_ir_t_above_196() -> None:
    # Beats SPY sharpe and DSR passes, but IR-t below 1.96 -> not significant.
    edges = _edges(**{"eq_sector": _edge(0.9, True, 1.5)})
    assert fh._prospects(edges, SPY) == []


def test_prospects_requires_beating_spy_sharpe() -> None:
    # DSR passes and IR-t significant, but sharpe does not exceed SPY's.
    edges = _edges(**{"sector_lo": _edge(0.5, True, 3.0)})
    assert fh._prospects(edges, SPY) == []


def test_prospects_nan_spy_sharpe_blocks_all() -> None:
    edges = _edges(**{"sector_lo": _edge(0.9, True, 3.0)})
    assert fh._prospects(edges, {"ann_sharpe": float("nan")}) == []


def test_prospects_across_modes_tags_pricemode_and_book() -> None:
    by_mode = {
        "adj": {
            "edges": _edges(**{"sector_lo": _edge(0.9, True, 2.5)}),
            "spy_edge": SPY,
        },
        "raw": {"edges": _edges(), "spy_edge": SPY},
    }
    assert fh._prospects_across_modes(by_mode) == ["adj:sector_lo"]


def test_overall_verdict_rejected_when_no_prospects() -> None:
    text = fh._overall_verdict([])
    assert "ALL books REJECTED" in text
    assert "stays ~0" in text


def test_overall_verdict_prospect_branch_locks_dead_path() -> None:
    # The branch that never fires in a real (all-REJECTED) run.
    text = fh._overall_verdict(["adj:sector_lo", "raw:sector_lo"])
    assert "PROSPECT" in text
    assert "2 book/price-mode" in text
    assert "adj:sector_lo, raw:sector_lo" in text
