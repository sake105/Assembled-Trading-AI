"""Realism metadata: label backtest/report outputs with realism assumptions.

Produces a compact metadata dict that documents which realism layers were
active during a backtest or paper run. This dict can be attached to any
run manifest, report, or artifact so that results can be interpreted
in light of their realism level.

M7-T04: realism metadata labeling.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Realism level constants
# ---------------------------------------------------------------------------

#: Ordered realism levels (ascending quality)
REALISM_LEVELS = ("none", "minimal", "standard", "high")

# Component mode constants
CALENDAR_MODES = ("none", "fallback", "nyse")
CORPORATE_ACTIONS_MODES = ("none", "splits_only", "splits+dividends")
COST_MODEL_MODES = ("none", "zero", "fixed_bps", "policy", "tca")
UNIVERSE_MODES = ("none", "watchlist", "snapshot")
DATA_SOURCE_MODES = ("synthetic", "real_partial", "real")


# ---------------------------------------------------------------------------
# Build label
# ---------------------------------------------------------------------------


def build_realism_label(
    calendar_mode: str = "none",
    corporate_actions_mode: str = "none",
    cost_model_mode: str = "none",
    universe_mode: str = "none",
    data_source: str = "synthetic",
    notes: str = "",
) -> dict[str, Any]:
    """Build a realism metadata label for a backtest or report output.

    Args:
        calendar_mode: Calendar used. One of: ``none``, ``fallback``, ``nyse``.
        corporate_actions_mode: CA handling. One of: ``none``, ``splits_only``,
            ``splits+dividends``.
        cost_model_mode: Cost model. One of: ``none``, ``zero``, ``fixed_bps``,
            ``policy``, ``tca``.
        universe_mode: Universe handling. One of: ``none``, ``watchlist``,
            ``snapshot``.
        data_source: Data origin. One of: ``synthetic``, ``real_partial``, ``real``.
        notes: Optional free-text notes (e.g. specific data feed, known gaps).

    Returns:
        Dict with:
            ``calendar_mode``, ``corporate_actions_mode``, ``cost_model_mode``,
            ``universe_mode``, ``data_source``, ``notes``,
            ``realism_score``: integer 0–6 (sum of individual scores),
            ``realism_level``: one of ``none`` / ``minimal`` / ``standard`` / ``high``.
    """
    # Compute a simple additive realism score
    score = 0
    score += _calendar_score(calendar_mode)
    score += _ca_score(corporate_actions_mode)
    score += _cost_score(cost_model_mode)
    score += _universe_score(universe_mode)
    score += _data_source_score(data_source)

    level = _score_to_level(score)

    return {
        "calendar_mode": calendar_mode,
        "corporate_actions_mode": corporate_actions_mode,
        "cost_model_mode": cost_model_mode,
        "universe_mode": universe_mode,
        "data_source": data_source,
        "notes": notes,
        "realism_score": score,
        "realism_level": level,
    }


def build_realism_label_from_policy(
    policy: dict[str, Any] | None = None,
    data_source: str = "synthetic",
    notes: str = "",
) -> dict[str, Any]:
    """Build a realism label by inspecting policy config automatically.

    Reads ``calendar``, ``corporate_actions``, ``cost_model``, and ``universe``
    sections from the policy dict to determine which features are enabled.

    Args:
        policy: Policy dict. Each section checked for ``enabled`` flag.
        data_source: Data source classification.
        notes: Optional notes.

    Returns:
        Realism label dict (same shape as ``build_realism_label``).
    """
    p = policy or {}

    # Calendar
    cal_sec = p.get("calendar") or {}
    if cal_sec.get("enabled", False):
        cal_mode = cal_sec.get("mode", "nyse")
    else:
        cal_mode = "none"

    # Corporate actions
    ca_sec = p.get("corporate_actions") or {}
    if ca_sec.get("enabled", False):
        ca_mode = ca_sec.get("mode", "splits+dividends")
    else:
        ca_mode = "none"

    # Cost model
    cm_sec = p.get("cost_model") or {}
    if cm_sec.get("enabled", False):
        cm_mode = cm_sec.get("mode", "policy")
    else:
        cm_mode = "none"

    # Universe
    univ_sec = p.get("universe") or {}
    if univ_sec.get("enabled", False):
        univ_mode = univ_sec.get("mode", "snapshot")
    else:
        univ_mode = "none"

    return build_realism_label(
        calendar_mode=cal_mode,
        corporate_actions_mode=ca_mode,
        cost_model_mode=cm_mode,
        universe_mode=univ_mode,
        data_source=data_source,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Scoring helpers (private)
# ---------------------------------------------------------------------------


def _calendar_score(mode: str) -> int:
    return {"none": 0, "fallback": 1, "nyse": 2}.get(mode, 0)


def _ca_score(mode: str) -> int:
    return {"none": 0, "splits_only": 1, "splits+dividends": 2}.get(mode, 0)


def _cost_score(mode: str) -> int:
    return {"none": 0, "zero": 0, "fixed_bps": 1, "policy": 1, "tca": 2}.get(mode, 0)


def _universe_score(mode: str) -> int:
    return {"none": 0, "watchlist": 1, "snapshot": 2}.get(mode, 0)


def _data_source_score(mode: str) -> int:
    return {"synthetic": 0, "real_partial": 1, "real": 2}.get(mode, 0)


def _score_to_level(score: int) -> str:
    # Max score = 2+2+2+2+2 = 10
    if score == 0:
        return "none"
    if score <= 3:
        return "minimal"
    if score <= 6:
        return "standard"
    return "high"


__all__ = [
    "build_realism_label",
    "build_realism_label_from_policy",
    "REALISM_LEVELS",
    "CALENDAR_MODES",
    "CORPORATE_ACTIONS_MODES",
    "COST_MODEL_MODES",
    "UNIVERSE_MODES",
    "DATA_SOURCE_MODES",
]
