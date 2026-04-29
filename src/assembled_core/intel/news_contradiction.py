"""News source contradiction detection.

Flags stories where different source classes (e.g. state media vs. mainstream
Western media) report materially different directions or severity. A contra-
diction lowers the aggregate confidence of the story — one side is likely
spinning, and we should not take either signal at face value.

Usage:
    detector = ContradictionDetector()
    report = detector.analyse(events)
    for key, entry in report.items():
        if entry.contradicts:
            print(f"story={key} disagreement={entry.direction_split}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.assembled_core.intel.news_classifier import get_source_bias, is_state_media
from src.assembled_core.intel.news_dedupe import content_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class ContradictionEntry:
    story_key: str
    contradicts: bool
    western_direction: str = "neutral"    # majority direction among Western mainstream
    state_direction: str = "neutral"      # majority direction among state media
    severity_delta: float = 0.0           # max severity gap between camps
    direction_split: str = ""             # e.g. "bearish_vs_bullish"
    sources: list[str] = field(default_factory=list)


# Source camps — informational labels, not value judgements.
# H11: add European, Japanese, Arab, Asian non-state outlets that
# previously fell through to "other" and silenced contradiction signals.
_WESTERN_MAINSTREAM = frozenset({
    "reuters", "ap", "apnews", "bbc", "bbc_world", "cnn", "nyt", "wsj",
    "ft", "guardian", "bloomberg", "wapo", "npr", "axios", "politico",
    "sky_news", "cnbc", "marketwatch", "dw", "france24",
    # H11 additions (non-state European / Asian / global outlets)
    "handelsblatt", "le_monde", "lemonde", "nikkei", "scmp",
    "spiegel", "faz", "zeit", "tagesschau", "reuters_de",
    "el_pais", "elpais", "corriere", "repubblica",
    "economist", "forbes", "barrons", "yahoo_finance",
})


def _source_camp(source_id: str) -> str:
    """Classify a source into 'state', 'western', or 'other'."""
    sid = (source_id or "").lower().strip()
    if is_state_media(sid):
        return "state"
    if sid in _WESTERN_MAINSTREAM:
        return "western"
    bias = get_source_bias(sid)
    editorial = bias.get("editorial_bias", "")
    if editorial in ("pro_government",):
        return "state"
    return "other"


class ContradictionDetector:
    """Compares event directions and severities across source camps."""

    def analyse(self, events: list) -> dict[str, ContradictionEntry]:
        """Return a story_key → ContradictionEntry map.

        Only stories seen by at least one Western AND one State source are
        evaluated. All others have `contradicts=False`.
        """
        grouped: dict[str, list] = {}
        for evt in events:
            try:
                title = getattr(evt, "title", "") or ""
                key = content_fingerprint(title, "")
                grouped.setdefault(key, []).append(evt)
            except Exception as exc:
                logger.debug("[SKIP] Contradiction group: %s", exc)

        out: dict[str, ContradictionEntry] = {}
        for key, grp in grouped.items():
            entry = self._analyse_group(key, grp)
            out[key] = entry
        return out

    def _analyse_group(self, key: str, group: list) -> ContradictionEntry:
        west_dirs: list[str] = []
        state_dirs: list[str] = []
        west_sev: list[float] = []
        state_sev: list[float] = []
        sources: set[str] = set()
        for evt in group:
            src = (getattr(evt, "source_id", "") or "").lower().strip()
            sources.add(src)
            camp = _source_camp(src)
            direction = getattr(evt, "market_direction", "neutral") or "neutral"
            severity = float(getattr(evt, "severity", 0.0) or 0.0)
            if camp == "western":
                west_dirs.append(direction)
                west_sev.append(severity)
            elif camp == "state":
                state_dirs.append(direction)
                state_sev.append(severity)

        if not west_dirs or not state_dirs:
            return ContradictionEntry(story_key=key, contradicts=False, sources=sorted(sources))

        west_majority = _majority(west_dirs)
        state_majority = _majority(state_dirs)

        contradicts = False
        split = ""
        if west_majority != state_majority and "neutral" not in (west_majority, state_majority):
            contradicts = True
            split = f"{west_majority}_vs_{state_majority}"
        elif abs((sum(west_sev) / len(west_sev)) - (sum(state_sev) / len(state_sev))) >= 3.0:
            contradicts = True
            split = "severity_gap"

        sev_delta = 0.0
        if west_sev and state_sev:
            sev_delta = round(abs(
                (sum(west_sev) / len(west_sev)) - (sum(state_sev) / len(state_sev))
            ), 2)

        return ContradictionEntry(
            story_key=key,
            contradicts=contradicts,
            western_direction=west_majority,
            state_direction=state_majority,
            severity_delta=sev_delta,
            direction_split=split,
            sources=sorted(sources),
        )


def _majority(items: list[str]) -> str:
    if not items:
        return "neutral"
    counts: dict[str, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=lambda k: counts[k])


__all__ = ["ContradictionDetector", "ContradictionEntry"]
