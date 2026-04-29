"""Disclosure event and health models (v1 minimal)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

DisclosuresStatus = Literal["OK", "DEGRADED", "ERROR"]


@dataclass
class DisclosureEvent:
    """Normalized disclosure event (v1 minimal)."""

    event_id: str
    source_id: str
    source_name: str
    source_domain: str
    published_utc: str
    fetched_utc: str
    person_or_entity: str = ""
    ticker: Optional[str] = None
    action_type: str = ""
    notional: Optional[float] = None
    raw: Dict[str, Any] = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.raw is None:
            self.raw = {}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class DisclosuresHealth:
    """Health summary for a disclosures pipeline run (like NewsHealth)."""

    status: DisclosuresStatus
    fetched_utc: str
    sources_total: int
    sources_ok: int
    sources_failed: int
    items_raw: int
    items_after_dedupe: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
