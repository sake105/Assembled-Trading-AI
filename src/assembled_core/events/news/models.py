from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

NewsStatus = Literal["OK", "DEGRADED", "ERROR"]


@dataclass
class NewsEvent:
    """Normalized news event used by the NEWS v1 pipeline."""

    event_id: str
    source_id: str
    title: str
    url: str
    canonical_url: str
    source_name: str
    source_domain: str
    published_utc: str
    fetched_utc: str
    summary: Optional[str] = None
    language: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    fingerprint64: str = ""
    entities: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsHealth:
    """Health summary for a NEWS v1 pipeline run."""

    status: NewsStatus
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
