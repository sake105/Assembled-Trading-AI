"""Disclosures v1 pipeline (contract & skeleton).

- Fetch from House PTR (PDF) and SEC EDGAR (Form 4/13D/13G) — stubs in v0
- Normalize into DisclosureEvent schema
- Dedupe by fingerprint, emit JSON under output/intel/disclosures/
"""

from __future__ import annotations

from .dedupe import dedupe_events
from .emit import emit_json_artifact
from .evidence import summarize_evidence
from .fetch_edgar import fetch_edgar, fetch_edgar_form4
from .fetch_house_ptr import fetch_house_ptr, fetch_house_ptr_filings
from .health import compute_health
from .models import DisclosureEvent, DisclosuresHealth, DisclosuresStatus
from .normalize import normalize_raw_item, now_utc_iso
from .pipeline import run_disclosures_pipeline
from .sources import DisclosureSource, load_disclosures_params, load_sources_registry
from .triggers import apply_qc_caps, score_disclosure_triggers

__all__ = [
    "DisclosureEvent",
    "DisclosureSource",
    "DisclosuresHealth",
    "DisclosuresStatus",
    "apply_qc_caps",
    "compute_health",
    "dedupe_events",
    "emit_json_artifact",
    "fetch_edgar",
    "fetch_edgar_form4",
    "fetch_house_ptr",
    "fetch_house_ptr_filings",
    "load_disclosures_params",
    "load_sources_registry",
    "normalize_raw_item",
    "now_utc_iso",
    "run_disclosures_pipeline",
    "score_disclosure_triggers",
    "summarize_evidence",
]
