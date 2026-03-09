"""Disclosures v1 pipeline (contract & skeleton).

- Fetch from House PTR (PDF) and SEC EDGAR (Form 4/13D/13G) — stubs in v0
- Normalize into DisclosureEvent schema
- Dedupe by fingerprint, emit JSON under output/intel/disclosures/
"""

from __future__ import annotations

from .pipeline import run_disclosures_pipeline

__all__ = ["run_disclosures_pipeline"]
