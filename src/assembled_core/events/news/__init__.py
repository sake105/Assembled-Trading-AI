"""News v1 pipeline (MVP, free & robust).

This package implements a minimal but robust NEWS v1 pipeline:

- Fetch from RSS (and optionally GDELT)
- Normalize into a clean NewsEvent schema
- Dedupe events
- Compute NewsHealth
- Emit JSON artifacts under output/intel/news/
"""

from __future__ import annotations

from .pipeline import run_news_pipeline

__all__ = ["run_news_pipeline"]

