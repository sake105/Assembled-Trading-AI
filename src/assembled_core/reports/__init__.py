"""Report generation modules.

This package handles:
- Daily QA reports
- Performance reports
- Risk reports
- Portfolio reports
"""

from __future__ import annotations

from src.assembled_core.reports.daily_qa_report import (
    generate_qa_report,
    generate_qa_report_from_files,
)
from src.assembled_core.reports.metrics_export import export_metrics_json

__all__ = [
    "generate_qa_report",
    "generate_qa_report_from_files",
    "export_metrics_json",
]
