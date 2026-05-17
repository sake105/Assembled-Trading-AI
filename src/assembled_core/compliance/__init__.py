"""Compliance module — GDPR, PDT rule, rate limits, tax reporting.

From 50_COMPLIANCE_RECHT.md (personal-use only, German tax context).
"""

from __future__ import annotations

from src.assembled_core.compliance.gdpr import (
    anonymize_news_headline,
    pseudonymize_user,
    should_retain,
)
from src.assembled_core.compliance.pdt import (
    PDT_EQUITY_THRESHOLD_USD,
    PDT_MAX_DAY_TRADES_IN_5_DAYS,
    can_day_trade,
    count_day_trades,
)
from src.assembled_core.compliance.rate_limits import (
    ALLOWED_PERSONAL_USE,
    PROHIBITED_SOURCES,
    get_min_delay_seconds,
)
from src.assembled_core.compliance.tax_report import (
    EFFECTIVE_TAX_RATE,
    SPARER_PAUSCHBETRAG_EUR,
    TaxReportSummary,
    summarize_closed_lots,
)

__all__ = [
    # gdpr
    "pseudonymize_user",
    "should_retain",
    "anonymize_news_headline",
    # pdt
    "PDT_EQUITY_THRESHOLD_USD",
    "PDT_MAX_DAY_TRADES_IN_5_DAYS",
    "can_day_trade",
    "count_day_trades",
    # rate limits
    "ALLOWED_PERSONAL_USE",
    "PROHIBITED_SOURCES",
    "get_min_delay_seconds",
    # tax report
    "EFFECTIVE_TAX_RATE",
    "SPARER_PAUSCHBETRAG_EUR",
    "TaxReportSummary",
    "summarize_closed_lots",
]
