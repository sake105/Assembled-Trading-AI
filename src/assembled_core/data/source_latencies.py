# src/assembled_core/data/source_latencies.py
"""Single source of truth for alt-data publication latencies (audit C4-082).

Each constant is the conservative *minimum* time between the underlying
event (transaction, filing intent, earnings release) and the moment the
data is publicly available — i.e. what backtests MUST shift their
``as_of`` by to avoid look-ahead bias.

Sources / rationale:

- ``INSIDER_DAYS = 2``       SEC Form 4 must be filed within two business
                              days of the trade (15 USC §78p(a)(2)(C)).
- ``CONGRESS_DAYS = 45``     STOCK Act §107 requires periodic transaction
                              reports within 30-45 days; we use the
                              conservative upper bound.
- ``EARNINGS_DAYS = 0``      Earnings releases are public the moment they
                              are issued; the timestamp on the alt-data
                              feed already encodes the press-release time.
- ``ACLED_DAYS = 1``         ACLED publishes daily on a 24h lag.
- ``GDELT_DAYS = 1``         GDELT 15-minute snapshots are typically
                              consumable on a 1-day batch lag.
- ``EDGAR_DAYS = 1``         SEC EDGAR filings carry an acceptance-date
                              timestamp; conservative round-up to one
                              calendar day so a same-day filing only
                              becomes visible at next-bar boundary
                              (audit C4-025).
- ``FINRA_DAYS = 1``         FINRA short-sale-volume reports are
                              published T+1 after end of trading day
                              (audit C4-026).
- ``WIKIPEDIA_DAYS = 1``     Wikipedia page-view counts roll over at
                              UTC midnight; complete counts for trading
                              day T are first available at UTC midnight
                              that day, i.e. ~02:00 ET of T+1
                              (audit C4-027).

Callers MUST prefer these constants over inline magic numbers so that any
adjustment propagates atomically across feature builders. Tests should
import the constants directly so a regression cannot drift one builder
out of sync with the others.
"""

from __future__ import annotations

# Disclosure / publication latencies in calendar days (audit C4-082).
INSIDER_DAYS: int = 2
CONGRESS_DAYS: int = 45
EARNINGS_DAYS: int = 0
ACLED_DAYS: int = 1
GDELT_DAYS: int = 1
EDGAR_DAYS: int = 1
FINRA_DAYS: int = 1
WIKIPEDIA_DAYS: int = 1

# Optional registry for callers that need to look up by source name.
LATENCY_DAYS: dict[str, int] = {
    "insider": INSIDER_DAYS,
    "congress": CONGRESS_DAYS,
    "earnings": EARNINGS_DAYS,
    "acled": ACLED_DAYS,
    "gdelt": GDELT_DAYS,
    "edgar": EDGAR_DAYS,
    "finra": FINRA_DAYS,
    "wikipedia": WIKIPEDIA_DAYS,
}


def latency_for(source: str) -> int:
    """Return the canonical latency in days for ``source`` (case-insensitive).

    Raises ``KeyError`` for unknown sources so misspellings fail loud.
    """
    key = source.strip().lower()
    return LATENCY_DAYS[key]
