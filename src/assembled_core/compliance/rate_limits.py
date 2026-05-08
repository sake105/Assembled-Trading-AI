"""Data-source rate-limit constants.

From 50_COMPLIANCE_RECHT.md §50.3.
Respecting rate limits protects us from IP bans and ToS violations.
All values are conservative estimates; reduce further if issues occur.
"""

from __future__ import annotations

# ── requests per unit ───────────────────────────────────────────────────────
YFINANCE_MAX_REQ_PER_HOUR: int = 500
STOOQ_MAX_REQ_PER_MINUTE: int = 10
WIKIPEDIA_MAX_REQ_PER_SEC: int = 100
REDDIT_MAX_REQ_PER_MINUTE: int = 60  # PRAW enforces this
SEC_EDGAR_MAX_REQ_PER_SEC: int = 10  # Official limit — violations cause bans
FINNHUB_MAX_REQ_PER_MINUTE: int = 60  # Free-tier limit
ALPACA_MAX_REQ_PER_MINUTE: int = 200  # Free-tier Alpaca
NEWSAPI_ORG_MAX_REQ_PER_DAY: int = 100  # Free-tier newsapi.org — not newsapi.ai
NEWSAPI_AI_MAX_REQ_PER_DAY: int = 1000  # newsapi.ai default free plan
POLYGON_MAX_REQ_PER_MINUTE: int = 5  # Free-tier Polygon.io
ALPHA_VANTAGE_MAX_REQ_PER_MINUTE: int = 5  # Free-tier Alpha Vantage

# ── source licences ─────────────────────────────────────────────────────────
# Allowed for personal-use (no re-sale, no redistribution)
ALLOWED_PERSONAL_USE: tuple[str, ...] = (
    "yfinance",
    "stooq",
    "reddit_praw",
    "wikipedia",
    "stocktwits_public",
    "sec_edgar",
    "fred",
    "finnhub_free",
    "alpha_vantage_free",
    "eodhd_free",
    "alpaca",
)

# Explicitly prohibited (ToS violation even for personal use)
PROHIBITED_SOURCES: tuple[str, ...] = (
    "linkedin",
    "indeed",
    "glassdoor",
    "nitter",  # Twitter scraping proxy — often hacked
    "seeking_alpha_scraper",
    "pytrends",  # Google sends fake data to bot-detection
)


def get_min_delay_seconds(source: str) -> float:
    """Return the minimum delay in seconds between requests for *source*.

    Returns 0.0 for unknown sources (caller must apply own policy).
    """
    mapping: dict[str, float] = {
        "yfinance": 3600 / YFINANCE_MAX_REQ_PER_HOUR,
        "stooq": 60 / STOOQ_MAX_REQ_PER_MINUTE,
        "wikipedia": 1 / WIKIPEDIA_MAX_REQ_PER_SEC,
        "reddit_praw": 60 / REDDIT_MAX_REQ_PER_MINUTE,
        "sec_edgar": 1 / SEC_EDGAR_MAX_REQ_PER_SEC,
        "finnhub_free": 60 / FINNHUB_MAX_REQ_PER_MINUTE,
        "alpaca": 60 / ALPACA_MAX_REQ_PER_MINUTE,
        "newsapi_org": 86400 / NEWSAPI_ORG_MAX_REQ_PER_DAY,  # 864s between calls
        "newsapi_ai": 86400 / NEWSAPI_AI_MAX_REQ_PER_DAY,  # 86.4s between calls
        "polygon_free": 60 / POLYGON_MAX_REQ_PER_MINUTE,
        "alpha_vantage_free": 60 / ALPHA_VANTAGE_MAX_REQ_PER_MINUTE,
    }
    return mapping.get(source, 0.0)
