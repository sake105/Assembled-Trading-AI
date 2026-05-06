"""News event taxonomy — maps trigger types and event labels to 6 economic categories.

Categories: FINANZEN, KONFLIKTE, GEOPOLITIK, ROHSTOFFE, TECHNOLOGIE, POLITIK, SONSTIGE
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import pandas as pd

CATEGORIES: list[str] = [
    "FINANZEN",
    "KONFLIKTE",
    "GEOPOLITIK",
    "ROHSTOFFE",
    "TECHNOLOGIE",
    "POLITIK",
    "SONSTIGE",
]

# Priority order (higher index = lower priority in tie-breaks)
_PRIORITY: dict[str, int] = {cat: i for i, cat in enumerate(CATEGORIES[:-1])}

TRIGGER_TO_CATEGORY: dict[str, str] = {
    # FINANZEN
    "BANKING_CRISIS": "FINANZEN",
    "RATE_SURPRISE": "FINANZEN",
    "CREDIT_DOWNGRADE": "FINANZEN",
    "PEG_STRESS": "FINANZEN",
    "RESERVE_DRAIN": "FINANZEN",
    "FISCAL_CLIFF": "FINANZEN",
    # KONFLIKTE
    "WAR_ESCALATION": "KONFLIKTE",
    "MILITARY_BUILDUP": "KONFLIKTE",
    "CASUALTY_SPIKE": "KONFLIKTE",
    "TERRITORIAL_ESCALATION": "KONFLIKTE",
    "NUCLEAR_THREAT": "KONFLIKTE",
    "CAPABILITY_SHIFT": "KONFLIKTE",
    "PROXY_WAR_EXPANSION": "KONFLIKTE",
    # GEOPOLITIK
    "SANCTIONS_ESCALATION": "GEOPOLITIK",
    "TRADE_WAR_ESCALATION": "GEOPOLITIK",
    "ALLIANCE_SHIFT": "GEOPOLITIK",
    "DIPLOMATIC_CRISIS": "GEOPOLITIK",
    "HEGEMONIC_CHALLENGE": "GEOPOLITIK",
    "RESOURCE_NATIONALIZATION": "GEOPOLITIK",
    "STRAIT_BLOCKADE": "GEOPOLITIK",
    # ROHSTOFFE
    "ENERGY_SUPPLY_RISK": "ROHSTOFFE",
    "CHOKEPOINT_STRESS": "ROHSTOFFE",
    "SHIPPING_DISRUPTION": "ROHSTOFFE",
    "SUPPLY_CHAIN_BREAK": "ROHSTOFFE",
    "LOGISTICS_DISRUPTION": "ROHSTOFFE",
    "SEVERE_WEATHER_ALERT": "ROHSTOFFE",
    # TECHNOLOGIE
    "NEW_EXPORT_CONTROL": "TECHNOLOGIE",
    "CYBER_ESCALATION": "TECHNOLOGIE",
    "ENTITY_LISTING": "TECHNOLOGIE",
    "TECHNOLOGY_GAP_WIDENING": "TECHNOLOGIE",
    "ZERO_DAY_DISCLOSURE": "TECHNOLOGIE",
    "MAJOR_BREACH_DETECTED": "TECHNOLOGIE",
    "STATE_ACTOR_ACTIVITY": "TECHNOLOGIE",
    # POLITIK
    "POLICY_SHIFT": "POLITIK",
    "COUP_RISK": "POLITIK",
}

# Keyword → category fallback for free-text event_types labels
_KEYWORD_CATEGORY: dict[str, str] = {
    "bank": "FINANZEN",
    "rate": "FINANZEN",
    "credit": "FINANZEN",
    "fiscal": "FINANZEN",
    "debt": "FINANZEN",
    "inflation": "FINANZEN",
    "recession": "FINANZEN",
    "war": "KONFLIKTE",
    "military": "KONFLIKTE",
    "conflict": "KONFLIKTE",
    "attack": "KONFLIKTE",
    "strike": "KONFLIKTE",
    "nuclear": "KONFLIKTE",
    "sanction": "GEOPOLITIK",
    "trade_war": "GEOPOLITIK",
    "tariff": "GEOPOLITIK",
    "diplomatic": "GEOPOLITIK",
    "alliance": "GEOPOLITIK",
    "oil": "ROHSTOFFE",
    "energy": "ROHSTOFFE",
    "gas": "ROHSTOFFE",
    "shipping": "ROHSTOFFE",
    "supply_chain": "ROHSTOFFE",
    "commodity": "ROHSTOFFE",
    "tech": "TECHNOLOGIE",
    "cyber": "TECHNOLOGIE",
    "export_control": "TECHNOLOGIE",
    "ai": "TECHNOLOGIE",
    "chip": "TECHNOLOGIE",
    "policy": "POLITIK",
    "election": "POLITIK",
    "coup": "POLITIK",
    "regulation": "POLITIK",
}


def _keyword_cat(label: str) -> str | None:
    lbl = label.lower()
    for kw, cat in _KEYWORD_CATEGORY.items():
        if kw in lbl:
            return cat
    return None


def categorize_event(
    event_types: Sequence[str] | None = None,
    trigger_type: str | None = None,
    *,
    fallback: str = "SONSTIGE",
) -> str:
    """Assign a single category to an event.

    Priority:
    1. Direct trigger_type → TRIGGER_TO_CATEGORY lookup.
    2. Majority vote across event_types labels (keyword matching).
    3. Tie-break by _PRIORITY order (FINANZEN > KONFLIKTE > ...).
    4. fallback if no match.
    """
    # Path 1 — direct trigger_type hit
    if trigger_type:
        cat = TRIGGER_TO_CATEGORY.get(trigger_type.upper())
        if cat:
            return cat

    # Path 2 — vote across event_types
    if event_types:
        votes: Counter[str] = Counter()
        for label in event_types:
            cat = TRIGGER_TO_CATEGORY.get(label.upper())
            if not cat:
                cat = _keyword_cat(label)
            if cat:
                votes[cat] += 1

        if votes:
            max_count = max(votes.values())
            candidates = [c for c, n in votes.items() if n == max_count]
            if len(candidates) == 1:
                return candidates[0]
            # Tie-break by priority
            return min(candidates, key=lambda c: _PRIORITY.get(c, 99))

    return fallback


def aggregate_categories_by_window(
    events_df: pd.DataFrame,
    window_hours: int = 24,
) -> pd.DataFrame:
    """Count news events by category per (date, symbol) window.

    Parameters
    ----------
    events_df : DataFrame with columns [published_at, tickers/symbol, category]
    window_hours : rolling look-back window in hours

    Returns
    -------
    DataFrame with columns: date, symbol, news_count_finanzen_24h, ..._konflikte_24h, etc.
    """
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()

    # Normalize timestamp column
    ts_col = next((c for c in ["published_at", "timestamp", "date"] if c in df.columns), None)
    if ts_col is None:
        return pd.DataFrame()
    df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"])
    df["_date"] = df["_ts"].dt.normalize()

    # Normalize symbol column
    sym_col = next((c for c in ["symbol", "ticker", "tickers"] if c in df.columns), None)
    if sym_col is None:
        df["_symbol"] = "MARKET"
    else:
        # Explode if list
        if df[sym_col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(sym_col)
        df["_symbol"] = df[sym_col].fillna("MARKET").astype(str).str.upper()

    # Ensure category column
    if "category" not in df.columns:
        df["category"] = "SONSTIGE"

    # Pivot into category counts
    cats = [c for c in CATEGORIES if c != "SONSTIGE"]
    rows = []
    for (date, sym), grp in df.groupby(["_date", "_symbol"]):
        cat_counts = grp["category"].value_counts()
        row: dict = {"date": date, "symbol": sym}
        for cat in cats:
            col = f"news_count_{cat.lower()}_{window_hours}h"
            row[col] = int(cat_counts.get(cat, 0))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result
