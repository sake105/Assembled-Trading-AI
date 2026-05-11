"""Material-Event-Klassifizierung für SEC 8-K Filings.

8-K Item-Codes (SEC)
--------------------
Form 8-K hat strukturierte "Items" — jedes Item = bestimmter Event-Typ.

Relevante Items (Trading-Impact-sortiert)
-----------------------------------------
- **1.01** Entry into Material-Agreement (oft positiv)
- **1.02** Termination of Material-Agreement
- **2.01** Completion of Acquisition / Disposition
- **2.02** Results of Operations (Earnings) — sehr volatile
- **2.05** Costs of Exit / Disposal-Activities (Restructuring)
- **2.06** Material Impairment
- **3.01** Notice of Delisting (sehr negativ)
- **3.02** Unregistered Sales of Securities (Dilution)
- **3.03** Material Modifications to Rights of Securities Holders
- **4.01** Changes in Registrant's Certifying Accountant (red flag)
- **4.02** Non-Reliance on Previously Issued Financial Statements (red flag)
- **5.01** Changes in Control of Registrant (M&A signal)
- **5.02** Departure of Directors / Officers (CEO change)
- **5.03** Amendments to Articles
- **7.01** Regulation FD Disclosure
- **8.01** Other Events
- **9.01** Financial Statements and Exhibits

Reference
---------
SEC Form 8-K: https://www.sec.gov/about/forms/form8-k.pdf
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


# Item-Code → (Event-Category, Typical-Direction, Material-Score)
ITEM_CATEGORIES: dict[str, tuple[str, int, float]] = {
    "1.01": ("agreement_entry", +1, 0.6),
    "1.02": ("agreement_termination", -1, 0.6),
    "1.03": ("bankruptcy", -1, 1.0),
    "2.01": ("acquisition_disposition", +1, 0.7),
    "2.02": ("results_of_operations", 0, 0.9),  # direction depends on surprise
    "2.03": ("creation_obligation", -1, 0.5),
    "2.04": ("triggering_event", -1, 0.6),
    "2.05": ("exit_disposal_costs", -1, 0.5),
    "2.06": ("material_impairment", -1, 0.8),
    "3.01": ("notice_of_delisting", -1, 1.0),
    "3.02": ("unregistered_sales", -1, 0.6),
    "3.03": ("modifications_rights", -1, 0.4),
    "4.01": ("changes_accountant", -1, 0.7),
    "4.02": ("non_reliance_financials", -1, 0.9),
    "5.01": ("change_in_control", +1, 0.9),  # M&A
    "5.02": ("director_officer_change", 0, 0.6),
    "5.03": ("amendments_articles", 0, 0.3),
    "5.07": ("submitted_matters_vote", 0, 0.3),
    "7.01": ("regulation_fd", 0, 0.5),
    "8.01": ("other_events", 0, 0.4),
    "9.01": ("financial_statements", 0, 0.2),
}


_ITEM_PATTERN = re.compile(r"\bitem[\s\-]*(\d+\.\d+)\b", re.IGNORECASE)


@dataclass
class EventClassification:
    accession: str
    items: list[str]
    categories: list[str]
    expected_direction: int  # +1 / -1 / 0
    material_score: float  # max over items
    explanation: str


def extract_items(text: str) -> list[str]:
    """Extract 8-K item-codes from filing-text."""
    if not isinstance(text, str):
        return []
    matches = _ITEM_PATTERN.findall(text)
    # Normalize
    return sorted(set([m for m in matches]))


def classify_filing(items: list[str], accession: str = "") -> EventClassification:
    """Map list of item-codes to event-classification.

    Args:
        items: list of item-codes (e.g. ["2.02", "9.01"]).
        accession: optional accession-id.

    Returns:
        EventClassification.
    """
    categories: list[str] = []
    directions: list[int] = []
    scores: list[float] = []
    explanations: list[str] = []

    for it in items:
        if it in ITEM_CATEGORIES:
            cat, direction, score = ITEM_CATEGORIES[it]
            categories.append(cat)
            directions.append(direction)
            scores.append(score)
            explanations.append(f"Item {it}: {cat}")
        else:
            categories.append("unknown")
            scores.append(0.1)

    # Aggregate direction: sum of signed-score, weighted by material-score
    if directions and scores:
        weighted_dir = sum(d * s for d, s in zip(directions, scores))
        if weighted_dir > 0.2:
            agg_direction = +1
        elif weighted_dir < -0.2:
            agg_direction = -1
        else:
            agg_direction = 0
    else:
        agg_direction = 0

    return EventClassification(
        accession=accession,
        items=items,
        categories=categories,
        expected_direction=agg_direction,
        material_score=max(scores) if scores else 0.0,
        explanation=" | ".join(explanations) if explanations else "no_items",
    )


def classify_filing_text(text: str, accession: str = "") -> EventClassification:
    """End-to-end: extract items from text → classify."""
    items = extract_items(text)
    return classify_filing(items, accession)


def filter_high_material_events(
    filings: pd.DataFrame,
    classifications: list[EventClassification],
    score_threshold: float = 0.7,
) -> pd.DataFrame:
    """Return only filings with material_score above threshold.

    Args:
        filings: DataFrame from edgar_stream.
        classifications: list aligned with filings.
        score_threshold: e.g. 0.7 = only most material events.

    Returns:
        Filtered DataFrame with classification columns added.
    """
    rows = []
    for i, row in filings.iterrows():
        if i >= len(classifications):
            continue
        c = classifications[i]
        if c.material_score < score_threshold:
            continue
        rec = row.to_dict()
        rec["categories"] = "|".join(c.categories)
        rec["expected_direction"] = c.expected_direction
        rec["material_score"] = c.material_score
        rows.append(rec)
    return pd.DataFrame(rows)


__all__ = [
    "ITEM_CATEGORIES",
    "EventClassification",
    "extract_items",
    "classify_filing",
    "classify_filing_text",
    "filter_high_material_events",
]
