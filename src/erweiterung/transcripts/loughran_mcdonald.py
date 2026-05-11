"""Loughran-McDonald Finance-Sentiment-Dictionary (LM 2011).

Reference
---------
Loughran, T. & McDonald, B. (2011). When is a Liability not a Liability? Textual
Analysis, Dictionaries, and 10-Ks. *Journal of Finance* 66.

Idee
----
General-Purpose-Sentiment-Lexika (Harvard-IV, etc.) sind für Finance unreliable.
Beispiel: "tax", "liability", "vice" sind in Finance neutral, aber in
General-Purpose negativ. Loughran/McDonald bauen Finance-spezifisches Lexikon
auf 10-K-Filings.

6 Kategorien
------------
- ``positive`` : optimistisch (z.B. "outperform", "strong", "exceeded")
- ``negative`` : pessimistisch (z.B. "loss", "decline", "weakness")
- ``uncertainty``: hedge-words (z.B. "may", "could", "approximately")
- ``litigious``: Klage-/Rechts-Sprache
- ``strong_modal``: bekräftigend ("will", "must")
- ``weak_modal``: abschwächend ("could", "might")

Implementation
--------------
Word-list-Subset (essentielle Wörter pro Kategorie). Vollständige Liste
~10k Wörter beim LM-Website-Download verfügbar; hier eingebettete Top-Wörter.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


# Subset der häufigsten Loughran-McDonald-Wörter (vollständiges Lexikon
# unter https://sraf.nd.edu/textual-analysis/resources/)
LM_POSITIVE = frozenset(
    [
        "outperform",
        "strong",
        "exceeded",
        "improved",
        "profitable",
        "successful",
        "advantage",
        "achieve",
        "benefit",
        "beneficial",
        "boost",
        "confidence",
        "constructive",
        "deliver",
        "delivered",
        "effective",
        "encouraging",
        "enhance",
        "enhanced",
        "enthusiasm",
        "excellent",
        "favorable",
        "gain",
        "gained",
        "growth",
        "improve",
        "improving",
        "innovative",
        "leadership",
        "milestone",
        "opportunities",
        "optimistic",
        "outperformed",
        "positive",
        "profitability",
        "progress",
        "record",
        "resilient",
        "robust",
        "solid",
        "strength",
        "stronger",
        "success",
        "successfully",
        "surpassed",
        "tremendous",
        "upbeat",
        "upside",
    ]
)

LM_NEGATIVE = frozenset(
    [
        "adverse",
        "adversely",
        "bankruptcy",
        "bearish",
        "below",
        "burden",
        "cancelled",
        "challenging",
        "claim",
        "concern",
        "concerned",
        "concerns",
        "declined",
        "declining",
        "decreased",
        "decreases",
        "decreasing",
        "deficit",
        "delay",
        "delayed",
        "deteriorate",
        "deteriorated",
        "difficult",
        "diminished",
        "disappointing",
        "disrupt",
        "disruption",
        "doubt",
        "downturn",
        "drag",
        "drop",
        "exposure",
        "fail",
        "failed",
        "failure",
        "fell",
        "fraud",
        "headwind",
        "headwinds",
        "hurt",
        "impairment",
        "inability",
        "ineffective",
        "investigation",
        "lawsuit",
        "layoffs",
        "litigation",
        "loss",
        "losses",
        "missed",
        "misstatement",
        "negative",
        "negatively",
        "outflow",
        "overstated",
        "penalty",
        "pressure",
        "recession",
        "restructure",
        "restructuring",
        "shortfall",
        "shrink",
        "slowdown",
        "softness",
        "stagnation",
        "subdued",
        "suspended",
        "underperform",
        "unfavorable",
        "weak",
        "weakened",
        "weakness",
        "writedown",
        "write-off",
        "writeoff",
    ]
)

LM_UNCERTAINTY = frozenset(
    [
        "almost",
        "anticipate",
        "anticipated",
        "appear",
        "appeared",
        "appears",
        "approximate",
        "approximately",
        "approximated",
        "assume",
        "assumed",
        "believe",
        "believed",
        "believes",
        "could",
        "depend",
        "depending",
        "depends",
        "estimate",
        "estimated",
        "estimates",
        "expect",
        "expected",
        "expects",
        "guidance",
        "indefinite",
        "intend",
        "intended",
        "intends",
        "likelihood",
        "may",
        "maybe",
        "might",
        "perhaps",
        "possibility",
        "possible",
        "possibly",
        "predict",
        "predicted",
        "predicts",
        "preliminary",
        "presumably",
        "probably",
        "projected",
        "roughly",
        "seem",
        "seemed",
        "seems",
        "suggest",
        "suggested",
        "suggests",
        "tentative",
        "tentatively",
        "uncertain",
        "uncertainty",
    ]
)

LM_LITIGIOUS = frozenset(
    [
        "alleged",
        "alleging",
        "appeal",
        "appealed",
        "appellate",
        "arbitration",
        "attorney",
        "civil",
        "claim",
        "claimed",
        "claims",
        "compel",
        "complaint",
        "complaints",
        "compliance",
        "consent",
        "contention",
        "convicted",
        "court",
        "courts",
        "crime",
        "criminal",
        "damage",
        "damages",
        "decree",
        "defendant",
        "defendants",
        "deposition",
        "disclosure",
        "dismiss",
        "dismissed",
        "dismissal",
        "felony",
        "guilty",
        "indemnification",
        "indemnify",
        "infringe",
        "infringement",
        "infringing",
        "injunction",
        "judgement",
        "judgment",
        "judicial",
        "jurisdiction",
        "justice",
        "lawsuit",
        "lawsuits",
        "lawyer",
        "legal",
        "legality",
        "legally",
        "liabilities",
        "liability",
        "liable",
        "litigant",
        "litigants",
        "litigation",
        "petition",
        "plaintiff",
        "plaintiffs",
        "probation",
        "prosecute",
        "prosecuted",
        "prosecution",
        "regulation",
        "regulations",
        "regulatory",
        "settlement",
        "settle",
        "sued",
        "summons",
        "testimony",
        "tribunal",
        "violation",
        "violations",
        "violator",
        "wrongful",
    ]
)

LM_STRONG_MODAL = frozenset(
    [
        "always",
        "must",
        "never",
        "shall",
        "undisputed",
        "undoubtedly",
        "unequivocal",
        "will",
    ]
)

LM_WEAK_MODAL = frozenset(["could", "may", "might", "perhaps", "possibly", "would"])


_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-']+\b")


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def lm_count_tokens(text: str) -> dict:
    """Count tokens in each Loughran-McDonald category.

    Returns:
        dict with counts per category + total-tokens.
    """
    tokens = tokenize(text)
    counts: Counter = Counter()
    n_total = len(tokens)
    for t in tokens:
        if t in LM_POSITIVE:
            counts["positive"] += 1
        if t in LM_NEGATIVE:
            counts["negative"] += 1
        if t in LM_UNCERTAINTY:
            counts["uncertainty"] += 1
        if t in LM_LITIGIOUS:
            counts["litigious"] += 1
        if t in LM_STRONG_MODAL:
            counts["strong_modal"] += 1
        if t in LM_WEAK_MODAL:
            counts["weak_modal"] += 1
    return {
        "positive": int(counts["positive"]),
        "negative": int(counts["negative"]),
        "uncertainty": int(counts["uncertainty"]),
        "litigious": int(counts["litigious"]),
        "strong_modal": int(counts["strong_modal"]),
        "weak_modal": int(counts["weak_modal"]),
        "n_tokens": n_total,
    }


def lm_sentiment_score(text: str) -> dict:
    """Compute LM-Sentiment-Score = (pos − neg) / n_tokens.

    Returns:
        dict with raw counts + normalized scores per category.
    """
    counts = lm_count_tokens(text)
    n = max(counts["n_tokens"], 1)
    return {
        "sentiment": (counts["positive"] - counts["negative"]) / n,
        "positive_pct": counts["positive"] / n,
        "negative_pct": counts["negative"] / n,
        "uncertainty_pct": counts["uncertainty"] / n,
        "litigious_pct": counts["litigious"] / n,
        "modality_ratio": counts["strong_modal"] / max(counts["weak_modal"], 1),
        "n_tokens": int(counts["n_tokens"]),
    }


def lm_score_documents(docs: list[str], ids: list | None = None) -> pd.DataFrame:
    """Apply LM-scoring to a list of documents."""
    rows = []
    for i, doc in enumerate(docs):
        rec = lm_sentiment_score(doc)
        rec["doc_id"] = ids[i] if ids and i < len(ids) else i
        rows.append(rec)
    return pd.DataFrame(rows)


__all__ = [
    "LM_POSITIVE",
    "LM_NEGATIVE",
    "LM_UNCERTAINTY",
    "LM_LITIGIOUS",
    "LM_STRONG_MODAL",
    "LM_WEAK_MODAL",
    "tokenize",
    "lm_count_tokens",
    "lm_sentiment_score",
    "lm_score_documents",
]
