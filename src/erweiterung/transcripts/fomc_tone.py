"""FOMC-Tone-Analyse — Hawkish vs Dovish Scoring.

Theorie
-------
FOMC-Statements und Press-Conferences sind extrem market-moving. Akademische
Studien (Lucca/Trebbi 2009, Hansen et al. 2018) zeigen, dass:
- **Hawkish** language (raise, tighten, inflation-fighting) → bond-yields up,
  equities down kurzfristig.
- **Dovish** (accommodate, stimulus, support) → opposite.

Methodik
--------
Domain-specific keyword-list + Counts. Optional Sentence-Level mit
Context-Window für Negation.

Reference
---------
- Lucca, D. & Trebbi, F. (2009). Measuring Central Bank Communication.
- Hansen, S., McMahon, M. & Prat, A. (2018). Transparency and Deliberation.
"""

from __future__ import annotations


import pandas as pd

from erweiterung.transcripts.loughran_mcdonald import tokenize


HAWKISH = frozenset(
    [
        "raise",
        "raising",
        "raised",
        "increase",
        "increasing",
        "increased",
        "hike",
        "hikes",
        "hiked",
        "hiking",
        "tighten",
        "tightening",
        "tightened",
        "tight",
        "elevated",
        "restrictive",
        "restrict",
        "restrictive",
        "contraction",
        "contractionary",
        "remove",
        "removing",
        "withdraw",
        "withdrawing",
        "inflation",
        "inflationary",
        "overheating",
        "wage-pressure",
        "wage",
        "robust",
        "strong",
        "resilient",
        "tight-labor",
        "firm",
        "firming",
        "above-target",
        "exceed-target",
    ]
)

DOVISH = frozenset(
    [
        "lower",
        "lowering",
        "lowered",
        "cut",
        "cuts",
        "cutting",
        "ease",
        "easing",
        "eased",
        "accommodate",
        "accommodative",
        "accommodation",
        "support",
        "supporting",
        "supportive",
        "stimulate",
        "stimulus",
        "stimulative",
        "expansion",
        "expansionary",
        "patient",
        "patience",
        "wait",
        "waiting",
        "hold",
        "holding",
        "pause",
        "paused",
        "below-target",
        "weak",
        "weaker",
        "weakening",
        "slack",
        "sluggish",
        "subdued",
        "softening",
        "decelerating",
        "moderating",
        "elevated-unemployment",
    ]
)


_NEGATION_WINDOW = 3
_NEGATIONS = frozenset(["not", "no", "never", "without", "neither", "nor"])


def hawkish_dovish_score(text: str, negation_aware: bool = True) -> dict:
    """Score hawkish/dovish per text.

    Args:
        text: speech / statement text.
        negation_aware: if True, flip sign if negation within window.

    Returns:
        dict mit hawkish_count, dovish_count, hd_score = (h − d) / (h + d).
        hd_score ∈ [-1, 1]: positive = hawkish-leaning, negative = dovish.
    """
    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return {"hawkish_count": 0, "dovish_count": 0, "hd_score": 0.0, "n_tokens": 0}

    h_count = 0
    d_count = 0
    for i, t in enumerate(tokens):
        is_hawk = t in HAWKISH
        is_dove = t in DOVISH
        if not (is_hawk or is_dove):
            continue
        sign_flip = False
        if negation_aware:
            start = max(0, i - _NEGATION_WINDOW)
            window = tokens[start:i]
            if any(neg in window for neg in _NEGATIONS):
                sign_flip = True
        if is_hawk:
            if sign_flip:
                d_count += 1
            else:
                h_count += 1
        elif is_dove:
            if sign_flip:
                h_count += 1
            else:
                d_count += 1

    total = h_count + d_count
    score = (h_count - d_count) / total if total > 0 else 0.0
    return {
        "hawkish_count": h_count,
        "dovish_count": d_count,
        "hd_score": score,
        "n_tokens": n,
        "intensity": total / n,
    }


def score_fomc_statements(
    statements: list[str], dates: list | None = None
) -> pd.DataFrame:
    """Score a list of FOMC statements.

    Returns:
        DataFrame [date, hawkish_count, dovish_count, hd_score, n_tokens, intensity].
    """
    rows = []
    for i, txt in enumerate(statements):
        rec = hawkish_dovish_score(txt)
        rec["date"] = dates[i] if dates and i < len(dates) else i
        rows.append(rec)
    return pd.DataFrame(rows)


def fomc_change_signal(scored_df: pd.DataFrame) -> pd.Series:
    """Detect changes in hawkish/dovish-stance between meetings.

    Δhd_score > 0.2 = significantly more hawkish than last → bond-yields up.
    """
    s = pd.Series(scored_df["hd_score"].values, index=scored_df["date"])
    return s.diff()


__all__ = [
    "HAWKISH",
    "DOVISH",
    "hawkish_dovish_score",
    "score_fomc_statements",
    "fomc_change_signal",
]
