"""Attention-Signal: Wikipedia + Google Trends + Reddit-Mentions.

Theorie
-------
Mehrere akademische Studien zeigen: ungewöhnliche Aufmerksamkeit für ein Asset
prognostiziert Kursbewegungen kurzfristig.

- Da/Engelberg/Gao (2011, *J Finance*): Google-Search-Volume → Returns nächste 2 Wochen
- Moat et al. (2013): Wikipedia-Pageviews → DJI-Bewegungen
- Bollen/Mao/Zeng (2011): Twitter-Sentiment → DJIA-Vorhersage

Signal-Konstruktion
-------------------
1. Multi-Source-Composite-Z-Score: ``(z_wiki + z_trends + z_reddit) / N_sources``.
2. PIT-Shift je Quelle (Wiki: T+1, Trends: T, Reddit: T+0).
3. Rolling-Z-Score über 30 Tage entfernt Saison-/Ticker-Bias.

Interpretation
--------------
- Sehr hohe Aufmerksamkeit (Z > 2): typisch nach Earnings oder bei Hype —
  oft bärisches Signal für nächste 5-10 Tage (Mean-Reversion).
- Steigende Aufmerksamkeit moderat (0 < Z < 1.5): bullishes Momentum-Signal.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def composite_attention_score(
    wiki_df: Optional[pd.DataFrame] = None,
    trends_df: Optional[pd.DataFrame] = None,
    reddit_df: Optional[pd.DataFrame] = None,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Aggregiere mehrere Aufmerksamkeitsquellen zu einem Score.

    Args:
        wiki_df: Output von ``wikipedia_pageviews.attention_score``,
            erwartet Spalten ``[date, symbol, attention_score]``.
        trends_df: Output von ``google_trends.trends_zscore``,
            erwartet Spalten ``[date, keyword, svi_z]``. ``keyword`` muss als
            ``symbol`` umbenannt sein (oder beides als 'symbol' gemerged).
        reddit_df: DataFrame [date, ticker, mention_z] (mention-z aus
            ``reddit_pushshift.reddit_mention_panel`` + Z-Skalierung).
        weights: Gewichte (default: 0.4 wiki, 0.4 trends, 0.2 reddit).

    Returns:
        DataFrame [date, symbol, attention_composite].
    """
    weights = weights or {"wiki": 0.4, "trends": 0.4, "reddit": 0.2}
    parts: list[pd.DataFrame] = []

    if (
        wiki_df is not None
        and not wiki_df.empty
        and "attention_score" in wiki_df.columns
    ):
        w = wiki_df[["date", "symbol", "attention_score"]].copy()
        w = w.rename(columns={"attention_score": "_wiki_z"})
        parts.append(w)
    if trends_df is not None and not trends_df.empty:
        if "keyword" in trends_df.columns:
            t = trends_df.rename(columns={"keyword": "symbol", "svi_z": "_trends_z"})
        else:
            t = trends_df.rename(columns={"svi_z": "_trends_z"})
        t = t[["date", "symbol", "_trends_z"]].copy()
        parts.append(t)
    if reddit_df is not None and not reddit_df.empty:
        r = reddit_df.rename(columns={"ticker": "symbol", "mention_z": "_reddit_z"})
        r = r[["date", "symbol", "_reddit_z"]].copy()
        parts.append(r)

    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "attention_composite"])

    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on=["date", "symbol"], how="outer")

    merged["attention_composite"] = 0.0
    used_weight = 0.0
    if "_wiki_z" in merged.columns:
        merged["attention_composite"] += weights["wiki"] * merged["_wiki_z"].fillna(0)
        used_weight += weights["wiki"]
    if "_trends_z" in merged.columns:
        merged["attention_composite"] += weights["trends"] * merged["_trends_z"].fillna(
            0
        )
        used_weight += weights["trends"]
    if "_reddit_z" in merged.columns:
        merged["attention_composite"] += weights["reddit"] * merged["_reddit_z"].fillna(
            0
        )
        used_weight += weights["reddit"]
    if used_weight > 0:
        merged["attention_composite"] /= used_weight

    return merged[["date", "symbol", "attention_composite"]]


def attention_meanrev_signal(
    composite: pd.DataFrame, threshold: float = 2.0
) -> pd.DataFrame:
    """Konvertiere extreme Aufmerksamkeit in Short-Signal.

    Z > threshold => -1 (mean-reversion long-short). Akademische Evidenz
    zeigt: Hype-Spikes verkehren sich i. d. R. innerhalb von 5-10 Handelstagen.
    """
    if composite.empty:
        return composite.assign(att_meanrev_signal=pd.Series(dtype=float))
    out = composite.copy()
    out["att_meanrev_signal"] = 0.0
    out.loc[out["attention_composite"] > threshold, "att_meanrev_signal"] = -1.0
    out.loc[out["attention_composite"] < -threshold, "att_meanrev_signal"] = +1.0
    return out


__all__ = ["composite_attention_score", "attention_meanrev_signal"]
