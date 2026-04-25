"""Patent Activity Features (M38b).

Extracts innovation signals from patent filing data. Uses publicly available
patent data (e.g. USPTO PAIR, Google Patents) to build company-level
innovation metrics.

Features produced:
    patent_count_12m         — patent filings in trailing 12 months
    patent_growth_yoy        — year-over-year growth rate of filings
    patent_citation_score    — average forward citations per patent
    patent_breadth           — number of distinct IPC classes filed
    patent_recency           — days since most recent filing
    innovation_momentum      — composite innovation signal

All features use filing dates (not grant dates) for PIT safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatentConfig:
    """Configuration for patent feature computation."""

    lookback_months: int = 12
    citation_decay_years: float = 3.0
    min_patents: int = 1
    breadth_weight: float = 0.3
    recency_weight: float = 0.2
    growth_weight: float = 0.3
    citation_weight: float = 0.2


def compute_patent_features(
    filings: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: PatentConfig | None = None,
    symbol_col: str = "symbol",
    filing_date_col: str = "filing_date",
    ipc_col: str = "ipc_class",
    citation_col: str = "forward_citations",
) -> pd.DataFrame:
    """Compute patent-based innovation features per symbol.

    Args:
        filings: DataFrame with patent filing records.
        as_of: Reference date (PIT cutoff).
        config: PatentConfig.
        symbol_col: Symbol / company column.
        filing_date_col: Filing date column (PIT-safe).
        ipc_col: IPC class column (for breadth).
        citation_col: Forward citation count column.

    Returns:
        DataFrame with one row per symbol and patent feature columns.
    """
    cfg = config or PatentConfig()
    as_of_dt = pd.Timestamp(as_of)

    if filings.empty:
        return pd.DataFrame(columns=[
            symbol_col, "patent_count_12m", "patent_growth_yoy",
            "patent_citation_score", "patent_breadth", "patent_recency_days",
            "innovation_momentum",
        ])

    df = filings.copy()
    df[filing_date_col] = pd.to_datetime(df[filing_date_col])

    # PIT filter: only filings before as_of
    df = df[df[filing_date_col] <= as_of_dt]

    if df.empty:
        return pd.DataFrame(columns=[
            symbol_col, "patent_count_12m", "patent_growth_yoy",
            "patent_citation_score", "patent_breadth", "patent_recency_days",
            "innovation_momentum",
        ])

    cutoff_12m = as_of_dt - pd.DateOffset(months=cfg.lookback_months)
    cutoff_24m = as_of_dt - pd.DateOffset(months=cfg.lookback_months * 2)

    rows = []
    for sym, grp in df.groupby(symbol_col):
        recent = grp[grp[filing_date_col] > cutoff_12m]
        prior = grp[(grp[filing_date_col] > cutoff_24m) & (grp[filing_date_col] <= cutoff_12m)]

        count_12m = len(recent)
        count_prior = len(prior)

        if count_12m < cfg.min_patents and count_prior < cfg.min_patents:
            continue

        # Growth YoY
        if count_prior > 0:
            growth = (count_12m - count_prior) / count_prior
        else:
            growth = 1.0 if count_12m > 0 else 0.0

        # Citation score
        if citation_col in grp.columns and count_12m > 0:
            citations = recent[citation_col].fillna(0).values
            cite_score = float(np.mean(citations))
        else:
            cite_score = 0.0

        # Breadth: distinct IPC classes
        if ipc_col in grp.columns and count_12m > 0:
            breadth = recent[ipc_col].nunique()
        else:
            breadth = 0

        # Recency: days since last filing
        last_filing = grp[filing_date_col].max()
        recency_days = (as_of_dt - last_filing).days

        # Composite innovation momentum
        growth_norm = np.clip(growth, -1.0, 3.0) / 3.0
        breadth_norm = min(breadth / 10.0, 1.0)
        recency_norm = max(1.0 - recency_days / 365.0, 0.0)
        cite_norm = min(cite_score / 10.0, 1.0)

        innovation = (
            cfg.growth_weight * growth_norm
            + cfg.breadth_weight * breadth_norm
            + cfg.recency_weight * recency_norm
            + cfg.citation_weight * cite_norm
        )

        rows.append({
            symbol_col: sym,
            "patent_count_12m": count_12m,
            "patent_growth_yoy": round(growth, 4),
            "patent_citation_score": round(cite_score, 2),
            "patent_breadth": breadth,
            "patent_recency_days": recency_days,
            "innovation_momentum": round(float(innovation), 4),
        })

    result = pd.DataFrame(rows)
    logger.info("[Patent] Computed features for %d symbols as of %s", len(result), as_of)
    return result


__all__ = [
    "PatentConfig",
    "compute_patent_features",
]
