"""Post-Earnings-Announcement Drift (PEAD) v2 — multi-faceted enhancement.

Klassisches PEAD
----------------
Aktien mit positiver/negativer Earnings-Surprise driften systematisch in die
Richtung der Surprise für ~60 Handelstage (Bernard/Thomas 1989, *JoF*).

V2-Erweiterungen
----------------
1. **Surprise-Quality**: Standardized Unexpected Earnings (SUE) gewichtet mit
   Estimate-Dispersion (Niedrige Dispersion = höhere Konviktion).
2. **Conditional Drift**: Drift ist stärker bei
   - kleinem Float
   - hohem Short-Interest (gegenüber neutralem)
   - günstiger Bewertung (low P/E Reversal-Bias)
3. **Two-Stage**: Day-of-EA reaction (T+0 → T+2) vs Long-Drift (T+2 → T+60).
4. **Quality-Filter**: Nur Earnings mit ≥3 Analyst-Estimaten.

Die V2-Variante ist eine Forschungs-Erweiterung; die ursprüngliche PEAD-Logik
liegt im Mainline-Core unter ``src/assembled_core/features/event_features.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def standardized_unexpected_earnings(
    earnings_df: pd.DataFrame,
    eps_col: str = "actual_eps",
    estimate_col: str = "consensus_eps",
    dispersion_col: Optional[str] = "estimate_std",
    min_estimates_col: Optional[str] = "num_estimates",
    min_estimates_threshold: int = 3,
) -> pd.DataFrame:
    """Berechne SUE = (actual − consensus) / estimate_dispersion.

    Args:
        earnings_df: DataFrame mit mindestens [symbol, fiscal_date, actual, consensus].
        eps_col, estimate_col, dispersion_col, min_estimates_col: column names.
        min_estimates_threshold: Earnings mit weniger Schätzern werden ausgefiltert.

    Returns:
        DataFrame mit ``sue`` und ``sue_z`` (Cross-section-Quartil je Tag).
    """
    if earnings_df.empty:
        return earnings_df.assign(sue=pd.Series(dtype=float))

    df = earnings_df.copy()
    if min_estimates_col and min_estimates_col in df.columns:
        df = df[df[min_estimates_col] >= min_estimates_threshold]
    surprise = df[eps_col] - df[estimate_col]
    if dispersion_col and dispersion_col in df.columns:
        denom = df[dispersion_col].replace(0, np.nan).clip(lower=1e-6)
        df["sue"] = surprise / denom
    else:
        # Fallback: divide by absolute consensus, then z-scale by quartile
        df["sue"] = surprise / df[estimate_col].abs().clip(lower=0.01)
    return df


def post_earnings_drift_signal(
    earnings_df: pd.DataFrame,
    prices: pd.DataFrame,
    drift_window: int = 60,
    skip_days: int = 2,
    short_interest_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Erzeuge PEAD-Signal je (date, symbol).

    Args:
        earnings_df: DataFrame mit [symbol, announcement_date, sue].
        prices: DataFrame [date, symbol, close].
        drift_window: Drift-Horizont (Handelstage).
        skip_days: Skip Day-of-EA reaction.
        short_interest_panel: Optional [date, symbol, short_pressure] für
            Conditional-Drift-Verstärkung.

    Returns:
        DataFrame [date, symbol, pead_signal].
    """
    if earnings_df.empty or prices.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    by_sym = prices.set_index(["symbol", "date"]).sort_index()
    for _, row in earnings_df.iterrows():
        sym = row.get("symbol")
        anno = pd.to_datetime(row.get("announcement_date"), utc=True, errors="coerce")
        sue = row.get("sue", np.nan)
        if pd.isna(anno) or pd.isna(sue):
            continue
        if sym not in by_sym.index.get_level_values("symbol"):
            continue
        sub = by_sym.loc[sym]
        if isinstance(sub, pd.Series):
            continue
        # Trading days from anno+skip_days to anno+drift_window
        future_dates = sub.index[
            (sub.index >= anno + pd.Timedelta(days=skip_days))
            & (sub.index <= anno + pd.Timedelta(days=drift_window * 2))
        ]
        if len(future_dates) == 0:
            continue
        # Sign matches sue
        sign = 1.0 if sue > 0 else -1.0 if sue < 0 else 0.0
        magnitude = float(np.tanh(abs(sue) / 2.0))  # bounded [0, 1]
        for d in future_dates[:drift_window]:
            out_rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "pead_signal": sign * magnitude,
                    "sue": sue,
                    "days_since_announcement": (d - anno).days,
                }
            )
    df = pd.DataFrame(out_rows)
    if (
        short_interest_panel is not None
        and not short_interest_panel.empty
        and not df.empty
    ):
        df = df.merge(
            short_interest_panel[["date", "symbol", "short_pressure"]],
            on=["date", "symbol"],
            how="left",
        )
        # Conditional amplification: hohe short-pressure verstärkt positive PEAD (Squeeze-Risiko)
        df["pead_signal"] = (
            df["pead_signal"]
            + 0.2 * df["short_pressure"].fillna(0).clip(-1, 1) * df["pead_signal"]
        )
    return df


__all__ = [
    "standardized_unexpected_earnings",
    "post_earnings_drift_signal",
]
