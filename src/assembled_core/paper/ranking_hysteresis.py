"""Ranking hysteresis for paper track runner.

Reduces symbol rotation churn by applying different rank thresholds
for entering vs. holding positions:
- New symbols must rank within entry_n (strict)
- Held symbols stay as long as they rank within hold_n (lenient)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


def apply_ranking_hysteresis(
    signals: pd.DataFrame,
    held_symbols: Set[str],
    *,
    entry_n: int = 5,
    hold_n: int = 7,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter LONG signals with hysteresis based on currently held positions.

    Args:
        signals: DataFrame with columns: symbol, direction, score.
                 Expected to contain latest signal per symbol.
        held_symbols: Set of symbols currently held in the portfolio.
        entry_n: Max rank for new entries (strict threshold).
        hold_n: Max rank for keeping held positions (lenient threshold).

    Returns:
        Tuple of (adjusted_signals, meta_dict).
        Signals for symbols outside the hysteresis band get direction='FLAT'.
    """
    meta: Dict[str, Any] = {
        "entry_n": entry_n,
        "hold_n": hold_n,
        "kept_by_hysteresis": 0,
        "blocked_entry": 0,
    }

    if signals.empty or "direction" not in signals.columns:
        return signals, meta

    out = signals.copy()

    long_mask = out["direction"] == "LONG"
    long_df = out[long_mask].copy()

    if long_df.empty:
        return out, meta

    if "score" in long_df.columns:
        long_df = long_df.sort_values("score", ascending=False).reset_index(drop=True)
    long_df["_rank"] = range(1, len(long_df) + 1)

    rank_map = dict(zip(long_df["symbol"], long_df["_rank"]))

    symbols_to_flat = []
    for sym, rank in rank_map.items():
        is_held = sym in held_symbols

        if is_held:
            if rank > hold_n:
                symbols_to_flat.append(sym)
        else:
            if rank > entry_n:
                symbols_to_flat.append(sym)
                meta["blocked_entry"] += 1

    for sym, rank in rank_map.items():
        is_held = sym in held_symbols
        if is_held and rank > entry_n and rank <= hold_n:
            meta["kept_by_hysteresis"] += 1

    if symbols_to_flat:
        mask = out["symbol"].isin(symbols_to_flat)
        out.loc[mask, "direction"] = "FLAT"
        if "score" in out.columns:
            out.loc[mask, "score"] = 0.0

    return out, meta
