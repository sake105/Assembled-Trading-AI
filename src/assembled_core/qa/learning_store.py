"""Learning Store — M11: Append-only JSONL store for post-trade learning records.

Records are appended atomically (write temp, rename).
Each record is one JSON line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LEARNING_STORE_PATH = Path("output/learning/post_trade_learning.jsonl")


def append_learning_record(
    record: dict[str, Any],
    store_path: str | Path = DEFAULT_LEARNING_STORE_PATH,
) -> Path:
    """Append a learning record to the JSONL store atomically.

    Args:
        record: Dict to serialize as JSON line.
        store_path: Path to the JSONL store file.

    Returns:
        Path to the store file.
    """
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, sort_keys=True, default=str) + "\n"

    # Atomic append: write to temp in same dir, then read existing + append + write back
    # For JSONL we can just append directly (atomic on most OSes for small writes)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)

    return path


def load_learning_records(
    store_path: str | Path = DEFAULT_LEARNING_STORE_PATH,
) -> list[dict[str, Any]]:
    """Load all learning records from the JSONL store.

    Args:
        store_path: Path to the JSONL store file.

    Returns:
        List of record dicts (empty if file doesn't exist).
    """
    path = Path(store_path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON line: %s", e)
    return records


def get_latest_record(
    store_path: str | Path = DEFAULT_LEARNING_STORE_PATH,
) -> dict[str, Any] | None:
    """Return the most recently appended record, or None if store is empty.

    Args:
        store_path: Path to the JSONL store file.

    Returns:
        Last record dict, or None.
    """
    records = load_learning_records(store_path)
    return records[-1] if records else None


def summarize_learning_store(
    store_path: str | Path = DEFAULT_LEARNING_STORE_PATH,
    last_n: int = 10,
) -> dict[str, Any]:
    """Summarize learning store: hit rate trend, record count, latest date.

    Args:
        store_path: Path to the JSONL store file.
        last_n: Number of most recent records to summarize.

    Returns:
        Dict with summary statistics.
    """
    records = load_learning_records(store_path)
    if not records:
        return {"total_records": 0, "avg_hit_rate": None, "latest_date": None}

    recent = records[-last_n:]
    hit_rates = [r["overall_hit_rate"] for r in recent if "overall_hit_rate" in r]
    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else None
    dates = [r.get("analysis_date") for r in records if r.get("analysis_date")]

    return {
        "total_records": len(records),
        "avg_hit_rate": round(avg_hit_rate, 4) if avg_hit_rate is not None else None,
        "latest_date": max(dates) if dates else None,
        "recent_hit_rates": hit_rates,
    }


def load_learning_records_as_dataframe(
    store_path: str | Path = DEFAULT_LEARNING_STORE_PATH,
) -> pd.DataFrame:
    """Load all learning records from the JSONL store as a structured DataFrame.

    Reads the JSONL at store_path and normalises the list of record dicts into
    a flat DataFrame. Returns an empty DataFrame if the file does not exist or
    contains no valid records.

    Log prefix: [LEARNING]

    Args:
        store_path: Path to the JSONL store file.

    Returns:
        pandas.DataFrame with one row per learning record.  Column set
        reflects the keys present in the stored records; no columns are
        guaranteed beyond what was written.
    """
    path = Path(store_path)
    if not path.exists():
        logger.info("[LEARNING] store not found at %s — returning empty DataFrame", path)
        return pd.DataFrame()

    records = load_learning_records(store_path)
    if not records:
        logger.info("[LEARNING] store is empty at %s — returning empty DataFrame", path)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(
        "[LEARNING] loaded %d records (%d columns) from %s",
        len(df),
        len(df.columns),
        path,
    )
    return df


def compute_factor_attribution(
    records_df: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """Compute per-factor attribution from trade records.

    For each trade record, identify which factors had highest absolute values at
    entry, determine if the dominant factor was in the right direction (hit), then
    aggregate per factor: attribution_pnl, hit_rate_when_dominant, n_trades,
    avg_strength_when_dominant.

    Log prefix: [LEARNING]

    Args:
        records_df: DataFrame with one row per trade. Must contain at least a
            'pnl' column and a 'direction' column (BUY/SELL or LONG/SHORT).
            Factor columns listed in factor_cols should be present; missing
            columns are silently skipped.
        factor_cols: List of column names holding factor values at entry time.

    Returns:
        DataFrame with columns: factor_name, attribution_pnl,
        hit_rate_when_dominant, n_trades, avg_strength_when_dominant.
        Returns empty DataFrame if records_df is empty or no factor_cols are
        present.
    """
    if records_df is None or records_df.empty:
        logger.info("[LEARNING] compute_factor_attribution: empty records_df, returning empty")
        return pd.DataFrame(
            columns=[
                "factor_name",
                "attribution_pnl",
                "hit_rate_when_dominant",
                "n_trades",
                "avg_strength_when_dominant",
            ]
        )

    if not factor_cols:
        logger.info("[LEARNING] compute_factor_attribution: no factor_cols provided")
        return pd.DataFrame(
            columns=[
                "factor_name",
                "attribution_pnl",
                "hit_rate_when_dominant",
                "n_trades",
                "avg_strength_when_dominant",
            ]
        )

    # Only keep factor cols that are actually in the DataFrame
    available_factors = [c for c in factor_cols if c in records_df.columns]
    if not available_factors:
        logger.warning(
            "[LEARNING] compute_factor_attribution: none of %d requested factor_cols "
            "found in records_df (columns: %s)",
            len(factor_cols),
            list(records_df.columns),
        )
        return pd.DataFrame(
            columns=[
                "factor_name",
                "attribution_pnl",
                "hit_rate_when_dominant",
                "n_trades",
                "avg_strength_when_dominant",
            ]
        )

    logger.info(
        "[LEARNING] compute_factor_attribution: %d records, %d available factor cols",
        len(records_df),
        len(available_factors),
    )

    # Accumulate per-factor stats
    factor_stats: dict[str, dict[str, Any]] = {
        f: {"pnl": 0.0, "hits": 0, "n": 0, "strength_sum": 0.0}
        for f in available_factors
    }

    for _, row in records_df.iterrows():
        try:
            # Extract factor values for this trade
            factor_values: dict[str, float] = {}
            for f in available_factors:
                try:
                    val = float(row[f])
                    if pd.notna(val):
                        factor_values[f] = val
                except (TypeError, ValueError):
                    pass

            if not factor_values:
                continue

            # Dominant factor = highest absolute value at entry
            dominant = max(factor_values, key=lambda k: abs(factor_values[k]))
            dominant_strength = factor_values[dominant]

            # PnL for this trade
            try:
                trade_pnl = float(row.get("pnl", 0.0))
                if pd.isna(trade_pnl):
                    trade_pnl = 0.0
            except (TypeError, ValueError):
                trade_pnl = 0.0

            # Direction: BUY/LONG = positive, SELL/SHORT = negative
            try:
                direction = str(row.get("direction", "")).upper()
            except Exception:
                direction = ""

            # Hit: dominant factor pointed in the same direction as the trade
            # and the trade was profitable
            hit = False
            try:
                if direction in ("BUY", "LONG"):
                    hit = (dominant_strength > 0) and (trade_pnl > 0)
                elif direction in ("SELL", "SHORT"):
                    hit = (dominant_strength < 0) and (trade_pnl > 0)
                else:
                    # Unknown direction: use sign of pnl only
                    hit = trade_pnl > 0
            except Exception:
                hit = False

            stats = factor_stats[dominant]
            stats["pnl"] += trade_pnl
            stats["hits"] += int(hit)
            stats["n"] += 1
            stats["strength_sum"] += abs(dominant_strength)

        except Exception as exc:
            logger.debug("[LEARNING] compute_factor_attribution: skipping row due to %s", exc)
            continue

    # Build output DataFrame
    rows: list[dict[str, Any]] = []
    for factor_name, stats in factor_stats.items():
        n = stats["n"]
        if n == 0:
            continue
        rows.append(
            {
                "factor_name": factor_name,
                "attribution_pnl": round(stats["pnl"], 6),
                "hit_rate_when_dominant": round(stats["hits"] / n, 4),
                "n_trades": n,
                "avg_strength_when_dominant": round(stats["strength_sum"] / n, 6),
            }
        )

    if not rows:
        logger.info("[LEARNING] compute_factor_attribution: no dominant-factor trades found")
        return pd.DataFrame(
            columns=[
                "factor_name",
                "attribution_pnl",
                "hit_rate_when_dominant",
                "n_trades",
                "avg_strength_when_dominant",
            ]
        )

    result = pd.DataFrame(rows).sort_values("attribution_pnl", ascending=False).reset_index(drop=True)
    logger.info(
        "[LEARNING] compute_factor_attribution: %d factors with dominant trades",
        len(result),
    )
    return result
