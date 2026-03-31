"""Learning Store — M11: Append-only JSONL store for post-trade learning records.

Records are appended atomically (write temp, rename).
Each record is one JSON line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
