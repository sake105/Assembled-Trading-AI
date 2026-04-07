"""Experience Log — Append-only JSONL audit trail for trading cycles.

Records one entry per cycle with execution stats, equity, risk state,
and reconciliation status. Used for long-term analysis and fine-tuning.

Location: output/experience/experience_log.jsonl
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("output") / "experience" / "experience_log.jsonl"


def _log_path(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else _DEFAULT_LOG_PATH


def append_experience(
    entry: dict[str, Any],
    *,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append a single experience entry to the JSONL log.

    Auto-adds timestamp_utc if missing.

    Args:
        entry: Dict with cycle data (execution_mode, equity, etc.).
        log_path: Override path. Defaults to output/experience/experience_log.jsonl.

    Returns:
        The entry as written (with timestamp added if missing).
    """
    p = _log_path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if "timestamp_utc" not in entry:
        entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    line = json.dumps(entry, ensure_ascii=True, default=str) + "\n"

    # File locking for concurrent-write safety (consistent with paper_ledger)
    lock = None
    lock_path = p.with_suffix(p.suffix + ".lock")
    try:
        from filelock import FileLock

        lock = FileLock(str(lock_path), timeout=5)
    except ImportError:
        pass

    def _do_write() -> None:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line)

    if lock is not None:
        with lock:
            _do_write()
    else:
        _do_write()

    logger.info("[experience_log] appended entry for %s", entry.get("cycle_date", "?"))
    return entry


def load_experience(
    *,
    days: int | None = None,
    log_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load experience log as DataFrame.

    Args:
        days: If set, only return entries from the last N days.
        log_path: Override path.

    Returns:
        DataFrame with one row per cycle. Empty DataFrame if no log exists.
    """
    p = _log_path(log_path)
    if not p.exists():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("[experience_log] skipping malformed line %d", i)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if days is not None and "timestamp_utc" in df.columns:
        try:
            df["_ts"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=days)
            df = df[df["_ts"] >= cutoff].drop(columns=["_ts"])
        except Exception:
            pass

    return df


def compute_experience_summary(
    *,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute summary statistics over the full experience log.

    Returns dict with: total_cycles, date_range, avg_equity,
    win_rate (cycles with positive return), max_drawdown, etc.
    """
    df = load_experience(log_path=log_path)
    if df.empty:
        return {"total_cycles": 0}

    summary: dict[str, Any] = {
        "total_cycles": len(df),
    }

    if "cycle_date" in df.columns:
        summary["first_date"] = str(df["cycle_date"].iloc[0])
        summary["last_date"] = str(df["cycle_date"].iloc[-1])

    if "broker_equity" in df.columns:
        eq = pd.to_numeric(df["broker_equity"], errors="coerce").dropna()
        if len(eq) > 0:
            summary["latest_equity"] = float(eq.iloc[-1])
            summary["avg_equity"] = float(eq.mean())
            summary["min_equity"] = float(eq.min())
            summary["max_equity"] = float(eq.max())

            # Simple return series
            if len(eq) > 1:
                returns = eq.pct_change().dropna()
                summary["avg_daily_return_pct"] = float(returns.mean() * 100)
                summary["win_rate_pct"] = float((returns > 0).mean() * 100)
                if returns.std() > 0:
                    summary["sharpe_approx"] = float(
                        returns.mean() / returns.std() * (252**0.5)
                    )

                # Max drawdown
                cummax = eq.cummax()
                drawdown = (eq - cummax) / cummax
                summary["max_drawdown_pct"] = float(drawdown.min() * 100)

    if "exit_code" in df.columns:
        summary["success_rate_pct"] = float(
            (df["exit_code"] == 0).mean() * 100
        )

    if "execution_mode" in df.columns:
        summary["mode_counts"] = df["execution_mode"].value_counts().to_dict()

    return summary


__all__ = [
    "append_experience",
    "load_experience",
    "compute_experience_summary",
]
