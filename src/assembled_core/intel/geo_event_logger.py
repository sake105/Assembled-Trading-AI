"""EDCL Phase C — Geo-Event Logger.

Persists TriggerBasket-fired events to data/intel/geo_events_historical.parquet
so that compute_event_betas.py can train the event_beta FeatureStore view.

Each call to log_basket_event() appends one row per fired trigger type.
The parquet file uses append-via-read-modify-write (atomic rename) to avoid
partial writes. On first call the file is created.

Schema:
    event_date      datetime64[ns, UTC]  — time the basket was computed (as_of)
    trigger_type    str                  — TriggerType.name
    conviction      float                — basket conviction at fire time
    source_tier     int                  — 1=raw_news_events, 2=active_triggers
    geo_tags        str                  — comma-joined ISO codes
    n_events        int                  — n_events in basket
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/intel/geo_events_historical.parquet")


def log_basket_event(
    basket: Any,
    conviction: float,
    as_of: datetime | None = None,
    source_tier: int = 2,
    output_path: str | Path | None = None,
) -> bool:
    """Append fired trigger events from basket to the historical parquet log.

    Args:
        basket: TriggerBasket (must have .fired_triggers, .geo_tags, .n_events).
        conviction: EDCL conviction score at time of fire.
        as_of: Event timestamp. Defaults to UTC now.
        source_tier: 1 = raw NewsEvent objects, 2 = active_triggers fallback.
        output_path: Override default path.

    Returns:
        True if at least one row was written, False otherwise.
    """
    if basket is None or not getattr(basket, "is_active", lambda: False)():
        return False

    fired = getattr(basket, "fired_triggers", None) or []
    if not fired:
        return False

    try:
        import pandas as pd
        from src.assembled_core.utils.atomic_io import atomic_write_json  # noqa: F401
    except ImportError as e:
        log.debug("geo_event_logger: missing deps (%s) — skipping", e)
        return False

    ts = (as_of or datetime.now(timezone.utc)).replace(tzinfo=timezone.utc)
    geo_tags_str = ",".join(sorted(getattr(basket, "geo_tags", set()) or set()))
    n_events = int(getattr(basket, "n_events", len(fired)))

    rows = [
        {
            "event_date": ts,
            "trigger_type": ttype.name,
            "conviction": float(conviction),
            "source_tier": int(source_tier),
            "geo_tags": geo_tags_str,
            "n_events": n_events,
        }
        for ttype, _ in fired
    ]
    new_df = pd.DataFrame(rows)
    new_df["event_date"] = pd.to_datetime(new_df["event_date"], utc=True)

    path = Path(output_path) if output_path else _DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Deduplicate: same event_date + trigger_type
            combined = combined.drop_duplicates(
                subset=["event_date", "trigger_type"], keep="last"
            ).sort_values("event_date")
        else:
            combined = new_df

        tmp = path.with_suffix(".tmp.parquet")
        combined.to_parquet(tmp, index=False)
        tmp.replace(path)

        log.debug(
            "[GEO-LOG] appended %d trigger rows at %s → %s (%d total)",
            len(rows), ts.isoformat(), path, len(combined),
        )
        return True
    except Exception as exc:
        log.warning("geo_event_logger write failed (%s): %s", path, exc)
        return False


def read_geo_event_log(
    path: str | Path | None = None,
    min_conviction: float = 0.0,
) -> "pd.DataFrame":
    """Read the historical geo-event log. Returns empty DataFrame on any error."""
    try:
        import pandas as pd
        _path = Path(path) if path else _DEFAULT_PATH
        if not _path.exists():
            return pd.DataFrame(columns=["event_date", "trigger_type", "conviction",
                                         "source_tier", "geo_tags", "n_events"])
        df = pd.read_parquet(_path)
        if min_conviction > 0.0:
            df = df[df["conviction"] >= min_conviction]
        return df
    except Exception as exc:
        log.warning("geo_event_logger read failed: %s", exc)
        import pandas as pd
        return pd.DataFrame()


__all__ = ["log_basket_event", "read_geo_event_log"]
