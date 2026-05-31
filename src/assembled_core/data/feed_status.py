"""Feed fetch-outcome status stamping (audit DAT-005, E-025 family).

A feed fetch that returns an empty frame is, by itself, ambiguous: it can mean
*the upstream feed is down / errored* (an **outage**) or *the requested window
legitimately contains no rows* (an **empty window**). The audit (DAT-005, the
E-025 fail-open family) flags that this masking happens at the **return-type**
level — both cases return the same empty ``DataFrame`` — so no caller is able to
react differently. There is a WARN log in most paths, but the distinction is not
expressible on the value a caller receives.

This module adds a non-invasive, behaviour-preserving distinction: a fetch
function stamps its outcome onto the returned frame's :attr:`pandas.DataFrame.attrs`
under the ``feed_status`` key. The frame's *content* is unchanged (same rows,
dtypes, ``.empty``), so every existing caller that ignores ``attrs`` is wholly
unaffected. A caller that wants to tell an outage from an empty window reads the
stamp at the return boundary via :func:`get_feed_status` / :func:`is_feed_outage`.

Honest limit (same framing as the OPS-07 / R2-6 / R2-7 observability fixes):
``DataFrame.attrs`` is best-effort metadata that pandas drops on most operations
(concat / merge / copy). The stamp is therefore a **catchable signal at the
return boundary**, not a guarantee that survives downstream reshaping, and there
is no operational consumer wired yet — an ingestion-level reader that checks the
stamp before further processing is a separate follow-up. What this delivers today
is the distinction that previously did not exist *at all* on the returned value.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Outcome vocabulary. Kept deliberately tiny — three mutually-exclusive states.
FEED_OK = "ok"  # at least one real row was fetched
FEED_EMPTY = "empty"  # fetch succeeded but the window genuinely had no rows
FEED_ERROR = "error"  # fetch failed (key/import/client/network/total outage)

_FEED_STATUS_KEY = "feed_status"
_VALID_STATUSES = frozenset({FEED_OK, FEED_EMPTY, FEED_ERROR})


def stamp_feed_status(
    df: pd.DataFrame,
    source: str,
    status: str,
    *,
    reason: str | None = None,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Stamp a feed fetch outcome onto ``df.attrs['feed_status']`` and return df.

    The stamp is a dict ``{source, status, reason, n_rows}``. Only ``attrs`` is
    touched — row content, dtypes and ``df.empty`` are never altered, so this is
    behaviour-preserving for every caller that does not read the stamp.

    Never raises: a non-DataFrame ``df`` or an unknown ``status`` is logged at
    DEBUG and the object is returned unchanged. A ``FEED_ERROR`` stamp also logs
    at WARNING so a total outage that collapses to an empty frame is visible even
    without a stamp-aware consumer.
    """
    if not isinstance(df, pd.DataFrame):
        logger.debug(
            "[DAT-005] feed_status: %s is not a DataFrame — not stamped (%s)",
            source,
            status,
        )
        return df
    if status not in _VALID_STATUSES:
        logger.debug(
            "[DAT-005] feed_status: unknown status %r for %s — not stamped",
            status,
            source,
        )
        return df
    rows = int(n_rows) if n_rows is not None else int(len(df))
    df.attrs[_FEED_STATUS_KEY] = {
        "source": str(source),
        "status": status,
        "reason": reason,
        "n_rows": rows,
    }
    if status == FEED_ERROR:
        logger.warning(
            "[DAT-005] %s: fetch OUTAGE (reason=%s) — the empty result is an "
            "ERROR, not a legitimate empty window",
            source,
            reason,
        )
    return df


def get_feed_status(df: pd.DataFrame) -> dict[str, Any] | None:
    """Return the ``feed_status`` stamp on ``df``, or ``None`` if absent.

    Defensive: a non-DataFrame, an unstamped frame, or a non-dict stamp all
    return ``None`` rather than raising.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    val = df.attrs.get(_FEED_STATUS_KEY)
    return val if isinstance(val, dict) else None


def is_feed_outage(df: pd.DataFrame) -> bool:
    """True iff ``df`` carries a ``feed_status`` stamp with status ``error``.

    An unstamped or empty-window frame returns ``False`` — only an explicitly
    recorded outage is an outage.
    """
    stamp = get_feed_status(df)
    return bool(stamp is not None and stamp.get("status") == FEED_ERROR)


__all__ = [
    "FEED_OK",
    "FEED_EMPTY",
    "FEED_ERROR",
    "stamp_feed_status",
    "get_feed_status",
    "is_feed_outage",
]
