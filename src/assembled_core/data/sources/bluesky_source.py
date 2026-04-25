"""Bluesky Jetstream — Twitter/X replacement for social sentiment.

From 10_FREE_DATEN.md §10.17.
AT-Protocol public firehose, no key needed, ~30% of Twitter volume (growing).

Twitter/X removed from primary pipeline: pay-per-use since 2026-02-06.
Nitter: dead. twitterapi.io / twscrape: ToS violation.

Install: pip install atproto

Cashtag pattern: $AAPL, $MSFT etc. — extract and count velocity.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_JETSTREAM_HOST = "jetstream2.us-east.bsky.network"


def _try_atproto():
    try:
        from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message
        from atproto.exceptions import FirehoseError
        return FirehoseSubscribeReposClient, parse_subscribe_repos_message, FirehoseError
    except ImportError:
        logger.warning("atproto not installed — pip install atproto")
        return None, None, None


async def stream_cashtag_mentions(
    duration_seconds: int = 60,
    max_posts: int = 5000,
) -> dict[str, int]:
    """Stream Bluesky firehose and count cashtag mentions.

    Args:
        duration_seconds: How long to stream (default 60s)
        max_posts: Maximum posts to process before stopping

    Returns:
        Dict mapping ticker → mention count over the streaming window.
    """
    FirehoseClient, parse_message, FirehoseError = _try_atproto()
    if FirehoseClient is None:
        return {}

    counts: dict[str, int] = defaultdict(int)
    processed = 0

    client = FirehoseClient()

    def on_message(message) -> None:
        nonlocal processed
        if processed >= max_posts:
            return
        try:
            commit = parse_message(message)
            if not hasattr(commit, "ops"):
                return
            for op in commit.ops:
                if op.action == "create" and op.path.startswith("app.bsky.feed.post"):
                    record = op.record
                    if hasattr(record, "text"):
                        for match in _CASHTAG_RE.finditer(record.text):
                            counts[match.group(1)] += 1
                        processed += 1
        except Exception:
            pass

    try:
        async def _run():
            await asyncio.wait_for(
                asyncio.to_thread(client.start, on_message),
                timeout=duration_seconds,
            )
        await _run()
    except asyncio.TimeoutError:
        pass
    except Exception as exc:
        logger.debug("Bluesky stream error: %s", exc)
    finally:
        try:
            client.stop()
        except Exception:
            pass

    return dict(counts)


def get_cashtag_velocity(
    duration_seconds: int = 60,
    min_mentions: int = 3,
) -> dict[str, int]:
    """Synchronous wrapper for cashtag mention velocity.

    Returns tickers with at least min_mentions in the streaming window.
    """
    try:
        counts = asyncio.run(stream_cashtag_mentions(duration_seconds=duration_seconds))
    except RuntimeError:
        # Already in an event loop — use thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, stream_cashtag_mentions(duration_seconds=duration_seconds))
            counts = fut.result(timeout=duration_seconds + 10)

    return {k: v for k, v in counts.items() if v >= min_mentions}


__all__ = [
    "stream_cashtag_mentions",
    "get_cashtag_velocity",
]
