"""Universe management (symbol lists with date ranges)."""

from __future__ import annotations

import pandas as pd


def get_universe_members(
    universe_name: str = "default",
    as_of: pd.Timestamp | str | None = None,
) -> list[str]:
    """Return the list of symbols in the universe at *as_of*.

    Falls back to watchlist.txt if no stored universe history exists.
    """
    from pathlib import Path

    wl = Path("watchlist.txt")
    if wl.exists():
        return [
            line.strip()
            for line in wl.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []


def store_universe_history(
    members: list[str],
    universe_name: str = "default",
    valid_from: pd.Timestamp | None = None,
) -> None:
    """Persist a universe snapshot (no-op stub)."""
    pass
