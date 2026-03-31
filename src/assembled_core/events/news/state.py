from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .emit import emit_json_artifact
from .normalize import now_utc_iso


def load_fetch_state(path: Path) -> Dict[str, Any]:
    """Load persistent fetch_state for NEWS v1.

    Returns a dict with at least keys: rss, gdelt.
    Missing file or parse errors result in an empty-but-structured state.
    """
    state: Dict[str, Any] = {
        "schema_version": "news.fetch_state.v1",
        "updated_utc": now_utc_iso(),
        "rss": {},
        "gdelt": {},
    }
    if not path.exists():
        return state

    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return state

    if isinstance(data, dict):
        state.update(
            {
                "rss": data.get("rss") or {},
                "gdelt": data.get("gdelt") or {},
            }
        )
    return state


def save_fetch_state(state: Dict[str, Any], path: Path) -> None:
    """Persist fetch_state atomically via emit_json_artifact."""
    state = dict(state)
    state["schema_version"] = "news.fetch_state.v1"
    state["updated_utc"] = now_utc_iso()
    emit_json_artifact(state, path)
