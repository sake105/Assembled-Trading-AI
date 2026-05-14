"""Per-turn marker to prevent re-triggering the review chain.

The Stop hook blocks once. After the main agent dispatches the review chain
and writes the marker, the next Stop event sees the marker and lets the stop
through.

Turn-ID derivation: SHA1 of the transcript byte length + last assistant UUID.
This changes whenever the transcript grows, which it does on every new turn.
"""

from __future__ import annotations

import hashlib  # nosec B324 - used for stable short hashes, not cryptographic
import json
from pathlib import Path

MARKER_DIR_NAME = ".review_markers"


def turn_id_from_transcript(transcript_path: Path) -> str:
    """Derive a stable per-turn ID from the transcript file."""
    if not transcript_path.exists():
        return "no-transcript"
    raw = transcript_path.read_bytes()
    size = len(raw)
    last_uuid = ""
    for line in reversed(raw.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "assistant":
                last_uuid = obj.get("uuid", "")
                break
        except json.JSONDecodeError:
            continue
    payload = f"{size}:{last_uuid}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()  # nosec B324
    return digest[:16]


def _marker_path(turn_id: str, state_dir: Path) -> Path:
    return state_dir / f"{turn_id}.done"


def has_review_marker(turn_id: str, state_dir: Path) -> bool:
    return _marker_path(turn_id, state_dir).exists()


def write_review_marker(turn_id: str, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _marker_path(turn_id, state_dir).write_text("done", encoding="utf-8")
