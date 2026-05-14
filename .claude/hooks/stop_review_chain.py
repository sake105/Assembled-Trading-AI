#!/usr/bin/env python3
"""Stop hook: enforce the review chain after coding steps in protected paths.

Flow per spec §4.2.1:
1. Read Stop event JSON from stdin (session_id, transcript_path, stop_hook_active).
2. If stop_hook_active is True, exit (never loop).
3. Check explicit skip marker file (one-shot, auto-consumed, requires reason).
4. Parse transcript trailing turn for edited paths.
5. Classify diff: skip / docs-only / test-only / full.
6. If kind is 'skip' or 'docs-only': allow stop.
7. Check review-marker for current turn_id.
8. If marker exists: allow stop (chain already ran for this turn).
9. Otherwise: output {"decision":"block","reason": REVIEW_INSTRUCTIONS}.

REVIEW_INSTRUCTIONS embed:
- list of Stage-1 specialists to dispatch (based on edited paths)
- reminder to dispatch senior-code-reviewer (Stage 2) and task-completion-auditor (Stage 3)
- reminder to write the review-marker via the `write_review_marker` utility before next stop

Env overrides for testability:
- CLAUDE_HOOKS_STATE_DIR: where to read/write review markers (default: .claude/.review_markers)
- CLAUDE_HOOKS_SKIP_FILE: path to the one-shot skip marker file (default: .claude/.review_skip)
- CLAUDE_HOOKS_SKIP_LOG: path to skip-audit JSONL (default: .claude/.review_skip_log.jsonl)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_utils.path_classifier import specialists_for_paths  # noqa: E402
from hook_utils.transcript_parser import edited_paths_in_last_turn  # noqa: E402
from hook_utils.diff_classifier import classify_diff  # noqa: E402
from hook_utils.review_marker import (  # noqa: E402
    turn_id_from_transcript,
    has_review_marker,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _state_dir() -> Path:
    override = os.environ.get("CLAUDE_HOOKS_STATE_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / ".claude" / ".review_markers"


def _skip_file() -> Path:
    override = os.environ.get("CLAUDE_HOOKS_SKIP_FILE")
    if override:
        return Path(override)
    return REPO_ROOT / ".claude" / ".review_skip"


def _skip_log() -> Path:
    override = os.environ.get("CLAUDE_HOOKS_SKIP_LOG")
    if override:
        return Path(override)
    return REPO_ROOT / ".claude" / ".review_skip_log.jsonl"


def _check_and_consume_skip() -> bool:
    """Check for one-shot skip marker. Returns True if skip is honored.

    Skip is honored only when:
    - The skip file exists
    - It contains a non-empty (post-strip) reason

    When honored, the skip is logged to the audit JSONL and the file is deleted.
    """
    sf = _skip_file()
    if not sf.exists():
        return False
    try:
        raw = sf.read_text(encoding="utf-8")
    except OSError:
        return False
    reason = raw.strip()
    if not reason:
        # Empty / whitespace-only → don't honor (force conscious skip)
        return False

    # Append audit entry (best-effort, never raises)
    try:
        log_path = _skip_log()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass

    # Consume the marker (one-shot)
    try:
        sf.unlink()
    except OSError:
        pass

    return True


def _allow_stop() -> int:
    # Empty stdout (or explicit approve) → Claude Code proceeds with stop
    return 0


def _build_review_instructions(
    edited_paths: list[str], specialists: set[str], turn_id: str
) -> str:
    spec_block = "\n".join(f"  - `{s}`" for s in sorted(specialists))
    paths_block = "\n".join(f"  - {p}" for p in edited_paths[:20])  # cap at 20
    if len(edited_paths) > 20:
        paths_block += f"\n  - ... ({len(edited_paths) - 20} more)"

    # Fix F-senior-4: embed absolute marker path so the snippet works regardless
    # of the agent's cwd (worktrees, subprocess chdir, etc.).
    marker_dir_abs = str((REPO_ROOT / ".claude" / ".review_markers").as_posix())

    return f"""REVIEW-CHAIN-REQUIRED — Step nicht abgeschlossen, bis Review-Kette gelaufen ist.

Du hast Code in geschützten Pfaden geändert:
{paths_block}

Pflicht-Ablauf jetzt:

**Stage 1 (parallel):** Dispatche diese Spezialist-Subagents über das Agent-Tool:
{spec_block}

Jeder bekommt: den Diff der oben gelisteten Pfade + die ursprüngliche Task-Beschreibung.
Verlange strukturierte Findings im YAML-Schema aus
`docs/superpowers/specs/2026-05-14-review-chain-design.md` §5.

**Stage 2:** Dispatche `senior-code-reviewer` mit:
- Diff
- ursprüngliche Task-Beschreibung
- konsolidierte Stage-1-Findings
- Top-10 Anti-Patterns sind bereits im Kontext (SessionStart-Hook)

**Stage 3:** Dispatche `task-completion-auditor` mit:
- ursprüngliche Task-Beschreibung
- Diff
- alle bisherigen Findings (Stage 1 + 2)
Erwarte Verdict: PASS / CONDITIONAL / FAIL.

**Konsolidierung:** Schreibe einen Findings-Block (Spec §5 Format) mit:
- BLOCKER / MAJOR / MINOR / INFO Listen
- Follow-ups (Adjacent-Themen)
- Errors-Log-Vorschläge (falls vorhanden)
- Verdict

**Marker schreiben (Pflicht, damit nächstes Stop nicht erneut blockt):**

```python
import sys
sys.path.insert(0, r"{(REPO_ROOT / ".claude" / "hooks").as_posix()}")
from hook_utils.review_marker import write_review_marker
from pathlib import Path
write_review_marker("{turn_id}", Path(r"{marker_dir_abs}"))
```

**Dann:**
- BLOCKER → adressieren bevor du an User zurückmeldest.
- MAJOR → adressieren ODER explizit dokumentieren/akzeptieren.
- MINOR/INFO → optional dokumentieren.
- Errors-Log-Vorschläge → in `docs/CLAUDE_CODING_ERRORS.md` appenden (append-only).
- Erst dann an User zurückmelden.

Vermeide: ungeprüfte Behauptungen „passt schon", Review-Theater, Scope-Erweiterung.
"""


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        # Malformed input → fail-open, allow stop
        return _allow_stop()

    # Avoid infinite loop: if Claude Code already invoked us in a Stop-hook loop, allow.
    if event.get("stop_hook_active") is True:
        return _allow_stop()

    transcript_path_str = event.get("transcript_path", "")
    if not transcript_path_str:
        return _allow_stop()

    transcript_path = Path(transcript_path_str)
    edited = edited_paths_in_last_turn(transcript_path, repo_root=REPO_ROOT)
    classification = classify_diff(edited)

    if not classification["run_full_chain"]:
        return _allow_stop()

    # Fix F-senior-1: only consume the skip marker when the chain WOULD have
    # blocked. Order matters — classification must run first so we don't waste
    # the user's one-shot on stops that would have allowed themselves anyway.
    if _check_and_consume_skip():
        return _allow_stop()

    # Check marker
    turn_id = turn_id_from_transcript(transcript_path)
    state_dir = _state_dir()
    if has_review_marker(turn_id, state_dir):
        return _allow_stop()

    # Block and instruct
    specialists = specialists_for_paths(edited)
    reason = _build_review_instructions(edited, specialists, turn_id)
    payload = {"decision": "block", "reason": reason}
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
