#!/usr/bin/env python3
"""Stop hook: enforce the review chain after coding steps in protected paths.

Flow per spec §4.2.1:
1. Read Stop event JSON from stdin (session_id, transcript_path, stop_hook_active).
2. If stop_hook_active is True, exit (never loop).
3. Parse transcript trailing turn for edited paths.
4. Classify diff: skip / docs-only / test-only / full.
5. If kind is 'skip' or 'docs-only': allow stop.
6. Check review-marker for current turn_id.
7. If marker exists: allow stop (chain already ran for this turn).
8. Otherwise: output {"decision":"block","reason": REVIEW_INSTRUCTIONS}.

REVIEW_INSTRUCTIONS embed:
- list of Stage-1 specialists to dispatch (based on edited paths)
- reminder to dispatch senior-code-reviewer (Stage 2) and task-completion-auditor (Stage 3)
- reminder to write the review-marker via the `write_review_marker` utility before next stop

Env overrides for testability:
- CLAUDE_HOOKS_STATE_DIR: where to read/write review markers (default: .claude/.review_markers)
"""

from __future__ import annotations

import json
import os
import sys
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
sys.path.insert(0, ".claude/hooks")
from hook_utils.review_marker import write_review_marker
from pathlib import Path
write_review_marker("{turn_id}", Path(".claude/.review_markers"))
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
