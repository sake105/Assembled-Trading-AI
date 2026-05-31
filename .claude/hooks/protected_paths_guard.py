#!/usr/bin/env python3
"""PreToolUse hook (Bash only): block destructive shell commands + writes into protected zones.

Envelope: **exit 2 + reason on stderr, NO JSON.**
(Claude Code ignores JSON stdout when the hook exits 2 — see hooks-guide.)
PreToolUse hooks fire BEFORE the permission-mode check, so this blocks even under
`defaultMode=bypassPermissions`.

Path protection for Edit/Write is handled DECLARATIVELY in `.claude/settings.json`
(`permissions.deny`). This hook only covers the **Bash** residual surface: destructive
commands and the most common shell writes into protected zones.

Residual gaps (documented, NOT covered):
  - Arbitrary subprocess writes: a `python`/`perl` script, here-doc generator, or any
    interpreter that writes into a protected path without a recognized
    redirect/cp/mv/tee token.
  - Command-substitution evasion: a destructive command nested inside `$(...)` or
    backticks of an otherwise-benign command may not be isolated by tokenization.
  - The separate **PowerShell** tool is a different `tool_name` and is NOT routed
    through this Bash hook.

Destructive patterns (path-independent), checked per sub-command:
  rm with recursive + force   (combined `-rf`/`-fr`, split `-r -f`, long `--recursive`/`--force`)
  git reset --hard
  git push --force / --force-with-lease / -f
  git clean -f / --force      (any combo, e.g. -fd, -fdx)
  sed -i / --in-place
  dd of=...
  find ... -delete  /  find ... -exec rm

Writes into protected zones (best-effort, per sub-command):
  redirect  >  >>  2>  &>   tee   cp DEST   mv DEST

Protected zones:
  src/assembled_core/{execution,risk,accounting,pipeline,paper}/
  .github/workflows/

One-shot authorization (consumed after first use, audit-logged):
  echo "Begründung" > .claude/.destructive_bash_authorized

Env overrides (testability):
  CLAUDE_GUARD_AUTH_FILE   auth marker path
  CLAUDE_GUARD_AUTH_LOG    audit log path
  CLAUDE_GUARD_REPO_ROOT   repo root override
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# Repo root + configurable paths
# ---------------------------------------------------------------------------

_ENV_ROOT = os.environ.get("CLAUDE_GUARD_REPO_ROOT")
REPO_ROOT = Path(_ENV_ROOT) if _ENV_ROOT else Path(__file__).resolve().parents[2]


def _auth_file() -> Path:
    override = os.environ.get("CLAUDE_GUARD_AUTH_FILE")
    return (
        Path(override)
        if override
        else REPO_ROOT / ".claude" / ".destructive_bash_authorized"
    )


def _auth_log() -> Path:
    override = os.environ.get("CLAUDE_GUARD_AUTH_LOG")
    return (
        Path(override)
        if override
        else REPO_ROOT / ".claude" / ".destructive_bash_auth_log.jsonl"
    )


# ---------------------------------------------------------------------------
# Protected zones
# ---------------------------------------------------------------------------

PROTECTED_ZONES: tuple[str, ...] = (
    "src/assembled_core/execution/",
    "src/assembled_core/risk/",
    "src/assembled_core/accounting/",
    "src/assembled_core/pipeline/",
    "src/assembled_core/paper/",
    ".github/workflows/",
)

# Sub-command separators: && || |& then single ; newline | &
_SEPARATORS = re.compile(r"&&|\|\||\|&|[;\n|&]")

# Separate redirect operator token (e.g. ">", ">>", "2>", "&>", ">|")
_REDIR_OP = re.compile(r"\d*&?>{1,2}\|?")
# Attached redirect token (e.g. ">file", "2>file", "&>file")
_REDIR_ATTACHED = re.compile(r"^\d*&?>{1,2}\|?(.+)$")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _norm_path(path: str) -> str:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    return str(PurePosixPath(p)) if p else p


def _targets_protected(path: str) -> bool:
    """True if `path` points into a protected zone (substring, post-normalization)."""
    p = _norm_path(path)
    if not p:
        return False
    hay = p + "/"
    return any(z in hay for z in PROTECTED_ZONES)


def _short_flag_chars(token: str) -> str:
    """For '-rf' -> 'rf', for '-i.bak' -> 'i.bak'; '' if not a short-flag group."""
    if len(token) >= 2 and token[0] == "-" and token[1] != "-":
        return token[1:]
    return ""


def _cmd_basename(token: str) -> str:
    return os.path.basename(token)


# ---------------------------------------------------------------------------
# Per-pattern detectors (operate on tokenized sub-command)
# ---------------------------------------------------------------------------

_PREFIX_WRAPPERS = ("sudo", "env", "command", "nice", "ionice", "doas")


def _strip_prefix(tokens: list[str]) -> list[str]:
    idx = 0
    while idx < len(tokens) and tokens[idx] in _PREFIX_WRAPPERS:
        idx += 1
    return tokens[idx:]


def _is_destructive_rm(tokens: list[str]) -> bool:
    toks = _strip_prefix(tokens)
    if not toks or _cmd_basename(toks[0]) != "rm":
        return False
    has_recursive = False
    has_force = False
    for tok in toks[1:]:
        if tok == "--":
            break
        if tok.startswith("--"):
            if tok == "--recursive":
                has_recursive = True
            elif tok == "--force":
                has_force = True
        else:
            chars = _short_flag_chars(tok)
            if "r" in chars or "R" in chars:
                has_recursive = True
            if "f" in chars:
                has_force = True
    return has_recursive and has_force


def _is_destructive_git(tokens: list[str]) -> str | None:
    toks = _strip_prefix(tokens)
    if not toks or _cmd_basename(toks[0]) != "git":
        return None
    rest = toks[1:]
    if "reset" in rest and "--hard" in rest:
        return "git reset --hard"
    if "push" in rest and any(t == "-f" or t.startswith("--force") for t in rest):
        return "git push --force"
    if "clean" in rest:
        for t in rest:
            if t == "--force" or "f" in _short_flag_chars(t):
                return "git clean -f"
    return None


def _is_sed_inplace(tokens: list[str]) -> bool:
    toks = _strip_prefix(tokens)
    if not toks or _cmd_basename(toks[0]) != "sed":
        return False
    for t in toks[1:]:
        if t.startswith("--in-place"):
            return True
        if "i" in _short_flag_chars(t):
            return True
    return False


def _is_dd_write(tokens: list[str]) -> bool:
    toks = _strip_prefix(tokens)
    if not toks or _cmd_basename(toks[0]) != "dd":
        return False
    return any(t.startswith("of=") for t in toks[1:])


def _is_find_delete(tokens: list[str]) -> bool:
    toks = _strip_prefix(tokens)
    if not toks or _cmd_basename(toks[0]) != "find":
        return False
    if "-delete" in toks:
        return True
    if "-exec" in toks and any(_cmd_basename(t) == "rm" for t in toks):
        return True
    return False


def _writes_protected(tokens: list[str]) -> bool:
    toks = _strip_prefix(tokens)
    if not toks:
        return False
    # Redirect targets (separate operator or attached form)
    for i, tok in enumerate(toks):
        if _REDIR_OP.fullmatch(tok):
            if i + 1 < len(toks) and _targets_protected(toks[i + 1]):
                return True
            continue
        m = _REDIR_ATTACHED.match(tok)
        if m and _targets_protected(m.group(1)):
            return True
    # tee FILE...
    for i, t in enumerate(toks):
        if _cmd_basename(t) == "tee":
            for arg in toks[i + 1 :]:
                if arg.startswith("-"):
                    continue
                if _targets_protected(arg):
                    return True
    # cp/mv DEST (last non-flag argument)
    if _cmd_basename(toks[0]) in ("cp", "mv"):
        non_flags = [t for t in toks[1:] if not t.startswith("-")]
        if non_flags and _targets_protected(non_flags[-1]):
            return True
    return False


# ---------------------------------------------------------------------------
# Sub-command + command level checks (pure, no I/O — unit-testable)
# ---------------------------------------------------------------------------


def _check_subcommand(sub: str) -> str | None:
    try:
        tokens = shlex.split(sub, posix=True)
    except ValueError:
        try:
            tokens = shlex.split(sub, posix=False)
        except ValueError:
            # Genuinely unparseable (e.g. unbalanced quotes) → fail-closed.
            return "unparseable shell sub-command (fail-closed)"
    if not tokens:
        return None
    if _is_destructive_rm(tokens):
        return "rm with recursive + force"
    git_label = _is_destructive_git(tokens)
    if git_label:
        return git_label
    if _is_sed_inplace(tokens):
        return "sed -i (in-place edit)"
    if _is_dd_write(tokens):
        return "dd of= (raw device/file write)"
    if _is_find_delete(tokens):
        return "find -delete / -exec rm"
    if _writes_protected(tokens):
        return "shell write into a protected zone"
    return None


def check_command(command: str) -> str | None:
    """Return a block reason for `command`, or None if allowed. Pure (no I/O)."""
    if not command or not command.strip():
        return None
    for sub in _SEPARATORS.split(command):
        sub = sub.strip()
        if not sub:
            continue
        reason = _check_subcommand(sub)
        if reason:
            return reason
    return None


# ---------------------------------------------------------------------------
# One-shot authorization
# ---------------------------------------------------------------------------


def _check_consume_auth() -> bool:
    """Consume the one-shot auth marker. Returns True if a valid marker was present."""
    f = _auth_file()
    if not f.exists():
        return False
    try:
        reason = f.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not reason:
        return False
    try:
        log = _auth_log()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"ts": datetime.now(timezone.utc).isoformat(), "reason": reason}
                )
                + "\n"
            )
    except OSError:
        pass
    try:
        f.unlink()
    except OSError:
        pass
    return True


# ---------------------------------------------------------------------------
# I/O wrapper
# ---------------------------------------------------------------------------

_AUTH_HINT = (
    "\n\nZum Entsperren (one-shot, wird nach Nutzung konsumiert):\n"
    '  echo "Begründung für diesen Befehl" > .claude/.destructive_bash_authorized\n'
    "Nur bei explizitem Auftrag und bewusster Entscheidung setzen."
)


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        # Unparseable hook envelope → cannot determine tool/command.
        # Edit/Write remain covered declaratively; allow to avoid bricking reads.
        return 0

    if not isinstance(event, dict) or event.get("tool_name", "") != "Bash":
        return 0

    tool_input = event.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}
    command = tool_input.get("command", "")
    if not isinstance(command, str):
        command = ""

    try:
        reason = check_command(command)
    except Exception as exc:  # noqa: BLE001 — fail-closed: never silently allow on bug
        sys.stderr.write(
            "DESTRUCTIVE-BASH-GUARD interner Fehler — fail-closed, Befehl blockiert: "
            f"{exc}{_AUTH_HINT}\n"
        )
        return 2

    if reason is None:
        return 0

    if _check_consume_auth():
        return 0

    sys.stderr.write(
        f"DESTRUCTIVE BASH BLOCKED — erkanntes Muster: {reason}\n\n"
        "Dieser PreToolUse-Hook blockt destruktive Shell-Befehle und Schreibzugriffe "
        "in Schutzzonen — auch unter bypassPermissions, da PreToolUse vor dem "
        "Permission-Check feuert." + _AUTH_HINT + "\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
