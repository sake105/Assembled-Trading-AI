"""Unit tests for .claude/hooks/protected_paths_guard.py.

The hook is Bash-only: it blocks destructive shell commands and shell writes into
protected zones with envelope **exit 2 + stderr, NO JSON stdout** (Edit/Write path
protection lives declaratively in .claude/settings.json, not here).

Tests cover:
  - check_command()  pure detection (block vs allow)
  - main()           envelope via subprocess (exit code + stream discipline)
  - one-shot auth override consumption + audit log
All tests run without performing any actual dangerous operation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_DIR = REPO_ROOT / ".claude" / "hooks"
HOOK_PATH = HOOK_DIR / "protected_paths_guard.py"
sys.path.insert(0, str(HOOK_DIR))

from protected_paths_guard import (  # noqa: E402
    PROTECTED_ZONES,
    check_command,
)


# ---------------------------------------------------------------------------
# check_command — must BLOCK
# ---------------------------------------------------------------------------

BLOCK_CASES = [
    # rm recursive + force, all flag shapes
    "rm -rf build",
    "rm -fr build",
    "rm -Rf build",
    "rm -r -f build",
    "rm -f -r build",
    "rm -r /tmp/x -f",
    "rm --recursive --force build",
    "rm --force --recursive build",
    "sudo rm -rf /tmp/x",
    "env rm -rf /tmp/x",
    # git
    "git reset --hard",
    "git reset --hard HEAD~1",
    "git push --force origin main",
    "git push -f",
    "git push --force-with-lease",
    "git clean -fdx",
    "git clean -fd",
    "git clean -f",
    "git clean --force",
    # sed in-place
    "sed -i s/a/b/ file.txt",
    "sed -i.bak s/a/b/ file.txt",
    "sed --in-place s/a/b/ f",
    # dd / find
    "dd if=/dev/zero of=/dev/sda",
    "find . -name '*.tmp' -delete",
    "find . -name '*.pyc' -exec rm {} ;",
    # writes into protected zones
    "echo hi > src/assembled_core/risk/x.py",
    "echo hi >> src/assembled_core/pipeline/y.py",
    "echo hi 2> src/assembled_core/execution/log.txt",
    "cp a.py src/assembled_core/execution/b.py",
    "mv a.py src/assembled_core/accounting/b.py",
    "tee src/assembled_core/paper/z.py",
    "echo x > ./.github/workflows/ci.yml",
    # compound: destructive in any part
    "ls && rm -rf build",
    "cat foo ; git reset --hard",
    "true || rm -rf build",
    # unparseable → fail-closed
    'echo "unbalanced',
]


@pytest.mark.fast
@pytest.mark.parametrize("cmd", BLOCK_CASES)
def test_check_command_blocks(cmd):
    assert check_command(cmd) is not None, f"expected block: {cmd!r}"


# ---------------------------------------------------------------------------
# check_command — must ALLOW
# ---------------------------------------------------------------------------

ALLOW_CASES = [
    "ls -la",
    "git status",
    "git push origin main",
    "git clean -n",  # dry-run, no -f
    "rm build.txt",  # no recursive
    "rm -f build.txt",  # force, not recursive
    "rm -r build",  # recursive, not force
    "sed s/a/b/ file.txt",  # no -i
    "sed -n '1,5p' file.txt",  # -n, not -i
    "echo hi > output/x.txt",  # not protected
    "cp a.py src/assembled_core/features/b.py",  # not protected zone
    "grep -rf pattern .",  # grep, not rm
    "python scripts/run_daily.py",
    "git commit -m 'msg'",
    "",  # empty → allow
    "   ",  # whitespace → allow
]


@pytest.mark.fast
@pytest.mark.parametrize("cmd", ALLOW_CASES)
def test_check_command_allows(cmd):
    assert check_command(cmd) is None, f"expected allow: {cmd!r}"


@pytest.mark.fast
def test_protected_zones_present():
    assert "src/assembled_core/risk/" in PROTECTED_ZONES
    assert ".github/workflows/" in PROTECTED_ZONES


# ---------------------------------------------------------------------------
# main() envelope — exit 2 + stderr, NO JSON on stdout
# ---------------------------------------------------------------------------


def _run_hook(event: dict, env_extra: dict | None = None):
    import os

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.mark.fast
def test_envelope_destructive_exit2_stderr_no_stdout():
    rc, out, err = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf build"}}
    )
    assert rc == 2
    assert out.strip() == ""  # NO JSON on stdout when exiting 2
    assert "BLOCKED" in err


@pytest.mark.fast
def test_envelope_benign_exit0():
    rc, out, err = _run_hook({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    assert rc == 0
    assert out.strip() == ""


@pytest.mark.fast
def test_envelope_non_bash_tool_allowed():
    rc, out, err = _run_hook(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "src/assembled_core/risk/x.py"},
        }
    )
    assert rc == 0


@pytest.mark.fast
def test_envelope_edit_into_protected_not_handled_here():
    # Edit/Write protection is declarative (settings.json), NOT this hook.
    rc, out, err = _run_hook(
        {
            "tool_name": "Write",
            "tool_input": {"file_path": "src/assembled_core/risk/x.py"},
        }
    )
    assert rc == 0


@pytest.mark.fast
def test_envelope_bash_write_into_protected_blocked():
    rc, out, err = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "echo hi > src/assembled_core/risk/x.py"},
        }
    )
    assert rc == 2
    assert "BLOCKED" in err


@pytest.mark.fast
def test_envelope_malformed_input_allows():
    proc = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input="not json",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


# ---------------------------------------------------------------------------
# One-shot authorization override
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_auth_override_consumes_and_allows(tmp_path):
    auth = tmp_path / "auth"
    log = tmp_path / "auth_log.jsonl"
    auth.write_text("explicit operator authorization", encoding="utf-8")

    rc, out, err = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf build"}},
        env_extra={
            "CLAUDE_GUARD_AUTH_FILE": str(auth),
            "CLAUDE_GUARD_AUTH_LOG": str(log),
        },
    )
    assert rc == 0  # destructive but authorized → allowed
    assert not auth.exists()  # one-shot consumed
    assert log.exists()
    entry = json.loads(log.read_text(encoding="utf-8").strip())
    assert entry["reason"] == "explicit operator authorization"


@pytest.mark.fast
def test_auth_override_empty_marker_still_blocks(tmp_path):
    auth = tmp_path / "auth"
    auth.write_text("   ", encoding="utf-8")  # whitespace-only → invalid

    rc, out, err = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf build"}},
        env_extra={"CLAUDE_GUARD_AUTH_FILE": str(auth)},
    )
    assert rc == 2


@pytest.mark.fast
def test_auth_override_not_consumed_when_command_benign(tmp_path):
    auth = tmp_path / "auth"
    auth.write_text("standing authorization", encoding="utf-8")

    rc, out, err = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
        env_extra={"CLAUDE_GUARD_AUTH_FILE": str(auth)},
    )
    assert rc == 0
    assert auth.exists()  # benign command → marker untouched
