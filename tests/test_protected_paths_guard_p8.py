"""Regression guards for the 2026-07-22 GESAMTBEWERTUNG P8 hook changes
(.claude/hooks/protected_paths_guard.py).

  K7a — PowerShell tool is now guarded: write/delete cmdlets with a
        protected-zone path, recursive+force deletes, redirects into zones.
  K7d — quote-aware sub-command splitting: a '|' or unbalanced-looking
        content INSIDE quotes must no longer produce a fail-closed false
        positive (two harmless reads were blocked during the 2026-07-19
        audit alone).
  Literal-stripping: commit messages that MENTION zone paths / 'copy' /
        'Remove-Item' must not be blocked (this repo's commit style).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

GUARD = (
    Path(__file__).resolve().parents[1]
    / ".claude"
    / "hooks"
    / "protected_paths_guard.py"
)


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("ppg_p8", GUARD)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ppg_p8"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- K7d: quote-aware splitting (bash path) --------------------------------


def test_k7d_pipe_inside_quotes_is_not_split(guard):
    # The 2026-07-19 false-positive class: quoted content containing | / &&.
    assert guard.check_command("python -c \"print('a|b && c')\"") is None


def test_k7d_commit_message_with_operators_passes(guard):
    cmd = "git commit -m 'fix: a && b | c; keep .github/workflows/ mention'"
    assert guard.check_command(cmd) is None


def test_k7d_real_separators_still_split(guard):
    assert guard.check_command("echo ok && rm -rf /tmp/x") is not None
    assert guard.check_command("echo ok; git reset --hard") is not None


def test_bash_write_into_zone_still_blocked(guard):
    assert guard.check_command("echo boom > src/assembled_core/risk/x.py") is not None


def test_bash_harmless_read_passes(guard):
    assert guard.check_command("cat src/assembled_core/risk/x.py") is None


# --- K7a: PowerShell guard ---------------------------------------------------


def test_ps_write_cmdlet_into_zone_blocks(guard):
    assert (
        guard.check_powershell_command(
            'Set-Content "src/assembled_core/risk/x.py" -Value "boom"'
        )
        is not None
    )


def test_ps_read_from_zone_passes(guard):
    assert (
        guard.check_powershell_command(
            "Get-Content src/assembled_core/risk/kill_switch.py -TotalCount 5"
        )
        is None
    )


def test_ps_recursive_force_delete_blocks_anywhere(guard):
    assert (
        guard.check_powershell_command("Remove-Item -Recurse -Force output/tmp")
        is not None
    )


def test_ps_redirect_into_zone_blocks(guard):
    assert (
        guard.check_powershell_command("echo boom > src/assembled_core/execution/x.py")
        is not None
    )


def test_ps_git_add_zone_path_passes(guard):
    # Staging zone files is not a write INTO the zone.
    assert (
        guard.check_powershell_command(
            "git add src/assembled_core/execution/broker_adapter.py 2>&1"
        )
        is None
    )


def test_ps_commit_message_mentioning_zone_and_copy_passes(guard):
    # This repo's commit style constantly mentions zone paths and words like
    # "copy"/"Remove-Item" inside here-strings — must not block.
    cmd = (
        "git commit -m @'\n"
        "feat(execution): stable root-level copy of reconcile artifact\n"
        "touches src/assembled_core/execution/ and .github/workflows/ docs;\n"
        "mentions Remove-Item -Recurse -Force in prose only.\n"
        "'@"
    )
    assert guard.check_powershell_command(cmd) is None


def test_ps_pytest_and_pipelines_pass(guard):
    assert (
        guard.check_powershell_command(
            ".venv\\Scripts\\python -m pytest tests -q 2>&1 | Select-Object -Last 5"
        )
        is None
    )


# --- main() envelope: PowerShell events are now routed ----------------------


def test_main_blocks_powershell_event(guard, monkeypatch, capsys, tmp_path):
    import io
    import json

    monkeypatch.setenv("CLAUDE_GUARD_AUTH_FILE", str(tmp_path / "auth"))
    monkeypatch.setenv("CLAUDE_GUARD_AUTH_LOG", str(tmp_path / "auth.log"))
    payload = {
        "tool_name": "PowerShell",
        "tool_input": {
            "command": 'Set-Content "src/assembled_core/risk/x.py" -Value "x"'
        },
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    rc = guard.main()
    assert rc == 2
    assert "PowerShell" in capsys.readouterr().err


def test_main_allows_harmless_powershell_event(guard, monkeypatch):
    import io
    import json

    payload = {
        "tool_name": "PowerShell",
        "tool_input": {"command": "git status --short"},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert guard.main() == 0
