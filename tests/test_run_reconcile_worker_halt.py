"""OPS-07 regression: the RECONCILE worker can HALT (non-zero exit) on a
sim-to-real mismatch instead of always passing green.

Before OPS-07 ``scripts/run_reconcile_worker.py`` returned exit 0 on a reconcile
mismatch (``ok=False`` only logged a WARNING), and the daily workflow wrapped it
in ``--dry-run || true`` — so the documented 2026-04-10 $412.54 reconciliation
break "reported exit_code=0 reconcile=OK" and never halted.

OPS-07 adds an opt-in ``--halt-on-mismatch`` flag with a fail-CLOSED exit-code
contract:

    0  OK (or mismatch with the flag OFF — preserves the read-only default)
    1  internal error / exception
    2  reconcile MISMATCH and the flag is set
    3  the flag is set but a ledger/broker input is missing (a match cannot be
       proven against absent data — fail-closed, consistent with R2-1/R2-2)

These tests pin that contract. The ledger/broker loaders are monkeypatched so
the test controls the reconcile inputs without building a real ledger parquet;
the fail-closed *input* tests deliberately leave the loaders real because the
guard must fire BEFORE any load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_reconcile_worker as rw  # noqa: E402


def _match_frame() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["AAPL"], "qty": [10.0]})


def _mismatch_broker_frame() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["AAPL"], "qty": [5.0]})


def _touch(p: Path) -> Path:
    p.write_text("", encoding="utf-8")
    return p


def _patch_loaders(
    monkeypatch, ledger_df: pd.DataFrame, broker_df: pd.DataFrame
) -> None:
    monkeypatch.setattr(rw, "_load_ledger_positions", lambda _p: ledger_df.copy())
    monkeypatch.setattr(rw, "_load_broker_snapshot", lambda _p: broker_df.copy())


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    base = ["run_reconcile_worker.py", "--output-dir", str(tmp_path)]
    base.extend(extra)
    return base


def test_mismatch_without_flag_exits_zero(monkeypatch, tmp_path: Path) -> None:
    """Default (flag OFF): a mismatch still exits 0 — read-only diagnostic."""
    ledger = _touch(tmp_path / "ledger.parquet")
    broker = _touch(tmp_path / "broker.csv")
    _patch_loaders(monkeypatch, _match_frame(), _mismatch_broker_frame())
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, "--ledger-path", str(ledger), "--broker-path", str(broker)),
    )
    assert rw.main() == 0


def test_match_with_flag_exits_zero(monkeypatch, tmp_path: Path) -> None:
    """Flag ON + ledger matches broker → exit 0 (the common clean case)."""
    ledger = _touch(tmp_path / "ledger.parquet")
    broker = _touch(tmp_path / "broker.csv")
    _patch_loaders(monkeypatch, _match_frame(), _match_frame())
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            tmp_path,
            "--ledger-path",
            str(ledger),
            "--broker-path",
            str(broker),
            "--ledger-cash",
            "100000.0",
            "--broker-cash",
            "100000.0",
            "--halt-on-mismatch",
        ),
    )
    assert rw.main() == 0


def test_mismatch_with_flag_halts_exit_two(monkeypatch, tmp_path: Path) -> None:
    """Flag ON + real mismatch → exit 2 (the OPS-07 halt)."""
    ledger = _touch(tmp_path / "ledger.parquet")
    broker = _touch(tmp_path / "broker.csv")
    _patch_loaders(monkeypatch, _match_frame(), _mismatch_broker_frame())
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            tmp_path,
            "--ledger-path",
            str(ledger),
            "--broker-path",
            str(broker),
            "--halt-on-mismatch",
        ),
    )
    assert rw.main() == 2


def test_flag_without_inputs_fails_closed_exit_three(
    monkeypatch, tmp_path: Path
) -> None:
    """Flag ON but NO --ledger-path/--broker-path → exit 3, fail-closed."""
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--halt-on-mismatch"))
    assert rw.main() == 3


def test_flag_with_missing_paths_fails_closed_exit_three(
    monkeypatch, tmp_path: Path
) -> None:
    """Flag ON but the supplied paths do not exist → exit 3, fail-closed."""
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            tmp_path,
            "--ledger-path",
            str(tmp_path / "nope.parquet"),
            "--broker-path",
            str(tmp_path / "nope.csv"),
            "--halt-on-mismatch",
        ),
    )
    assert rw.main() == 3


def test_failclosed_manifest_is_written(monkeypatch, tmp_path: Path) -> None:
    """The exit-3 fail-closed path leaves an auditable manifest with the reason."""
    import json

    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--halt-on-mismatch"))
    assert rw.main() == 3
    manifests = list(tmp_path.glob("reconcile_manifest_*.json"))
    assert manifests, "fail-closed run must still write a manifest"
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert data["halt_reason"] == "insufficient_inputs"


def test_internal_error_exits_one(monkeypatch, tmp_path: Path) -> None:
    """An exception inside the run → exit 1 (distinct from a mismatch halt)."""
    ledger = _touch(tmp_path / "ledger.parquet")
    broker = _touch(tmp_path / "broker.csv")
    _patch_loaders(monkeypatch, _match_frame(), _match_frame())

    def _boom(*args, **kwargs):
        raise RuntimeError("reconcile blew up")

    monkeypatch.setattr(rw, "reconcile_ledger_vs_broker", _boom)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            tmp_path,
            "--ledger-path",
            str(ledger),
            "--broker-path",
            str(broker),
            "--halt-on-mismatch",
        ),
    )
    assert rw.main() == 1
