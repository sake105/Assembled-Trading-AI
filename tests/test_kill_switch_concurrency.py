"""OPS-04 regression: concurrent kill-switch writers must not fork the audit chain.

``execution/kill_switch.py`` persists state and appends to a SHA-256 hash-chained
audit log via read-modify-write. Before OPS-04 there was no lock, so two
concurrent writers (the DMS daemon and the runner drawdown check are *separate
processes*) could read the same ``prev_hash`` and fork the chain, which
``verify_audit_chain`` then reports as tampered. These tests hammer the writers
from many threads and assert the chain stays valid and the state file stays
well-formed.

Threads (not processes) are used as a fast in-process proxy: ``filelock.FileLock``
serializes contending threads on the OS-level lock just as it does processes.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution import kill_switch as ks  # noqa: E402


def _point_env_at_tmp(monkeypatch, tmp_path: Path) -> Path:
    """Redirect state/audit/lock into an isolated tmp dir."""
    audit = tmp_path / "kill_switch_audit.jsonl"
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(audit))
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_LOCK", str(tmp_path / ".kill_switch.lock")
    )
    return audit


def test_concurrent_audit_appends_keep_chain_valid(monkeypatch, tmp_path: Path) -> None:
    """N threads each append M audit records; the chain must verify with no fork."""
    audit = _point_env_at_tmp(monkeypatch, tmp_path)

    n_threads = 8
    per_thread = 15
    expected = n_threads * per_thread

    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []

    def _worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=10.0)  # maximise contention on the first append
            for j in range(per_thread):
                ks._append_audit({"action": "TEST", "worker": worker_id, "seq": j})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert errors == [], f"workers raised: {errors}"

    ok, n = ks.verify_audit_chain(audit)
    assert ok is True, "audit hash-chain forked under concurrent appends (OPS-04)"
    assert n == expected, f"expected {expected} records, chain has {n} (dropped/forked)"


def test_concurrent_activate_and_guard_chain_valid(monkeypatch, tmp_path: Path) -> None:
    """Race the public activate + guard API; chain valid, state file parses."""
    import pandas as pd

    audit = _point_env_at_tmp(monkeypatch, tmp_path)
    state_file = tmp_path / "state.json"

    barrier = threading.Barrier(6)
    errors: list[Exception] = []

    def _activator(worker_id: int) -> None:
        try:
            barrier.wait(timeout=10.0)
            for _ in range(5):
                ks.activate_kill_switch(
                    throttle_pct=0.0, reason="race", actor=f"w{worker_id}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def _guard(worker_id: int) -> None:
        try:
            barrier.wait(timeout=10.0)
            orders = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [10, 20]})
            for _ in range(5):
                ks.guard_orders_with_kill_switch(orders)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_activator, args=(i,)) for i in range(3)]
    threads += [threading.Thread(target=_guard, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert errors == [], f"workers raised: {errors}"

    ok, _n = ks.verify_audit_chain(audit)
    assert ok is True, (
        "audit hash-chain forked under concurrent activate/guard (OPS-04)"
    )

    # State file is well-formed JSON (no torn .tmp clobber).
    assert state_file.exists()
    parsed = json.loads(state_file.read_text(encoding="utf-8"))
    assert parsed.get("engaged") is True


def test_lock_path_defaults_beside_state(monkeypatch, tmp_path: Path) -> None:
    """Default lock co-locates with the state file; env override wins."""
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH_LOCK", raising=False)
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "sub" / "s.json"))
    assert ks._lock_path() == (tmp_path / "sub" / ".kill_switch.lock")

    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_LOCK", str(tmp_path / "explicit.lock"))
    assert ks._lock_path() == (tmp_path / "explicit.lock")


def test_missing_filelock_degrades_but_writer_completes(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """filelock absent (ImportError) → writer still appends; chain valid; warned."""
    import logging

    audit = _point_env_at_tmp(monkeypatch, tmp_path)
    # A ``None`` entry in sys.modules makes ``from filelock import ...`` raise
    # ImportError, exercising the degradation branch without uninstalling the pkg.
    monkeypatch.setitem(sys.modules, "filelock", None)

    with caplog.at_level(logging.WARNING):
        for j in range(3):
            ks._append_audit({"action": "TEST", "seq": j})

    ok, n = ks.verify_audit_chain(audit)
    assert ok is True, "chain must stay valid even with the lock inactive"
    assert n == 3
    assert "filelock not installed" in caplog.text


def test_lock_timeout_degrades_but_writer_completes(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """Acquire Timeout → writer proceeds unlocked; chain valid; error logged."""
    import logging

    import filelock

    audit = _point_env_at_tmp(monkeypatch, tmp_path)

    class _TimeoutLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def acquire(self, *args, **kwargs):
            raise filelock.Timeout("forced contention")

        def release(self, *args, **kwargs) -> None:  # pragma: no cover - guard
            raise AssertionError("release must not run when acquire failed")

    monkeypatch.setattr(filelock, "FileLock", _TimeoutLock)

    with caplog.at_level(logging.ERROR):
        ks._append_audit({"action": "TEST", "seq": 0})

    ok, n = ks.verify_audit_chain(audit)
    assert ok is True
    assert n == 1
    assert "proceeding WITHOUT" in caplog.text


def test_lock_manager_yields_exactly_once_on_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression for the double-yield hazard: acquire Timeout must yield once."""
    import filelock

    _point_env_at_tmp(monkeypatch, tmp_path)

    class _TimeoutLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def acquire(self, *args, **kwargs):
            raise filelock.Timeout("forced contention")

        def release(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(filelock, "FileLock", _TimeoutLock)

    entries = 0
    with ks._kill_switch_lock():
        entries += 1
    assert entries == 1, "contextmanager must yield exactly once on Timeout"
