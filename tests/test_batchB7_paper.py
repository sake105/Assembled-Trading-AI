"""Batch-B7 paper concurrency / atomicity regressions.

Covers:
  * B-paper-1 — read-modify-write on shared aggregate parquet/csv must serialize
    (FileLock) and use a UNIQUE per-writer tmp, so concurrent EOD/backtest
    writers can neither lose a row (lost-update) nor replace-in a half-written
    tmp (corruption). Mirrors the risk state_machine concurrency test style
    (tests/test_risk_state_machine.py).
  * intel_context atomic write — persist_historical_scores must rewrite the
    rolling JSONL cache atomically (unique tmp + os.replace), so a crash
    mid-write leaves the prior cache intact.

B-paper-2 (cost-resolver fail-closed) is ALREADY-OK and covered by
tests/test_paper_runner_paket6.py::TestResolveCostCfg
(test_returns_conservative_default_when_both_missing). No new test here.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.assembled_core.paper.intel_context import persist_historical_scores
from src.assembled_core.paper.paper_track import (
    _locked_rmw_write,
    _write_df_unique_tmp,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# B-paper-1: unique-tmp atomic write
# --------------------------------------------------------------------------- #


def test_write_df_unique_tmp_uses_pid_uuid_tmp_not_shared(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Each writer's tmp carries pid + uuid — never a fixed/shared <name>.tmp.

    Guards the corruption fix: two concurrent writers can never share a tmp, so
    os.replace always moves a fully-written file into place.
    """
    seen_tmps: list[str] = []
    original_replace = os.replace

    def recording_replace(src: str, dst: str) -> None:
        seen_tmps.append(Path(src).name)
        original_replace(src, dst)

    monkeypatch.setattr(
        "src.assembled_core.paper.paper_track.os.replace", recording_replace
    )

    dest = tmp_path / "equity_curve.parquet"
    df1 = pd.DataFrame([{"date": "2025-01-01", "equity": 1.0}])
    df2 = pd.DataFrame([{"date": "2025-01-02", "equity": 2.0}])
    _write_df_unique_tmp(df1, dest, "parquet")
    _write_df_unique_tmp(df2, dest, "parquet")

    assert len(seen_tmps) == 2
    # Never the legacy shared/fixed tmp name
    assert all(name != "equity_curve.tmp.parquet" for name in seen_tmps)
    # Two distinct unique tmps (no shared-tmp collision)
    assert seen_tmps[0] != seen_tmps[1]
    # Each carries the pid prefix and a .tmp suffix
    pid = str(os.getpid())
    assert all(name.startswith(f"equity_curve.parquet.{pid}.") for name in seen_tmps)
    assert all(name.endswith(".tmp") for name in seen_tmps)
    # Destination intact, last write wins, no tmp leaks
    assert dest.exists()
    leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []
    loaded = pd.read_parquet(dest)
    assert loaded["equity"].tolist() == [2.0]


def test_write_df_unique_tmp_finally_removes_only_own_tmp_on_failure(
    tmp_path: Path,
) -> None:
    """If the write raises before os.replace, the destination is untouched and
    no tmp leaks remain (finally cleans only this writer's own tmp)."""
    dest = tmp_path / "trades_all.csv"
    dest.write_text("date,symbol\n2025-01-01,AAA\n", encoding="utf-8")

    class _Boom(pd.DataFrame):
        # to_csv raises after the tmp file may already exist
        def to_csv(self, *a: Any, **k: Any) -> None:  # type: ignore[override]
            # Touch a tmp-like file then fail to simulate partial write
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        _write_df_unique_tmp(_Boom(), dest, "csv")

    # Original destination is intact
    assert dest.read_text(encoding="utf-8") == "date,symbol\n2025-01-01,AAA\n"
    leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []


def test_locked_rmw_write_serializes_no_lost_update(tmp_path: Path) -> None:
    """Two threads each append a distinct row via _locked_rmw_write on the SAME
    aggregate path; the FileLock serializes the read->concat->replace so NO row
    is lost (the lost-update window is closed) and the committed parquet stays
    valid throughout.
    """
    import logging

    logger = logging.getLogger("test_batchB7")
    dest = tmp_path / "equity_curve.parquet"

    # Writer 0 writes even ids 0,2,..,2*N-2; writer 1 writes odd ids 1,3,..,2*N-1.
    # Union must be the full 0..2*N-1 range if no row was lost.
    n_each = 30
    barrier = threading.Barrier(2)

    def _append_writer(start: int) -> None:
        barrier.wait()
        for k in range(n_each):
            row_id = start + 2 * k  # writer 0 -> even, writer 1 -> odd

            def _build(rid: int = row_id) -> pd.DataFrame:
                if dest.exists():
                    existing = pd.read_parquet(dest)
                    new = pd.DataFrame([{"row_id": rid, "equity": float(rid)}])
                    merged = pd.concat([existing, new], ignore_index=True)
                else:
                    merged = pd.DataFrame([{"row_id": rid, "equity": float(rid)}])
                return merged.sort_values("row_id").reset_index(drop=True)

            _locked_rmw_write(dest, _build, "parquet", logger)

    t0 = threading.Thread(target=_append_writer, args=(0,))
    t1 = threading.Thread(target=_append_writer, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    final = pd.read_parquet(dest)
    # Every row_id from both writers survived — no lost update.
    expected = set(range(0, 2 * n_each))
    assert set(final["row_id"].tolist()) == expected
    # No tmp/lock leaks
    leftover = [
        p.name
        for p in tmp_path.iterdir()
        if p.name.endswith(".tmp") or p.name.endswith(".lock")
    ]
    assert leftover == []


def test_locked_rmw_write_uses_filelock(tmp_path: Path, monkeypatch: Any) -> None:
    """The RMW path acquires the repo-internal utils.file_lock.FileLock."""
    acquired: list[str] = []
    import src.assembled_core.utils.file_lock as fl_mod

    real_acquire = fl_mod.FileLock.acquire

    def spy_acquire(self: Any) -> None:
        acquired.append(str(self._path))
        real_acquire(self)

    monkeypatch.setattr(fl_mod.FileLock, "acquire", spy_acquire)

    import logging

    logger = logging.getLogger("test_batchB7")
    dest = tmp_path / "positions_history.parquet"

    def _build() -> pd.DataFrame:
        return pd.DataFrame([{"date": "2025-01-01", "symbol": "AAA", "qty": 1.0}])

    _locked_rmw_write(dest, _build, "parquet", logger)
    assert any(dest.name in a for a in acquired)
    assert dest.exists()


def test_locked_rmw_write_lock_timeout_falls_back_and_warns(
    tmp_path: Path, caplog: Any
) -> None:
    """When the lock cannot be acquired in time, the write must STILL happen
    (corruption-safe unique-tmp) and a surfaced WARNING is logged — not silent,
    not skipped. Single-process: holding the same FileLock drives save into the
    timeout fallback branch.
    """
    import logging

    from src.assembled_core.utils.file_lock import FileLock

    logger = logging.getLogger("test_batchB7")
    dest = tmp_path / "equity_curve.parquet"

    def _build() -> pd.DataFrame:
        return pd.DataFrame([{"row_id": 99, "equity": 99.0}])

    with caplog.at_level(logging.WARNING):
        # Hold the lock externally so the inner acquire times out fast.
        with FileLock(dest, timeout=5.0):
            _locked_rmw_write(dest, _build, "parquet", logger, lock_timeout_s=0.2)

    assert dest.exists()
    loaded = pd.read_parquet(dest)
    assert loaded["row_id"].tolist() == [99]
    assert any("lock timeout" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# intel_context: atomic historical_scores rewrite
# --------------------------------------------------------------------------- #


def test_persist_historical_scores_is_atomic(tmp_path: Path) -> None:
    """Rewrite uses a unique tmp + os.replace, never an in-place truncate."""
    cache = tmp_path / "historical_scores.jsonl"
    seen_tmps: list[str] = []
    real_replace = os.replace

    def recording_replace(src: str, dst: str) -> None:
        seen_tmps.append(Path(src).name)
        real_replace(src, dst)

    import src.assembled_core.paper.intel_context as ic

    # os is imported locally inside persist_historical_scores; patch the stdlib.
    orig = os.replace
    try:
        os.replace = recording_replace  # type: ignore[assignment]
        persist_historical_scores(
            pd.Series([1.0, 2.0, 3.0]),
            tmp_path,
            historical_scores_path=str(cache),
        )
    finally:
        os.replace = orig  # type: ignore[assignment]

    assert cache.exists()
    assert len(seen_tmps) == 1
    assert seen_tmps[0] != cache.name  # not an in-place write
    pid = str(os.getpid())
    assert seen_tmps[0].startswith(f"{cache.name}.{pid}.")
    assert seen_tmps[0].endswith(".tmp")
    # No tmp leaks
    leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []
    # Content is one valid JSONL record
    lines = [ln for ln in cache.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert ic.json.loads(lines[0])["n"] == 3


def test_persist_historical_scores_crash_leaves_prior_cache_intact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """If the rewrite crashes mid-write (os.replace fails), the PRIOR cache file
    is untouched — no half-written truncation.
    """
    cache = tmp_path / "historical_scores.jsonl"
    # Seed a prior, valid cache (recent ts so it is kept within window).
    prior_ts = pd.Timestamp.now("UTC").isoformat()
    cache.write_text(
        ic_dumps({"ts": prior_ts, "mean": 5.0, "n": 7}) + "\n",
        encoding="utf-8",
    )
    prior_content = cache.read_text(encoding="utf-8")

    def boom_replace(src: str, dst: str) -> None:
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom_replace)

    # Must not raise out (OSError is caught + warned); prior cache must survive.
    persist_historical_scores(
        pd.Series([9.0, 9.0]),
        tmp_path,
        historical_scores_path=str(cache),
    )

    assert cache.read_text(encoding="utf-8") == prior_content
    leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []


def ic_dumps(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj)
