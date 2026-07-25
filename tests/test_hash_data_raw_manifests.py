"""Regression guards for scripts/ops/hash_data_raw_manifests.py (2026-07-25).

Persisted per Stage-3 F-auditor-1: the Stage-1 manipulation tests were run
transiently and must live in the repo (Rule 40). Covers the non-trivial
verify logic: write->verify roundtrip, tamper->changed, delete->missing,
added file, corrupt sidecar (fail-closed, no traceback), missing-dir
SKIP-vs-MISS asymmetry, determinism, no self-hash drift.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "hash_data_raw_manifests.py"


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hash_raw_test_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    pull = tmp_path / "pull_a"
    (pull / "sub").mkdir(parents=True)
    (pull / "a.csv").write_text("col\n1\n", encoding="utf-8")
    (pull / "sub" / "b.json").write_text('{"x": 1}', encoding="utf-8")
    (pull / "notes.txt").write_text("decoy — not a data suffix", encoding="utf-8")
    monkeypatch.setattr(m, "PULL_DIRS", [pull])
    return m, pull


def test_write_then_verify_roundtrip_clean(mod):
    m, pull = mod
    assert m.write_sidecars() == 0
    sidecar = pull / m.SIDECAR_NAME
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["n_files"] == 2  # decoy .txt excluded
    assert set(data["files"]) == {"a.csv", "sub/b.json"}  # POSIX keys
    assert m.verify_sidecars() == 0


def test_tampered_file_same_size_is_detected(mod):
    m, pull = mod
    m.write_sidecars()
    (pull / "a.csv").write_text("col\n2\n", encoding="utf-8")  # same size
    assert m.verify_sidecars() == 1


def test_deleted_file_is_detected(mod):
    m, pull = mod
    m.write_sidecars()
    (pull / "sub" / "b.json").unlink()
    assert m.verify_sidecars() == 1


def test_added_file_is_detected(mod):
    m, pull = mod
    m.write_sidecars()
    (pull / "new.parquet").write_bytes(b"PAR1")
    assert m.verify_sidecars() == 1


def test_corrupt_sidecar_fails_closed_without_traceback(mod):
    m, pull = mod
    m.write_sidecars()
    (pull / m.SIDECAR_NAME).write_text("{not json", encoding="utf-8")
    assert m.verify_sidecars() == 1  # clean FAIL, no exception


def test_missing_sidecar_is_rc1(mod):
    m, pull = mod
    assert m.verify_sidecars() == 1  # never written


def test_missing_dir_skip_in_write_miss_in_verify(mod, tmp_path, monkeypatch):
    m, _ = mod
    ghost = tmp_path / "does_not_exist"
    monkeypatch.setattr(m, "PULL_DIRS", [ghost])
    assert m.write_sidecars() == 0  # SKIP
    assert m.verify_sidecars() == 1  # MISS — documented asymmetry


def test_rewrite_is_deterministic_and_no_self_hash_drift(mod):
    m, pull = mod
    m.write_sidecars()
    first = json.loads((pull / m.SIDECAR_NAME).read_text(encoding="utf-8"))["files"]
    # Second write happens WITH the sidecar present — it must not hash itself.
    m.write_sidecars()
    second = json.loads((pull / m.SIDECAR_NAME).read_text(encoding="utf-8"))["files"]
    assert first == second
    assert list(first.keys()) == list(second.keys())  # ordering stable
    assert m.SIDECAR_NAME not in first
    assert m.verify_sidecars() == 0
