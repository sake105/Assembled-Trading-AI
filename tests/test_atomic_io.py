"""Tests for utils/atomic_io.py — atomic JSON write helper (B2).

Verifies:
- Normal write produces valid JSON at the target path
- Crash during write (simulated via monkeypatch) leaves original intact
- Tmp file is cleaned up on success
- Backward-compat alias works
"""

from __future__ import annotations

import json
import os

import pytest

from src.assembled_core.utils.atomic_io import (
    atomic_write_json,
    atomic_write_json_with_retry,
)


class TestAtomicWriteJson:
    def test_normal_write_creates_valid_json(self, tmp_path):
        target = tmp_path / "state.json"
        data = {"status": "ok", "value": 42}
        atomic_write_json(target, data)

        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded == data

    def test_no_tmp_file_remains_after_success(self, tmp_path):
        target = tmp_path / "state.json"
        atomic_write_json(target, {"x": 1})

        tmp = tmp_path / "state.json.tmp"
        assert not tmp.exists()

    def test_parent_directory_created_if_missing(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "state.json"
        atomic_write_json(target, {"ok": True})
        assert target.exists()

    def test_existing_file_is_replaced_atomically(self, tmp_path):
        target = tmp_path / "state.json"
        target.write_text(json.dumps({"old": True}))
        atomic_write_json(target, {"new": True})
        loaded = json.loads(target.read_text())
        assert loaded == {"new": True}

    def test_crash_during_replace_raises_and_leaves_no_partial_file(
        self, tmp_path, monkeypatch
    ):
        """Simulate os.replace failing: original file must remain intact."""
        target = tmp_path / "state.json"
        original = {"safe": "value"}
        target.write_text(json.dumps(original))

        call_count = [0]
        real_replace = os.replace

        def fail_replace(src, dst):
            call_count[0] += 1
            raise OSError("Simulated crash during replace")

        monkeypatch.setattr("os.replace", fail_replace)

        with pytest.raises(OSError, match="Simulated crash"):
            atomic_write_json(target, {"unsafe": "value"}, retries=1)

        # Original file must still be intact
        assert json.loads(target.read_text()) == original

    def test_crash_on_json_dump_raises(self, tmp_path, monkeypatch):
        """Simulate crash during json.dump (via bad data): function must raise."""
        target = tmp_path / "state.json"

        class _Unserializable:
            pass

        # json.dump with default=str should handle most types, but we override default
        import json as _json

        original_dump = _json.dump

        def fail_dump(obj, fp, **kwargs):
            raise OSError("Simulated json.dump failure")

        monkeypatch.setattr("src.assembled_core.utils.atomic_io.json.dump", fail_dump)

        with pytest.raises(OSError, match="Simulated json.dump failure"):
            atomic_write_json(target, {"key": "val"}, retries=1)

        # Target file must NOT exist (write never completed)
        assert not target.exists()

    def test_sort_keys_option(self, tmp_path):
        target = tmp_path / "sorted.json"
        atomic_write_json(target, {"z": 1, "a": 2}, sort_keys=True)
        raw = target.read_text()
        # "a" must appear before "z" in sorted output
        assert raw.index('"a"') < raw.index('"z"')

    def test_backward_compat_alias(self, tmp_path):
        """atomic_write_json_with_retry is the old name — must still work."""
        target = tmp_path / "compat.json"
        atomic_write_json_with_retry(target, {"alias": True})
        assert json.loads(target.read_text()) == {"alias": True}


class TestAtomicWriteJsonIntegration:
    def test_pit_store_uses_atomic_write(self, tmp_path, monkeypatch):
        """pit_store.archive() must call atomic_write_json, not write_text."""
        calls = []

        def recording_write(path, data, **kwargs):
            calls.append(str(path))

        monkeypatch.setattr(
            "src.assembled_core.utils.atomic_io.atomic_write_json",
            recording_write,
        )

        from src.assembled_core.intel.pit_store import PITStore

        store = PITStore(root=tmp_path)

        # archive creates the directory structure and calls atomic_write_json
        store.archive(
            source="test_source",
            run_id="run001",
            artifact_type="signals",
            data={"signal": 1.0},
        )

        assert any(
            "signals.json" in c for c in calls
        ), f"atomic_write_json not called for signals.json; calls={calls}"

    def test_factor_store_manifest_uses_atomic_write(self, tmp_path, monkeypatch):
        """factor_store._write_manifest() must use atomic_write_json."""
        import pandas as pd

        calls = []

        def recording_write(path, data, **kwargs):
            calls.append(str(path))

        monkeypatch.setattr(
            "src.assembled_core.utils.atomic_io.atomic_write_json",
            recording_write,
        )

        from src.assembled_core.data.factor_store import _write_manifest

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "momentum": [0.1, 0.2, 0.3],
            }
        )

        panel_dir = tmp_path / "panel"
        panel_dir.mkdir()
        _write_manifest(
            panel_dir, df, factor_group="test", freq="1d", universe_key="US"
        )

        assert any(
            "_metadata.json" in c for c in calls
        ), f"atomic_write_json not called for _metadata.json; calls={calls}"
