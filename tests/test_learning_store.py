"""Tests for M11: Post-Trade Learning Loop — learning store."""

from __future__ import annotations

import pytest

from src.assembled_core.qa.learning_store import (
    append_learning_record,
    load_learning_records,
    get_latest_record,
    summarize_learning_store,
)


@pytest.mark.phase12
@pytest.mark.phase13
class TestAppendLearningRecord:
    def test_creates_file(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"run_id": "r1"}, store_path=store)
        assert store.exists()

    def test_appends_multiple_records(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"run_id": "r1"}, store_path=store)
        append_learning_record({"run_id": "r2"}, store_path=store)
        records = load_learning_records(store)
        assert len(records) == 2

    def test_record_is_readable_json(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"run_id": "r1", "value": 42}, store_path=store)
        records = load_learning_records(store)
        assert records[0]["run_id"] == "r1"
        assert records[0]["value"] == 42

    def test_creates_parent_dirs(self, tmp_path):
        store = tmp_path / "nested" / "deep" / "test.jsonl"
        append_learning_record({"run_id": "r1"}, store_path=store)
        assert store.exists()


@pytest.mark.phase12
@pytest.mark.phase13
class TestLoadLearningRecords:
    def test_empty_if_file_not_exists(self, tmp_path):
        records = load_learning_records(tmp_path / "nonexistent.jsonl")
        assert records == []

    def test_returns_list(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"a": 1}, store_path=store)
        records = load_learning_records(store)
        assert isinstance(records, list)

    def test_preserves_order(self, tmp_path):
        store = tmp_path / "test.jsonl"
        for i in range(5):
            append_learning_record({"n": i}, store_path=store)
        records = load_learning_records(store)
        assert [r["n"] for r in records] == [0, 1, 2, 3, 4]


@pytest.mark.phase12
@pytest.mark.phase13
class TestGetLatestRecord:
    def test_none_if_empty(self, tmp_path):
        assert get_latest_record(tmp_path / "nonexistent.jsonl") is None

    def test_returns_last(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"run_id": "first"}, store_path=store)
        append_learning_record({"run_id": "last"}, store_path=store)
        rec = get_latest_record(store)
        assert rec["run_id"] == "last"


@pytest.mark.phase12
@pytest.mark.phase13
class TestSummarizeLearningStore:
    def test_empty_store(self, tmp_path):
        summary = summarize_learning_store(tmp_path / "nonexistent.jsonl")
        assert summary["total_records"] == 0
        assert summary["avg_hit_rate"] is None

    def test_counts_records(self, tmp_path):
        store = tmp_path / "test.jsonl"
        for i in range(3):
            append_learning_record(
                {
                    "overall_hit_rate": 0.6,
                    "analysis_date": f"2024-01-0{i + 1}",
                },
                store_path=store,
            )
        summary = summarize_learning_store(store)
        assert summary["total_records"] == 3

    def test_avg_hit_rate_computed(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"overall_hit_rate": 0.6}, store_path=store)
        append_learning_record({"overall_hit_rate": 0.4}, store_path=store)
        summary = summarize_learning_store(store)
        assert abs(summary["avg_hit_rate"] - 0.5) < 0.01

    def test_latest_date_set(self, tmp_path):
        store = tmp_path / "test.jsonl"
        append_learning_record({"analysis_date": "2024-01-01"}, store_path=store)
        append_learning_record({"analysis_date": "2024-01-10"}, store_path=store)
        summary = summarize_learning_store(store)
        assert summary["latest_date"] == "2024-01-10"
