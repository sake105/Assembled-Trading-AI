"""Tests for experience_log.py — append, load, summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.ops.experience_log import (
    append_experience,
    compute_experience_summary,
    load_experience,
)


@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "test_experience.jsonl"


def test_append_and_load(tmp_log):
    entry = {
        "cycle_date": "2026-04-07",
        "execution_mode": "broker",
        "broker_equity": 10500.0,
        "exit_code": 0,
    }
    result = append_experience(entry, log_path=tmp_log)
    assert "timestamp_utc" in result

    df = load_experience(log_path=tmp_log)
    assert len(df) == 1
    assert df.iloc[0]["cycle_date"] == "2026-04-07"


def test_append_multiple(tmp_log):
    for i in range(5):
        append_experience(
            {
                "cycle_date": f"2026-04-0{i+1}",
                "broker_equity": 10000 + i * 100,
                "exit_code": 0,
            },
            log_path=tmp_log,
        )
    df = load_experience(log_path=tmp_log)
    assert len(df) == 5


def test_load_empty(tmp_log):
    df = load_experience(log_path=tmp_log)
    assert df.empty


def test_load_nonexistent(tmp_path):
    df = load_experience(log_path=tmp_path / "does_not_exist.jsonl")
    assert df.empty


def test_summary_empty(tmp_log):
    summary = compute_experience_summary(log_path=tmp_log)
    assert summary["total_cycles"] == 0


def test_summary_with_data(tmp_log):
    equities = [10000, 10100, 10050, 10200, 10300]
    for i, eq in enumerate(equities):
        append_experience(
            {
                "cycle_date": f"2026-04-0{i+1}",
                "broker_equity": eq,
                "exit_code": 0,
                "execution_mode": "broker",
            },
            log_path=tmp_log,
        )
    summary = compute_experience_summary(log_path=tmp_log)
    assert summary["total_cycles"] == 5
    assert summary["latest_equity"] == 10300
    assert summary["success_rate_pct"] == 100.0
    assert "sharpe_approx" in summary
    assert "max_drawdown_pct" in summary


def test_malformed_lines_skipped(tmp_log):
    # Write one valid and one malformed line
    with open(tmp_log, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"cycle_date": "2026-04-01", "exit_code": 0}) + "\n")
        fh.write("this is not json\n")
        fh.write(json.dumps({"cycle_date": "2026-04-02", "exit_code": 0}) + "\n")
    df = load_experience(log_path=tmp_log)
    assert len(df) == 2
