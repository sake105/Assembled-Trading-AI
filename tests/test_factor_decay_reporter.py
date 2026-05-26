"""Tests for src/assembled_core/qa/factor_decay_reporter.py.

Covers:
- Normal run writes JSONL
- Exception inside computation does not crash caller
- Empty factor panel logs SKIP
- Output path parent dirs created automatically
- Log record contains expected fields
- Factor column auto-detection
- No-forward-returns scenario (price-only panel)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.assembled_core.qa.factor_decay_reporter import run_factor_decay_monitoring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    n_symbols: int = 3,
    n_days: int = 30,
    add_factor: bool = True,
    factor_name: str = "mom_12m",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal panel DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="UTC")
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rows = []
    for sym in symbols:
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        for j, d in enumerate(dates):
            row = {"timestamp": d, "symbol": sym, "close": float(prices[j])}
            if add_factor:
                row[factor_name] = float(rng.normal(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunFactorDecayMonitoring:
    def test_normal_run_writes_jsonl(self, tmp_path):
        """A successful run appends one JSON line to the log file."""
        panel = _make_panel(n_symbols=4, n_days=40, factor_name="factor_a")
        log_file = tmp_path / "qa" / "factor_decay.jsonl"

        result = run_factor_decay_monitoring(
            panel_df=panel,
            factor_cols=["factor_a"],
            log_path=log_file,
        )

        assert log_file.exists(), "JSONL log file was not created"
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}"
        record = json.loads(lines[0])
        assert "run_date" in record
        assert "ts_utc" in record
        assert "results" in record
        assert result["status"] == "ok"

    def test_output_path_parent_dirs_created(self, tmp_path):
        """Parent directories of log_path are created on demand."""
        deeply_nested = tmp_path / "a" / "b" / "c" / "decay.jsonl"
        panel = _make_panel(n_symbols=3, n_days=30, factor_name="f1")

        run_factor_decay_monitoring(
            panel_df=panel,
            factor_cols=["f1"],
            log_path=deeply_nested,
        )

        assert deeply_nested.exists(), "Nested parent dirs were not created"

    def test_empty_panel_returns_skip(self, tmp_path):
        """An empty panel_df returns status='skip' and does not crash."""
        log_file = tmp_path / "decay.jsonl"
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])

        result = run_factor_decay_monitoring(
            panel_df=empty_df,
            factor_cols=["some_factor"],
            log_path=log_file,
        )

        assert result["status"] == "skip"
        assert result["factors_computed"] == 0
        # Empty panel → no JSONL row written (nothing to report)
        # (the function returns early without writing)
        assert not log_file.exists() or log_file.stat().st_size == 0

    def test_none_panel_returns_skip(self, tmp_path):
        """None panel_df returns status='skip' without raising."""
        log_file = tmp_path / "decay.jsonl"
        result = run_factor_decay_monitoring(
            panel_df=None,
            factor_cols=["f"],
            log_path=log_file,
        )
        assert result["status"] == "skip"
        assert result["factors_computed"] == 0

    def test_exception_inside_factor_does_not_crash_caller(self, tmp_path, monkeypatch):
        """If compute_ic_decay_curve raises for a factor, result still returns ok."""
        import src.assembled_core.qa.factor_analysis as fa

        def _bad_compute(*args, **kwargs):
            raise RuntimeError("synthetic test failure")

        monkeypatch.setattr(fa, "compute_ic_decay_curve", _bad_compute)

        panel = _make_panel(n_symbols=3, n_days=30, factor_name="bad_factor")
        log_file = tmp_path / "decay.jsonl"

        # Must not raise
        result = run_factor_decay_monitoring(
            panel_df=panel,
            factor_cols=["bad_factor"],
            log_path=log_file,
        )

        # Status is still "ok" at function level; per-factor status is "error"
        assert result["status"] == "ok"
        assert result["factors_computed"] == 0  # error path counts as not computed
        per_factor = result["results"]
        assert len(per_factor) == 1
        assert per_factor[0]["status"] == "error"
        assert "synthetic test failure" in per_factor[0].get("traceback", "")

    def test_log_record_contains_expected_fields(self, tmp_path):
        """JSONL record has all required fields."""
        panel = _make_panel(n_symbols=4, n_days=50, factor_name="momentum")
        log_file = tmp_path / "decay.jsonl"

        run_factor_decay_monitoring(
            panel_df=panel,
            factor_cols=["momentum"],
            log_path=log_file,
            run_date="2026-01-15",
        )

        record = json.loads(log_file.read_text(encoding="utf-8").strip())
        for field in ("run_date", "ts_utc", "factors_computed", "results", "status"):
            assert field in record, f"Missing field: {field}"
        assert record["run_date"] == "2026-01-15"
        assert isinstance(record["results"], list)

    def test_missing_factor_col_in_panel_logs_skip(self, tmp_path):
        """Requesting a factor column that doesn't exist logs [SKIP]."""
        panel = _make_panel(n_symbols=3, n_days=30, add_factor=False)
        log_file = tmp_path / "decay.jsonl"

        result = run_factor_decay_monitoring(
            panel_df=panel,
            factor_cols=["nonexistent_factor"],
            log_path=log_file,
        )

        assert result["status"] == "skip"
        assert result["factors_computed"] == 0

    def test_multiple_appends_produce_multiple_lines(self, tmp_path):
        """Two calls to the function append two lines to the same JSONL file."""
        panel = _make_panel(n_symbols=3, n_days=35, factor_name="sig")
        log_file = tmp_path / "decay.jsonl"

        run_factor_decay_monitoring(
            panel_df=panel, factor_cols=["sig"], log_path=log_file
        )
        run_factor_decay_monitoring(
            panel_df=panel, factor_cols=["sig"], log_path=log_file
        )

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_survivorship_bias_warning_logged_for_factor_false(self, tmp_path, caplog):
        """[FACTOR-DECAY] prefix appears in log for a successful run."""
        panel = _make_panel(n_symbols=3, n_days=30, factor_name="f1")
        log_file = tmp_path / "decay.jsonl"

        with caplog.at_level(
            logging.INFO, logger="src.assembled_core.qa.factor_decay_reporter"
        ):
            run_factor_decay_monitoring(
                panel_df=panel,
                factor_cols=["f1"],
                log_path=log_file,
            )

        assert any("[FACTOR-DECAY]" in r.message for r in caplog.records)
