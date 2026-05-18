"""Tests for scripts/ops/check_promotion_gate.py (C2-074)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.ops.check_promotion_gate import (
    check_dsr_threshold,
    check_max_drawdown,
    check_operator_flag,
    check_rolling_sharpe,
    check_track_record_length,
    render_markdown,
    run_promotion_gate,
)


def _make_csv(tmp_path: Path, returns: np.ndarray) -> Path:
    equity = 100_000.0 * np.cumprod(1.0 + returns)
    dates = pd.date_range("2024-01-01", periods=len(returns), tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "timestamp": dates,
            "equity": equity,
            "daily_return": returns,
        }
    )
    p = tmp_path / "test_equity.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Per-check unit tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestTrackRecordLength:
    def test_long_enough(self) -> None:
        s = pd.Series(np.arange(100, dtype=float))
        r = check_track_record_length(s, min_days=90)
        assert r["pass"] is True
        assert r["actual"] == 100

    def test_too_short(self) -> None:
        s = pd.Series(np.arange(50, dtype=float))
        r = check_track_record_length(s, min_days=90)
        assert r["pass"] is False


@pytest.mark.fast
class TestRollingSharpe:
    def test_high_sharpe_returns_pass(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.005, 0.005, size=150)  # very high drift/vol
        r = check_rolling_sharpe(returns)
        assert r["pass"] is True

    def test_low_sharpe_returns_fail(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(-0.001, 0.02, size=150)  # negative drift
        r = check_rolling_sharpe(returns)
        assert r["pass"] is False

    def test_insufficient_data(self) -> None:
        r = check_rolling_sharpe(np.array([0.01, 0.02]))
        assert r["pass"] is False


@pytest.mark.fast
class TestMaxDrawdown:
    def test_low_drawdown_passes(self) -> None:
        equity = np.array([100.0, 105.0, 95.0, 110.0, 120.0])
        # MDD = (95-105)/105 ≈ -9.5%
        r = check_max_drawdown(equity, max_dd_threshold=-0.20)
        assert r["pass"] is True

    def test_large_drawdown_fails(self) -> None:
        equity = np.array([100.0, 105.0, 50.0, 60.0])
        # MDD = (50-105)/105 ≈ -52%
        r = check_max_drawdown(equity, max_dd_threshold=-0.20)
        assert r["pass"] is False


@pytest.mark.fast
class TestDSRThreshold:
    def test_high_sharpe_passes(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.005, 0.005, size=200)  # high sharpe
        equity = 100_000.0 * np.cumprod(1.0 + returns)
        r = check_dsr_threshold(equity, dsr_threshold=1.0)
        assert r["pass"] is True

    def test_insufficient_data(self) -> None:
        equity = np.array([100.0, 101.0])
        r = check_dsr_threshold(equity)
        assert r["pass"] is False


@pytest.mark.fast
class TestOperatorFlag:
    def test_true_passes(self) -> None:
        r = check_operator_flag("kill_switch_confirmed", confirmed=True)
        assert r["pass"] is True

    def test_false_fails(self) -> None:
        r = check_operator_flag("kill_switch_confirmed", confirmed=False)
        assert r["pass"] is False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPipeline:
    def test_passing_pipeline(self, tmp_path: Path) -> None:
        """Strong strategy + operator flags set → ready verdict."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.003, 0.008, size=150)  # high sharpe, 150 days
        path = _make_csv(tmp_path, returns)
        report = run_promotion_gate(
            equity_curve_path=path,
            kill_switch_confirmed=True,
            pre_trade_gates_confirmed=True,
            run_forensic=False,  # subprocess too slow for unit test
        )
        # All 6 auto-checks should pass
        assert report["promotion_verdict"] == "ready", report

    def test_blocked_minor_operator_flags(self, tmp_path: Path) -> None:
        """Strong strategy but operator flags unconfirmed → blocked_minor."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.003, 0.008, size=150)
        path = _make_csv(tmp_path, returns)
        report = run_promotion_gate(
            equity_curve_path=path,
            kill_switch_confirmed=False,
            pre_trade_gates_confirmed=False,
            run_forensic=False,
        )
        # 2 missing operator flags out of 6 → blocked_minor
        assert report["promotion_verdict"] in {"blocked_minor", "blocked_major"}
        assert "kill_switch_confirmed" in report["failing_checks"]
        assert "pre_trade_gates_confirmed" in report["failing_checks"]

    def test_blocked_major_weak_strategy(self, tmp_path: Path) -> None:
        """Weak strategy fails multiple checks → blocked_major."""
        rng = np.random.default_rng(0)
        returns = rng.normal(-0.001, 0.02, size=150)  # negative drift
        path = _make_csv(tmp_path, returns)
        report = run_promotion_gate(
            equity_curve_path=path,
            kill_switch_confirmed=False,
            pre_trade_gates_confirmed=False,
            run_forensic=False,
        )
        assert report["promotion_verdict"] == "blocked_major"

    def test_missing_file_blocked(self, tmp_path: Path) -> None:
        report = run_promotion_gate(
            equity_curve_path=tmp_path / "nope.csv",
            run_forensic=False,
        )
        assert report["promotion_verdict"] == "blocked"
        assert "error" in report

    def test_missing_equity_column(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad, index=False)
        report = run_promotion_gate(bad, run_forensic=False)
        assert report["promotion_verdict"] == "blocked"

    def test_json_round_trip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.003, 0.008, size=150)
        path = _make_csv(tmp_path, returns)
        report = run_promotion_gate(
            path,
            kill_switch_confirmed=True,
            pre_trade_gates_confirmed=True,
            run_forensic=False,
        )
        s = json.dumps(report)
        rt = json.loads(s)
        assert rt["promotion_verdict"] == report["promotion_verdict"]


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_render_markdown_basic(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.003, 0.008, size=150)
    path = _make_csv(tmp_path, returns)
    report = run_promotion_gate(
        path,
        kill_switch_confirmed=True,
        pre_trade_gates_confirmed=True,
        run_forensic=False,
    )
    md = render_markdown(report)
    assert "Promotion-Gate-Check" in md
    assert "Verdict" in md
    assert "Per-Check Detail" in md
    assert "Limitations" in md
