"""Tests for scripts/forensic/out_of_regime_test.py (C2-019)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.forensic.out_of_regime_test import (
    _annualised_sharpe,
    check_edge_consistency,
    classify_regimes,
    per_regime_metrics,
    render_markdown,
    run_out_of_regime_test,
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


@pytest.mark.fast
class TestClassifyRegimes:
    def test_short_series_all_warmup(self) -> None:
        equity = np.linspace(100.0, 110.0, 50)
        labels = classify_regimes(equity, rolling_window=120)
        assert all(label == "warmup" for label in labels)

    def test_steady_bull_regime(self) -> None:
        # 200 days, equity rises linearly from 100 → 200 (100% return)
        equity = np.linspace(100.0, 200.0, 200)
        labels = classify_regimes(equity, rolling_window=120, threshold=0.05)
        # First 120 days are warmup, days 120+ should be bull
        post_warmup = labels[120:]
        bull_count = sum(1 for lab in post_warmup if lab == "bull")
        assert bull_count > 0
        assert bull_count >= len(post_warmup) // 2

    def test_steady_bear_regime(self) -> None:
        equity = np.linspace(100.0, 50.0, 200)
        labels = classify_regimes(equity, rolling_window=120, threshold=0.05)
        post_warmup = labels[120:]
        bear_count = sum(1 for lab in post_warmup if lab == "bear")
        assert bear_count > 0

    def test_sideways_below_threshold(self) -> None:
        # Tiny upward drift (well below 5% per 120d)
        rng = np.random.default_rng(0)
        equity = 100.0 + rng.normal(0, 0.01, size=200).cumsum() * 0.1
        labels = classify_regimes(equity, rolling_window=120, threshold=0.05)
        post_warmup = labels[120:]
        sideways_count = sum(1 for lab in post_warmup if lab == "sideways")
        # At least SOME sideways days (random noise + tight threshold)
        assert sideways_count > 0


@pytest.mark.fast
class TestPerRegimeMetrics:
    def test_empty_regime_returns_zero_count(self) -> None:
        equity = np.array([100.0, 101.0, 102.0])
        returns = np.array([0.01, 0.01])
        labels = np.array(["warmup", "warmup", "warmup"], dtype=object)
        out = per_regime_metrics(equity, returns, labels)
        assert out["bull"]["n_days"] == 0
        assert out["bear"]["n_days"] == 0
        assert out["sideways"]["n_days"] == 0
        assert out["warmup"]["n_days"] == 2  # aligned_labels = labels[1:]

    def test_bull_only_metrics(self) -> None:
        equity = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        returns = np.diff(equity) / equity[:-1]
        labels = np.array(
            ["warmup", "bull", "bull", "bull", "bull", "bull"], dtype=object
        )
        out = per_regime_metrics(equity, returns, labels)
        assert out["bull"]["n_days"] == 5
        # All-positive returns → positive Sharpe
        assert out["bull"]["sharpe"] > 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="len"):
            per_regime_metrics(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.1, 0.1]),
                np.array(["bull", "bull"], dtype=object),  # wrong length
            )


@pytest.mark.fast
class TestEdgeConsistency:
    def test_robust_when_all_above_threshold(self) -> None:
        per_regime = {
            "bull": {"n_days": 100, "sharpe": 1.5},
            "bear": {"n_days": 80, "sharpe": 0.8},
            "sideways": {"n_days": 50, "sharpe": 0.6},
        }
        verdict = check_edge_consistency(per_regime, sharpe_min_in_each_regime=0.5)
        assert verdict["verdict"] == "robust"
        assert verdict["regimes_below_threshold"] == []

    def test_regime_dependent_when_one_fails(self) -> None:
        per_regime = {
            "bull": {"n_days": 100, "sharpe": 2.5},
            "bear": {"n_days": 80, "sharpe": -0.5},  # fails
        }
        verdict = check_edge_consistency(per_regime, sharpe_min_in_each_regime=0.5)
        assert "regime_dependent" in verdict["verdict"]
        assert "bear" in verdict["regimes_below_threshold"]

    def test_ignores_warmup(self) -> None:
        per_regime = {
            "bull": {"n_days": 100, "sharpe": 1.5},
            "warmup": {"n_days": 50, "sharpe": -2.0},  # ignored
        }
        verdict = check_edge_consistency(per_regime)
        assert verdict["verdict"] == "robust"

    def test_ignores_small_samples(self) -> None:
        """n_days < 20 should not factor into the verdict."""
        per_regime = {
            "bull": {"n_days": 100, "sharpe": 1.5},
            "bear": {"n_days": 10, "sharpe": -3.0},  # too few samples
        }
        verdict = check_edge_consistency(per_regime)
        assert verdict["verdict"] == "robust"
        assert "bear" not in verdict["regime_sharpes"]

    def test_insufficient_data_verdict(self) -> None:
        per_regime = {
            "bull": {"n_days": 10, "sharpe": 1.0},  # too few
        }
        verdict = check_edge_consistency(per_regime)
        assert verdict["verdict"] == "insufficient_data"


@pytest.mark.fast
class TestRunOutOfRegimeTest:
    def test_basic_run(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.001, 0.012, size=400)
        path = _make_csv(tmp_path, returns)
        report = run_out_of_regime_test(path, rolling_window=60)
        assert "input_path" in report
        assert "per_regime" in report
        assert "consistency" in report
        assert report["n_periods"] == 400

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_out_of_regime_test(tmp_path / "nope.csv")

    def test_no_equity_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="equity"):
            run_out_of_regime_test(bad)

    def test_render_markdown_includes_disclosure(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(seed=1)
        returns = rng.normal(0.001, 0.012, size=300)
        path = _make_csv(tmp_path, returns)
        report = run_out_of_regime_test(path, rolling_window=60)
        md = render_markdown(report)
        # Honesty disclosure must be in the output
        assert "self-referential" in md.lower() or "self-defined" in md.lower()
        # Verdict must appear
        assert "Verdict" in md or "verdict" in md

    def test_json_round_trip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(seed=3)
        returns = rng.normal(0.0005, 0.012, size=400)
        path = _make_csv(tmp_path, returns)
        report = run_out_of_regime_test(path, rolling_window=60)
        s = json.dumps(report)
        round_trip = json.loads(s)
        assert round_trip["n_periods"] == 400


@pytest.mark.fast
def test_helper_annualised_sharpe_zero_std() -> None:
    """Edge: constant returns → undefined Sharpe (NaN or huge)."""
    s = _annualised_sharpe(np.zeros(50))
    assert np.isnan(s)
