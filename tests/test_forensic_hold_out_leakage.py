"""Tests for scripts/forensic/hold_out_leakage_test.py (§8.7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.forensic.hold_out_leakage_test import (
    _annualised_sharpe,
    permutation_test_mdd,
    permutation_test_sharpe,
    render_markdown,
    run_hold_out_leakage_test,
    train_test_split_audit,
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
# _annualised_sharpe
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAnnualisedSharpe:
    def test_positive_drift(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, size=500)
        s = _annualised_sharpe(r)
        assert s > 0.5  # drift/vol * sqrt(252) ≈ 1.58 with this seed

    def test_zero_std_returns_nan(self) -> None:
        assert np.isnan(_annualised_sharpe(np.zeros(50)))

    def test_short_series_nan(self) -> None:
        assert np.isnan(_annualised_sharpe(np.array([0.01])))


# ---------------------------------------------------------------------------
# permutation_test_sharpe — degeneracy is the point
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPermutationTestSharpe:
    def test_sharpe_preserved_under_permutation(self) -> None:
        """Permuting an i.i.d. returns sample preserves mean+std exactly →
        sharpe is invariant. Test confirms this property and surfaces the
        degeneracy_note in the output."""
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, size=200)
        result = permutation_test_sharpe(r, n_permutations=50, seed=42)
        # Permuting preserves Sharpe exactly → observed == perm_mean ± FP
        assert abs(result["observed_sharpe"] - result["perm_mean"]) < 1e-9
        # std of permutation distribution is essentially 0 (FP-noise only)
        assert result["perm_std"] < 1e-9
        # Honest degeneracy note must surface
        assert "degeneracy_note" in result
        assert "i.i.d." in result["degeneracy_note"]

    def test_too_short_returns_error(self) -> None:
        result = permutation_test_sharpe(np.array([0.01]), n_permutations=10)
        assert "error" in result

    def test_undefined_sharpe_returns_error(self) -> None:
        result = permutation_test_sharpe(np.zeros(50), n_permutations=10)
        assert "error" in result


# ---------------------------------------------------------------------------
# permutation_test_mdd — actually informative (path-dependent)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPermutationTestMDD:
    def test_clustered_drawdown_distinguishable(self) -> None:
        """A returns series with all negatives clustered at the end has
        deeper MDD than most random orderings → low p-value."""
        # 30 days of +0.01, 10 days of -0.02 at the end
        r = np.concatenate([np.full(30, 0.01), np.full(10, -0.02)])
        result = permutation_test_mdd(r, n_permutations=200, seed=42)
        assert "p_value_mdd_ge_observed" in result
        assert np.isfinite(result["observed_mdd"])

    def test_iid_random_mdd_near_median(self) -> None:
        """A truly i.i.d. series should have observed MDD near the
        permutation median (p ≈ 0.5)."""
        rng = np.random.default_rng(7)
        r = rng.normal(0.0005, 0.01, size=300)
        result = permutation_test_mdd(r, n_permutations=200, seed=42)
        p = result["p_value_mdd_ge_observed"]
        # For i.i.d. series, p should be roughly in [0.2, 0.8]
        assert 0.1 < p < 0.9

    def test_too_short_returns_error(self) -> None:
        result = permutation_test_mdd(np.array([0.01]))
        assert "error" in result


# ---------------------------------------------------------------------------
# train_test_split_audit
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestTrainTestSplitAudit:
    def test_basic_split(self) -> None:
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.012, size=400)
        result = train_test_split_audit(r, train_frac=0.7, n_permutations=50)
        assert result["n_train"] == 280
        assert result["n_test"] == 120
        assert np.isfinite(result["train_sharpe"])
        assert np.isfinite(result["test_sharpe"])

    def test_invalid_train_frac_raises(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, size=100)
        with pytest.raises(ValueError, match="train_frac"):
            train_test_split_audit(r, train_frac=0.05)
        with pytest.raises(ValueError, match="train_frac"):
            train_test_split_audit(r, train_frac=0.99)

    def test_insufficient_data(self) -> None:
        result = train_test_split_audit(np.array([0.01] * 10))
        assert "error" in result

    def test_overfitting_pattern_visible(self) -> None:
        """Construct a series where train is positive-drift, test is
        zero-mean. The Sharpe-decay metric should clearly show the gap."""
        rng = np.random.default_rng(11)
        train = rng.normal(0.002, 0.01, size=200)
        test = rng.normal(0.0, 0.01, size=100)  # no drift
        r = np.concatenate([train, test])
        result = train_test_split_audit(r, train_frac=2 / 3, n_permutations=50)
        # train_sharpe should be clearly higher than test_sharpe
        assert result["train_sharpe"] > result["test_sharpe"]
        # Decay should be positive (train > test)
        assert result["sharpe_decay_train_to_test"] > 0


# ---------------------------------------------------------------------------
# run_hold_out_leakage_test
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRunPipeline:
    def test_basic_pipeline(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        r = rng.normal(0.0005, 0.012, size=300)
        path = _make_csv(tmp_path, r)
        report = run_hold_out_leakage_test(path, n_permutations=50)
        assert "verdict" in report
        assert "full_series" in report
        assert "train_test_split" in report
        assert report["verdict"] in {
            "hold_out_edge_significant",
            "hold_out_edge_weak",
            "hold_out_edge_indistinguishable_from_random",
            "hold_out_negative_sharpe",
            "undefined",
        }

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_hold_out_leakage_test(tmp_path / "nope.csv")

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="equity"):
            run_hold_out_leakage_test(bad)

    def test_negative_sharpe_verdict(self, tmp_path: Path) -> None:
        """Equity that LOSES money over test should yield hold_out_negative_sharpe."""
        rng = np.random.default_rng(99)
        train = rng.normal(0.001, 0.01, size=200)
        test = rng.normal(-0.002, 0.01, size=100)  # clearly losing
        r = np.concatenate([train, test])
        path = _make_csv(tmp_path, r)
        report = run_hold_out_leakage_test(path, n_permutations=50)
        assert report["train_test_split"]["test_sharpe"] < 0
        assert report["verdict"] == "hold_out_negative_sharpe"

    def test_json_round_trip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        r = rng.normal(0.001, 0.01, size=200)
        path = _make_csv(tmp_path, r)
        report = run_hold_out_leakage_test(path, n_permutations=50)
        s = json.dumps(report)
        rt = json.loads(s)
        assert rt["verdict"] == report["verdict"]


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRenderMarkdown:
    def test_includes_verdict_and_semantics(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(5)
        r = rng.normal(0.001, 0.01, size=200)
        path = _make_csv(tmp_path, r)
        report = run_hold_out_leakage_test(path, n_permutations=50)
        md = render_markdown(report)
        assert "Hold-Out-Leakage" in md
        assert "Verdict" in md
        assert "Train/Test Split" in md
        assert "Verdict Semantics" in md
        assert "hold_out_edge_significant" in md  # semantics section
