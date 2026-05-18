"""Tests for scripts/forensic/equity_curve_audit.py (C3-030)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.forensic.equity_curve_audit import (
    _annualised_sharpe,
    _cagr,
    _drawdown_duration_distribution,
    _max_drawdown,
    audit_equity_curve,
    render_markdown,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic equity curves
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, returns: np.ndarray) -> Path:
    """Build a minimal equity_curve CSV from a returns array."""
    equity = 100_000.0 * np.cumprod(1.0 + returns)
    dates = pd.date_range("2024-01-01", periods=len(returns), tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "timestamp": dates,
            "equity": equity,
            "daily_return": returns,
            "cash": equity,
        }
    )
    p = tmp_path / "test_equity.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestStatisticalHelpers:
    def test_annualised_sharpe_zero_for_no_excess(self) -> None:
        # All-zero returns → undefined Sharpe (NaN), guard.
        returns = np.zeros(100)
        s = _annualised_sharpe(returns)
        assert np.isnan(s)

    def test_annualised_sharpe_constant_returns_inf_or_nan(self) -> None:
        """Constant returns produce std ≈ 0 (FP noise possible). Sharpe is
        either huge (FP-noise std) or NaN — both signal "undefined". Just
        check it is NOT a normal finite value."""
        returns = np.full(100, 0.001)
        s = _annualised_sharpe(returns)
        # Result is either NaN or extremely large — both ≠ a sane Sharpe.
        assert np.isnan(s) or abs(s) > 1e6

    def test_annualised_sharpe_noisy_positive(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=500)
        s = _annualised_sharpe(returns)
        # Drift 0.001/0.01 * sqrt(252) ≈ 1.59 — should be near that
        assert 1.0 < s < 2.5

    def test_max_drawdown_negative(self) -> None:
        equity = np.array([100, 110, 105, 90, 95, 120])
        mdd, duration, peak = _max_drawdown(equity)
        # peak at 110, trough at 90 ⇒ MDD = (90-110)/110 = -0.1818
        assert mdd == pytest.approx(-0.18181818, abs=1e-6)
        assert duration == 2  # 110 (idx=1) to 90 (idx=3)

    def test_cagr_for_doubled_equity_one_year(self) -> None:
        # 252 daily returns, equity doubles → CAGR ≈ 100% per year
        equity = np.linspace(100, 200, 252)
        c = _cagr(equity, periods_per_year=252)
        assert abs(c - 1.0) < 0.05

    def test_drawdown_duration_distribution(self) -> None:
        # Equity walks down for 4 days then recovers; harness counts
        # contiguous under-running-max periods.
        equity = np.array([100, 110, 105, 102, 108, 115, 110, 108, 105, 102, 120])
        dist = _drawdown_duration_distribution(equity)
        assert dist["count"] >= 1
        # At least one drawdown episode of length >= 3 days
        assert dist["max"] >= 3


# ---------------------------------------------------------------------------
# Full audit pipeline
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAuditPipeline:
    def test_audit_basic_synthetic(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(seed=2024)
        returns = rng.normal(0.0008, 0.012, size=500)
        path = _make_csv(tmp_path, returns)
        report = audit_equity_curve(path, n_bootstrap=50)
        assert report["n_periods"] == 500
        assert "sharpe" in report
        assert "cagr" in report
        assert "max_drawdown_pct" in report
        assert "returns_distribution" in report
        assert "probabilistic_sharpe_ratio" in report

    def test_audit_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            audit_equity_curve(tmp_path / "nonexistent.csv")

    def test_audit_no_equity_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="equity"):
            audit_equity_curve(bad)

    def test_render_markdown_returns_string(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(seed=1)
        returns = rng.normal(0.001, 0.012, size=300)
        path = _make_csv(tmp_path, returns)
        report = audit_equity_curve(path, n_bootstrap=50)
        md = render_markdown(report)
        assert isinstance(md, str)
        assert "Sharpe" in md
        assert "CAGR" in md
        assert "Bootstrap" in md

    def test_audit_json_serialisable(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(seed=3)
        returns = rng.normal(0.0005, 0.012, size=400)
        path = _make_csv(tmp_path, returns)
        report = audit_equity_curve(path, n_bootstrap=50)
        # Must round-trip through JSON
        s = json.dumps(report)
        round_trip = json.loads(s)
        assert round_trip["n_periods"] == 400


# ---------------------------------------------------------------------------
# Negative control: a flat equity curve must NOT show edge
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_flat_returns_yield_undefined_sharpe(tmp_path: Path) -> None:
    """Equity with zero returns (NaN-producing constant equity) is the
    canonical edge case — Sharpe must be NaN, MDD = 0, and the audit must
    not crash."""
    returns = np.zeros(200)
    path = _make_csv(tmp_path, returns)
    report = audit_equity_curve(path, n_bootstrap=50)
    assert np.isnan(report["sharpe"]) or report["sharpe"] == 0.0
    assert report["max_drawdown_pct"] == 0.0
