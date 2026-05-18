"""Tests for scripts/ops/check_tax_loss_harvest.py (C2-064)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.ops.check_tax_loss_harvest import (
    analyse_realized_pnl,
    compute_offset_potential,
    find_harvesting_candidates,
    render_markdown,
    run_tax_loss_check,
)


# ---------------------------------------------------------------------------
# analyse_realized_pnl
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAnalyseRealizedPnl:
    def test_mixed_gains_losses(self) -> None:
        ledger = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "close_date": [
                    "2025-03-15",
                    "2025-06-22",
                    "2025-08-30",
                    "2025-11-10",
                ],
                "pnl_eur": [1000.0, -500.0, 2000.0, -300.0],
            }
        )
        result = analyse_realized_pnl(ledger, tax_year=2025)
        assert result["total_gains_eur"] == 3000.0
        assert result["total_losses_eur"] == -800.0
        assert result["net_pnl_eur"] == 2200.0
        assert result["n_winning_trades"] == 2
        assert result["n_losing_trades"] == 2

    def test_only_target_year(self) -> None:
        ledger = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "close_date": ["2024-12-31", "2025-01-02"],
                "pnl_eur": [1000.0, 2000.0],
            }
        )
        result = analyse_realized_pnl(ledger, tax_year=2025)
        # Only the 2025 trade counts
        assert result["total_gains_eur"] == 2000.0
        assert result["n_winning_trades"] == 1

    def test_empty_ledger(self) -> None:
        ledger = pd.DataFrame(columns=["symbol", "close_date", "pnl_eur"])
        result = analyse_realized_pnl(ledger, tax_year=2025)
        assert result["total_gains_eur"] == 0.0
        assert result["total_losses_eur"] == 0.0
        assert result["n_winning_trades"] == 0

    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="close_date"):
            analyse_realized_pnl(bad, tax_year=2025)

    def test_by_symbol_breakdown(self) -> None:
        ledger = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "close_date": ["2025-01-01", "2025-06-01", "2025-08-01"],
                "pnl_eur": [500.0, 300.0, -200.0],
            }
        )
        result = analyse_realized_pnl(ledger, tax_year=2025)
        assert result["by_symbol"]["AAPL"] == 800.0
        assert result["by_symbol"]["MSFT"] == -200.0


# ---------------------------------------------------------------------------
# find_harvesting_candidates
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestFindHarvestingCandidates:
    def test_only_losers_returned(self) -> None:
        positions = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "qty": [10, 10, 10],
                "entry_price": [100.0, 50.0, 200.0],
                "current_price": [120.0, 45.0, 180.0],  # gain, loss, loss
            }
        )
        candidates = find_harvesting_candidates(positions)
        symbols = [c["symbol"] for c in candidates]
        assert "A" not in symbols  # winner excluded
        assert "B" in symbols
        assert "C" in symbols

    def test_sorted_by_worst_loss_first(self) -> None:
        positions = pd.DataFrame(
            {
                "symbol": ["small_loss", "big_loss"],
                "qty": [10, 10],
                "entry_price": [100.0, 100.0],
                "current_price": [95.0, 50.0],
            }
        )
        candidates = find_harvesting_candidates(positions)
        assert candidates[0]["symbol"] == "big_loss"
        assert candidates[1]["symbol"] == "small_loss"

    def test_unrealized_eur_correct(self) -> None:
        positions = pd.DataFrame(
            {
                "symbol": ["X"],
                "qty": [10.0],
                "entry_price": [100.0],
                "current_price": [80.0],
            }
        )
        candidates = find_harvesting_candidates(positions)
        # (80-100)*10 = -200
        assert candidates[0]["unrealized_eur"] == -200.0

    def test_fx_conversion(self) -> None:
        positions = pd.DataFrame(
            {
                "symbol": ["X"],
                "qty": [10.0],
                "entry_price": [100.0],
                "current_price": [80.0],
            }
        )
        # USD → EUR at 0.9
        candidates = find_harvesting_candidates(positions, fx_rate_usd_eur=0.9)
        assert candidates[0]["unrealized_eur"] == -180.0  # -200 * 0.9

    def test_empty_positions(self) -> None:
        positions = pd.DataFrame(
            columns=["symbol", "qty", "entry_price", "current_price"]
        )
        candidates = find_harvesting_candidates(positions)
        assert candidates == []

    def test_missing_column_raises(self) -> None:
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            find_harvesting_candidates(bad)

    def test_entry_price_zero_does_not_crash(self) -> None:
        """F-senior-1: entry_price=0 (corp-action artefact / bad CSV) must
        not raise ZeroDivisionError. pct_loss becomes NaN; the position
        stays in the candidate list because unrealized_eur is still
        well-defined."""
        import numpy as np

        positions = pd.DataFrame(
            {
                "symbol": ["BAD"],
                "qty": [10.0],
                "entry_price": [0.0],
                "current_price": [-1.0],  # contrived loss
            }
        )
        candidates = find_harvesting_candidates(positions)
        assert len(candidates) == 1
        assert candidates[0]["symbol"] == "BAD"
        assert np.isnan(candidates[0]["pct_loss"])


# ---------------------------------------------------------------------------
# compute_offset_potential
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeOffsetPotential:
    def test_zero_realized_net_zero_target(self) -> None:
        realized = {"total_gains_eur": 100.0, "total_losses_eur": -100.0}
        result = compute_offset_potential(realized, [])
        assert result["realized_net_pnl_eur"] == 0.0
        assert result["target_loss_to_offset_gains_eur"] == 0.0

    def test_positive_net_targets_match(self) -> None:
        realized = {"total_gains_eur": 1000.0, "total_losses_eur": -200.0}
        # net = 800; target_loss = 800
        result = compute_offset_potential(realized, [])
        assert result["target_loss_to_offset_gains_eur"] == 800.0

    def test_negative_net_target_zero(self) -> None:
        """If we already have net losses, no harvest is needed."""
        realized = {"total_gains_eur": 200.0, "total_losses_eur": -500.0}
        result = compute_offset_potential(realized, [])
        assert result["target_loss_to_offset_gains_eur"] == 0.0

    def test_cumulative_path_meets_target(self) -> None:
        realized = {"total_gains_eur": 1000.0, "total_losses_eur": 0.0}
        candidates = [
            {"symbol": "BIG", "unrealized_eur": -800.0},
            {"symbol": "SMALL", "unrealized_eur": -300.0},
        ]
        result = compute_offset_potential(realized, candidates)
        # First candidate alone: cumulative -800, target -1000 → not enough
        # Second: cumulative -1100, target -1000 → meets
        assert result["min_n_positions_to_neutralise"] == 2

    def test_cumulative_path_first_alone_enough(self) -> None:
        realized = {"total_gains_eur": 500.0, "total_losses_eur": 0.0}
        candidates = [
            {"symbol": "HUGE", "unrealized_eur": -1000.0},
        ]
        result = compute_offset_potential(realized, candidates)
        assert result["min_n_positions_to_neutralise"] == 1


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRunPipeline:
    def test_basic_pipeline(self, tmp_path: Path) -> None:
        ledger = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "close_date": ["2025-03-15", "2025-08-30"],
                "pnl_eur": [1000.0, -200.0],
            }
        )
        positions = pd.DataFrame(
            {
                "symbol": ["GOOGL", "AMZN"],
                "qty": [10, 5],
                "entry_price": [150.0, 200.0],
                "current_price": [140.0, 210.0],  # GOOGL loser, AMZN winner
            }
        )
        ledger_path = tmp_path / "ledger.csv"
        positions_path = tmp_path / "positions.csv"
        ledger.to_csv(ledger_path, index=False)
        positions.to_csv(positions_path, index=False)
        report = run_tax_loss_check(ledger_path, positions_path, tax_year=2025)
        assert report["tax_year"] == 2025
        assert "realized_pnl" in report
        assert "harvesting_candidates" in report
        assert "offset_potential" in report
        # GOOGL should be a candidate, AMZN should not
        symbols = [c["symbol"] for c in report["harvesting_candidates"]]
        assert "GOOGL" in symbols
        assert "AMZN" not in symbols

    def test_missing_files_graceful(self, tmp_path: Path) -> None:
        report = run_tax_loss_check(
            tmp_path / "nope_ledger.csv",
            tmp_path / "nope_positions.csv",
            tax_year=2025,
        )
        assert "missing:" in report["ledger_status"]
        assert "missing:" in report["positions_status"]
        assert report["harvesting_candidates"] == []

    def test_json_round_trip(self, tmp_path: Path) -> None:
        ledger = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "close_date": ["2025-01-01"],
                "pnl_eur": [500.0],
            }
        )
        positions = pd.DataFrame(
            {
                "symbol": ["BAD"],
                "qty": [10],
                "entry_price": [100.0],
                "current_price": [50.0],
            }
        )
        ledger_path = tmp_path / "l.csv"
        positions_path = tmp_path / "p.csv"
        ledger.to_csv(ledger_path, index=False)
        positions.to_csv(positions_path, index=False)
        report = run_tax_loss_check(ledger_path, positions_path, tax_year=2025)
        s = json.dumps(report)
        rt = json.loads(s)
        assert rt["tax_year"] == 2025


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_render_markdown_includes_sections(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "close_date": ["2025-06-01"],
            "pnl_eur": [1000.0],
        }
    )
    positions = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "qty": [10],
            "entry_price": [200.0],
            "current_price": [150.0],
        }
    )
    ledger_path = tmp_path / "l.csv"
    positions_path = tmp_path / "p.csv"
    ledger.to_csv(ledger_path, index=False)
    positions.to_csv(positions_path, index=False)
    report = run_tax_loss_check(ledger_path, positions_path, tax_year=2025)
    md = render_markdown(report)
    assert "Tax-Loss-Harvesting Report" in md
    assert "Realised P&L" in md
    assert "Offset Potential" in md
    assert "Harvesting Candidates" in md
    assert "Limitations" in md
    assert "MSFT" in md  # losing position should be in the table
