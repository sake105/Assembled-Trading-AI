"""Tests for scripts/forensic/survivorship_bias_check.py (§8.7 / C3-063)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.forensic.survivorship_bias_check import (
    KNOWN_US_DELISTINGS,
    assign_risk_level,
    check_start_date_clustering,
    compute_active_delisted_ratio,
    cross_check_known_delistings,
    render_markdown,
    run_survivorship_check,
)


def _make_watchlist_csv(
    tmp_path: Path,
    symbols: list[str],
    statuses: list[str] | None = None,
    start_dates: list[str] | None = None,
) -> Path:
    n = len(symbols)
    rows = {"symbol": symbols}
    if statuses is not None:
        rows["status"] = statuses
    if start_dates is not None:
        rows["start_date"] = start_dates
    p = tmp_path / "wl.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Indicator 1: active/delisted ratio
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestActiveDelistedRatio:
    def test_all_active(self) -> None:
        df = pd.DataFrame({"symbol": ["A", "B", "C"], "status": ["active"] * 3})
        result = compute_active_delisted_ratio(df)
        assert result["pct_active"] == 100.0
        assert result["n_delisted"] == 0

    def test_mixed(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "status": ["active", "active", "delisted", "delisted"],
            }
        )
        result = compute_active_delisted_ratio(df)
        assert result["pct_active"] == 50.0
        assert result["n_delisted"] == 2

    def test_no_status_column(self) -> None:
        df = pd.DataFrame({"symbol": ["A", "B"]})
        result = compute_active_delisted_ratio(df)
        assert "warning" in result
        assert result["n_active"] is None

    def test_empty(self) -> None:
        df = pd.DataFrame({"symbol": []})
        result = compute_active_delisted_ratio(df)
        assert result["n_total"] == 0


# ---------------------------------------------------------------------------
# Indicator 2: known delistings cross-check
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestCrossCheckKnownDelistings:
    def test_lehman_missing_in_2008_window(self) -> None:
        df = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})  # no LEH
        result = cross_check_known_delistings(
            df,
            pd.Timestamp("2008-01-01", tz="UTC"),
            pd.Timestamp("2008-12-31", tz="UTC"),
        )
        # LEH 2008-09-15 is in window
        missing_syms = [m["symbol"] for m in result["missing_delistings"]]
        assert "LEH" in missing_syms

    def test_event_outside_window_not_counted(self) -> None:
        df = pd.DataFrame({"symbol": ["AAPL"]})
        result = cross_check_known_delistings(
            df,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-12-31", tz="UTC"),
        )
        # All known delistings are pre-2024 — should be 0 in window
        assert result["n_events_in_window"] == 0
        assert result["n_missing"] == 0

    def test_present_symbol_not_flagged(self) -> None:
        df = pd.DataFrame({"symbol": ["LEH", "AAPL"]})
        result = cross_check_known_delistings(
            df,
            pd.Timestamp("2008-01-01", tz="UTC"),
            pd.Timestamp("2008-12-31", tz="UTC"),
        )
        missing_syms = [m["symbol"] for m in result["missing_delistings"]]
        assert "LEH" not in missing_syms

    def test_no_symbol_column(self) -> None:
        df = pd.DataFrame({"foo": ["bar"]})
        result = cross_check_known_delistings(
            df,
            pd.Timestamp("2008-01-01", tz="UTC"),
            pd.Timestamp("2024-12-31", tz="UTC"),
        )
        # No symbols at all → all known events flagged as missing
        assert result["n_missing"] == result["n_events_in_window"]


# ---------------------------------------------------------------------------
# Indicator 3: start-date clustering
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestStartDateClustering:
    def test_single_clustered_date(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E", "F"],
                "start_date": ["2008-09-02"] * 6,
            }
        )
        result = check_start_date_clustering(df)
        assert result["n_unique_start_dates"] == 1
        assert result["clustering_signal"] is True

    def test_varied_dates_no_signal(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "start_date": [
                    "2008-09-02",
                    "2010-03-15",
                    "2012-08-01",
                    "2015-01-10",
                    "2020-06-22",
                ],
            }
        )
        result = check_start_date_clustering(df)
        assert result["n_unique_start_dates"] == 5
        assert result["clustering_signal"] is False

    def test_no_start_date_column(self) -> None:
        df = pd.DataFrame({"symbol": ["A"]})
        result = check_start_date_clustering(df)
        assert "warning" in result


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAssignRiskLevel:
    def test_no_flags_low(self) -> None:
        result = assign_risk_level(
            ratio={"pct_active": 90.0},
            cross_check={"n_missing": 0},
            clustering={"clustering_signal": False},
        )
        assert result["risk_level"] == "low"
        assert result["n_flags"] == 0

    def test_one_flag_medium(self) -> None:
        result = assign_risk_level(
            ratio={"pct_active": 100.0},  # flag
            cross_check={"n_missing": 0},
            clustering={"clustering_signal": False},
        )
        assert result["risk_level"] == "medium"
        assert result["n_flags"] == 1

    def test_all_three_flags_high(self) -> None:
        result = assign_risk_level(
            ratio={"pct_active": 100.0},
            cross_check={"n_missing": 5},
            clustering={
                "clustering_signal": True,
                "most_common_count": 19,
                "most_common_start_date": "2008-09-02",
            },
        )
        assert result["risk_level"] == "high"
        assert result["n_flags"] == 3


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRunSurvivorshipCheck:
    def test_high_risk_baseline_repro(self, tmp_path: Path) -> None:
        """Reproduce the 'high' verdict on the real watchlist pattern:
        all-active + clustered start_date + no delisted names."""
        path = _make_watchlist_csv(
            tmp_path,
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
            statuses=["active"] * 6,
            start_dates=["2008-09-02"] * 6,
        )
        report = run_survivorship_check(
            path, expected_window_start="2007-01-01", expected_window_end="2026-12-31"
        )
        assert report["verdict"]["risk_level"] == "high"
        assert report["verdict"]["n_flags"] >= 2

    def test_low_risk_with_diverse_universe(self, tmp_path: Path) -> None:
        """A universe with mixed status + varied start_dates + only the
        known delisting that's in the window included should NOT trigger
        any flag. Window Jan-Mar 2008: only BSC (2008-03-17) is in scope
        and BSC is in the watchlist."""
        path = _make_watchlist_csv(
            tmp_path,
            symbols=["AAPL", "BSC", "MSFT", "GOOGL"],
            statuses=["active", "delisted", "active", "active"],
            start_dates=[
                "2008-09-02",
                "2003-11-08",
                "2010-01-10",
                "2015-04-22",
            ],
        )
        # Window: only BSC (Mar 17) in scope; BSC present in watchlist.
        # pct_active 75% (under 99%) → no flag-1.
        # BSC present → no flag-2.
        # Varied start_dates → no flag-3.
        # Verdict = low.
        report = run_survivorship_check(
            path,
            expected_window_start="2008-01-01",
            expected_window_end="2008-03-31",
        )
        assert report["verdict"]["risk_level"] == "low"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_survivorship_check(tmp_path / "nope.csv")

    def test_no_symbol_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": ["bar"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="symbol"):
            run_survivorship_check(bad)

    def test_real_watchlist_high_verdict(self) -> None:
        """Smoke against the actual repo watchlist — heute eindeutig 'high'.
        Wenn das je auf 'low' fällt, hat sich das Universum echt geändert
        (echte CRSP-Daten eingebaut → echter PIT-Universe) und der Test
        sollte dann angepasst werden."""
        path = Path("data/universe/watchlist_2007_2026.csv")
        if not path.exists():
            pytest.skip("real watchlist not present in test env")
        report = run_survivorship_check(path)
        assert report["verdict"]["risk_level"] == "high"

    def test_json_round_trip(self, tmp_path: Path) -> None:
        path = _make_watchlist_csv(
            tmp_path,
            symbols=["AAPL", "MSFT"],
            statuses=["active", "active"],
            start_dates=["2008-09-02"] * 2,
        )
        report = run_survivorship_check(path)
        s = json.dumps(report)
        rt = json.loads(s)
        assert rt["verdict"]["risk_level"] in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRenderMarkdown:
    def test_markdown_includes_verdict_and_indicators(self, tmp_path: Path) -> None:
        path = _make_watchlist_csv(
            tmp_path,
            symbols=["AAPL", "MSFT"],
            statuses=["active", "active"],
            start_dates=["2008-09-02"] * 2,
        )
        report = run_survivorship_check(path)
        md = render_markdown(report)
        assert "Survivorship-Bias-Check" in md
        assert "Verdict" in md
        assert "Indicator 1" in md
        assert "Indicator 2" in md
        assert "Indicator 3" in md
        assert "Limitations" in md  # honesty disclosure must surface


# ---------------------------------------------------------------------------
# Hardcoded sample sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_known_delistings_sample_is_non_empty() -> None:
    """The hardcoded sample of known US delistings should contain at least
    the canonical Lehman + Bear Stearns + Sears + JCPenney entries."""
    syms = {d.symbol for d in KNOWN_US_DELISTINGS}
    assert "LEH" in syms
    assert "BSC" in syms
    assert "SHLD" in syms
    assert "JCP" in syms
