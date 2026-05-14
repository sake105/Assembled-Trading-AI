"""Integration tests for scripts/run_event_study.py CLI wiring.

Verifies the script actually calls the three qa/event_study.py functions
and produces the documented output artifacts (returns CSV, aggregated CSV,
Markdown report).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_event_study import (
    _df_to_md_table,
    _load_events,
    _resolve_symbols,
    run_event_study_from_args,
)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """Daily close panel for 3 symbols over 200 trading days."""
    timestamps = pd.date_range("2024-01-01", periods=200, freq="B", tz="UTC")
    rows = []
    for sym, base in [("AAA", 100.0), ("BBB", 50.0), ("CCC", 200.0)]:
        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "close": base * (1.0 + 0.001 * i),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_events(tmp_path: Path) -> Path:
    """Events CSV with 2 events on a known day in the middle of the panel."""
    events = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-04-01", tz="UTC"),
                pd.Timestamp("2024-04-15", tz="UTC"),
            ],
            "symbol": ["AAA", "BBB"],
            "event_type": ["earnings", "earnings"],
        }
    )
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)
    return path


@pytest.fixture()
def symbols_file(tmp_path: Path) -> Path:
    path = tmp_path / "symbols.txt"
    path.write_text("AAA\nBBB\nCCC\n# this is a comment\n\n", encoding="utf-8")
    return path


# ─── Helper-level tests ─────────────────────────────────────────────────


def test_load_events_csv(synthetic_events: Path) -> None:
    df = _load_events(synthetic_events)
    assert {"timestamp", "symbol", "event_type"}.issubset(df.columns)
    assert len(df) == 2


def test_load_events_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_events(tmp_path / "nope.csv")


def test_load_events_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [], "symbol": []}).to_csv(bad, index=False)
    with pytest.raises(KeyError, match="event_type"):
        _load_events(bad)


def test_load_events_unsupported_format(tmp_path: Path) -> None:
    bad = tmp_path / "events.xyz"
    bad.write_text("anything", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        _load_events(bad)


def test_resolve_symbols_from_args() -> None:
    args = argparse.Namespace(symbols=["AAPL", "MSFT"], symbols_file=None)
    out = _resolve_symbols(args, pd.DataFrame({"symbol": ["XXX"]}))
    assert out == ["AAPL", "MSFT"]


def test_resolve_symbols_from_file(symbols_file: Path) -> None:
    args = argparse.Namespace(symbols=None, symbols_file=str(symbols_file))
    out = _resolve_symbols(args, pd.DataFrame({"symbol": ["XXX"]}))
    assert out == ["AAA", "BBB", "CCC"]  # comment + blank line stripped


def test_resolve_symbols_from_events() -> None:
    args = argparse.Namespace(symbols=None, symbols_file=None)
    out = _resolve_symbols(args, pd.DataFrame({"symbol": ["BBB", "AAA", "AAA"]}))
    assert out == ["AAA", "BBB"]  # unique + sorted


def test_resolve_symbols_missing_file(tmp_path: Path) -> None:
    args = argparse.Namespace(symbols=None, symbols_file=str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError):
        _resolve_symbols(args, pd.DataFrame({"symbol": ["XXX"]}))


def test_df_to_md_table_basic() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [1.234, 5.678]})
    lines = _df_to_md_table(df)
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 1.2340 |"
    assert lines[3] == "| 2 | 5.6780 |"


def test_df_to_md_table_empty() -> None:
    assert _df_to_md_table(pd.DataFrame()) == ["(empty)"]


# ─── End-to-end CLI test with mocked price source ───────────────────────


class _MockPriceSource:
    """Stub data source returning the synthetic_prices panel."""

    def __init__(self, panel: pd.DataFrame) -> None:
        self._panel = panel

    def get_history(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        freq: str = "1d",
    ) -> pd.DataFrame:
        return self._panel[self._panel["symbol"].isin(symbols)].copy()


def test_run_event_study_end_to_end(
    tmp_path: Path,
    synthetic_events: Path,
    synthetic_prices: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI wiring runs the three qa functions and writes all artifacts."""
    # Patch the data source factory so we don't touch the real config.
    monkeypatch.setattr(
        "scripts.run_event_study.get_price_data_source",
        lambda settings, data_source=None: _MockPriceSource(synthetic_prices),
    )
    monkeypatch.setattr(
        "scripts.run_event_study.get_settings",
        lambda: object(),
    )

    out_dir = tmp_path / "study_out"
    args = argparse.Namespace(
        events_file=str(synthetic_events),
        symbols=None,
        symbols_file=None,
        data_source="local",
        freq="1d",
        start_date="2024-01-01",
        end_date="2024-09-30",
        window_before=5,
        window_after=10,
        benchmark_col=None,
        return_type="log",
        use_abnormal=False,
        confidence_level=0.95,
        output_dir=str(out_dir),
        output_csv=None,
        output_md=None,
    )

    rc = run_event_study_from_args(args)
    assert rc == 0, "run should succeed end-to-end"

    # Output artifacts exist.
    returns_csv = out_dir / "event_returns.csv"
    agg_csv = out_dir / "event_study_aggregated.csv"
    md_report = out_dir / "event_study_report.md"
    assert returns_csv.exists(), f"missing {returns_csv}"
    assert agg_csv.exists(), f"missing {agg_csv}"
    assert md_report.exists(), f"missing {md_report}"

    # Aggregated CSV has the expected qa/event_study.py columns.
    agg = pd.read_csv(agg_csv)
    for col in [
        "rel_day",
        "avg_ret",
        "std_ret",
        "n_events",
        "se",
        "ci_lower",
        "ci_upper",
        "cum_ret",
    ]:
        assert col in agg.columns, f"aggregated missing column: {col}"
    # Window is [-5, +10] = 16 days.
    assert len(agg) == 16

    # Markdown report has structural sections.
    md = md_report.read_text(encoding="utf-8")
    assert "# Event Study Report" in md
    assert "## Headline" in md
    assert "CAR on event day" in md


def test_run_event_study_missing_events_file(tmp_path: Path) -> None:
    args = argparse.Namespace(
        events_file=str(tmp_path / "nope.csv"),
        symbols=None,
        symbols_file=None,
        data_source="local",
        freq="1d",
        start_date="2024-01-01",
        end_date="2024-12-31",
        window_before=5,
        window_after=5,
        benchmark_col=None,
        return_type="log",
        use_abnormal=False,
        confidence_level=0.95,
        output_dir=str(tmp_path / "out"),
        output_csv=None,
        output_md=None,
    )
    rc = run_event_study_from_args(args)
    assert rc == 2


def test_run_event_study_empty_prices(
    tmp_path: Path,
    synthetic_events: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty price panel from data source → exit code 2."""
    monkeypatch.setattr(
        "scripts.run_event_study.get_price_data_source",
        lambda settings, data_source=None: _MockPriceSource(
            pd.DataFrame(columns=["timestamp", "symbol", "close"])
        ),
    )
    monkeypatch.setattr(
        "scripts.run_event_study.get_settings",
        lambda: object(),
    )

    args = argparse.Namespace(
        events_file=str(synthetic_events),
        symbols=None,
        symbols_file=None,
        data_source="local",
        freq="1d",
        start_date="2024-01-01",
        end_date="2024-12-31",
        window_before=5,
        window_after=5,
        benchmark_col=None,
        return_type="log",
        use_abnormal=False,
        confidence_level=0.95,
        output_dir=str(tmp_path / "out"),
        output_csv=None,
        output_md=None,
    )
    assert run_event_study_from_args(args) == 2
