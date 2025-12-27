"""Tests for Paper-Track Risk Report Integration (full workflow)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import scripts.run_paper_track as rpt
from src.assembled_core.paper.paper_track import PaperTrackDayResult, PaperTrackState

pytestmark = pytest.mark.advanced


@pytest.fixture
def mock_run_paper_day(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock run_paper_day to simulate successful runs."""

    def _mock_run(config, as_of, state_path):
        state_after = PaperTrackState(
            strategy_name=config.strategy_name,
            last_run_date=as_of,
            cash=100000.0 + as_of.day,
            equity=100000.0 + as_of.day,
            seed_capital=100000.0,
        )
        return PaperTrackDayResult(
            date=as_of,
            config=config,
            state_before=PaperTrackState(
                strategy_name=config.strategy_name, last_run_date=None
            ),
            state_after=state_after,
            orders=pd.DataFrame(),
            daily_return_pct=0.1,
            daily_pnl=100.0,
            trades_count=1,
            buy_count=1,
            sell_count=0,
            status="success",
        )

    mock = MagicMock(side_effect=_mock_run)
    monkeypatch.setattr(rpt, "run_paper_day", mock)
    return mock


@pytest.fixture
def mock_write_paper_day_outputs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock write_paper_day_outputs to create minimal files."""

    def _mock_write(result, output_dir, config):
        run_date_str = result.date.strftime("%Y%m%d")
        run_dir = output_dir / "runs" / run_date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "daily_summary.json").write_text(
            json.dumps({"date": result.date.strftime("%Y-%m-%d"), "status": "success"})
        )
        # Create aggregated equity curve for risk report
        aggregates_dir = output_dir / "aggregates"
        aggregates_dir.mkdir(parents=True, exist_ok=True)
        equity_curve = pd.DataFrame(
            [
                {
                    "date": result.date.strftime("%Y-%m-%d"),
                    "timestamp": result.date.isoformat(),
                    "equity": result.state_after.equity,
                    "cash": result.state_after.cash,
                    "positions_value": 0.0,
                    "total_pnl": 0.0,
                    "total_return_pct": 0.0,
                    "daily_return_pct": result.daily_return_pct,
                    "daily_pnl": result.daily_pnl,
                }
            ]
        )
        equity_curve.to_csv(aggregates_dir / "equity_curve.csv", index=False)

    mock = MagicMock(side_effect=_mock_write)
    monkeypatch.setattr(rpt, "write_paper_day_outputs", mock)
    return mock


@pytest.fixture
def mock_save_paper_state(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock save_paper_state."""

    def _mock_save(state, state_path):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy_name": state.strategy_name,
                    "last_run_date": state.last_run_date.isoformat(),
                    "equity": state.equity,
                },
                f,
            )

    mock = MagicMock(side_effect=_mock_save)
    monkeypatch.setattr(rpt, "save_paper_state", mock)
    return mock


@pytest.fixture
def mock_load_paper_state(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock load_paper_state."""

    def _mock_load(state_path, strategy_name):
        return None  # No existing state

    mock = MagicMock(side_effect=_mock_load)
    # load_paper_state is imported from src.assembled_core.paper.paper_track
    monkeypatch.setattr("src.assembled_core.paper.paper_track.load_paper_state", mock)
    return mock


@pytest.fixture
def paper_track_cli_config_file(tmp_path: Path) -> Path:
    """Create a minimal config file for CLI tests."""
    config_content = """
strategy_name: test_strategy
strategy_type: trend_baseline
universe:
  file: {tmp_path}/universe.txt
trading:
  freq: 1d
portfolio:
  seed_capital: 100000.0
output:
  root: {tmp_path}/output/paper_track
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content.format(tmp_path=tmp_path), encoding="utf-8")
    (tmp_path / "universe.txt").write_text("AAPL\n", encoding="utf-8")
    return config_file


def test_risk_report_weekly_trigger_on_friday(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is generated on Friday when frequency is weekly."""
    # Friday 2025-01-17
    friday = pd.Timestamp("2025-01-17", tz="UTC")
    assert friday.weekday() == 4  # Friday

    # Mock _generate_risk_report_for_paper_track to track calls
    mock_generate_risk_report_func = MagicMock()
    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", mock_generate_risk_report_func
    )

    # Run for Friday
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        as_of="2025-01-17",
        generate_risk_report=True,
        risk_report_frequency="weekly",
    )

    assert exit_code == 0
    assert mock_run_paper_day.call_count == 1
    # Risk report should be generated (Friday)
    assert mock_generate_risk_report_func.call_count == 1


def test_risk_report_weekly_no_trigger_on_monday(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is NOT generated on Monday when frequency is weekly."""
    # Monday 2025-01-13
    monday = pd.Timestamp("2025-01-13", tz="UTC")
    assert monday.weekday() == 0  # Monday

    # Mock _generate_risk_report_for_paper_track to track calls
    mock_generate_risk_report_func = MagicMock()
    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", mock_generate_risk_report_func
    )

    # Run for Monday
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        as_of="2025-01-13",
        generate_risk_report=True,
        risk_report_frequency="weekly",
    )

    assert exit_code == 0
    assert mock_run_paper_day.call_count == 1
    # Risk report should NOT be generated (not Friday)
    assert mock_generate_risk_report_func.call_count == 0


def test_risk_report_range_with_friday_triggers_once(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is generated once for a 5-day range including Friday."""
    # Range: Monday 2025-01-13 to Friday 2025-01-17
    # Should trigger on Friday only

    # Mock _generate_risk_report_for_paper_track to track calls
    mock_generate_risk_report_func = MagicMock()
    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", mock_generate_risk_report_func
    )

    # Run for range
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        start_date="2025-01-13",
        end_date="2025-01-17",
        generate_risk_report=True,
        risk_report_frequency="weekly",
    )

    assert exit_code == 0
    assert mock_run_paper_day.call_count == 5  # 5 days
    # Risk report should be generated once (on Friday)
    assert mock_generate_risk_report_func.call_count == 1


def test_risk_report_off_no_generation(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is NOT generated when generate_risk_report=False."""
    # Friday 2025-01-17
    friday = pd.Timestamp("2025-01-17", tz="UTC")
    assert friday.weekday() == 4  # Friday

    # Mock _generate_risk_report_for_paper_track to track calls
    mock_generate_risk_report_func = MagicMock()
    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", mock_generate_risk_report_func
    )

    # Run for Friday with generate_risk_report=False
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        as_of="2025-01-17",
        generate_risk_report=False,  # Disabled
        risk_report_frequency="weekly",
    )

    assert exit_code == 0
    assert mock_run_paper_day.call_count == 1
    # Risk report should NOT be generated (disabled)
    assert mock_generate_risk_report_func.call_count == 0


def test_risk_report_monthly_trigger_on_month_end(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is generated on month-end when frequency is monthly."""
    # Month-end: 2025-01-31
    month_end = pd.Timestamp("2025-01-31", tz="UTC")

    # Mock _generate_risk_report_for_paper_track to track calls
    mock_generate_risk_report_func = MagicMock()
    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", mock_generate_risk_report_func
    )

    # Run for month-end
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        as_of="2025-01-31",
        generate_risk_report=True,
        risk_report_frequency="monthly",
    )

    assert exit_code == 0
    assert mock_run_paper_day.call_count == 1
    # Risk report should be generated (month-end)
    assert mock_generate_risk_report_func.call_count == 1


def test_risk_report_daily_triggers_every_day(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is generated every day when frequency is daily."""
    # Range: Monday to Friday
    # Should trigger on all 5 days

    # Mock _generate_risk_report_for_paper_track to track calls
    mock_generate_risk_report_func = MagicMock()
    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", mock_generate_risk_report_func
    )

    # Run for range with daily frequency
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        start_date="2025-01-13",
        end_date="2025-01-17",
        generate_risk_report=True,
        risk_report_frequency="daily",
    )

    assert exit_code == 0
    assert mock_run_paper_day.call_count == 5  # 5 days
    # Risk report should be generated for all 5 days
    assert mock_generate_risk_report_func.call_count == 5


def test_risk_report_output_directory_structure(
    mock_run_paper_day: MagicMock,
    mock_write_paper_day_outputs: MagicMock,
    mock_save_paper_state: MagicMock,
    mock_load_paper_state: MagicMock,
    paper_track_cli_config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that risk report is written to correct output directory."""
    # Friday 2025-01-17
    friday = pd.Timestamp("2025-01-17", tz="UTC")

    # Mock _generate_risk_report_for_paper_track to track calls and verify output_dir
    call_args_list = []

    def _mock_generate_risk_report(*args, **kwargs):
        call_args_list.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(
        rpt, "_generate_risk_report_for_paper_track", _mock_generate_risk_report
    )

    # Run for Friday
    exit_code = rpt.run_paper_track_from_cli(
        config_file=paper_track_cli_config_file,
        as_of="2025-01-17",
        generate_risk_report=True,
        risk_report_frequency="weekly",
    )

    assert exit_code == 0
    assert len(call_args_list) == 1

    # Verify call arguments
    call_kwargs = call_args_list[0]["kwargs"]
    output_root = call_kwargs.get("output_root")
    date = call_kwargs.get("date")
    assert output_root is not None
    assert date is not None
    assert date == friday
    # Verify output_root structure (risk_reports will be created inside _generate_risk_report_for_paper_track)
    assert "paper_track" in str(output_root) or "test_strategy" in str(output_root)

