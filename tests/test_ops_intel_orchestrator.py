"""Tests for OPS-11 intel orchestrator and paper_runner real/sim wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.assembled_core.ops.intel_orchestrator import run_intel_pipelines
from src.assembled_core.ops.kpi_artifacts import write_run_kpis

pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_intel_orchestrator_real_mode_calls_pipelines() -> None:
    """When mode=real and run_news/run_disclosures true, both pipelines are called and statuses propagated."""
    news_result = {"health": Mock(status="OK")}
    disc_result = {"health": Mock(status="DEGRADED")}
    app_cfg: dict[str, Any] = {
        "paper_runner": {
            "intel": {
                "mode": "real",
                "run_news_pipeline": True,
                "run_disclosures_pipeline": True,
                "news": {
                    "sources_path": "cfg/news/sources.yaml",
                    "config_path": "cfg/news/news.yaml",
                    "cadence": "hourly",
                    "output_dir": "out/news",
                },
                "disclosures": {
                    "sources_path": "cfg/disc/sources.yaml",
                    "config_path": "cfg/disc/disc.yaml",
                    "cadence": "daily",
                    "output_dir": "out/disc",
                },
            },
        },
    }
    root = Path("/tmp/ops11_test")
    with patch(
        "src.assembled_core.events.news.run_news_pipeline", return_value=news_result
    ) as m_news:
        with patch(
            "src.assembled_core.events.disclosures.run_disclosures_pipeline",
            return_value=disc_result,
        ) as m_disc:
            out = run_intel_pipelines(app_cfg, root=root)
    assert out["news"]["ran"] is True
    assert out["news"]["status"] == "OK"
    assert out["disclosures"]["ran"] is True
    assert out["disclosures"]["status"] == "DEGRADED"
    assert m_news.call_count == 1
    assert m_disc.call_count == 1
    m_news.assert_called_once()
    call_kw = m_news.call_args[1]
    assert call_kw.get("output_dir") == str(root / "out/news")
    call_disc_kw = m_disc.call_args[1]
    assert call_disc_kw.get("output_dir") == str(root / "out/disc")


def test_intel_orchestrator_uses_news_and_disclosures_output_dir_override() -> None:
    """NEWS-DEBUG-52: App overrides for paper_runner.intel.news.output_dir and disclosures.output_dir are used."""
    app_cfg: dict[str, Any] = {
        "paper_runner": {
            "intel": {
                "mode": "real",
                "run_news_pipeline": True,
                "run_disclosures_pipeline": True,
                "news": {"output_dir": "output/runs/_experiments/exp1/intel/news"},
                "disclosures": {
                    "output_dir": "output/runs/_experiments/exp1/intel/disclosures"
                },
            },
        },
    }
    root = Path("/repo")
    with patch(
        "src.assembled_core.events.news.run_news_pipeline",
        return_value={"health": Mock(status="OK")},
    ) as m_news:
        with patch(
            "src.assembled_core.events.disclosures.run_disclosures_pipeline",
            return_value={"health": Mock(status="OK")},
        ) as m_disc:
            run_intel_pipelines(app_cfg, root=root)
    assert m_news.call_count == 1
    assert m_disc.call_count == 1
    assert m_news.call_args[1]["output_dir"] == str(
        root / "output/runs/_experiments/exp1/intel/news"
    )
    assert m_disc.call_args[1]["output_dir"] == str(
        root / "output/runs/_experiments/exp1/intel/disclosures"
    )
    """Pipeline exceptions are caught and reported as ERROR status."""
    app_cfg = {
        "paper_runner": {
            "intel": {
                "mode": "real",
                "run_news_pipeline": True,
                "run_disclosures_pipeline": False,
                "news": {
                    "sources_path": "n",
                    "config_path": "n",
                    "cadence": "hourly",
                    "output_dir": "out",
                },
            },
        },
    }
    with patch(
        "src.assembled_core.events.news.run_news_pipeline",
        side_effect=RuntimeError("network error"),
    ):
        out = run_intel_pipelines(app_cfg, root=Path("/tmp"))
    assert out["news"]["ran"] is True
    assert out["news"]["status"] == "ERROR"
    assert out["disclosures"]["status"] == "SKIPPED"


def test_intel_orchestrator_non_real_returns_skipped() -> None:
    """When mode is not 'real', both return SKIPPED and no pipelines run."""
    app_cfg = {"paper_runner": {"intel": {"mode": "sim"}}}
    with patch("src.assembled_core.events.news.run_news_pipeline") as m_news:
        with patch(
            "src.assembled_core.events.disclosures.run_disclosures_pipeline"
        ) as m_disc:
            out = run_intel_pipelines(app_cfg)
    assert out["news"]["status"] == "SKIPPED"
    assert out["news"]["ran"] is False
    assert out["disclosures"]["status"] == "SKIPPED"
    assert out["disclosures"]["ran"] is False
    m_news.assert_not_called()
    m_disc.assert_not_called()


def test_run_paper_daily_one_real_mode_skips_intel_sim() -> None:
    """When paper_runner.intel.mode=real, apply_intel_sim is not called."""
    from src.assembled_core.ops.paper_runner import run_paper_daily_one

    app_cfg: dict[str, Any] = {
        "paper_runner": {
            "intel": {
                "mode": "real",
                "run_news_pipeline": False,
                "run_disclosures_pipeline": False,
            },
            "intel_sim": {
                "enabled": True,
                "mode": "stress_based",
                "disclosures_confirm_every_n_days": 5,
            },
            "ledger_path": None,
            "strategy": {"name": "none"},
        },
    }
    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-06-26", tz="UTC")],
            "symbol": ["X"],
            "close": [100.0],
        }
    )
    as_of = pd.Timestamp("2025-06-26", tz="UTC")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "src.assembled_core.ops.intel_orchestrator.run_intel_pipelines",
            lambda *a, **k: {
                "news": {"ran": False, "status": "SKIPPED"},
                "disclosures": {"ran": False, "status": "SKIPPED"},
            },
        )
        apply_sim_mock = Mock()
        m.setattr("src.assembled_core.ops.intel_sim.apply_intel_sim", apply_sim_mock)
        with temp_dir() as tmp:
            out_dir = tmp / "runs" / "2025-06-26"
            out_dir.mkdir(parents=True)
            ledger_dir = tmp / "ledger"
            ledger_dir.mkdir(parents=True)
            (ledger_dir / "ledger_state.json").write_text(
                '{"cash": 10000, "positions": {}, "equity_curve": []}'
            )
            app_cfg["paper_runner"]["ledger_path"] = str(
                ledger_dir / "ledger_state.json"
            )
            exit_code, _ = run_paper_daily_one(
                as_of, out_dir, "paper", app_cfg, prices, root=tmp, day_index=0
            )
        assert exit_code == 0
        apply_sim_mock.assert_not_called()


def test_run_kpis_contains_intel_orchestration() -> None:
    """write_run_kpis includes intel_orchestration from result.meta."""

    class Ctx:
        risk_state = None
        news_geo = None
        market_stress = None

    class Result:
        meta = {
            "intel_orchestration": {
                "news": {"ran": True, "status": "OK"},
                "disclosures": {"ran": False, "status": "SKIPPED"},
            }
        }
        target_positions = pd.DataFrame()

    with pytest.MonkeyPatch.context() as m:
        m.setattr(
            "src.assembled_core.ops.kpi_artifacts.compute_exposure_multiplier",
            lambda *a, **k: 1.0,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = write_run_kpis(d, Ctx(), Result(), {}, "paper")
            data = __import__("json").loads(path.read_text(encoding="utf-8"))
        assert "intel_orchestration" in data
        assert data["intel_orchestration"]["news"]["status"] == "OK"
        assert data["intel_orchestration"]["disclosures"]["status"] == "SKIPPED"


from contextlib import contextmanager


@contextmanager
def temp_dir():
    import tempfile

    d = tempfile.mkdtemp()
    try:
        yield Path(d)
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
